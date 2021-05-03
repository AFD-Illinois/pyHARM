#!/usr/bin/env python3

# This script calculates analysis reductions over an entire run, i.e. every zone of every dump file.
# It makes use of multiprocessing, but the effect is limited due to high memory usage

# Usage: python analysis.py /path/to/dump/folder/ [tstart] tavg_start tavg_end [tend]
# All dump_*.h5 files in the path are assumed to be HARM fluid dump files
# tavg_* are the interval in time (r_g/c) over which to average any time-averaged variables
# EHT MAD comparison standard is 6000-10000M for a 10kM run. SANE comparison was 5-10kM
# * t* are the start and end times for all processing -- dumps before/after this will be ignored
# This is useful if multiple copies of analysis.py are being run and the results later merged
# (see merge_ana.py)

# Alternate usage: analysis.py /path/to/dump.h5
# Test analysis over just 1 dump file
# used to make sure it completes successfully/correctly before running at scale.

import os
import sys
from glob import glob
import numpy as np
import h5py

# Suppress runtime math warnings
# We will be dividing by zero, and don't want to hear about it
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# For memory usage stats
import psutil

import pyHARM
# Spam base namespace so we don't have to type pyHARM or even ph
from pyHARM.ana.variables import *
from pyHARM.ana.reductions import *

import pyHARM.io as io
import pyHARM.util as util

# Specific useful things 
from pyHARM.defs import Loci
from pyHARM.util import i_of

# Whether to augment or replace existing results
# Augmenting behaves *very* badly when jobs time out or are cancelled
# TODO add more file checks before appending to make augmenting more reliable
augment = False

# Whether to calculate each set of variables
# Once performed once, calculations will be ported to each new output file
calc_basic = True # Fluxes at horizon, often needed before movies, as basic check, etc.
calc_ravgs = True # Radial averages in the disk for code comparisons
calc_efluxes = False # Fluxes in places other than the horizon, to judge infall equilibrium etc.

# Stuff written specifically for the MAD code comparison
# A little long but useful
calc_madcc = False
calc_madcc_optional = False
# Field fluxes away from EH
calc_phi = False

# Maxima/minima over the grid that might prove useful diagnostics
calc_diagnostics = False

# Specialized calculations
calc_thavgs = False
calc_omega_bz = False
calc_jet_profile = False
calc_jet_cuts = False
calc_lumproxy = False
calc_gridtotals = False
calc_outfluxes = False

calc_pdfs = False
pdf_nbins = 200

params = {}
parallel = True

# This doesn't seem like the _right_ way to do optional args
# Skips everything before tstart, averages between tavg_start and tavg_end
tstart = None
tavg_start = None
tavg_end = None
tend = None
path = sys.argv[1]
if ".h5" not in path:
    dumps = io.get_fnames(path)
    if len(sys.argv) > 5:
        tstart = float(sys.argv[2])
        tavg_start = float(sys.argv[3])
        tavg_end = float(sys.argv[4])
        tend = float(sys.argv[5])
    elif len(sys.argv) > 3:
        tavg_start = float(sys.argv[2])
        tavg_end = float(sys.argv[3])
    else:
        print("Format: python analysis.py {dump_to_analyze.h5|/path/to/dumps/} [tstart] tavg_start tavg_end [tend]")
        sys.exit()
    debug = False
else:
    # Allow debugging new analysis over one dump with minimal arguments
    dumps = [path]
    path = os.path.dirname(path)
    tavg_start = 0
    tavg_end = 1e7
    debug = True

ND = len(dumps)

# Require averaging times as arguments, but default to running over all dumps
if tstart is None:
    tstart = io.get_dump_time(dumps[0])

if tend is None:
    tend = io.get_dump_time(dumps[-1])
if tend == 0.:
    tend = float(ND)

# Get header and geometry stuff from the first dump on the list
dump = pyHARM.load_dump(dumps[0], params=params, calc_derived=False)
hdr = dump.header

if dump['r'].ndim == 3:
    r1d = dump['r'][:, hdr['n2']//2, 0]
elif dump['r'].ndim == 2:
    r1d = dump['r'][:, hdr['n2']//2]
elif dump['r'].ndim == 1:
    r1d = dump['r']

jmin, jmax = get_eht_disk_j_vals(dump)

del dump

# Leave several extra zones if using MKS3 coordinates
if hdr['coordinates'] == "mks3":
    iEH = i_of(r1d, hdr['r_eh']) + 4
else:
    iEH = i_of(r1d, hdr['r_eh'])

# Measure fluxes at event horizon. TODO options here?
iF = iEH

# Max radius when computing "total" energy
iEmax = i_of(r1d, 40)

# BZ luminosity
# 100M seems like the standard measuring spot (or at least, BHAC does it that way)
# L_BZ seems constant* after that, but much higher within ~50M
if hdr['r_out'] < 100 or r1d[-1] < 100:  # If in theory or practice the sim is small...
    if r1d[-1] > 40:
        iBZ = i_of(r1d, 40)  # most SANEs
    else:
        iBZ = len(r1d) - 6 # Otherwise measure it just inside the boundary
else:
    iBZ = i_of(r1d, 100)  # most MADs

print("Running from t={} to {}, averaging from t={} to {}".format(tstart, tend, tavg_start, tavg_end))
print("Using EH at zone {}, Fluxes at zone {}, Emax within zone {}, L_BZ at zone {}".format(iEH, iF, iEmax, iBZ))


def avg_dump(n):
    out = {}

    t = io.get_dump_time(dumps[n])
    # When we don't know times, fudge
    # TODO accept -1 as "Not available" flag in the HDF5 spec
    if t == 0 and n != 0:
        t = 10*n
    # Record
    out['coord/t'] = t

    if t < tstart or t > tend:
        # Still return the time
        return out

    print("Loading {} / {}: t = {}".format((n + 1), len(dumps), int(t)), file=sys.stderr)
    # TODO Add only what we need here...
    dump = pyHARM.load_dump(dumps[n], params=params, calc_derived=True, add_jcon=True, add_fails=True, add_floors=True)

    # Should we compute the time-averaged quantities?
    do_tavgs = (tavg_start <= t <= tavg_end)

    # EHT Radial profiles: Average only over the disk portion (excluding first & last pi/3 ~ "poles")
    if calc_ravgs:
        for var in ['rho', 'Pg', 'u^r', 'u^th', 'u^3', 'b^r', 'b^th', 'b^3', 'b', 'betainv', 'Ptot']:
            out['rt/' + var] = shell_avg(dump, var, j_slice=(jmin, jmax))
            out['rt/' + var + '_notdisk'] = shell_avg(dump, var, j_slice=(0, jmin)) + \
                                        shell_avg(dump, var, j_slice=(jmax, dump.header['n2']))
            if do_tavgs:
                out['r/' + var] = out['rt/' + var]
                out['r/' + var + '_notdisk'] = out['rt/' + var + '_notdisk']

    if calc_thavgs:
        if do_tavgs:
            # THETA AVERAGES
            for var in ['betainv', 'sigma']:
                out['th/' + var + '_25'] = theta_av(dump, var, i_of(r1d, 25), 5, fold=False)

    if calc_basic:
        # FIELD STRENGTHS
        # The HARM B_unit is sqrt(4pi)*c*sqrt(rho), and this is standard for EHT comparisons
        out['t/Phi_b'] = 0.5 * shell_sum(dump, np.fabs(dump['B1']), at_zone=iEH)

        # FLUXES
        # Radial profiles of Mdot and Edot, and their particular values
        # EHT code-comparison normalization has all these values positive
        for var, flux in [['Edot', 'FE'], ['Mdot', 'FM'], ['Ldot', 'FL']]:
            out['rt/' + flux] = shell_sum(dump, flux)
            if do_tavgs:
                out['r/' + flux] = shell_sum(dump, flux)
            out['t/'+var] = shell_sum(dump, flux, at_zone=iF)
        # Mdot and Edot are defined inward/positive at EH
        out['t/Mdot'] *= -1
        out['t/Edot'] *= -1

    if calc_diagnostics:
        # Maxima (for gauging floors)
        for var in ['sigma', 'betainv', 'Theta', 'U']:
            out['t/' + var + '_max'] = np.max(dump[var])
        # Minima
        for var in ['rho', 'U']:
            out['t/' + var + '_min'] = np.min(dump[var])
        out['rt/total_floors'] = np.sum(dump['floors'] != 0, axis=(1,2))
        out['rt/total_fails'] = np.sum(dump['fails'] != 0, axis=(1,2))

    if calc_phi:
        out['rt/Phi_b_sph'] = 0.5 * shell_sum(dump, np.fabs(dump['B1']))
        out['rt/Phi_b_mid'] = np.zeros_like(out['rt/Phi_b_sph'])
        for i in range(out['rt/Phi_b_mid'].shape[0]):
            out['rt/Phi_b_mid'][i] = midplane_sum(dump, -dump['B2'], r_slice=(0,i))

    if calc_madcc:
        out['rt/thrho'] = (shell_sum(dump, dump['rho']*np.abs(np.pi/2 - dump.grid.coords.th(dump.grid.coord_all()))) /
                           shell_sum(dump, dump['rho']))

        if do_tavgs:
            for var in ['rho', 'u^r', 'u^th', 'u^3', 'b^r', 'b^th', 'b^3', 'b', 'Pg', 'betainv', 'sigma']:
                out['rth/' + var] = dump[var].mean(axis=-1)

    if calc_madcc_optional:
        # Wavelength of fastest MRI mode for calculating suppression factor
        out['rt/lam_MRI'] = (shell_sum(dump, dump['rho']*dump['lam_MRI']) /
                             shell_sum(dump, dump['rho']))

        # Correlation functions at specific radii
        for var in ['rho', 'betainv']:
            out['phit/' + var + '_cf10'] = corr_midplane(dump[var], at_i1=i_of(r1d, 10))
            out['phit/' + var + '_cf20'] = corr_midplane(dump[var], at_i1=i_of(r1d, 20))
            out['phit/' + var + '_cf30'] = corr_midplane(dump[var], at_i1=i_of(r1d, 30))
            out['phit/' + var + '_cf50'] = corr_midplane(dump[var], at_i1=i_of(r1d, 50))

        # Jet profile moments/ellipse
        for w_r in [50, 100]:
            for w_pole, w_slice in [('north', (0, jmin)), ('south', (jmax, dump.header['n2']))]:
                # CM
                out['thphit/jet_psi_'+w_pole+'_'+str(w_r)] = dump['jet_psi'][i_of(r1d, w_r), :, :]
                out['t/M_'+w_pole+'_'+str(w_r)] = M = shell_sum(dump, dump['jet_psi'] * np.cos(dump['th']), j_slice=w_slice, at_r=w_r)
                out['t/X_'+w_pole+'_'+str(w_r)] = X = 1/M * shell_sum(dump, dump['x']*dump['jet_psi'] * np.cos(dump['th']), j_slice=w_slice, at_r=w_r)
                out['t/Y_'+w_pole+'_'+str(w_r)] = Y = 1/M * shell_sum(dump, dump['y']*dump['jet_psi'] * np.cos(dump['th']), j_slice=w_slice, at_r=w_r)
                # Moments
                out['t/Ixx_'+w_pole+'_'+str(w_r)] = shell_sum(dump, (dump['x'] - X)**2 * dump['jet_psi'] * np.cos(dump['th']), j_slice=w_slice, at_r=w_r)
                out['t/Iyy_'+w_pole+'_'+str(w_r)] = shell_sum(dump, (dump['y'] - Y)**2 * dump['jet_psi'] * np.cos(dump['th']), j_slice=w_slice, at_r=w_r)
                out['t/Ixy_'+w_pole+'_'+str(w_r)] = shell_sum(dump, (dump['x'] - X) * (dump['y'] - Y) * dump['jet_psi'] * np.cos(dump['th']), j_slice=w_slice, at_r=w_r)
                del M, X, Y

        if do_tavgs:
            # Full midplane correlation function, time-averaged
            for var in ['rho', 'betainv']:
                out['rphi/' + var + '_cf'] = corr_midplane(dump[var])


    # Polar profiles of different fluxes and variables
    if calc_jet_profile:
        for var in ['rho', 'bsq', 'b^r', 'b^th', 'b^3', 'u^r', 'u^th', 'u^3', 'FM', 'FE', 'FE_EM', 'FE_Fl', 'FL', 'FL_EM', 'FL_Fl', 'betagamma', 'Be_nob', 'Be_b']:
            out['tht/' + var + '_100'] = np.sum(dump[var][iBZ], axis=-1)
            if do_tavgs:
               out['th/' + var + '_100'] = out['tht/' + var + '_100']
               out['thphi/' + var + '_100'] = dump[var][iBZ]
               #out['rth/' + var] = dump[var].mean(axis=-1)

    # Blandford-Znajek Luminosity L_BZ
    # This is a lot of luminosities!
    if calc_jet_cuts:
        # TODO cut on phi/t averages? -- needs 2-pass cut...
        cuts = {'sigma1': lambda dump: (dump['sigma'] > 1),
                'Be_b0': lambda dump: (dump['Be_b'] > 0.02),
                'Be_b1': lambda dump: (dump['Be_b'] > 1),
                'Be_nob0': lambda dump: (dump['Be_nob'] > 0.02),
                'Be_nob1': lambda dump: (dump['Be_nob'] > 1),
                # 'mu1' : lambda dump : (dump['mu'] > 1),
                'bg1': lambda dump: (dump['betagamma'] > 1.0),
                'bg05': lambda dump: (dump['betagamma'] > 0.5),
                'allp': lambda dump: (dump['FE'] > 0)}

        # Terminology:
        # LBZ = E&M energy only, any cut
        # Lj = full E flux, any cut
        # Ltot = Lj_allp = full luminosity wherever it is positive
        for lum, flux in [['LBZ', 'FE_EM'], ['Lj', 'FE']]:
            for cut in cuts.keys():
                out['rt/' + lum + '_' + cut] = shell_sum(dump, flux, mask=cuts[cut](dump))
                out['t/' + lum + '_' + cut] = out['rt/' + lum + '_' + cut][iBZ]
                if do_tavgs:
                    out['r/' + lum + '_' + cut] = out['rt/' + lum + '_' + cut]

    else:
        # Use the default cut from Paper V
        # These are the powers for the MADCC
        is_jet = dump['Be_b'] > 1
        for lum, flux in [['Mdot_jet', 'FM'], ['P_jet', 'FE'], ['P_EM_jet', 'FE_EM'], ['P_PAKE_jet', 'FE_PAKE'], ['P_EN_jet', 'FE_EN'], ['Area_jet', '1']]:
            out['rt/' + lum] = shell_sum(dump, flux, mask=is_jet)
        for lum, flux in [['Area_mag', '1']]:
            out['rt/' + lum] = shell_sum(dump, flux, mask=(dump['sigma'] > 1))
        for var in ['rho', 'Pg', 'u^r', 'u^th', 'u^3', 'b^r', 'b^th', 'b^3', 'b', 'betainv', 'Ptot']:
            out['rt/' + var + '_jet'] = shell_avg(dump, var, mask=is_jet)
        del is_jet
        

    if calc_lumproxy:
        rho, Pg, B = dump['rho'], dump['Pg'], dump['b']
        # See EHT code comparison paper
        j = rho ** 3 / Pg ** 2 * np.exp(-0.2 * (rho ** 2 / (B * Pg ** 2)) ** (1. / 3.))
        out['rt/Lum'] = shell_sum(dump, j, j_slice=(jmin, jmax))

    if calc_gridtotals:
        # Total energy and current, summed by shells to allow cuts on radius
        for tot_name, var_name in [['Etot', 'JE0'], ['Jsq_inv', 'jsq'], ['Jsq_loc', 'current']]:
            out['rt/'+tot_name] = shell_sum(dump, var_name)

    if calc_efluxes:
        # Conserved (maybe; in steady state) 2D energy flux
        for var in ['JE0', 'JE1', 'JE2']:
            out['rt/'+var] = shell_sum(dump, var)
            if do_tavgs:
                out['rth/' + var] = dump[var].mean(axis=-1)

    # Total outflowing portions of variables
    if calc_outfluxes:
        for name, var in [['outflow', 'FM'], ['outEflow', 'FE']]:
            var_tmp = dump[var]
            out['rt/'+name] = shell_sum(dump, var_tmp, mask=(var_tmp > 0))
            if do_tavgs:
                out['r/'+name] = out['rt/'+name]

    if calc_pdfs:
        for var, pdf_range in [ ['betainv', [-3.5, 3.5]], ['rho', [-7, 1]] ]:
            # TODO handle negatives, pass on the range & bins
            var_tmp = np.log10(dump[var])
            out['pdft/' + var], _ = np.histogram(var_tmp, bins=pdf_nbins, range=pdf_range,
                                                 weights=np.repeat(dump['gdet'], var_tmp.shape[2]).reshape(var_tmp.shape),
                                                 density=True)
            del var_tmp

    if calc_omega_bz and do_tavgs:
        Fcov01, Fcov13 = Fcov(dump, 0, 1), Fcov(dump, 1, 3)
        Fcov02, Fcov23 = Fcov(dump, 0, 2), Fcov(dump, 2, 3)
        vr, vth, vphi = dump['u^1']/dump['u^0'], dump['u^2']/dump['u^0'], dump['u^3']/dump['u^0']
        out['rhth/omega'] = np.zeros((hdr['n1'],hdr['n2']//2))
        out['rhth/omega_alt_num'] = np.zeros((hdr['n1'],hdr['n2']//2))
        out['rhth/omega_alt_den'] = np.zeros((hdr['n1'],hdr['n2']//2))
        out['rhth/omega_alt'] = np.zeros((hdr['n1'],hdr['n2']//2))
        out['rhth/vphi'] = np.zeros((hdr['n1'],hdr['n2']//2))
        out['rhth/F13'] = np.zeros((hdr['n1'],hdr['n2']//2))
        out['rhth/F01'] = np.zeros((hdr['n1'],hdr['n2']//2))
        out['rhth/F23'] = np.zeros((hdr['n1'],hdr['n2']//2))
        out['rhth/F02'] = np.zeros((hdr['n1'],hdr['n2']//2))
        coord_hth = dump.grid.coord_all()[:,:,:hdr['n2']//2,0]
        alpha_over_omega =  dump.grid.lapse[Loci.CENT.value, :, :hdr['n2']//2] / (hdr['r_eh'] * np.sin(dump.grid.coords.th(coord_hth)))
        for i in range(hdr['n1']):
            out['rhth/F01'][i] = theta_av(dump, Fcov01, i, 1)
            out['rhth/F13'][i] = theta_av(dump, Fcov13, i, 1)
            out['rhth/F02'][i] = theta_av(dump, Fcov02, i, 1)
            out['rhth/F23'][i] = theta_av(dump, Fcov23, i, 1)
            out['rhth/omega'][i] =  out['rhth/F01'][i] / out['rhth/F13'][i]
            out['rhth/omega_alt_num'][i] = theta_av(dump, vr * dump['B3']*dump['B2'] + vth * dump['B3']*dump['B1'], i, 1)
            out['rhth/omega_alt_den'][i] = theta_av(dump, dump['B2']*dump['B1'], i, 1)
            out['rhth/omega_alt'][i] = theta_av(dump, vr * dump['B3']/dump['B1'] + vth * dump['B3']/dump['B2'], i, 1)
            out['rhth/vphi'][i] = theta_av(dump, vphi, i, 1)

        out['rhth/omega_alt'] *= -alpha_over_omega

        del Fcov01, Fcov13, vr, vth, vphi

    this_process = psutil.Process(os.getpid())
    print("Memory use: {} GB".format(this_process.memory_info().rss / 10**9))

    del dump

    return out


def merge_dict(n, out, out_full):
    # Merge the output dicts, translate ending tags from above into HDF5 groups for easier merge/read
    for key in list(out.keys()):
        tag = key.split('/')[0]
        if key not in out_full:
            # Add the destination ndarray if not present
            # TODO this can probably be reduced to zeros(ND+out[key[-1]].shape)
            if tag == 'rt':
                out_full[key] = np.zeros((ND, hdr['n1']))
            elif tag == 'htht':
                out_full[key] = np.zeros((ND, hdr['n2'] // 2))
            elif tag == 'tht':
                out_full[key] = np.zeros((ND, hdr['n2']))
            elif tag == 'phit':
                out_full[key] = np.zeros((ND, hdr['n3']))
            elif tag == 'rtht':
                out_full[key] = np.zeros((ND, hdr['n1'], hdr['n2']))
            elif tag == 'thphit':
                out_full[key] = np.zeros((ND, hdr['n2'], hdr['n3']))
            elif tag == 'rphit':
                out_full[key] = np.zeros((ND, hdr['n1'], hdr['n3']))
            elif tag == 'pdft':
                out_full[key] = np.zeros((ND, pdf_nbins))
            elif tag in ['r', 'hth', 'rhth', 'th', 'phi', 'rth', 'rphi', 'thphi', 'pdf']:
                out_full[key] = np.zeros_like(out[key])
            else:
                out_full[key] = np.zeros(ND)
        # Average the averaged tags, slot in the time-dep tags
        if tag in ['r', 'hth', 'th', 'phi', 'rth', 'rhth', 'rphi', 'thphi', 'pdf']:
            # Weight the average correctly for _us_.  Full weighting will be done on merge w/the key 'avg/w'
            if my_avg_range > 0:
                try:
                    out_full[key][()] += out[key] / my_avg_range
                except TypeError as e:
                    print("Encountered error when updating {}: {}".format(key, e))
        else:
            try:
                if ND > 1:
                    out_full[key][n] = out[key]
                else:
                    # Array created above will only have 1D
                    out_full[key][:] = out[key]
            except TypeError as e:
                print("Encountered error when updating {}: {}".format(key, e))


# TODO this, properly, some other day
if ND < 200:
    nstart, nmin, nmax, nend = 0, 0, ND - 1, ND - 1
elif ND < 300:
    nstart, nmin, nmax, nend = 0, ND // 2, ND - 1, ND - 1
else:
    nstart, nmin, nmax, nend = int(tstart) // 5, int(tavg_start) // 5, int(tavg_end) // 5, int(tend) // 5

full_avg_range = nmax - nmin

if nmin < nstart: nmin = nstart
if nmin > nend: nmin = nend
if nmax < nstart: nmax = nstart
if nmax > nend: nmax = nend

my_avg_range = nmax - nmin

# If we're testing over just 1 dump, keep radial "averages" for reference
if full_avg_range < 1:
    full_avg_range = 1
    my_avg_range = 1

print("nstart = {}, nmin = {}, nmax = {} nend = {}".format(nstart, nmin, nmax, nend))

# Deduce the name of the output file
if tstart > 0 or tend < 10000:
    outfname = "eht_out_{:08d}_{:08d}.h5".format(int(tstart), int(tend))
else:
    outfname = "eht_out.h5"

# Open the output file -- note we keep it open the *whole time*, as we write each dump's
# results ~immediately to save memory. (TODO could we write per-process with a lock?)
# Delete the current one unless we're specially "augmenting" current results to save time
if not augment:
    outf = h5py.File(outfname, 'w')
    print("Replacing existing output: {}".format(outfname))
else:
    try:
        # This will append if file exists, otherwise create
        outf = h5py.File(outfname, 'a')
    except OSError:
        # If it exists but isn't valid HDF5, there's no saving it
        # Truncate it instead
        outf = h5py.File(outfname, 'w')
        print("Replacing existing output: {}".format(outfname))

inf = h5py.File(dumps[0],'r')
if 'header' in inf:
    hdr_preserve = io.hdr.hdf5_to_dict(inf['header'])
    if not 'header' in outf:
        outf.create_group('header')
    io.hdr.dict_to_hdf5(hdr_preserve, outf['header'])
inf.close()

# Fill the output dict with all per-dump or averaged stuff
# Hopefully in a way that doesn't keep too much of it around in memory
if parallel:
    nthreads = util.calc_nthreads(hdr, n_mkl=16, pad=0.2)
    #nthreads = 5
    util.iter_parallel(avg_dump, merge_dict, outf, ND, nthreads)
else:
    for n in range(ND):
        out = avg_dump(n)
        merge_dict(n, out, outf)

# Toss in anything else we want to keep, including all the diagnostics
vars = {'avg/start': tavg_start,
        'avg/end': tavg_end,
        'avg/w': my_avg_range / full_avg_range}
diag = io.load_log(path)
# Move diags into a subfolder
if diag is not None:
    for key in diag:
        vars['diag/'+key] = diag[key]

for key in vars:
    if key not in outf:
        outf[key] = vars[key]
    else:
        try:
            outf[key][()] = vars[key]
        except TypeError as e:
            print("Error adding diag {} from HARM log to outfile: {}".format(key, e))

print("Merge operation will weight averages by {}".format(outf["avg/w"][()]))

outf.close()
