# File I/O.  Should be self-explanatory

import os
import h5py
import numpy as np

# This gets used from wherever, so delay OpenCL errors/dependencies
try:
    import pyopencl.array as cl_array
    from pyHARM.phys import mhd_gamma_calc
except ModuleNotFoundError:
    #print("Loading h5io.py without OpenCL array support.")
    pass

from pyHARM.defs import Loci

# FORMAT SPEC
# Constants corresponding to the HARM format as written in HDF5.  Useful for checking compliance,
# and writing files correctly
# Keys in the base directory
base_keys = ['t', 'dt', 'dump_cadence', 'full_dump_cadence', 'is_full_dump', 'n_dump', 'n_step']
# Keys that should be in the header:
header_keys = ['cour', 'gam', 'tf', #'fel0', 'gam_e', 'gam_p', 'tptemax', 'tptemin', # when we do e- later
               'gridfile', 'metric', 'reconstruction', 'version', 'prim_names',
               'n1', 'n2', 'n3', 'n_prim', 'n_prims_passive']
               #'has_electrons', 'has_radiation']
# Keys in header/geom
geom_keys = ['dx1', 'dx2', 'dx3', 'startx1', 'startx2', 'startx3', 'n_dim']
# Keys in potential geom/mks
mks_keys = ['r_eh', 'r_in', 'r_out', 'a', 'hslope']
fmks_keys = ['mks_smooth', 'poly_alpha', 'poly_xt']
# A generally useful set of translations: names of pyHARM parameters,
# with their counterparts in the HDF5 format
# TODO standardize the last one, at least
translations = {'n1': 'n1tot', 'n2': 'n2tot', 'n3': 'n3tot', 'metric': 'coordinates'}

# Static dump counter (easier than passing from caller...)
n_dump = 1

def write_grid(G, fname="dumps/grid.h5"):
    """Dump a file containing grid zones.
    This will primarily be of archival use soon -- see grid.py, coordinates.py for
    a good way of reconstructing all common grids on the fly.
    """
    outf = h5py.File(fname, "w")
    # Output grid in double precision
    out_type = np.float32

    x = G.coord_bulk(Loci.CENT).reshape(4, G.N[1], G.N[2], G.N[3])
    coords = G.coords

    outf['x'] = coords.cart_x(x).astype(out_type)
    outf['y'] = coords.cart_y(x).astype(out_type)
    outf['z'] = coords.cart_z(x).astype(out_type)
    outf['r'] = coords.r(x).astype(out_type)
    outf['th'] = coords.th(x).astype(out_type)
    outf['phi'] = coords.phi(x).astype(out_type)

    # Native coordinate output
    outf['X1'] = x[1].astype(out_type)
    outf['X2'] = x[2].astype(out_type)
    outf['X3'] = x[3].astype(out_type)

    # Return only the CENT values, repeated over the N3 axis
    if G.NG > 0:
        b = slice(G.NG, -G.NG)
    else:
        b = slice(None, None)
    gcon3 = G.gcon[Loci.CENT.value, :, :, b, b, None].repeat(G.NTOT[3], axis=-1).transpose((2, 3, 4, 0, 1))
    gcov3 = G.gcov[Loci.CENT.value, :, :, b, b, None].repeat(G.NTOT[3], axis=-1).transpose((2, 3, 4, 0, 1))
    gdet3 = G.gdet[Loci.CENT.value, b, b, None].repeat(G.NTOT[3], axis=-1)
    lapse3 = G.lapse[Loci.CENT.value, b, b, None].repeat(G.NTOT[3], axis=-1)

    outf['gcon'] = gcon3.astype(out_type)
    outf['gcov'] = gcov3.astype(out_type)
    outf['gdet'] = gdet3.astype(out_type)
    outf['lapse'] = lapse3.astype(out_type)

    outf.close()


def write_dump(params, G, P, t, dt, n, fname, dump_gamma=True, out_type=np.float32):
    global n_dump
    s = G.slices

    outf = h5py.File(fname, "w")

    write_hdr(G, params, outf)

    # Per-write_dump single variables
    outf['t'] = t
    outf['dt'] = dt
    outf['dump_cadence'] = params['dump_cadence']
    outf['full_dump_cadence'] = params['dump_cadence']
    outf['is_full_dump'] = 1
    outf['n_dump'] = n_dump
    outf['n_step'] = n

    # TODO jcon

    # Arrays corresponding to actual data
    if G.NG > 0:
        outf["prims"] = np.einsum("p...->...p", P[s.allv + s.bulk]).astype(out_type)
    else:
        outf["prims"] = np.einsum("p...->...p", P).astype(out_type)
    if 'queue' in params and dump_gamma:
        P_d = cl_array.to_device(params['queue'], P)
        outf["gamma"] = mhd_gamma_calc(params['queue'], G, P_d, Loci.CENT).astype(out_type).get()[s.bulk]
    elif dump_gamma:
        raise NotImplementedError("Implement gamma calc without OpenCL!")


    # Extra in-situ calculations or custom debugging additions
    if "extras" not in outf:
        outf.create_group("extras")

    outf.close()
    
    n_dump += 1


def read_dump(fname, get_gamma=False, get_jcon=False, zones_first=False, read_type=np.float32):
    """Read the header and primitives from a write_dump.
    No analysis or extra processing is performed
    """
    infile = h5py.File(fname, "r")

    params = read_hdr(infile['/header'])

    # Per-write_dump single variables.  TODO more?
    for key in ['t', 'dt', 'dump_cadence', 'full_dump_cadence']:
        if key in infile:
            params[key] = infile[key][()]

    P = infile["prims"][()]

    out_list = [P, params]

    # For keeping track of which elements' indices we should reverse later
    i_of_jcon = -1

    if get_gamma:
        if "gamma" in infile:
            out_list.append(infile["gamma"][()])
        else:
            print("Requested gamma, but not present in write_dump!")

    if get_jcon:
        if "jcon" in infile:
            out_list.append(infile["jcon"][()])
            i_of_jcon = len(out_list) - 1
        else:
            print("Requested jcon, but not present in write_dump!")
            out_list.append(None)

    # TODO divB? failures?

    infile.close()

    # Reverse indices on P since most pyHARM tooling expects p,i,j,k
    # See iharm_dump for analysis interface that restores i,j,k,p order
    if not zones_first:
        # Switch P, jcon
        switch_list = [0]
        if i_of_jcon > 0:
            switch_list.append(i_of_jcon)

        for el in switch_list:
            out_list[el] = np.einsum("...m->m...", out_list[el])

    # Also upconvert to doubles if necessary
    if read_type != np.float32:
        for i in range(len(out_list)):
            out_list[i] = (out_list[i]).astype(read_type)

    # Return immutable to ensure unpacking
    return tuple(out_list)


# For cutting on time without loading everything
def get_dump_time(fname):
    dfile = h5py.File(fname, 'r')

    if 't' in dfile.keys():
        t = dfile['t'][()]
    else:
        t = 0

    dfile.close()
    return t


def write_checkpoint(params, P, fname):
    outf = h5py.File(fname, "w")
    outf["version"] = "pyHARM-alpha-0.01"

    outf.create_group("params")
    for key in [p for p in params if p not in ['ctx', 'queue']]:
        # Convert strings to ascii for HDF5
        if isinstance(params[key], list):
            outf["params"][key] = [n.encode("ascii", "ignore") for n in params[key]]
        elif isinstance(params[key], str):
            outf["params"][key] = params[key].encode("ascii", "ignore")
        else:
            outf["params"][key] = params[key]
    outf["prims"] = P.get()

    last_restart_link = os.path.join(os.path.dirname(fname), "restart.last")
    if os.path.exists(last_restart_link):
        os.remove(last_restart_link)
    os.symlink(fname, last_restart_link) # TODO doesn't work


def read_checkpoint(params, fname):
    outf = h5py.File(fname, "r")
    for key in outf["params"]:
        params[key] = outf["params"][key][()]
    return cl_array.to_device(params['queue'], outf["prims"][()])  # TODO other stuff

def write_hdf5(outf, value, name):
    """Write a single value to HDF5 file outf, automatically converting Python3 strings & lists"""
    if isinstance(value, list):
        if isinstance(value[0], str):
            load = [n.upper().encode("ascii", "ignore") for n in value]
        else:
            load = value
    elif isinstance(value, str):
        load = value.upper().encode("ascii", "ignore")
    else:
        load = value

    # If key exists just overwrite it
    if name not in outf:
        outf[name] = load
    else:
        outf[name][()] = load

def _write_param_grp(params, key_list, outgrp):
    for key in key_list:
        if key in translations.keys():
            write_hdf5(outgrp, params[translations[key]], key)
        elif key in params:
            write_hdf5(outgrp, params[key], key)
        else:
            print("WARNING: Format specifies writing key {} to {}, but not present!".format(key, outgrp.name))

def write_hdr(G, params, outf):
    if 'header' not in outf:
        outf.create_group("header")
    _write_param_grp(params, header_keys, outf['header'])
    
    geom_params = {'dx1': G.dx[1],
                   'dx2': G.dx[2],
                   'dx3': G.dx[3],
                   'startx1': G.startx[1],
                   'startx2': G.startx[2],
                   'startx3': G.startx[3],
                   'n_dim': 4}
    
    if 'geom' not in outf['header']:
        outf['header'].create_group("geom")
    _write_param_grp(geom_params, geom_keys, outf['header/geom'])


    if params['coordinates'] in ["cartesian", "minkowski"]:
        # No special geometry to record
        pass
    elif params['coordinates'] == "mks":
        if 'mks' not in outf['header/geom']:
            outf['header/geom'].create_group("mks")
        _write_param_grp(params, mks_keys, outf['header/geom/mks'])
    elif params['coordinates'] == "fmks":
        _write_param_grp(params, mks_keys+fmks_keys, outf['header/geom/fmks'])
        # TEMPORARY:
        _write_param_grp(params, mks_keys+fmks_keys, outf['header/geom/mmks'])
    else:
        raise NotImplementedError("Fluid dump files in {} coordinates not implemented!".format(params['coordinates']))

    # Write everything except the pointers to an archival copy
    if "extras" not in outf:
        outf.create_group("extras")
    if "extras/pyharm_params" not in outf:
        outf['extras'].create_group("pyharm_params")
    for key in [p for p in params if p not in ['ctx', 'queue']]:
        write_hdf5(outf, params[key], 'header/'+key)

def read_hdr(grp):
    hdr = {}
    try:
        # Scoop all the keys that are data, leave the sub-groups
        for key in [key for key in list(grp) if isinstance(grp[key], h5py.Dataset)]:
            hdr[key] = grp[key][()]

        # The 'geom' group should contain at most one more layer of sub-group
        geom = grp['geom']
        for key in geom:
            if isinstance(geom[key], h5py.Dataset):
                hdr[key] = geom[key][()]
            else:
                for sub_key in geom[key]:
                    hdr[sub_key] = geom[key+'/'+sub_key][()]

    except KeyError as e:
        raise KeyError("{}: File is corrupted or older than supported by pyHARM.".format(e))

    # Fix up all the nasty non-Unicode strings
    decode_all(hdr)

    # EXTRAS AND WORKAROUNDS
    # Turn the version string into components
    if 'version' not in hdr:
        hdr['version'] = "iharm-alpha-3.6"
        print("Unknown version: defaulting to {}".format(hdr['version']))

    hdr['codename'], hdr['codestatus'], hdr['vnum'] = hdr['version'].split("-")
    hdr['vnum'] = [int(x) for x in hdr['vnum'].split(".")]

    # iharm3d-specific workarounds:
    if hdr['codename'] == "iharm":
        # Work around naming bug before output v3.4
        if hdr['vnum'] < [3, 4]:
            names = []
            for name in hdr['prim_names'][0]:
                names.append(name)
            hdr['prim_names'] = names

        # Work around bad radius names before output v3.6
        if ('r_in' not in hdr) and ('Rin' in hdr):
            hdr['r_in'], hdr['r_out'] = hdr['Rin'], hdr['Rout']

        # Grab the git revision if that's something we output
        if 'extras' in grp.parent and 'git_version' in grp.parent['extras']:
            hdr['git_version'] = grp.parent['/extras/git_version'][()].decode('UTF-8')

    # Renames we've done for pyHARM  or are useful
    # hdr_key -> likely name in existing header
    # param_key -> desired name in conformant/pyHARM header
    for hdr_key in translations:
        param_key = translations[hdr_key]
        if hdr_key in hdr:
            if isinstance(hdr[hdr_key], str):
                hdr[param_key] = hdr[hdr_key].lower()
                # iharm3d called FMKS (radial dependence) by the name MMKS (which we use for cylindrification only)
                if hdr[param_key] == "mmks":
                    hdr[param_key] = "fmks"
            else:
                hdr[param_key] = hdr[hdr_key]

    # Patch things that sometimes people forget to put in the header
    if 'n_dim' not in hdr:
        hdr['n_dim'] = 4
    if 'prim_names' not in hdr:
        if hdr['n_prims'] == 10:
            hdr['prim_names'] = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3", "KEL", "KTOT"]
        else:
            hdr['prim_names'] = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3"]
    if 'has_electrons' not in hdr:
        hdr['has_electrons'] = (hdr['n_prims'] == 10)
    if 'r_eh' not in hdr and "ks" in hdr['coordinates']:
        hdr['r_eh'] = (1. + np.sqrt(1. - hdr['a'] ** 2))
    if 'poly_norm' not in hdr and hdr['coordinates'] in ["mmks", "fmks"]:
        hdr['poly_norm'] = 0.5 * np.pi * 1. / (1. + 1. / (hdr['poly_alpha'] + 1.) *
                                               1. / np.power(hdr['poly_xt'], hdr['poly_alpha']))

    return hdr


def hdf5_to_dict(h5grp):
    """Recursively load group contents into nested dictionaries.
    Used to load analysis output while keeping shapes straight
    """
    do_close = False
    if isinstance(h5grp, str):
        h5grp = h5py.File(h5grp, "r")
        do_close = True

    ans = {}
    for key, item in h5grp.items():
        if isinstance(item, h5py._hl.group.Group):
            # Call recursively
            ans[key] = hdf5_to_dict(h5grp[key])
        elif isinstance(item, h5py._hl.dataset.Dataset):
            # Otherwise read the dataset
            ans[key] = item[()]

    if do_close:
        h5grp.close()

    # This runs the un-bytes-ing too much, but somehow not enough
    decode_all(ans)
    return ans

def dict_to_hdf5(wdict, h5grp):
    """Write nested dictionaries of Python values to HDF5 groups nested within the group/file h5grp
    If a filename is specified, automatically opens/closes the file.
    """
    do_close = False
    if isinstance(h5grp, str):
        h5grp = h5py.File(h5grp, "r+")
        do_close = True

    for key, item in wdict.items():
        if isinstance(item, dict):
            # Call recursively
            h5grp.create_group(key)
            dict_to_hdf5(wdict[key], h5grp[key])
        else:
            # Otherwise write the value to a dataset
            write_hdf5(h5grp, wdict[key], key)

    if do_close:
        h5grp.close()

# Function to recursively un-bytes all the dumb HDF5 strings
def decode_all(bytes_dict):
    for key in bytes_dict:
        # Decode bytes/numpy bytes
        if isinstance(bytes_dict[key], (bytes, np.bytes_)):
            bytes_dict[key] = bytes_dict[key].decode('UTF-8')
        # Split ndarray of bytes into list of strings
        elif isinstance(bytes_dict[key], np.ndarray):
            if bytes_dict[key].dtype.kind == 'S':
                bytes_dict[key] = [el.decode('UTF-8') for el in bytes_dict[key]]
        # Recurse for any subfolders
        elif isinstance(bytes_dict[key], dict):
            decode_all(bytes_dict[key])

    return bytes_dict
