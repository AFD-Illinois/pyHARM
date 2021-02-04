
import numpy as np
import glob
import h5py

import pyHARM.parameters as parameters
from pyHARM.grid import Grid
from .phdf import phdf

def read_hdr(fname, params=None):
    params_was_none = False
    if params is None:
        params = {}
        params_was_none = True

    # If there's just one file in the same directory, it's probably the right one.
    globpath = "/".join(fname.split("/")[:-1] + ["*.par"])
    if len(glob.glob(globpath)) == 1:
        parameters.parse_parthenon_dat(glob.glob(globpath)[0], params)
        return params
    elif parameters.parse_parthenon_dat(fname.split(".")[0] + ".par", params) is not None:
        return params
    elif parameters.parse_parthenon_dat(fname.split("/")[-1].split(".")[0] + ".par", params) is not None:
        return params
    elif params_was_none:
        raise RuntimeError("No parameter file could be found for KHARMA dump {}".format(fname))
    # But if params had elements, assume it was taken care of by the user and don't require a file
    return params

def get_dump_time(fname):
    dfile = h5py.File(fname, 'r')

    if 'Info' in dfile.keys():
        t = dfile['Info'].attrs['Time']
    else:
        t = 0

    dfile.close()
    return t

def read_dump(fname, add_ghosts=False, params=None):
    f = phdf(fname)

    params = read_hdr(fname, params)
    G = Grid(params)

    # First, quit if we've been asked for ghost zones but can't produce them
    if add_ghosts and not f.IncludesGhost:
        raise ValueError("Ghost zones aren't available in file {}".format(fname))

    # This is the number of ghost zones present in the file
    ng_f = f.IncludesGhost * f.NGhost

    if add_ghosts:
        params['ng'] = f.NGhost
        # This is the number we should include on output
        ng_ix = ng_f
    else:
        params['ng'] = 0
        # This is the number we should include on output
        ng_ix = 0

    # Trivial dimensions don't have ghosts
    if params['n2'] == 1:
        ng_iy = 0
    else:
        ng_iy = ng_ix

    if params['n3'] == 1:
        ng_iz = 0
    else:
        ng_iz = ng_ix

    # Set incidental parameters from what we've read
    params['t'] = f.Time
    params['n_step'] = f.NCycle
    params['n_dump'] = int(fname.split("/")[-1].split(".")[2]) # This assumes the usual Parthenon naming
    params['startx1'] = G.startx[1]
    params['startx2'] = G.startx[2]
    params['startx3'] = G.startx[3]
    params['dx1'] = dx = G.dx[1]
    params['dx2'] = dy = G.dx[2]
    params['dx3'] = dz = G.dx[3]

    # Lay out the blocks and determine total mesh size
    bounds = []
    for ib in range(f.NumBlocks):
        bb = f.BlockBounds[ib]
        # Internal location of the block i.e. starting/stopping indices in the final, big mesh
        bound = [int((bb[0]+dx/2 - G.startx[1])/dx)-ng_ix, int((bb[1]+dx/2 - G.startx[1])/dx)+ng_ix,
                 int((bb[2]+dy/2 - G.startx[2])/dy)-ng_iy, int((bb[3]+dy/2 - G.startx[2])/dy)+ng_iy,
                 int((bb[4]+dz/2 - G.startx[3])/dz)-ng_iz, int((bb[5]+dz/2 - G.startx[3])/dz)+ng_iz]
        bounds.append(bound)

    # Optionally allocate enough space for ghost zones
    P = np.zeros((8, G.NTOT[1]+2*ng_ix, G.NTOT[2]+2*ng_iy, G.NTOT[3]+2*ng_iz))

    # Read blocks into their slices of the full mesh
    for ib in range(f.NumBlocks):
        b = bounds[ib]
        # Exclude ghost zones if we don't want them
        if ng_ix == 0 and ng_f != 0:
            if params['n2'] == 1:
                o = [ng_f, -ng_f, None, None, None, None]
            elif params['n3'] == 1:
                o = [ng_f, -ng_f, ng_f, -ng_f, None, None]
            else:
                o = [ng_f, -ng_f, ng_f, -ng_f, ng_f, -ng_f]
        else:
            o = [None, None, None, None, None, None]
        # False == don't flatten into 1D array
        #print("Bound ", b)
        P[:, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = \
            f.Get('c.c.bulk.prims', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1],:].transpose(3,2,1,0)

    return (P, params)

def read_jcon(fname, add_ghosts=False):
    f = phdf(fname)

    params = read_hdr(fname)
    G = Grid(params)

    # First, quit if we've been asked for ghost zones but can't produce them
    if add_ghosts and not f.IncludesGhost:
        raise ValueError("Ghost zones aren't available in file {}".format(fname))

    # This is the number of ghost zones present in the file
    ng_f = f.IncludesGhost * f.NGhost

    if add_ghosts:
        params['ng'] = f.NGhost
        # This is the number we should include on output
        ng_ix = ng_f
    else:
        params['ng'] = 0
        # This is the number we should include on output
        ng_ix = 0

    # Trivial dimensions don't have ghosts
    if params['n2'] == 1:
        ng_iy = 0
    else:
        ng_iy = ng_ix

    if params['n3'] == 1:
        ng_iz = 0
    else:
        ng_iz = ng_ix

    # Lay out the blocks and determine total mesh size
    bounds = []
    for ib in range(f.NumBlocks):
        bb = f.BlockBounds[ib]
        # Internal location of the block i.e. starting/stopping indices in the final, big mesh
        bound = [int((bb[0]+dx/2 - G.startx[1])/dx)-ng_ix, int((bb[1]+dx/2 - G.startx[1])/dx)+ng_ix,
                 int((bb[2]+dy/2 - G.startx[2])/dy)-ng_iy, int((bb[3]+dy/2 - G.startx[2])/dy)+ng_iy,
                 int((bb[4]+dz/2 - G.startx[3])/dz)-ng_iz, int((bb[5]+dz/2 - G.startx[3])/dz)+ng_iz]
        bounds.append(bound)

    # Optionally allocate enough space for ghost zones
    jcon = np.zeros((4, G.NTOT[1]+2*ng_ix, G.NTOT[2]+2*ng_iy, G.NTOT[3]+2*ng_iz))

    # Read blocks into their slices of the full mesh
    for ib in range(f.NumBlocks):
        b = bounds[ib]
        # Exclude ghost zones if we don't want them
        if ng_ix == 0 and ng_f != 0:
            if params['n2'] == 1:
                o = [ng_f, -ng_f, None, None, None, None]
            elif params['n3'] == 1:
                o = [ng_f, -ng_f, ng_f, -ng_f, None, None]
            else:
                o = [ng_f, -ng_f, ng_f, -ng_f, ng_f, -ng_f]
        else:
            o = [None, None, None, None, None, None]
        # False == don't flatten into 1D array
        jcon[:, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = \
            f.Get('c.c.bulk.jcon', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1],:].transpose(3,2,1,0)

    return jcon

# TODO abstract the mesh thing, add c.c.bulk.pflag, c.c.bulk.fflag.  ctop?