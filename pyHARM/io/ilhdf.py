# File I/O.  Should be self-explanatory

import os
import h5py
import numpy as np

from pyHARM.defs import Loci
from pyHARM.grid import Grid
import pyHARM.parameters as parameters

from .hdr import *

def read_dump(fname, params=None, add_ghosts=False, **kwargs):
    """Read the header and primitives from a write_dump.
    No analysis or extra processing is performed
    @return P, params
    """

    with h5py.File(fname, "r") as infile:
        if params is not None:
            params.update(read_hdr(infile['/header']))
        else:
            params = read_hdr(infile['/header'])

        # Per-write_dump single variables.  TODO more?
        for key in ['t', 'dt', 'n_step', 'n_dump', 'is_full_dump', 'dump_cadence', 'full_dump_cadence']:
            if key in infile:
                params[key] = infile[key][()]

        P = _prep_array(infile['/prims'][()], **kwargs)

    return P, params

# Other readers, let users use as desired
def read_gamma(fname, **kwargs):
    with h5py.File(fname, "r") as infile:
        return _prep_array(infile['/gamma'][()], **kwargs)
def read_jcon(fname, **kwargs):
    with h5py.File(fname, "r") as infile:
        return _prep_array(infile['/jcon'][()], **kwargs)
def read_divb(fname, **kwargs):
    with h5py.File(fname, "r") as infile:
        return _prep_array(infile['/extras/divB'][()], **kwargs)
def read_fail_flags(fname, **kwargs):
    with h5py.File(fname, "r") as infile:
        if 'fail' in infile:
            return _prep_array(infile['/fail'][()], **kwargs)
        else:
            return _prep_array(infile['/extras/fail'][()], **kwargs)
def read_floor_flags(fname, **kwargs):
    with h5py.File(fname, "r") as infile:
        return _prep_array(infile['/extras/fixup'][()], **kwargs)

def _prep_array(arr, zones_first=False, as_double=False, add_ghosts=False):
    """Re-order and optionally up-convert an array from a file,
    to put it in usual pyHARM order/format
    """
    # Reverse indices on vectors, since most pyHARM tooling expects p,i,j,k
    # See iharm_dump for analysis interface that restores i,j,k,p order
    if (not zones_first) and (len(arr.shape) > 3):
        arr = np.einsum("...m->m...", arr)

    # Also upconvert to doubles if necessary
    # TODO could add other types?  Not really necessary yet
    if as_double:
        arr = arr.astype(np.float64)
    
    return arr


# For cutting on time without loading everything
def get_dump_time(fname):
    dfile = h5py.File(fname, 'r')

    if 't' in dfile.keys():
        t = dfile['t'][()]
    else:
        t = 0

    dfile.close()
    return t

# TODO wrapper that takes an IharmDump object
def write_dump(params, G, P, t, dt, n_step, n_dump, fname, out_type=np.float32):
    s = G.slices

    outf = h5py.File(fname, "w")

    write_hdr(params, outf)

    # Per-write_dump single variables
    outf['t'] = t
    outf['dt'] = dt
    outf['dump_cadence'] = params['dump_cadence']
    outf['full_dump_cadence'] = params['dump_cadence']
    outf['is_full_dump'] = 0
    outf['n_dump'] = n_dump
    outf['n_step'] = n_step

    # Arrays corresponding to actual data
    #if G.NG > 0:
    #    outf["prims"] = np.einsum("p...->...p", P[s.allv + s.bulk]).astype(out_type)
    #else:
    outf["prims"] = np.einsum("p...->...p", P).astype(out_type)

    # Extra in-situ calculations or custom debugging additions
    if "extras" not in outf:
        outf.create_group("extras")
    
    # TODO current, flags

    outf.close()