# Functions for reading iharm2d_v3 dumps in ASCII format

import numpy as np

from pyHARM.parameters import _fix

"""
Read iharm2d_v3 dumps

This can also serve as a basis for reading ASCII dumps from the other HARM-alikes
"""

def read_hdr(fname, params=None):
    if params is None:
        params = {}

    # Stolen from iharm2d_v3 
    # open data file and get data
    data = np.loadtxt(fname, max_rows=1)

    # first get line 0 header info
    params['t']       = data[0]
    params['n1']      = int(data[1])
    params['n2']      = int(data[2])
    params['startx1'] = data[3]
    params['startx2'] = data[4]
    params['dx1']    = data[5]
    params['dx2']     = data[6]
    params['tf'] 	= data[7]
    params['nstep'] 	= int(data[8])
    params['gam']     = data[9]
    params['cour'] 	= data[10]
    params['DTd']	= data[11]  # dump freq
    params['DTl'] 	= data[12]  # log freq
    params['DTi'] 	= data[13]  # imag freq
    params['DTr'] 	= data[14]  # restart freq
    params['dump_cnt'] = int(data[15])
    params['image_cnt'] = int(data[16])
    params['rdump_cnt'] = int(data[17])
    params['dt']  	= data[18]

    # Then add anything else we need.
    params['startx3'] = 0
    params['dx3'] = 0
    params['n3'] = 1
    # TODO try to read these
    params['coordinates'] = "mks"
    params['r_out'] = np.exp(params['startx1'] + params['n1']*params['dx1'])
    params['a'] = 0.9375
    params['hslope'] = 0.3
    #params['coordinates'] = "cartesian"

    return _fix(params)

def read_dump(fname, params=None, **kwargs):
    """Read the header and primitives from an iharm2d v3 ASCII file.
    No analysis or extra processing is performed
    @return P, params in standard pyHARM
    """
    # Stolen from iharm2d_v3
    # Just fetch the primitives.  iharm2d caches u/b, we could fetch those too but meh.
    params = read_hdr(fname)
    data = np.loadtxt(fname, skiprows=1)
    P 	= data[:,2:10].reshape(params['n1'], params['n2'], params['n3'], params['n_prim']).transpose(3,0,1,2)
    return P, params

def read_jcon(fname, **kwargs):
    params = read_hdr(fname)
    data = np.loadtxt(fname, skiprows=1)
    return data[:,32:36].reshape(params['n1'], params['n2'], params['n3'], 4).transpose(3,0,1,2)

def read_divb(fname, **kwargs):
    params = read_hdr(fname)
    data = np.loadtxt(fname, skiprows=1)
    return data[:,10].reshape(params['n1'], params['n2'], params['n3'])

# For cutting on time without loading everything
def get_dump_time(fname):
    data = np.loadtxt(fname, max_rows=1)
    return data[0]