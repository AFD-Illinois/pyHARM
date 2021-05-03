#!/usr/bin/env python3

# Script to write an arbitrary gridfile

import sys
import h5py
import numpy as np
from pyHARM.defs import Loci
from pyHARM.grid import Grid
from pyHARM import load_dump

dname = sys.argv[1]
dump = load_dump(dname, add_derived=False, add_fail=False)
G = dump.grid
coords = G.coords

outf = h5py.File(dname[:-16] + "grid_vis.h5", "w")

# Cell coordinates
x = G.coord_bulk(Loci.CENT).reshape(4, G.N[1], G.N[2], G.N[3])
outf['Xharm'] = x.transpose(1,2,3,0)
outf['Xcart'] = np.array([np.zeros([G.N[1],G.N[2],G.N[3]]), *coords.cart_coord(x)]).transpose(1,2,3,0)
outf['Xbl'] = np.array([np.zeros([G.N[1],G.N[2],G.N[3]]), *coords.ks_coord(x)]).transpose(1,2,3,0)

# Face coordinates
xf = G.coord_bulk_mesh().reshape(4, G.N[1]+1, G.N[2]+1, G.N[3]+1)
outf['XFharm'] = xf.transpose(1,2,3,0)
outf['XFcart'] = np.array([np.zeros([G.N[1]+1,G.N[2]+1,G.N[3]+1]), *coords.cart_coord(xf)]).transpose(1,2,3,0)

# Return only the CENT values, repeated over the N3 axis
if G.NG > 0:
    b = slice(G.NG, -G.NG)
else:
    b = slice(None, None)
gamma = G.conn[:, :, :, b, b, None].repeat(G.NTOT[3], axis=-1).transpose((3, 4, 5, 0, 1, 2))
gcon3 = G.gcon[Loci.CENT.value, :, :, b, b, None].repeat(G.NTOT[3], axis=-1).transpose((2, 3, 4, 0, 1))
gcov3 = G.gcov[Loci.CENT.value, :, :, b, b, None].repeat(G.NTOT[3], axis=-1).transpose((2, 3, 4, 0, 1))
gdet3 = G.gdet[Loci.CENT.value, b, b, None].repeat(G.NTOT[3], axis=-1)
lapse3 = G.lapse[Loci.CENT.value, b, b, None].repeat(G.NTOT[3], axis=-1)

outf['Gamma'] = gamma
outf['gcon'] = gcon3
outf['gcov'] = gcov3
outf['gdet'] = gdet3
outf['alpha'] = lapse3

dxdX = np.einsum("ij...,jk...->...ik", coords.dxdX_cartesian(x), coords.dxdX(x))
outf['Lambda_h2cart_con'] = dxdX
outf['Lambda_h2cart_cov'] = np.linalg.inv(dxdX)

outf.close()

#TODO not used for VisIt but for completeness:
#Lambda_bl2cart_con       Dataset {32, 32, 1, 4, 4}
#Lambda_bl2cart_cov       Dataset {32, 32, 1, 4, 4}
#Lambda_h2bl_con          Dataset {32, 32, 1, 4, 4}
#Lambda_h2bl_cov          Dataset {32, 32, 1, 4, 4}


