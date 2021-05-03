# Read a HARM output file, and output a file with the zone center locations in BL coordinates

import sys
import h5py
import numpy as np

import pyHARM


dump = pyHARM.load_dump(sys.argv[1])

outf = h5py.File(sys.argv[2], "w")

# Get native coordinates of zone centers,
X = dump.grid.coord_all()
outf['X'] = X
# Convert r,th to KS==BL
r = dump.grid.coords.r(X)
outf['r'] = r
th = dump.grid.coords.th(X)
outf['th'] = th
# convert phi to BL coordinates
outf['phi'] = dump.grid.coords.phi(X) - dump['a'] / (r**2 - 2.*r + dump['a']**2) * r

# 4-vectors
bl = pyHARM.coordinates.BL({'a': dump['a']})
#ks = pyHARM.coordinates.KS({'a': dump['a']})
zero = np.zeros_like(r)
# Transform native->KS,
#gcov_ks = ks.gcov(np.array([zero,r,th,zero]))
ucon_ks = np.einsum("ij...,j...->i...", dump.grid.coords.dxdX(dump.grid.coord_all()), dump['ucon'])
#ucov_ks = dump.grid.dot(ucon_ks, gcov_ks)
#print(dump.grid.dot(ucov_ks, ucon_ks))
# Then KS->BL
#gcov_bl = bl.gcov(np.array([zero,r,th,zero]))
ucon_bl = np.einsum("ij...,j...->i...", bl.dXdx(np.array([zero,r,zero,zero])), ucon_ks)
#ucov_bl = dump.grid.dot(ucon_bl, gcov_bl)
#print(dump.grid.dot(ucov_bl, ucon_bl))
outf['ucon'] = ucon_bl

bcon_ks = dump.grid.dot(dump['bcon'], dump.grid.coords.dxdX(dump.grid.coord_all()))
bcon_bl = np.einsum("ij...,j...->i...", bl.dXdx(np.array([zero,r,zero,zero])), bcon_ks)
#bcov_bl = dump.grid.dot(bcon_bl, gcov_bl)
#print(dump.grid.dot(bcov_bl, bcon_bl))
outf['Bcon'] = bcon_bl

outf.close()