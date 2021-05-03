# Read an ipole trace output file, convert each r,th,phi point to BL, and re-write

import sys
import h5py
import numpy as np

import pyHARM


trace = h5py.File(sys.argv[1], "r")
outf = h5py.File(sys.argv[2], "w")

# r,th are in KS==BL
r = trace['r'][()]
outf['r'] = r
outf['th'] = trace['th'][()]
# convert phi to BL
a = trace['fluid_header/geom/mks/a'][()]
outf['phi'] = trace['phi'][()] - a / (r**2 - 2.*r + a**2) * r

outf.close()