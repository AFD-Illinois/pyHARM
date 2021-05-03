#!/usr/bin/env python3

# Script to write an arbitrary gridfile

import sys
import h5py
import pyHARM.grid as grid
import pyHARM.io.gridfile as h5io

system=sys.argv[1]
spin=float(sys.argv[2])
r_out=float(sys.argv[3])
if len(sys.argv) > 5:
    n1 = int(sys.argv[4])
    n2 = int(sys.argv[5])
    n3 = int(sys.argv[6])
else:
    n1 = 192
    n2 = 128
    n3 = 128

print("Building grid: {}, a = {}, {}x{}x{} to r_out of {}".format(
      system, spin, n1, n2, n3, r_out))

G = grid.make_some_grid(system, n1, n2, n3, a=spin, r_out=r_out)
h5io.write_grid(G, fname=sys.argv[-1])
