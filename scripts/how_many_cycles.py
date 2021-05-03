#!/usr/bin/env python3

# Script to calculate how many zone-cycles are required to complete a simulation
# TODO bonus points if it 

# Usage: ./how_many_cycles.py spin r_out n1 n2 n3 tf

import sys
import pyHARM.grid as grid

spin=float(sys.argv[1])
r_out=float(sys.argv[2])
n1 = int(sys.argv[3])
n2 = int(sys.argv[4])
n3 = int(sys.argv[5])
tf = float(sys.argv[6])
if len(sys.argv) > 7:
    system = sys.argv[7]
else:
    system = 'fmks'

print("Building grid: {}, a = {}, {}x{}x{} to r_out of {}".format(
      system, spin, n1, n2, n3, r_out))

G = grid.make_some_grid(system, n1, n2, n3, a=spin, r_out=r_out)

dt = G.dt_light()

print("Somewhat generous total ZC estimate: {}".format(tf/dt*n1*n2*n3))
