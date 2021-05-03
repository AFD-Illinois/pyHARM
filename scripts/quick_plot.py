################################################################################
#                                                                              #
#  PLOT ONE PRIMITIVE                                                          #
#                                                                              #
################################################################################

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyHARM
import pyHARM.ana.plot as pplt
from pyHARM import pretty
from pyHARM.units import get_units_M87

FIGX = 10
FIGY = 10

# TODO parse these instead of hard-coding
USEARRSPACE = False

if USEARRSPACE:
    window = (0, 1, 0, 1)
else:
    SIZE = 100
    #window = (0, SIZE, 0, SIZE)
    window = (-SIZE, SIZE, -SIZE, SIZE)
    # window=(-SIZE/4, SIZE/4, 0, SIZE)

pdf_window = (-10, 0)

dumpfile = sys.argv[1]
var = sys.argv[2]
# Optionally take extra name, otherwise just set it to var
name = sys.argv[-1]

if len(sys.argv) > 4:
    munit = float(sys.argv[4])
    cgs = get_units_M87(munit)
    print("Uisng M_unit: ", munit)
    unit = cgs[sys.argv[3]]
    print("Will multiply by unit {} with value {}".format(sys.argv[3], unit))
    name = var + "_units"
else:
    unit = 1

dump = pyHARM.load_dump(dumpfile)

if "pdf_" in var:
    fig = plt.figure(figsize=(FIGX, FIGY))
    plt.title(pretty(var))
    d_var, d_var_bins = dump[var]
    plt.plot(d_var_bins[:-1], d_var)
    if "_log_" in var:
        plt.xlabel("Log10 value")
    elif "_ln_" in var:
        plt.xlabel("Ln value")
    else:
        plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.savefig(name+".png", dpi=100)
    plt.close(fig)
    exit() # We already saved the figure, we don't need another

if dump['n3'] > 1:
    fig = plt.figure(figsize=(FIGX, FIGY))
    plt.title(pretty(var))
    # Plot XY
    # Plot vectors in 4-pane layout
    if var in ['jcon', 'jcov', 'ucon', 'ucov', 'bcon', 'bcov']:
        axes = [plt.subplot(2, 2, i) for i in range(1, 5)]
        for n in range(4):
            pplt.plot_xy(axes[n], dump, dump[var][n] * unit, arrayspace=USEARRSPACE, window=window)
    else:
        # TODO allow specifying vmin/max, average from command line or above
        ax = plt.subplot(1, 1, 1)
        pplt.plot_xy(ax, dump, dump[var] * unit, log=False, arrayspace=USEARRSPACE, window=window)

    plt.tight_layout()
    plt.savefig(name + "_xy.png", dpi=100)
    plt.close(fig)

# Plot XZ
fig = plt.figure(figsize=(FIGX, FIGY))
plt.title(pretty(var))

if var in ['jcon', 'jcov', 'ucon', 'ucov', 'bcon', 'bcov']:
    axes = [plt.subplot(2, 2, i) for i in range(1, 5)]
    for n in range(4):
        pplt.plot_xz(axes[n], dump, dump[var][n] * unit, arrayspace=USEARRSPACE, window=window)
else:
    ax = plt.subplot(1, 1, 1)
    pplt.plot_xz(ax, dump, dump[var] * unit, log=False, arrayspace=USEARRSPACE, window=window)

plt.tight_layout()

plt.savefig(name + "_xz.png", dpi=100)
plt.close(fig)
