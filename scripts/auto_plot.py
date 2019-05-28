#!/usr/bin/env python3

import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from pyHARM.ana.plot import pcolormesh_symlog
from pyHARM.ana.results import get_ivar

infile = h5py.File(sys.argv[1])

if not os.path.exists("auto_plots"):
    os.mkdir("auto_plots")

for ivar in [key for key in infile.keys() if key not in ['avg', 'coord', 'diag', 'header', 'pdf', 'pdft']]:
    # Split independent and dependent variable retrieval for efficiency
    ivar_d = get_ivar(infile, ivar, i_xy=True)

    for var in infile[ivar].keys():
        fname = "{}_{}.png".format(ivar, var)
        print("Plotting {}".format(fname))
    
        # A bunch of custom sizes here...
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
        plt.grid(True)
    
        var_d = infile[ivar][var][()]
        if not isinstance(ivar_d, list):
            plt.semilogy(ivar_d, var_d)
            plt.xlabel(ivar)
            plt.ylabel(var)
        else:
            if np.min(var_d) >= 0:
                # Only plot variables with any nonzero elements (hard to log-scale 0)
                vmax = np.nanmax(var_d)
                vmax = min(vmax, np.finfo('d').max)
                if vmax > 0:
                    # Floor to 8 orders of magnitude
                    vmin = min(1, vmax * 1e-8)
                    var_d = np.maximum(var_d, 1.01*vmin)
                    
                    pcm = plt.pcolormesh(*ivar_d, var_d, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap='jet')
                    plt.colorbar(pcm)
            else:
                pcm = pcolormesh_symlog(ax, *ivar_d, var_d)
            if ivar[-1:] == 't':
                plt.xlabel('t')
                if len(ivar) > 1:
                    plt.ylabel(ivar[:-1])
                    plt.title(var)
                else:
                    plt.ylabel(var)

            elif 'r' in ivar and ('phi' in ivar or 'th' in ivar):
                plt.xlabel('x')
                if 'phi' in ivar:
                    plt.ylabel('y')
                elif 'th' in ivar:
                    plt.ylabel('z')
                    fig.set_size_inches((10, 15))
                plt.title(var)

            elif ivar == 'thphi':
                plt.xlabel('th')
                plt.ylabel('phi')
                plt.title(var)
            else:
                plt.xlabel(ivar)
                plt.ylabel(var)

        fpath = os.path.join(os.path.dirname(infile.filename), "auto_plots", fname)
        plt.savefig(fpath, dpi=100)
        plt.close()

infile.close()
