# Functions for plotting analysis results

import numpy as np
import matplotlib

from pyHARM.ana.results import get_diag, get_result, get_ivar
from pyHARM.ana.variables import pretty

def plot_diag(ax, infile, ivar, var, tline=None,
              ylabel=None, ylim=None, logy=False,
              xlabel=None, xlim=None, logx=False,
              only_nonzero=True, **kwargs):
    """Plot a variable vs time, for movies"""

    # Fetch data if we were just pointed somewhere
    # Keep names for auto-naming axes below
    if isinstance(var, str):
        ivarname = ivar
        varname = var
        # TODO option here
        #ivar, var = get_diag(infile, varname, only_nonzero=only_nonzero, qui=False)
        #print(get_result(infile, ivar, var, only_nonzero=only_nonzero, qui=False))
        ivar, var = get_result(infile, ivar, var, only_nonzero=only_nonzero, qui=False)
    elif isinstance(ivar, str):
        ivarname = ivar
        ivar = get_ivar(infile, ivar)
        varname = None
    else:
        ivarname = None
        varname = None
    

    if ivar is None or var is None:
        print("Not plotting unknown analysis variables: {} as function of {}".format(varname, ivarname))
        return
    
    ax.plot(ivar, var, **kwargs)
    
    # Trace current t on finished plot
    if tline is not None:
        ax.axvline(tline, color='r')

    # Prettify
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif varname is not None:
        ax.set_ylabel(pretty(varname))

    if ylim is not None:
        ax.set_ylim(ylim)

    if logy:
        ax.set_yscale('log')
    
    if xlabel == True and ivarname is not None:
        ax.set_xlabel(pretty(ivarname))
    elif xlabel == False or xlabel is None:
        # Also nix time labels if we're stacking
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(xlabel)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(ivar[0], ivar[-1])

    if logx:
        ax.set_xscale('log')

def plot_t(ax, ivar, var, range=(5000, 10000), label=None, xticks=None):
    """Plot a variable vs time.  Separated because xticks default to false"""
    slc = np.nonzero(var)

    if label is not None:
        ax.plot(ivar[slc], var[slc], label=label)
    else:
        ax.plot(ivar[slc], var[slc])

    ax.set_xlim(range)
    if xticks is None:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(xticks)


def fit(x, y, log=False):
    if log:
        coeffs = np.polyfit(np.log(x), np.log(y), deg=1)
    else:
        coeffs = np.polyfit(x, y, deg=1)

    poly = np.poly1d(coeffs)
    if log:
        yfit = lambda xf: np.exp(poly(np.log(xf)))
    else:
        yfit = poly

    if log:
        fit_lab = r"{:.2g} * r^{:.2g}".format(np.exp(coeffs[1]), coeffs[0])
    else:
        fit_lab = r"{:2g}*x + {:2g}".format(coeffs[0], coeffs[1])

    return x, yfit(x), fit_lab