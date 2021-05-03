# Functions for reading old or non-HDF5 files for analysis

import os
import numpy as np


def load_log(logfpath):
    """Try to load an iharm3d text log file at 'logfpath' & return contents as a dict
    returns None on failure
    """
    if not os.path.isfile(logfpath):
        logfpath = os.path.join(logfpath, "log.out")
    if not os.path.exists(logfpath):
        return None
    dfile = np.loadtxt(logfpath).transpose()

    # TODO logs should have a header I read here,
    # and use these defaults on failure
    diag = {}
    diag['t'] = dfile[0]
    diag['rmed'] = dfile[1]
    diag['pp'] = dfile[2]
    diag['e'] = dfile[3]
    diag['uu_rho_gam_cent'] = dfile[4]
    diag['uu_cent'] = dfile[5]
    diag['mdot'] = dfile[6]
    diag['edot'] = dfile[7]
    diag['ldot'] = dfile[8]
    diag['mass'] = dfile[9]
    diag['egas'] = dfile[10]
    diag['Phi'] = dfile[11]
    diag['phi'] = dfile[12]
    diag['jet_EM_flux'] = dfile[13]
    diag['divbmax'] = dfile[14]
    diag['lum_eht'] = dfile[15]
    diag['mdot_eh'] = dfile[16]
    diag['edot_eh'] = dfile[17]
    diag['ldot_eh'] = dfile[18]

    return diag


def log_time(diag, var, t):
    """Get the value of column 'var' in a log dict corresponding to the time 't' in M"""
    # TODO Could use i_of here for flexibility
    if len(diag['t'].shape) < 1:
        return diag[var]
    else:
        i = 0
        while i < len(diag['t']) and diag['t'][i] < t:
            i += 1
        return diag[var][i - 1]
