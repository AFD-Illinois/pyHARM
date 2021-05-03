# Parse parameters files and sys.argv

import os
import sys
import numpy as np


def parse_iharm3d_dat(params, fname):
    """Parse the HARM params.dat format to produce a Python dict.
    params.dat format:
    [tag] name = value
    Where tag is in {dbl, float, int, str} corresponding to desired datatype.
    All lines not in this format are ignored, though conventionally comments begin with '# '
    """
    fp = open(fname, "r")
    for line in fp:
        # Trim out trailing newline, anything after '#', stray parentheses or extra spaces
        ls = [token.strip('()') for token in line[:-1].split("#")[0].split(" ") if token != '']
        # And blank lines
        if len(ls) == 0:
            continue
        # Parse according to tag
        if ls[0] in ["[dbl]", "[float]"]:
            params[ls[1]] = float(ls[-1])
        elif ls[0] == "[int]":
            params[ls[1]] = int(ls[-1])
        elif ls[0] == "[str]":
            params[ls[1]] = str(ls[-1])
    return _fix(params)

def parse_parthenon_dat(fname, params=None):
    """Parse the KHARMA params.dat format to produce a Python dict.
    params.dat format:
    <header/subheader>
    name = value
    All lines not in this format are ignored, though conventionally comments begin with '# '
    Note this parser is far less robust than Parthenon's
    """
    if params is None:
        params = {}
    try:
        fp = open(fname, "r")
    except OSError:
        return None

    # Things KHARMA will never use/modify but need to be *something* for IL HDF file header
    params['version'] = "kharma-alpha-0.1"
    params['gridfile'] = "grid.h5"
    params['n_prims_passive'] = 0

    for line in fp:
        # Trim out trailing newline, anything after '#', stray parentheses, headers
        ls = [token.strip().strip('()') for token in line[:-1].split("#")[0].split("<")[0].split("=") if token != '']
        # And blank lines
        if len(ls) == 0:
            continue
        # Parse, assuming float->int->str and taking the largest surviving numbers (to avoid block-specific nxN)
        try:
            if "." in ls[-1]:
                if ls[0] not in params or params[ls[0]] < float(ls[-1]):
                    params[ls[0]] = float(ls[-1])
            else:
                if ls[0] not in params or params[ls[0]] < int(ls[-1]):
                    params[ls[0]] = int(ls[-1])
        except ValueError:
            params[ls[0]] = ls[-1]
    
    # Now do any repairs specific to translating the Parthenon->iharm3d naming scheme
    for pair in (('nx1','n1'), ('nx2','n2'), ('nx3','n3'),
                 ('n1','n1tot'), ('n2','n2tot'), ('n3','n3tot'),
                 ('x1min', 'startx1'), ('x2min', 'startx2'), ('x3min', 'startx3'),
                 ('gamma', 'gam'), ('dt', 'dump_cadence'), ('tlim', 'tf'), ('cfl', 'cour')):
        if (pair[0] in params):
            params[pair[1]] = params[pair[0]]

    if 'x1min' in params:
        params['r_in'] = np.exp(params['x1min'])
        params['r_out'] = np.exp(params['x1max'])
        params['dx1'] = (params['x1max'] - params['x1min'])/params['nx1']
        params['dx2'] = (params['x2max'] - params['x2min'])/params['nx2']
        params['dx3'] = (params['x3max'] - params['x3min'])/params['nx3']

    if "cartesian" in params['base']:
        params['coordinates'] = "cartesian"
    elif "fmks" in params['transform'] or "funky" in params['transform']:
        params['coordinates'] = "fmks"
    elif "mks" in params['transform'] or "modif" in params['transform']:
        params['coordinates'] = "mks"
    else:
        print("Defaulting coordinate system...")
        params['coordinates'] = params['metric'] = "fmks"

    return _fix(params)


def _fix(params):
    # Fix common parameter mistakes
    if 'n_prim' not in params and 'n_prims' in params: # This got messed up *often*
        params['n_prim'] = params['n_prims']

    if ('derefine_poles' in params) and (params['derefine_poles'] == 1):
        params['metric'] = "fmks"
    if 'Rout' in params:
        params['r_out'] = params['Rout']
    if 'electrons' in params and params['electrons'] == 1:
        params['electrons'] = True
        params['n_prim'] = 10 # TODO be less aggressive about this stuff
        params['prim_names'] = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3", "KTOT", "KEL"]
    else:
        params['electrons'] = False
        params['n_prim'] = params['np'] = 8
        params['prim_names'] = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3"]

    # Lots of layers of legacy coordinate specification
    # To be clear the modern way is 'coordiantes' key as one of usually 'fmks','mks','cartesian'
    # The old 'metric' key was uppercase
    if 'metric' not in params and 'coordinates' not in params:
        if ('derefine_poles' in params) and (params['derefine_poles'] == 1):
            params['metric'] = "FMKS"
        else:
            print("Defaulting coordinate system...")
            params['metric'] = "MKS"

    if 'metric' in params and 'coordinates' not in params:
        params['coordinates'] = params['metric'].lower()

    if params['coordinates'] != "cartesian" and 'r_eh' not in params and 'a' in params:
        params['r_eh'] = (1. + np.sqrt(1. - params['a'] ** 2))

    # Metric defaults we're pretty safe in setting
    if not 'n_dim' in params:
        params['n_dim'] = 4
    if params['coordinates'] == "fmks":
        if not 'mks_smooth' in params:
            params['mks_smooth'] = 0.5
        if not 'poly_xt' in params:
            params['poly_xt'] = 0.82
        if not 'poly_alpha' in params:
            params['poly_alpha'] = 14.0
        if 'poly_norm' not in params:
            params['poly_norm'] = 0.5 * np.pi * 1. / (1. + 1. / (params['poly_alpha'] + 1.) *
                                        1. / np.power(params['poly_xt'], params['poly_alpha']))

    return params