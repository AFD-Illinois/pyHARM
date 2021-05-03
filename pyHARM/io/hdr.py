
import numpy as np
import h5py

from pyHARM.grid import Grid
import pyHARM.parameters as parameters

# FORMAT SPEC
# Constants corresponding to the HARM format as written in HDF5.  Useful for checking compliance,
# and writing files correctly
# Keys in the base directory
base_keys = ['t', 'dt', 'dump_cadence', 'full_dump_cadence', 'is_full_dump', 'n_dump', 'n_step']
# Keys that should be in the header:
header_keys = ['cour', 'gam', 'tf', #'fel0', 'gam_e', 'gam_p', 'tptemax', 'tptemin', # when we do e- later
               'gridfile', 'metric', 'reconstruction', 'version', 'prim_names',
               'n1', 'n2', 'n3', 'n_prim', 'n_prims_passive']
               #'has_electrons', 'has_radiation']
# Keys in header/geom
geom_keys = ['dx1', 'dx2', 'dx3', 'startx1', 'startx2', 'startx3', 'n_dim']
# Keys in potential geom/mks
mks_keys = ['r_eh', 'r_in', 'r_out', 'a', 'hslope']
mmks_keys = ['poly_alpha', 'poly_xt']
fmks_keys = ['mks_smooth', 'poly_alpha', 'poly_xt']
# A generally useful set of translations: names of pyHARM parameters,
# with their counterparts in the HDF5 format
# TODO standardize the last one, at least
translations = {'n1': 'n1tot', 'n2': 'n2tot', 'n3': 'n3tot', 'metric': 'coordinates'}
# TODO Parthenon translations/format here too


def write_hdr(params, outf):
    _write_param_grp(params, header_keys, 'header', outf)
    
    G = Grid(params)
    geom_params = {'dx1': G.dx[1],
                   'dx2': G.dx[2],
                   'dx3': G.dx[3],
                   'startx1': G.startx[1],
                   'startx2': G.startx[2],
                   'startx3': G.startx[3],
                   'n_dim': 4}

    _write_param_grp(geom_params, geom_keys, 'geom', outf['header'])


    if params['coordinates'] in ["cartesian", "minkowski"]:
        # No special geometry to record
        pass
    elif params['coordinates'] == "mks":
        _write_param_grp(params, mks_keys, 'mks', outf['header/geom'])
    elif params['coordinates'] == "fmks":
        _write_param_grp(params, mks_keys+fmks_keys, 'fmks', outf['header/geom'])
        # FOR NOW: Duplicate into "mmks" header because codes expect things there
        _write_param_grp(params, mks_keys+fmks_keys, 'mmks', outf['header/geom'])
    elif params['coordinates'] == "mmks":
        _write_param_grp(params, mks_keys+mmks_keys, 'mmks', outf['header/geom'])
    else:
        raise NotImplementedError("Fluid dump files in {} coordinates not implemented!".format(params['coordinates']))

    # Write everything except the pointers to an archival copy
    if "extras" not in outf:
        outf.create_group("extras")
    if "extras/pyharm_params" not in outf:
        outf['extras'].create_group("pyharm_params")
    for key in [p for p in params if p not in ['ctx', 'queue']]:
        _write_value(outf, params[key], 'header/'+key)

def read_hdr(grp, params=None):
    # Handle lots of different calling conventions
    if isinstance(grp, str):
        if grp[-5:] == ".phdf":
            return parameters.parse_parthenon_dat(grp.split("/")[-1].split(".")[0]+".par")
        fil = h5py.File(grp, "r")
        grp = fil['header']
        close_file = True
    elif 'header' in grp:
        grp = grp['header']
        close_file = False
    else:
        close_file = False
    
    if params is None:
        params = {}
    try:
        # Scoop all the keys that are data, leave the sub-groups
        for key in [key for key in list(grp) if isinstance(grp[key], h5py.Dataset)]:
            params[key] = grp[key][()]

        # The 'geom' group should contain at most one more layer of sub-group
        geom = grp['geom']
        for key in geom:
            if isinstance(geom[key], h5py.Dataset):
                params[key] = geom[key][()]
            else:
                for sub_key in geom[key]:
                    params[sub_key] = geom[key+'/'+sub_key][()]

    except KeyError as e:
        print("Warning: {}".format(e))
        print("File is older than supported, but pyHARM will attempt to continue".format(e))


    # Fix up all the nasty non-Unicode strings
    _decode_all(params)

    # EXTRAS AND WORKAROUNDS
    # Turn the version string into components
    if 'version' not in params:
        params['version'] = "iharm-alpha-3.6"
        print("Unknown version: defaulting to {}".format(params['version']))

    params['codename'], params['codestatus'], params['vnum'] = params['version'].split("-")
    params['vnum'] = [int(x) for x in params['vnum'].split(".")]

    # iharm3d-specific workarounds:
    if params['codename'] == "iharm":
        # Work around naming bug before output v3.4
        if params['vnum'] < [3, 4]:
            names = []
            for name in params['prim_names'][0]:
                names.append(name)
            params['prim_names'] = names

        # Work around bad radius names before output v3.6
        if ('r_in' not in params) and ('Rin' in params):
            params['r_in'], params['r_out'] = params['Rin'], params['Rout']

        # Grab the git revision if that's something we output
        if 'extras' in grp.parent and 'git_version' in grp.parent['extras']:
            params['git_version'] = grp.parent['/extras/git_version'][()].decode('UTF-8')

    # Renames we've done for pyHARM  or are useful
    # params_key -> likely name in existing header
    # param_key -> desired name in conformant/pyHARM header
    for params_key in translations:
        param_key = translations[params_key]
        if params_key in params:
            if isinstance(params[params_key], str):
                params[param_key] = params[params_key].lower()
                # iharm3d called FMKS (radial dependence) by the name MMKS (which we use for cylindrification only)
                if params[param_key] == "mmks":
                    params[param_key] = "fmks"
            else:
                params[param_key] = params[params_key]

    if close_file:
        fil.close()

    return parameters._fix(params)


def hdf5_to_dict(h5grp):
    """Recursively load group contents into nested dictionaries.
    Used to load analysis output while keeping shapes straight
    """
    do_close = False
    if isinstance(h5grp, str):
        h5grp = h5py.File(h5grp, "r")
        do_close = True

    ans = {}
    for key, item in h5grp.items():
        if isinstance(item, h5py._hl.group.Group):
            # Call recursively
            ans[key] = hdf5_to_dict(h5grp[key])
        elif isinstance(item, h5py._hl.dataset.Dataset):
            # Otherwise read the dataset
            ans[key] = item[()]

    if do_close:
        h5grp.close()

    # This runs the un-bytes-ing too much, but somehow not enough
    _decode_all(ans)
    return ans


def dict_to_hdf5(wdict, h5grp):
    """Write nested dictionaries of Python values to HDF5 groups nested within the group/file h5grp
    If a filename is specified, automatically opens/closes the file.
    """
    do_close = False
    if isinstance(h5grp, str):
        h5grp = h5py.File(h5grp, "r+")
        do_close = True

    for key, item in wdict.items():
        if isinstance(item, dict):
            # Call recursively
            if not key in h5grp:
                h5grp.create_group(key)
            dict_to_hdf5(wdict[key], h5grp[key])
        else:
            # Otherwise write the value to a dataset
            _write_value(h5grp, wdict[key], key)

    if do_close:
        h5grp.close()


def _decode_all(bytes_dict):
    """HDF5 strings are read as bytestrings.  This converts nested dicts of bytestrings to native UTF-8"""
    for key in bytes_dict:
        # Decode bytes/numpy bytes
        if isinstance(bytes_dict[key], (bytes, np.bytes_)):
            bytes_dict[key] = bytes_dict[key].decode('UTF-8')
        # Split ndarray of bytes into list of strings
        elif isinstance(bytes_dict[key], np.ndarray):
            if bytes_dict[key].dtype.kind == 'S':
                bytes_dict[key] = [el.decode('UTF-8') for el in bytes_dict[key]]
        # Recurse for any subfolders
        elif isinstance(bytes_dict[key], dict):
            _decode_all(bytes_dict[key])

    return bytes_dict

def _write_value(outf, value, name):
    """Write a single value to HDF5 file outf, automatically converting Python3 strings & lists"""
    if isinstance(value, list):
        if isinstance(value[0], str):
            load = [np.array(n.upper().encode("ascii", "ignore"), dtype='S20') for n in value]
        else:
            load = value
    elif isinstance(value, str):
        load = np.array(value.upper().encode("ascii", "ignore"), dtype='S20')
    else:
        load = value

    # If key exists just overwrite it
    if name not in outf:
        outf[name] = load
    else:
        outf[name][()] = load

def _write_param_grp(params, key_list, name, parent):
    if not name in parent:
        parent.create_group(name)
    outgrp = parent[name]
    for key in key_list:
        if key in translations.keys():
            _write_value(outgrp, params[translations[key]], key)
        elif key in params:
            _write_value(outgrp, params[key], key)
        else:
            print("WARNING: Format specifies writing key {} to {}, but not present!".format(key, outgrp.name))