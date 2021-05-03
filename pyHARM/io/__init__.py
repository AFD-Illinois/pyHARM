
# Wrap functions from individual io modules by detecting filenames
from glob import glob

import pyHARM.io.ilhdf as ilhdf
import pyHARM.io.kharma as kharma
import pyHARM.io.harm2d as harm2d

# Also import i/o for other filetypes directly
from .misc import *
from .gridfile import *

def get_fnames(path):
    files = np.sort(glob(os.path.join(path, "*.h5")))
    if len(files) == 0:
        files = np.sort(glob(os.path.join(path, "*out*.phdf")))
    if len(files) == 0:
        files = np.sort(glob(os.path.join(path, "dump*")))
    if len(files) == 0:
        raise FileNotFoundError("No dump files found at {}".format(path))
    return files

def get_filter(fname):
    # Choose an importer based on what we know of file contents, or failing that, names
    # TODO This can be smarter: test for HDF5 format at all, etc etc.
    if ".phdf" in fname:
        return kharma
    elif ".h5" in fname:
        if not 'header' in h5py.File(fname, 'r').keys():
            return kharma
        else:
            return ilhdf
    else:
        # HARM ASCII files are *usually* named "dump" with a number
        return harm2d

def read_dump(fname, params=None):
    """Try to automatically read a dump file by guessing the filetype.
    See individual implementations for more options
    """
    my_filter = get_filter(fname)
    return my_filter.read_dump(fname, params=params)

def read_hdr(fname, params=None):
    """Try to automatically read a dump's header by guessing the filetype.
    See individual implementations for more options
    """
    my_filter = get_filter(fname)
    return my_filter.read_hdr(fname, params=params)

def get_dump_time(fname):
    """Try to automatically read a dump's header by guessing the filetype.
    See individual implementations for more options
    """
    my_filter = get_filter(fname)
    return my_filter.get_dump_time(fname)