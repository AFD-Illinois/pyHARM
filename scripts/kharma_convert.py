#!/usr/bin/env python3

# This script converts KHARMA output to iharm3d-like IL HDF format,
# specifically a bunch of *.h5 files in a folder called dumps/
# Usage:
# $PATH_TO/kharma_convert.py *.phdf
# Note this places dumps/ and the output in the *current directory* when run.
# If you don't want that it should be simple to change.

import os
import sys
import pyHARM

os.makedirs("dumps", exist_ok=True)

dump_list = sys.argv[1:]

def convert(n):
    dumpname = dump_list[n]
    dump = pyHARM.load_dump(dumpname, add_grid_caches=False)
    hdr = dump.params
    # The zero is dt, which KHARMA does not keep
    pyHARM.io.ilhdf.write_dump(hdr, dump.grid, dump.prims, hdr['t'], 0.0, hdr['n_step'], hdr['n_dump'],
                                "dumps/"+dumpname.replace("dumps_kharma/","").replace(".phdf", ".h5"))


#pyHARM.util.run_parallel(convert, len(dump_list), 40)
for i in range(len(dump_list)):
    convert(i)
