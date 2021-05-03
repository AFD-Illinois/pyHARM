#!/usr/bin/env python3

import sys
import h5py
import numpy as np

# TODO can check overlaps, complain on different-sized ND, etc

files = []
for fname in np.sort(sys.argv[1:]):
    try:
        file = h5py.File(fname, "r")
        print("Loading {}".format(fname))
        files.append(file)
    except OSError as e:
        print("Error loading {}: {}".format(fname, e))

maxnkey = 0
for i,file in enumerate(files):
    nkey = sum([ len(file[key]) for key in file.keys() ])
    if nkey > maxnkey:
        master_file = file
        maxnkey = nkey

null_groups = []
copy_groups = ['header', 'diag', 'avg', 'coord', 'extras']
avg_groups = ['r', 'th', 'hth', 'phi', 'rth', 'rhth', 'rphi', 'thphi', 'pdf']
t_groups = ['t', 'rt', 'tht', 'phit', 'rtht', 'thphit', 'rphit', 'pdft']
groups = null_groups + copy_groups + avg_groups + t_groups
# Extra base-level keys, all 1D time-based vars
spare_keys = [key for key in master_file.keys() if key not in groups]

uni = h5py.File("eht_out.h5", "w")
for key in spare_keys:
    uni[key] = np.zeros_like(master_file[key][()])
    for file in [f for f in files if key in f]:
        if uni[key].size < file[key].size:
            uni[key][()] += file[key][:uni[key].size]
        else:
            uni[key][:file[key].size] += file[key]

for grp in [g for g in t_groups if g in master_file]:
    keys = ["/".join([grp, key]) for key in master_file[grp].keys()]
    for key in keys:
        uni[key] = np.zeros_like(master_file[key][()])
        for file in [f for f in files if key in f]:
            if uni[key].shape[0] < file[key].shape[0]:
                uni[key][()] += file[key][:uni[key].shape[0]]
            else:
                uni[key][:file[key].shape[0]] += file[key]

for grp in [g for g in avg_groups if g in master_file]:
    keys = ["/".join([grp, key]) for key in master_file[grp].keys()]
    for key in keys:
        uni[key] = np.zeros_like(master_file[key][()])
        for file in [f for f in files if key in f]:
            uni[key][()] += file[key][()] * file['avg/w'][()]

for grp in [g for g in copy_groups if g in master_file]:
    master_file.copy(grp, uni)

for file in files:
    file.close()
uni.close()
