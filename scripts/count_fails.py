#!/usr/bin/env python3

import sys
import numpy as np
import h5py

nfiles = len(sys.argv) - 1

all_fails = []
for fname in sys.argv[1:]:
    f = h5py.File(fname)
    if 'fail' in f:
        failures = f['fail'][()]
    else:
        failures = f['extras/fail'][()]
    fixups = f['extras/fixup'][()]
    total = f['header/n1'][()]*f['header/n2'][()]*f['header/n3'][()]
    f.close()
    
    failed = np.sum(failures != 0)
    fixedup = np.sum(fixups != 0)
    fail_percent = failed / total * 100
    fixup_percent = fixedup / total * 100

    if nfiles < 100:
        #print("All failure flags: {}".format(np.unique(failures)))
        print("Incoming -rho failures: {}".format(np.sum(failures == -100)))
        print("ITERMAX failures: {}".format(np.sum(failures == 1)))
        print("Negative u^2 failures: {}".format(np.sum(failures == 2)))

        print("Negative rho||U failures: {}".format(np.sum(failures == 5)))
        print("Negative rho failures: {}".format(np.sum(failures == 6)))
        print("Negative U failures: {}".format(np.sum(failures == 7)))
        print("Negative both failures: {}".format(np.sum(failures == 8)))

        print("All floor flags: {}".format(np.unique(fixups)))
        print("Geometric rho floors: {}".format(np.sum(fixups == 1)))
        print("Geometric U floors: {}".format(np.sum(fixups == 2)))
        print("Sigma rho floors: {}".format(np.sum(fixups == 4)))
        print("Sigma U floors: {}".format(np.sum(fixups == 8)))
        print("Temperature max floors: {}".format(np.sum(fixups == 16)))
        print("Gamma max floors: {}".format(np.sum(fixups == 32)))
        print("Entropy max floors: {}".format(np.sum(fixups == 64)))

    print("Total floored zones: {} or {}%".format(fixedup, fixup_percent))
    print("Total failed zones: {} or {}%".format(failed, fail_percent))
    all_fails.append(fail_percent)

print("Max rate: {}%".format(max(all_fails)))
print("Average rate: {}%".format(np.mean(all_fails)))
