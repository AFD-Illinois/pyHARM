

import sys
import h5py

import matplotlib.pyplot as plt

from pyHARM.ana.results import *

var = sys.argv[1]
ivar = sys.argv[2]
files = sys.argv[3:]

r_at = 3

plt.figure(figsize=(12,6))

for fil in files:
    name = fil.replace("/eht_out.h5","")
    if 't' in ivar:
        ivr, vr = get_result(fil, ivar, var, mesh=False)
        if len(ivr.shape) > 1:
            iv = ivr[0][:,0]
            v = np.array([vr[i,i_of(ivr[1][0,:], r_at)] for i in range(vr.shape[0])])
        else:
            iv = ivr
            v = vr
    else:
        iv, v = get_result(fil, ivar, var)
    plt.plot(iv, v, label=name)
    plt.xlim((0, 10000))
    print(name,"average 6k+ is: ", np.mean(v[i_of(iv,6000):]))


plt.legend()
plt.savefig("compare_"+var+"_"+ivar+".png")