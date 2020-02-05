#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import h5py

import dtcwt

mpl.rcParams['image.interpolation']='nearest'
mpl.rcParams['font.size'] = 20

inFn1=sys.argv[1]
inFn2=sys.argv[2]

h5FH1=h5py.File(inFn1)
coeffs1=np.copy(h5FH1['coefficients'])
h5FH1.close()
h5FH2=h5py.File(inFn2)
coeffs2=np.copy(h5FH2['coefficients'])
h5FH2.close()
wccs=[]
lambds=[]
for iSc in range(min(coeffs1.shape[0],coeffs2.shape[0])):
    cSub1=coeffs1[iSc,:,:,:,0]+1.j*coeffs1[iSc,:,:,:,1]
    cSub2=coeffs2[iSc,:,:,:,0]+1.j*coeffs2[iSc,:,:,:,1]
    # Compute wavelet cross-correlation over all sub-bands
    # (For general WCC explanation, Addison: Introduction to redundancy rules, 2018, A 29)
    crossTerm=np.absolute(np.sum(np.conj(cSub1)*cSub2))
    cSub1Term=np.sum(np.square(np.absolute(cSub1)))
    cSub2Term=np.sum(np.square(np.absolute(cSub2)))
    wcc=crossTerm/(np.sqrt(cSub1Term*cSub2Term))
    # For Mex hat, wavelength is appx 4*scale
    lambd=(2.0*np.pi/np.sqrt(2.5))*np.power(2.,(iSc-1))
    print(wcc,lambd)
    wccs.append(wcc)
    lambds.append(lambd)

np.savetxt("out_mex_wccs.txt",wccs,delimiter=",")
np.savetxt("out_mex_lambds.txt",lambds,delimiter=",")


