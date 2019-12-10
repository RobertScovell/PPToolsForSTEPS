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
data1=np.genfromtxt(inFn1,delimiter=",")[::-1,:]
data2=np.genfromtxt(inFn2,delimiter=",")[::-1,:]
print("Log transforming input.")
imageLn1=np.log2(data1)
imageLn2=np.log2(data2)
plt.imshow(imageLn1)
plt.show()
plt.imshow(imageLn2)
plt.show()

transform = dtcwt.Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
nLevelsT=int(np.ceil(np.log2(data1.shape[0])))
dataT1 = transform.forward(imageLn1,nlevels=nLevelsT)
dataT2 = transform.forward(imageLn2,nlevels=nLevelsT)

for iSc in range(len(dataT1.highpasses)-1):
    hp1=dataT1.highpasses[iSc]
    hp2=dataT2.highpasses[iSc]
    rAll=[]

    # Compute wavelet cross-correlation over all sub-bands
    # (Addison: Introduction to redundancy rules, 2018, A 29)
    crossTerm=np.absolute(np.sum(np.conj(hp1)*hp2))
    hp1Term=np.sum(np.square(np.absolute(hp1)))
    hp2Term=np.sum(np.square(np.absolute(hp2)))
    wcc=crossTerm/(np.sqrt(hp1Term*hp2Term))
    print(wcc)

