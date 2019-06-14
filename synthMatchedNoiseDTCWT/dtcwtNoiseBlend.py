#!/bin/env python2.7

# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# uses : DTDWT from https://pypi.python.org/pypi/dtcwt

import numpy as np
import numpy.random
import matplotlib as mpl
mpl.rcParams['image.interpolation']='nearest'
mpl.rcParams['font.size']=14
import scipy.ndimage
import h5py
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clr
import dtcwt

# Load rain image from CSV file and convert to log-R.
image=np.genfromtxt(sys.argv[1],delimiter=",")
imageLn=np.log2(image)
wetFrac=np.sum(image>np.min(image)+1.0e-4)/float(image.size)

# Load random realization from CSV and convert to log-R.
imageRnd=np.genfromtxt(sys.argv[2],delimiter=",")
imageLnRnd=np.log2(imageRnd)

# compute DTDWTs 
ny=image.shape[0]
nx=image.shape[1]
nLevels = int(np.ceil(np.log2(ny)))
# compute DTDWTs 
nyRnd=image.shape[0]
nxRnd=image.shape[1]
nLevelsRnd = int(np.ceil(np.log2(nyRnd)))

# Adjust random image to have the similar mean / std
imageLnRnd-=np.mean(imageLnRnd)
imageLnRnd/=np.std(imageLnRnd)
imageLnRnd*=2*np.std(imageLn[image>np.min(image)+1.0e-4])
imageLnRnd+=np.mean(imageLn[image>np.min(image)+1.0e-4])

# Display rain image and random image side-by-side
plt.subplot(1,2,1)
plt.xlabel("Distance East [km]")
plt.ylabel("Distance North [km]")
plt.imshow(imageLn)
cb=plt.colorbar()
cb.ax.set_ylabel("Log rainfall rate [$log_2 mm/h$]")
plt.subplot(1,2,2)
plt.xlabel("Distance East [km]")
plt.ylabel("Distance North [km]")
plt.imshow(imageLnRnd)
cb=plt.colorbar()
cb.ax.set_ylabel("Log rainfall rate [$log_2 mm/h$]")
plt.show()

# Do / do not use modified wavelets to achieve directional invariance at the cost of
# inaccurate reconstruction. 
#transform = dtcwt.Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
transform = dtcwt.Transform2d(biort='near_sym_b', qshift='qshift_b')
dataT = transform.forward(imageLn,nlevels=nLevels)
dataRndT = transform.forward(imageLnRnd,nlevels=nLevels)
coeffs=dataT.highpasses
coeffsRnd=dataRndT.highpasses
ln2Scales=[]

for iLev in range(nLevels):
    absLev=np.absolute(dataT.highpasses[iLev][:,:,:])
    powLev=np.mean(np.square(absLev)) # possibly want to use only the non-zero here
    powLevRnd=np.mean(np.square(np.absolute(dataRndT.highpasses[iLev][:,:,:])))
    for iOri in range(6):
        fracPowOri=np.square(np.absolute(dataT.highpasses[iLev][:,:,iOri]))/powLev
        fracPowOriRnd=np.square(np.absolute(dataRndT.highpasses[iLev][:,:,iOri]))/powLevRnd
        # Smooth
        fracPowOri=scipy.ndimage.gaussian_filter(fracPowOri,sigma=3.0)
        fracPowOriRnd=scipy.ndimage.gaussian_filter(fracPowOriRnd,sigma=3.0)
        # Adjust
        if iLev < 99: # change this to an integer value in [0,10], to replace with noise only up to that scale
            dataRndT.highpasses[iLev][:,:,iOri]/=fracPowOriRnd
            dataRndT.highpasses[iLev][:,:,iOri]*=fracPowOri
        else:
            dataRndT.highpasses[iLev][:,:,iOri]=dataT.highpasses[iLev][:,:,iOri]

# Overwrite the top-level low-pass coefficients from the random/blended transform with those from the rain image.
dataRndT.lowpass[:,:]=dataT.lowpass[:,:]

# Invert the DTCWT.
imageRec=transform.inverse(dataRndT)

# Display rain image and image with blended noise side-by-side.
plt.subplot(1,2,1)
plt.imshow(imageLn)
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(imageRec,vmin=np.min(imageLn),vmax=np.max(imageLn))
plt.colorbar()
plt.show()

