#!/usr/bin/env python3
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
#mpl.rcParams['font.size']=14
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

# Compute dry drift (mean intensity in each distance transform bin)
nzMask=imageLn>np.min(imageLn)+1.0e-4
dt=scipy.ndimage.morphology.distance_transform_edt(nzMask)

dtRange=int(np.max(dt))
dtBinMeans=np.zeros((dtRange))
for i in range(dtRange):
    dtBinMeans[i]=np.mean(imageLn[np.absolute(dt-i)<1.0e-3])
dtBinMeans[np.isnan(dtBinMeans)]=0.0

imageDD=np.zeros_like(imageLn)
dtMeanFlat=np.mean(dtBinMeans[-1])
for i in range(dtRange):
    imageDD[dt.astype(np.int64)==i]=dtBinMeans[i]
imageDD[dt>=dtRange]=dtMeanFlat

# Subtract the dry drift from the image before DTCWT, to remove some of the bias in the power estimates
imageLn-=imageDD

# Do / do not use modified wavelets to achieve directional invariance at the cost of
# inaccurate reconstruction.
#transform = dtcwt.Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
transform = dtcwt.Transform2d(biort='near_sym_b', qshift='qshift_b')
dataT = transform.forward(imageLn,nlevels=nLevels)
dataRndT = transform.forward(imageLnRnd,nlevels=nLevels)
coeffs=dataT.highpasses
coeffsRnd=dataRndT.highpasses
ln2Scales=[]

# Adjustment of noise image using fraction of power in each sub-band. 
fracPowInOri=np.empty((nLevels,6))
fracPowInOriRnd=np.empty((nLevels,6))

for iLev in range(nLevels):
        mask=np.sum(np.absolute(dataT.highpasses[iLev][:,:,:]),axis=2)<1.0e-4
        powInLev=np.mean(np.square(np.absolute(dataT.highpasses[iLev][~mask,:])))
        powInLevRnd=np.mean(np.square(np.absolute(dataRndT.highpasses[iLev][:,:,:])))
        for iOri in range(6):
            # If rotationally-varying wavelets are used, bear in mind power
            # will vary due to DTCWT, not just from the image. This is OK
            # because orientations aren't mixed in this approach.

            # Fraction of power in each orientation, for original image
            mask=np.absolute(dataT.highpasses[iLev][:,:,iOri])<1.0e-4
            powInOri=np.mean(np.square(np.absolute(dataT.highpasses[iLev][~mask,iOri])))
            fracPowInOri[iLev,iOri]=powInOri/powInLev

            # Fraction of power in each orientation, for random image
            powInOriRnd=np.mean(np.square(np.absolute(dataRndT.highpasses[iLev][:,:,iOri])))
            fracPowInOriRnd[iLev,iOri]=powInOriRnd/powInLevRnd

            # Power conversion factor actual im -> rnd im
            alpha=fracPowInOri[iLev,iOri]/fracPowInOriRnd[iLev,iOri]
            dataRndT.highpasses[iLev][:,:,iOri]*=alpha#np.sqrt(alpha)

# Invert transform
imageRec=transform.inverse(dataRndT)

## Adjust mean to original image mean
#imRecMean=np.mean(imageRec)
#imageRec-=imRecMean
#meanNZ=np.mean(image[imageLn>np.min(imageLn)+1.0e-4])
#imageRec+=meanNZ

# Show random image and adjusted random image side-by-side
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.subplot(1,2,1)
im=plt.imshow(imageLnRnd,vmin=-3.0,vmax=3.0)
plt.xlabel("x-pixels")
plt.ylabel("y-pixels")
ax=plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb=plt.colorbar(im,cax=cax)
cb.set_label("$\\log_{2}(R)$")
plt.subplot(1,2,2)
im=plt.imshow(imageRec,vmin=-3.0,vmax=3.0)
plt.xlabel("x-pixels")
plt.ylabel("y-pixels")
ax=plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb=plt.colorbar(im,cax=cax)
cb.set_label("$\\log_{2}(R)$")
plt.tight_layout()
plt.show()

