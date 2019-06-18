#!/usr/bin/env python2.7
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

import sys
import os
import numpy as np
import matplotlib as mpl
import scipy.ndimage
mpl.rcParams['image.interpolation']='nearest'
mpl.rcParams['font.size']=14
import matplotlib.pyplot as plt

import dtcwt

import colormaps # from dir ../pylib

image=np.genfromtxt(sys.argv[1],delimiter=",")

# compute DTDWT 
ny=image.shape[0]
nx=image.shape[1]
nLevels = int(np.ceil(np.log2(ny)))
imageLn=np.log2(image)

# Use modified wavelets to achieve directional invariance at the cost of
# inaccurate reconstruction
transform = dtcwt.Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
dataT = transform.forward(imageLn,nlevels=nLevels)

# Synthesis
fracPowInOri=np.empty((nLevels-1,6))

for iLev in range(nLevels-1):
    for iOri in range(6):
        # May want to avoid summing zeros but this is not done here.
        powInOri=np.sum(np.square(np.absolute(dataT.highpasses[iLev][:,:,iOri])))
        fracPowInOri[iLev,iOri]=powInOri/np.sum(np.square(np.absolute(dataT.highpasses[iLev][:,:,:])))

plt.subplot(1,2,1)
plt.xlabel("Distance East [km]")
plt.ylabel("Distance North [km]")
plt.imshow(np.log2(image),cmap=plt.get_cmap('viridis'))
cb=plt.colorbar()
cb.ax.set_ylabel("Log rainfall rate [$log_2 mm/h$]")

plt.subplot(1,2,2)
plt.imshow(fracPowInOri,cmap=plt.get_cmap('viridis'))
cb=plt.colorbar()
cb.ax.set_ylabel("Fraction of power in each subband")
plt.xticks([0.0,1.0,2.0,3.0,4.0,5.0],[
    "$+15^{\\circ}$",
    "$+45^{\\circ}$",
    "$+75^{\\circ}$",
    "$+105^{\\circ}$",
    "$+135^{\\circ}$",
    "$+165^{\\circ}$"],rotation=90)
plt.yticks([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0],[1,2,3,4,5,6,7,8])
plt.xlabel("Orientation [$^{\\circ}$ from +ve $x$-axis]")
plt.ylabel("Dyadic scale level")
plt.tight_layout()
plt.show()

# Coefficients correspond to directions: +15, +45, +75, +105, +135, +165.
# Wavelet basis function directions relative to positive x-axis
dirs=[np.pi/12.,3*np.pi/12.,5*np.pi/12.,7*np.pi/12.,9*np.pi/12.,11*np.pi/12.]

# Compute and display directional analysis
imCount=1
for iLev in range(2,nLevels-3):
    stdLev=np.std(np.absolute(dataT.highpasses[iLev][:,:,:]))
    stdOriLev=np.empty((nLevels,6,dataT.highpasses[iLev].shape[0],dataT.highpasses[iLev].shape[1]))
    xav=np.zeros_like(dataT.highpasses[iLev][:,:,0])
    yav=np.zeros_like(dataT.highpasses[iLev][:,:,0])   
    for iOri in range(6):
        stdOriLev[iLev,iOri,:,:]=np.absolute(dataT.highpasses[iLev][:,:,iOri])/stdLev
        ang=2*dirs[iOri] # convert 1/2 circle to circle for purposes of directional average
        xav+=stdOriLev[iLev,iOri]*np.cos(ang)
        yav+=stdOriLev[iLev,iOri]*np.sin(ang)
    xav/=6
    yav/=6
    arg=np.angle(xav+1.0j*yav)
    mod=np.sqrt(np.square(np.absolute(xav))+np.square(np.absolute(yav)))
    arg/=2.0 # go back to half circle
    xavhalf=mod*np.cos(arg)
    yavhalf=mod*np.sin(arg)
    xavhalfsm=scipy.ndimage.gaussian_filter(xavhalf,sigma=2.0)
    yavhalfsm=scipy.ndimage.gaussian_filter(yavhalf,sigma=2.0)
    s=2**(iLev+1)
    if s < 16:
        skip=16/s
        xavhalfsm=xavhalfsm[::skip,::skip]
        yavhalfsm=yavhalfsm[::skip,::skip]
    else:
        skip=1
    plt.subplot(2,2,imCount)
    imCount+=1
    iStep=2**iLev#iStepLen[iLev]
    plt.imshow(imageLn,cmap='viridis')
    ax=plt.gca()
    ax.invert_yaxis()
    zeroAbsVectorCoords=np.where(np.logical_and(np.absolute(xavhalfsm)>1.0e-6,np.absolute(yavhalfsm)>1.0e-6))
    zeroAbsVectorU=xavhalfsm[zeroAbsVectorCoords]
    zeroAbsVectorV=yavhalfsm[zeroAbsVectorCoords]
    plt.quiver(skip*2**(iLev+1)*zeroAbsVectorCoords[1],skip*2**(iLev+1)*zeroAbsVectorCoords[0],zeroAbsVectorU,zeroAbsVectorV,scale=10,color=[1.0,1.0,1.0,0.9])
    plt.quiver(skip*2**(iLev+1)*zeroAbsVectorCoords[1],skip*2**(iLev+1)*zeroAbsVectorCoords[0],-zeroAbsVectorU,-zeroAbsVectorV,scale=10,color=[1.0,1.0,1.0,0.9])
    plt.xlabel("Distance East [km]")
    plt.ylabel("Distance North [km]")
    plt.title("Level " + str(iLev+1))

plt.tight_layout()
plt.show()

