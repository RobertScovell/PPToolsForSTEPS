#!/bin/env python3
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

# Please cite the following paper, if using this code:
# Scovell, R. W. (2020) Applications of Directional Wavelets, Universal Multifractals and Anisotropic Scaling in Ensemble Nowcasting; A Review of Methods with Case Studies. Quarterly Journal of the Royal Meteorological Society. In Press. URL: http://dx.doi.org/abs/10.1002/qj.3780


import numpy as np
import numpy.random
import matplotlib as mpl
mpl.rcParams['image.interpolation']='nearest'
#mpl.rcParams['font.size']=12
import scipy.ndimage
import h5py
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clr
import dtcwt

# Load the random image, convert to log-R units and display
imageRnd=np.genfromtxt(sys.argv[1],delimiter=",")
nyRnd=imageRnd.shape[0]
nxRnd=imageRnd.shape[1]
nLevels = int(np.ceil(np.log2(nyRnd)))
imageLnRnd=np.log2(imageRnd)
plt.imshow(imageLnRnd)
plt.colorbar()
plt.show()

# Use modified wavelets to achieve directional invariance at the cost of
# inaccurate reconstruction
transform = dtcwt.Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
dataRndT = transform.forward(imageLnRnd,nlevels=nLevels,include_scale=True)

# Apply artificial global factors to achieve global anisotropy.
for iLev in range(nLevels):
  rotLevShift=0.0
  oriFacs=[1.0,4.0,1.0,0.5,0.0,0.5]
  for iOri in range(6):
    rotFac=oriFacs[iOri]*np.cos(rotLevShift)
    dataRndT.highpasses[iLev][:,:,iOri]*=rotFac

# Use modified wavelets to achieve directional invariance at the cost of
# inaccurate reconstruction
transform = dtcwt.Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
dataRnd3T = transform.forward(imageLnRnd,nlevels=nLevels,include_scale=True)


# Apply artificial local factors to achieve local anisotropy.
dirs=[np.pi/12.,3*np.pi/12.,5*np.pi/12.,7*np.pi/12.,9*np.pi/12.,11*np.pi/12.]

for iLev in range(nLevels):
  rotLevShift=0.25*np.pi*float(iLev)/float(nLevels-1)
  nxLev=dataRnd3T.highpasses[iLev].shape[0]
  nyLev=dataRnd3T.highpasses[iLev].shape[1]
  x=np.linspace(0.0,float(nxLev-1),num=nxLev,endpoint=True)
  y=np.linspace(0.0,float(nyLev-1),num=nyLev,endpoint=True)
  xx,yy=np.meshgrid(x,y)
  xx-=float(nxLev)/2.0
  yy-=float(nyLev)/2.0
  rotPosShift=np.mod(np.arctan2(yy,xx),np.pi) # rotate pi/2 then angle modulo pi
  for iRotPos in range(6):
    for iOri in range(6):
        dataRnd3T.highpasses[iLev][:,:,iOri]*=np.cos(0.5*np.pi+dirs[iOri]+rotPosShift+rotLevShift)

# Display both outputs, after adjustment, side-by-side.
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.subplot(1,2,1)
imageRec=transform.inverse(dataRndT)
im=plt.imshow(imageRec)
plt.xlabel("x-pixels")
plt.ylabel("y-pixels")
ax=plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb=plt.colorbar(im,cax=cax)
cb.set_label("$\\log_{2}(R)$")

plt.subplot(1,2,2)
imageRec=transform.inverse(dataRnd3T)
im=plt.imshow(imageRec)
plt.xlabel("x-pixels")
plt.ylabel("y-pixels")
ax=plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb=plt.colorbar(im,cax=cax)
cb.set_label("$\\log_{2}(R)$")
plt.tight_layout()
plt.show()

