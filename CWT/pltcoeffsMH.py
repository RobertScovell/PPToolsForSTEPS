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

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
import sys
import viridis

h5FH1=h5py.File(sys.argv[1])

coeffs=np.copy(h5FH1['coefficients'])

# Dataset {34/34, 1/1, 1024/1024, 1024/1024, 2/2}

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

scales=[0,1,2,3,4,5,6,7,8,9,10,11]

fig,axes=plt.subplots(3,2)
ax=axes.flat[0]

imCount=0
# Display real values at selected scales
for i in range(2,7,2):
    iScale = scales[i]
    s=np.power(2.0,iScale)
    lambd=s*(2.0*np.pi)/np.sqrt(2.0+0.5)
    imArr=np.copy(coeffs[iScale,iOri,:,:,0])
    ax=axes.flat[imCount]
    minIm=np.min(imArr)
    maxIm=np.max(imArr)
    maxAbs=max(minIm,maxIm)
    im=ax.imshow(imArr,cmap=plt.get_cmap('RdBu_r'),interpolation='nearest',vmin=-maxAbs,vmax=maxAbs)#,vmin=-80.0,vmax=80.0)
    ax.set_title("$W_{\psi,j}\,[a=%2.1f,\lambda=%3.1f]}$" % (s,lambd) )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    div=make_axes_locatable(ax)
    cax=div.append_axes("right",size="10%",pad=0.05)
    plt.colorbar(im,cax=cax)
    imCount+=2
imCount=1
# Display absolute values
for i in range(2,7,2):
    iScale=scales[i]
    s=np.power(2.0,iScale)
    lambd=s*(2.0*np.pi)/np.sqrt(2.0+0.5)
    imArr=np.copy(coeffs[iScale,iOri,:,:,0])
    ax=axes.flat[imCount]
    minIm=np.min(imArr)
    maxIm=np.max(imArr)
    maxAbs=max(minIm,maxIm)
    imd=np.absolute(np.square(imArr))
    imd[imd<0.125]=0.125
    im=ax.imshow(np.log2(imd),cmap=plt.get_cmap('viridis'),interpolation='nearest')#,vmin=-80.0,vmax=80.0)
    ax.set_title("$\log_{2}|W_{\psi,j}|^2\,[a=%2.1f,\lambda=%3.1f]}$" % (s,lambd) )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    div=make_axes_locatable(ax)
    cax=div.append_axes("right",size="10%",pad=0.05)
    plt.colorbar(im,cax=cax)
    imCount+=2
plt.tight_layout()
plt.show()

