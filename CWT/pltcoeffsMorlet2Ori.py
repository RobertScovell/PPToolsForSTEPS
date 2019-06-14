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
import matplotlib as mpl
mpl.rcParams['image.interpolation']='nearest'
mpl.rcParams['font.size'] = 10
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

for iOri in [0,4]: 
    imCount=1
    for i in (2,4,6):
        iScale = scales[i]
        s=np.power(2.0,iScale)
        lambd=s*(4.0*np.pi)/(6.0+np.sqrt(2+6.0*6.0))
        imArr0=np.copy(coeffs[iScale,iOri,:,:,0])
        imArr1=np.copy(coeffs[iScale,iOri,:,:,1])
        imArr=imArr0
        maxIm=np.max(imArr)
        plt.subplot(3,3,imCount)
        plt.imshow(imArr0,cmap=plt.get_cmap('RdBu_r'),interpolation='nearest',vmin=-1*maxIm,vmax=maxIm)#,vmin=-80.0,vmax=80.0)
        plt.ylabel("$a=%2.1f$" % (s),rotation=90,size='large')
        if i==2:
            plt.title("$Re(W_{\psi,jl})$")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.subplot(3,3,imCount+1)
        plt.imshow(imArr1,cmap=plt.get_cmap('RdBu_r'),interpolation='nearest',vmin=-1*maxIm,vmax=maxIm)#,vmin=-80.0,vmax=80.0)
        if i==2:
            plt.title("$Im(W_{\psi,jl})$")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()

        # Abs
        plt.subplot(3,3,imCount+2)
        imArr=np.square(np.absolute(imArr0+1.0j*imArr1))
        plt.imshow(np.log2(imArr),cmap=plt.get_cmap('viridis'),interpolation='nearest',vmin=-2.0)#,vmax=maxIm)#,vmin=-80.0,vmax=80.0)
        if i==2:
            plt.title("$\log_{2}|W_{\psi,jl}|^2$")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        imCount+=3
    plt.tight_layout()
    plt.show()


