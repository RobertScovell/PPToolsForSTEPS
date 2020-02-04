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
import matplotlib as mpl
mpl.use("Qt5Agg")
mpl.rcParams['image.interpolation']='nearest'
mpl.rcParams['font.size'] = 14
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import scipy.ndimage
from scipy.signal import convolve2d
import dtcwt


import fracInt

def pyramid(image,nLevs):
    outArr=[]
    for iLvl in range(nLevs):
        filtLen=2**iLvl
        box=np.ones((filtLen,filtLen))
        # Convolve with box of size 2**iLvl
        out=scipy.signal.fftconvolve(image,box,mode='valid')
        # Pick only disjoint boxes
        out=out[0::filtLen,0::filtLen]
        outArr.append(out)
    return outArr


def pyramidDTCWT(image,nLevs):
    # compute DTDWT, keeping approximation coefficients 
    transform = dtcwt.Transform2d(biort='near_sym_a', qshift='qshift_a')
    dataT = transform.forward(image,nlevels=nLevs,include_scale=True)
    arrs=[]
    for i in range(len(dataT.scales)):
        fac=np.power(2.,(1.0)*float(i))
        arrs.append(fac*np.absolute(dataT.scales[i]))

        #arrs.append(np.max(np.absolute(dataT.scales[i]),axis=2))
    return arrs

# Load noise
noise=np.genfromtxt(sys.argv[1],delimiter=",")
ny=noise.shape[0]
nx=noise.shape[1]

# compute DTDWT 
nLevels = int(np.floor(max(np.log2(ny),np.log2(nx))))

# Uncomment this to apply fractional integration before analysis.
# This can convert non-conservative to conservative multifractals.
#noise=fracInt.fractionalIntegration(noise,-1.0)
#noise[noise<0.0]=0.0 # fractional integration can lead to negative fluxes!

# Determine the number of decomposition levels needed
n0=noise.shape[0]
n1=noise.shape[1]
nLevs=int(np.floor(np.log2(np.min((n0,n1)))))

# Compute bare fluxes as pyramid
imPyramid=pyramidDTCWT(noise,nLevs)
#imPyramid2=pyramid(noise,nLevs)
#for i in range(len(imPyramid)):
#    plt.subplot(1,2,1)
#    plt.imshow(imPyramid[i])
#    plt.colorbar()
#    plt.subplot(1,2,2)
#    plt.imshow(imPyramid2[i])
#    plt.colorbar()
#    plt.show()

minScale=1
maxScale=6
D=2.
nScales=maxScale-minScale+1

# Flux at shortest scale considered
epsLambda0=imPyramid[minScale]
lambda0=np.power(2.,maxScale)/np.power(2.,minScale)

etaVals=np.exp(np.linspace(-2.0,1.0,num=31,endpoint=True))
qVals = [ 1.5 ]

tr=np.empty((len(qVals),len(etaVals),nScales))
kQEta=np.empty((len(qVals),len(etaVals)))

for (iEta,eta) in enumerate(etaVals):
    # Dress flux by raising to power eta
    epsEtaLambda0=np.power(epsLambda0,eta)
    lambdas=[]
    imPyramidEta=pyramidDTCWT(epsEtaLambda0,nScales)

    # Compute trace moments (as in Single Trace Moment method) but on on epsEtas, for a range of q and scale (for this value of eta)
    for iLvl in range(nScales):
        lambdaLvl=np.power(2.,maxScale)/np.power(2.,minScale+iLvl)
        lambdas.append(lambdaLvl)
        for (iq,q) in enumerate(qVals):
            dressedFlux=imPyramidEta[iLvl]*np.power(lambda0,D)
            dressedFlux[dressedFlux<0.0]=0.0 # supress rounding errors that lead to small negative values
            tr[iq,iEta,iLvl]=np.sum(np.power(dressedFlux,q))

    # Now estimate K(q,eta) from slopes of plots of log(Tr_lambda[epsilon_lamda^q]) against log(lambda)
    slopes=np.empty((len(qVals),len(etaVals)))
    for iq,q in enumerate(qVals):
        (m,c)=np.polyfit(np.log2(lambdas),np.log2(tr[iq,iEta,:]),1)
        kQEta[iq,iEta]=m+(q-1.)*D

# Evaluate the the slopes of log(K(q,eta)) vs log(eta) for fixed qs to get alpha
umfParamEsts=[]
for (iq,q) in enumerate(qVals):
    (alphaEst,intercept)=np.polyfit(np.log2(etaVals),np.log2(np.absolute(kQEta[iq,:])),1)
    plt.plot(np.log2(etaVals),np.log2(np.absolute(kQEta[iq,:])))
    plt.plot(np.log2(etaVals),alphaEst*np.log2(etaVals)+intercept)
    plt.show()
    # intercept=log2(k(q))
    kqEst=np.power(2.,intercept)
    fac=(np.power(q,alphaEst)-q)/(alphaEst-1)
    c1Est=kqEst/fac
    umfParamEsts.append((alphaEst,c1Est))
print(umfParamEsts)



