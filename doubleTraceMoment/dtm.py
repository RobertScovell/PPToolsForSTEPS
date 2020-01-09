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
mpl.rcParams['image.interpolation']='nearest'
mpl.rcParams['font.size'] = 14
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import scipy.ndimage
from scipy.signal import convolve2d

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


# Load data
data=np.genfromtxt(sys.argv[1],delimiter=",")
print("Mean:",np.mean(data))

# Uncomment this to apply fractional integration before analysis.
# This can convert non-conservative to conservative multifractals.
if len(sys.argv)>2:
    H=float(sys.argv[2])
    if np.abs(H) > 1.0e-3:
        print("Fractional integration, order=",-H)
        data=fracInt.fractionalIntegration(data,H)

# Fractional integration can lead to negative fluxes
# Set values below a small threshold to the small threshold value
data[data<1.0e-6]=1.0e-6

# Determine the number of decomposition levels needed
n0=data.shape[0]
n1=data.shape[1]
nLevs=int(np.floor(np.log2(np.min((n0,n1)))))

# Compute bare fluxes as pyramid
imPyramid=pyramid(data,nLevs)

minScale=1
maxScale=5
D=2.
nScales=maxScale-minScale+1

# Flux at shortest scale considered
epsLambda0=imPyramid[minScale]
lambda0=np.power(2.,maxScale)/np.power(2.,minScale)

etaVals=np.exp(np.linspace(-2.0,1.0,num=31,endpoint=True))
qVals = [ 1.5, 2.0, 2.5, 3.0 ]

tr=np.empty((len(qVals),len(etaVals),nScales))
kQEta=np.empty((len(qVals),len(etaVals)))

for (iEta,eta) in enumerate(etaVals):
    # Dress flux by raising to power eta
    epsEtaLambda0=np.power(epsLambda0,eta)
    lambdas=[]
    imPyramidEta=pyramid(epsEtaLambda0,nScales)

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



