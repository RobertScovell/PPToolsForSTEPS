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

# Please cite the following paper, if using this code:
# Scovell, R. W. (2020) Applications of Directional Wavelets, Universal Multifractals and Anisotropic Scaling in Ensemble Nowcasting; A Review of Methods with Case Studies. Quarterly Journal of the Royal Meteorological Society. In Press. URL: http://dx.doi.org/abs/10.1002/qj.3780


import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.ndimage
from scipy.signal import convolve2d

from dtm import dtm

sys.path.insert(0,os.path.dirname(__file__)+"../stochasticNoise")
import fracInt

def pyramid2(image,iDyadicScaleLevs):
    outArr=[]
    for iLvl in iDyadicScaleLevs:
        filtLen=2**iLvl
        box=np.ones((filtLen,filtLen))
        # Convolve with box of size 2**iLvl. Input must be positive
        out=scipy.signal.fftconvolve(np.absolute(image),box,mode='valid')
        # Pick only disjoint boxes
        out=out[0::filtLen,0::filtLen]
        # Normalize by # pixels in box = 2**iLvl * 2**iLvl
        outArr.append(out)
    return outArr

# Load two datasets for side-by-side comparison
data1=np.genfromtxt(sys.argv[1],delimiter=",")
print("Mean:",np.mean(data1))
data2=np.genfromtxt(sys.argv[2],delimiter=",")
print("Mean:",np.mean(data1))

minScale=0
maxScale=5
D=2.
dyLevs=np.arange(minScale,maxScale+1,1)
imPyramid=pyramid2(data1,dyLevs)
qVals = [ 1.5, 2.0, 2.5, 3.0 ]
markers = [ 'o','v','^','x' ]
lambda0=np.power(2.,maxScale)/np.power(2.,minScale)
# Compute moments of normalized bare fluxes vs scale
trSTM=np.empty((len(qVals),dyLevs.shape[0]))
lambdasSTM=[]
for i,iDyLvl in enumerate(dyLevs):
    # As iDyLvl increases, scale increases, lambda decreases
    lambdaLvl=np.power(2.,maxScale)/np.power(2.,iDyLvl)
    print(lambdaLvl)
    lambdasSTM.append(lambdaLvl)
    for (iq,q) in enumerate(qVals):
        # Normalize pyramid output so the ensemble mean = 1 
        # The DTM below uses the un-normalized
        bareFlux=imPyramid[i]/np.mean(imPyramid[i])#np.power(2.,iLvl+1)#*np.power(lambda0,D)
        bareFlux[bareFlux<0.0]=0.0 # supress rounding errors that lead to small negative values
        trSTM[iq,i]=np.mean(np.power(bareFlux,q))

plt.subplot(2,2,1)
for iq,q in enumerate(qVals):
    plt.scatter(np.log2(lambdasSTM[1:]),np.log2(trSTM[iq,1:]),label="$q=%3.1f$"%(q),marker=markers[iq],color='k')
    (m,c)=np.polyfit(np.log2(lambdasSTM[1:]),np.log2(trSTM[iq,1:]),1)
    plt.plot(np.log2(lambdasSTM[1:]),c+m*np.array(np.log2(lambdasSTM[1:])),color='k')
plt.ylim(0,8)
plt.xticks([0,1,2,3,4],[1,2,3,4,5])
plt.xlabel("$\\log_{2} \\lambda$") #\\Lambda / \\lambda$")
plt.ylabel("$\\log_{2}<\\epsilon_{\\lambda}^{q}>/<\\epsilon_{\\lambda}>$")
plt.title("(a)")
plt.legend(fontsize=8,loc='upper center',bbox_to_anchor=(0.5,1.02),ncol=2)

imPyramid=pyramid2(data2,dyLevs)
qVals = [ 1.5, 2.0, 2.5, 3.0 ]
markers = [ 'o','v','^','x' ]
lambda0=np.power(2.,maxScale)/np.power(2.,minScale)
# Compute moments of normalized bare fluxes vs scale
trSTM=np.empty((len(qVals),dyLevs.shape[0]))
lambdasSTM=[]
for i,iDyLvl in enumerate(dyLevs):
    # As iDyLvl increases, scale increases, lambda decreases
    lambdaLvl=np.power(2.,maxScale)/np.power(2.,iDyLvl)
    print(lambdaLvl)
    lambdasSTM.append(lambdaLvl)
    for (iq,q) in enumerate(qVals):
        # Normalize pyramid output so the ensemble mean = 1 
        # The DTM below uses the un-normalized
        bareFlux=imPyramid[i]/np.mean(imPyramid[i])#np.power(2.,iLvl+1)#*np.power(lambda0,D)
        bareFlux[bareFlux<0.0]=0.0 # supress rounding errors that lead to small negative values
        trSTM[iq,i]=np.mean(np.power(bareFlux,q))


plt.subplot(2,2,2)
for iq,q in enumerate(qVals):
    plt.scatter(np.log2(lambdasSTM[1:]),np.log2(trSTM[iq,1:]),label="$q=%3.1f$"%(q),marker=markers[iq],color='k')
    (m,c)=np.polyfit(np.log2(lambdasSTM[1:]),np.log2(trSTM[iq,1:]),1)
    plt.plot(np.log2(lambdasSTM[1:]),c+m*np.array(np.log2(lambdasSTM[1:])),color='k')
plt.ylim(0,24)
plt.xticks([0,1,2,3,4],[1,2,3,4,5])
plt.xlabel("$\\log_{2} \\lambda$")
plt.ylabel("$\\log_{2}<\\epsilon_{\\lambda}^{q}>/<\\epsilon_{\\lambda}>$")
plt.title("(b)")
plt.legend(fontsize=8,loc='upper center',bbox_to_anchor=(0.5,1.02),ncol=2)

etaMin1=float(sys.argv[3])
etaMax1=float(sys.argv[4])
etaMin2=float(sys.argv[5])
etaMax2=float(sys.argv[6])

## Uncomment this to apply fractional integration before analysis.
## This can convert non-conservative to conservative multifractals.
#
#if len(sys.argv)>4:
#    H=float(sys.argv[4])
#    if np.abs(H) > 1.0e-3:
#        print("Fractional integration, order=",-H)
#        data=fracInt.fractionalIntegration(data,H)

# Fractional integration can lead to negative fluxes
# Set values below a small threshold to the small threshold value


qVals = [ 1.5, 2.0, 2.5, 3.0 ]
markers = [ 'o','v','^','x' ]

plt.subplot(2,2,3)
dataThresh=np.copy(data1)
dataThresh[data1<=0.03125-1.0e-6]=0.0#1.0e-16
est=dtm(dataThresh,minScale=1,maxScale=5,etaMin=etaMin1,etaMax=etaMax1,qVals=qVals,debug=True,pltTitle="(c)",pltShow=False,legendStr="Doris")
plt.legend(fontsize=8)
plt.xticks(np.arange(etaMin2,etaMax2+1,1.0))
alphaEst=np.mean([x[0] for x in est])
alphaEstStd=np.std([x[0] for x in est])
C1Est=np.mean([x[1] for x in est])
C1EstStd=np.std([x[1] for x in est])
print(alphaEst,alphaEstStd,C1Est,C1EstStd)

plt.subplot(2,2,4)
dataThresh=np.copy(data2)
dataThresh[data2<=0.03125-1.0e-6]=0.0#1.0e-16
est=dtm(dataThresh,minScale=1,maxScale=5,etaMin=etaMin2,etaMax=etaMax2,qVals=qVals,debug=True,pltTitle="(d)",pltShow=False,legendStr="Birm.")
plt.legend(fontsize=8)
plt.xticks(np.arange(etaMin2,etaMax2+1,1.0))
alphaEst=np.mean([x[0] for x in est])
alphaEstStd=np.std([x[0] for x in est])
C1Est=np.mean([x[1] for x in est])
C1EstStd=np.std([x[1] for x in est])
print(alphaEst,alphaEstStd,C1Est,C1EstStd)

plt.tight_layout()
plt.show()

