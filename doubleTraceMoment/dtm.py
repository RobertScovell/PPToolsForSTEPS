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

# uses : PyWavelets from https://pywavelets.readthedocs.io/en/latest/

# Please cite the following paper, if using this code:
# Scovell, R. W. (2020) Applications of Directional Wavelets, Universal Multifractals and Anisotropic Scaling in Ensemble Nowcasting; A Review of Methods with Case Studies. Quarterly Journal of the Royal Meteorological Society. In Press. URL: http://dx.doi.org/abs/10.1002/qj.3780

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0,os.path.dirname(__file__)+"../stochasticNoise")
import fracInt

import pywt

def pyramid(image,nLevs):
    outArr=[]
    cA=image
    for iLvl in range(nLevs):
        cA,(cH,cV,cD)=pywt.dwt2(cA,'haar')
        outArr.append(np.absolute(cA))
    return outArr

def dtm(data,minScale=1,maxScale=10,etaMin=-2.,etaMax=1.,qVals=[1.5,2.0,2.5,3.0],debug=False,pltTitle=None,pltShow=True,legendStr=None):
    # Determine the number of decomposition levels needed
    n0=data.shape[0]
    n1=data.shape[1]
    nLevs=int(np.floor(np.log2(np.min((n0,n1)))))
    
    # Compute fluxes as pyramid
    imPyramid=pyramid(data,nLevs)
    
    D=2.
    nScales=maxScale-minScale+1
    
    lambda0=np.power(2.,maxScale)/np.power(2.,minScale)
    
    etaVals=np.exp(np.linspace(etaMin,etaMax,num=31,endpoint=True))
    if debug==True:
        print("log(eta):",np.log(etaVals))
        print("q:",qVals)
    
    tr=np.empty((len(qVals),len(etaVals),nScales))
    kQEta=np.empty((len(qVals),len(etaVals)))
    
    for (iEta,eta) in enumerate(etaVals):
        lambdas=[]
    
        # Compute trace moments (as in Single Trace Moment method) but on on epsEtas, for a range of q and scale (for this value of eta)
        for iLvl in range(nScales):
            lambdaLvl=np.power(2.,maxScale)/np.power(2.,minScale+iLvl)
            lambdas.append(lambdaLvl)
            for (iq,q) in enumerate(qVals):
                imPyramid[iLvl][imPyramid[iLvl]<1.0e-16]=1.0e-16
                tr[iq,iEta,iLvl]=np.mean(np.power(imPyramid[iLvl],eta*q))/np.power(np.mean(np.power(imPyramid[iLvl],eta)),q)

        # Now estimate K(q,eta) from slopes of plots of log(Tr_lambda[epsilon_lamda^q]) against log(lambda)
        for iq,q in enumerate(qVals):
            (m,c)=np.polyfit(np.log2(lambdas[:]),np.log2(tr[iq,iEta,:]),1)
            kQEta[iq,iEta]=m#+(q-1.)*D
    
    # Evaluate the the slopes of log(K(q,eta)) vs log(eta) for fixed qs to get alpha
    umfParamEsts=[]
    for (iq,q) in enumerate(qVals):
        alphaC1Ests=[]
        hw=2
        for i in range(hw,len(etaVals)-hw):
            alphaC1Ests.append(np.polyfit(np.log(etaVals[i-hw:i+hw]),np.log(np.absolute(kQEta[iq,i-hw:i+hw])),1))
        iEst=np.argmax([x[0] for x in alphaC1Ests])
        alphaEst=alphaC1Ests[iEst][0]
        intercept=alphaC1Ests[iEst][1]
        if debug == True and iq == 0:
            if legendStr is not None:
                plt.scatter(np.log(etaVals),np.log(np.absolute(kQEta[iq,:])),marker='^',color='k',s=10,label="%s data"%legendStr)
                plt.plot(np.log(etaVals),alphaEst*np.log(etaVals)+intercept,color='k',ls='--',label="%s fit"%legendStr)
            else:
                plt.scatter(np.log(etaVals),np.log(np.absolute(kQEta[iq,:])),marker='^',color='k',s=10)
                plt.plot(np.log(etaVals),alphaEst*np.log(etaVals)+intercept,color='k',ls='--')
            plt.xlabel("$\log\; \left[\eta\\right]$")
            plt.ylabel("$\log\; \left[K(q,\eta)\\right]$")
            if pltTitle is not None:
                plt.title(pltTitle)
            if pltShow == True:
                plt.show()
        kqEst=np.exp(intercept)
        fac=(np.power(q,alphaEst)-q)/(alphaEst-1)
        c1Est=kqEst/fac
        umfParamEsts.append((alphaEst,c1Est))
    return umfParamEsts
    

if __name__ == '__main__':
    
    # Load data
    data=np.genfromtxt(sys.argv[1],delimiter=",")#[175:190,290:305]
    
    if len(sys.argv)>2:
        etaMin=float(sys.argv[2])
        etaMax=float(sys.argv[3])
    else:
        etaMin=-2.
        etaMax=1.
    
    # Uncomment this to apply fractional integration before analysis.
    # This can convert non-conservative to conservative multifractals.
    if len(sys.argv)>4:
        H=float(sys.argv[4])
        if np.abs(H) > 1.0e-3:
            print("Fractional integration, order=",H)
            data=fracInt.fractionalIntegration(data,H)
    
    # Fractional integration can lead to negative fluxes
    # Set values below a small threshold to the small threshold value
    data[data<1.0e-16]=1.0e-16
    
    # Divide by mean to normalize
    data/=np.mean(data)
    
    (alpha,C1)=dtm(data,minScale=1,maxScale=5,etaMin=etaMin,etaMax=etaMax,qVals=[1.5],debug=True)[0]
    print(alpha,C1)

