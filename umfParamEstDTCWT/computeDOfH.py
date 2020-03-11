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

import numpy as np
import scipy.ndimage
import warnings
from scipy.optimize import least_squares

# uses : DTDWT from https://pypi.python.org/pypi/dtcwt
# uses : PyWavelets from https://pywavelets.readthedocs.io/en/latest/
import dtcwt
import pywt

# Please quote the following paper, if using this code:
# Scovell, R. W. (2020) Applications of Directional Wavelets, Universal Multifractals and Anisotropic Scaling in Ensemble Nowcasting; A Review of Methods with Case Studies. Quarterly Journal of the Royal Meteorological Society. In Press. URL: http://dx.doi.org/abs/10.1002/qj.3780

def wlsZeros(noiseIn,thresh0,maxSc,minSc=1,qmin=0.,qmax=8.,nq=81,decompmode='dtcwt',minZeroDist=10,mode='once',zeroMask=False):
    
    d=2.
    ny=noiseIn.shape[0]
    nx=noiseIn.shape[1]
    nLevelsMax = int(np.floor(max(np.log2(ny),np.log2(nx))))
    minScale=minSc
    maxScale=min(nLevelsMax,maxSc)
    qstep=(qmax-qmin)/nq
    qvals=np.linspace(qmin,qmax,num=nq,endpoint=True)
    noise=np.copy(noiseIn)

    # Compute distance transform from zero mask 
    # Set "zeros" to nan, so that that zero-contaminated coefficients can be identified
    if zeroMask == True:
        nzMask=noise<thresh0
        noise[noise<thresh0]=thresh0 # for those values that don't exceed minZeroDist
        dt=scipy.ndimage.morphology.distance_transform_edt(nzMask)
        # Only set to nan if distance is greater than minZeroDist from a non-zero
        noise[dt>minZeroDist]=np.nan
    
    ln2Scales=[]
    oriMax=[]

    if decompmode == 'dtcwt':
        # Compute DTDWT 
        # Do not use rotationally-invariant transform. Adantage is much shorter filters. Disadvantage is that max mod over ori may not be meaningful?
        # Use length 10 'a' filters here, rather than the length 14 'b' (or length 18 'b_bp'). 
#        transform = dtcwt.Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
        transform = dtcwt.Transform2d(biort='near_sym_a', qshift='qshift_a')
        dataT = transform.forward(noise,nlevels=maxScale+1)
        for iLev in range(maxScale+1):
            hp=dataT.highpasses[iLev]
            ln2Scales.append(iLev+1) # first set of detail coeffs has scale=2
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                oriMax.append(np.nanmax(np.absolute(hp),axis=2))
    elif decompmode == 'dwt':
        dataT = pywt.wavedec2(noise,'sym4',mode='periodization')
        for iPos in range(len(dataT)-1,0,-1):
            iLev=len(dataT)-1-iPos
            arrs=dataT[iPos]
            arrsStack=np.abs(np.stack(arrs,axis=2))
            tmp=np.nanmax(np.abs(arrsStack),axis=2)
            # Now do single-level approxiation to the 'leaders' bit
            tmp=scipy.ndimage.maximum_filter(tmp,size=3)
            oriMax.append(tmp)
            ln2Scales.append(iLev+1) # first set of detail coeffs has scale=2

    else:
        raise Exception("Invalid mode:",mode)

    # Convert ln2Scales to np.array
    ln2Scales=np.array(ln2Scales)
       
    # Compute maxima over orientations for each dyadic cube 
    oriMaxValid=[]
    for iLev in range(maxScale+1):
        if decompmode == 'dtcwt':
            validAtThisScale=np.absolute(oriMax[iLev][~np.isnan(oriMax[iLev])])
        elif decompmode == 'dwt':
            # This should be replaced with a neighbourhood maximum
            validAtThisScale=np.absolute(oriMax[iLev][oriMax[iLev]>1.0e-10])
        else:
            pass
        validAtThisScale/=np.mean(validAtThisScale)
        oriMaxValid.append(validAtThisScale)
    
    # Compute D(h) using thermodynamical approach (Partition Function, Free Energy, Entropy) 
    U=np.zeros((maxScale+1,nq))
    V=np.zeros((maxScale+1,nq))
    tmpu1=np.zeros((maxScale+1,nq))
    tmpu2=np.zeros((maxScale+1,nq))
    tmpv=np.zeros((maxScale+1,nq))
    nj=np.zeros((maxScale+1))
    S=np.zeros((maxScale+1,nq))
    SNN=np.zeros((maxScale+1,nq))
    sz=np.zeros((maxScale+1))
    for iLev in range(maxScale+1):
        epsArr=oriMaxValid[iLev]
        sz[iLev]=epsArr.size
        for iq,q in enumerate(qvals[:nq]):
            # Compute qth order structure functions
            qthMomArr=np.power(epsArr,q)
            S[iLev,iq]=np.sum(qthMomArr)
            # Compute intermediate quantities for h,D(h)
            s=S[iLev,iq]
            tmpv[iLev,iq]=np.sum(qthMomArr*np.log2(epsArr))
            tmpu1[iLev,iq]=np.sum(qthMomArr*np.log2(qthMomArr))
            tmpu2[iLev,iq]=np.sum(qthMomArr)
            tmpu=tmpu1[iLev,iq]-np.log2(s)*tmpu2[iLev,iq]
            U[iLev,iq]=tmpu/(s) + np.log2(sz[iLev])
            V[iLev,iq]=tmpv[iLev,iq]/(s)
    
    if mode == 'once':
        # Display h(q,a) vs log2(a) function at selected scales
        order=1
        hfit=np.empty((nq,order+1))
        dfit=np.empty((nq,order+1))
        for iq in range(nq):
            hfit[iq,:]=np.polyfit(ln2Scales[minScale:maxScale+1],V[minScale:maxScale+1,iq],order)
            dfit[iq,:]=np.polyfit(ln2Scales[minScale:maxScale+1],U[minScale:maxScale+1,iq],order)
        return hfit,dfit
    elif mode == 'batch':
        return sz,S,tmpv,tmpu1,tmpu2,ln2Scales

def cOfGamma(alpha,C1,gammaVals):
    if np.abs(alpha-1.)<1.0e-5:
        return C1*np.exp(gammaVals/C1-1.)
    else:
        alphap=1./(1.-1./alpha)
        tmp1=gammaVals/(C1*alphap)
        tmp2=1./alpha
        res=C1*np.power((tmp1+tmp2),alphap)
        # If nan values are generated (negative manissa) then reset to zero
        # c(gamma) must be positive-valued
        res[~np.isfinite(res)]=0.
        return res

def dOfHCF(hvals,alpha,C1,H):
    d=2.
    # Compute codimension function ( gamma = d - alpha )
    gammavals=d-hvals+H
    cOfDMinusAlpha=cOfGamma(alpha,C1,gammavals)

    ## Transform to f(alpha) representation (but with alpha=d-gamma)
    dOfH=d-cOfDMinusAlpha
    return dOfH

def umfModelFit(hfit,dfit,firstGuess):
    # use the spacing of hfit to determine fitting weights
    hfitdiffs=np.gradient(hfit[:,0])
    weights=np.abs(hfitdiffs[-1])/np.abs(hfitdiffs)
    weights=np.maximum.accumulate(weights)
    popt,pcov=scipy.optimize.curve_fit(dOfHCF,hfit[:,0],dfit[:,0]+2.,firstGuess,sigma=weights,bounds=[[0.0,0.0,-np.inf],[2.0,1.0,np.inf]],max_nfev=100000,ftol=1.0e-10)
    return popt,pcov


