#!/usr/bin/env python3
# encoding: utf-8

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

# This code is a basic generator of 2D realisations of the Fractionally Integrated Flux model.
# Parameters are:
# 1. Number of rows in image (not tested with unequal number of rows x columns )
# 2. Number of columns in image
# 3. Multifractality parameter ( alpha = 2.0 => lognormal cascade, alpha = 1.0 => Cauchy (not tested) ). Typically alpha = [1.5,1.8] for rain.
# 4. Codimension of the mean (see Lovejoy and Schertzer, 2013). This is the sparseness of the mean value. C1=0 => homogeneous.
# 5. Hurst parameter ( H=0.0 -> 1.0 ) used for fractional integration. Typically H=0.33 is reasonable for rain.
# 6. Output CSV file.

# This code is a generator of 2D realisations of the Fractionally Integrated Flux model.
# It is a direct port from the Matlab code provided by Schertzer and Lovejoy, available here: http://www.physics.mcgill.ca/~gang/software/index.html
# An extra step to do the fractional integration has been added at the end.

# Parameters are:
# 1. Number of rows in image (not tested with unequal number of rows x columns )
# 2. Number of columns in image
# 3. Multifractality parameter ( alpha = 2.0 => lognormal cascade, alpha = 1.0 => Cauchy (not tested) ). Typically alpha = [1.2,1.6] for rain.
# 4. Codimension of the mean (see Lovejoy and Schertzer, 2013). This is the sparseness of the mean value. C1=0 => homogeneous. A value of 0.1-0.2 is typical for rain.
# 5. Hurst parameter ( H=0.0 -> 1.0 ) used for fractional integration. H=0.33 is reasonable for rain.
# 6. Output CSV file.

import numpy as np
import sys
import numpy.fft
import scipy.stats
import fracInt

def Frac(scal=None,alpha=None,*args,**kwargs):
    lambda_=np.min(scal.shape)

    if lambda_ == 1:
        lambda_=np.max(scal.shape)

    rat=2

    alphap=1 / (1 - 1 / alpha)

    lcut=lambda_ / 2

    lcut2=lcut / rat

    exlambda=np.exp(- (scal / lcut) ** 4)

    exlambda[exlambda<1.0e-16]=1.0e-16
    
    exlambda2=np.exp(- (scal / lcut2) ** 4)

    exlambda2[exlambda2<1.0e-16]=1.0e-16
    
    sing=np.power(scal, (- 1 / alphap))

    singsmooth=np.multiply(sing,exlambda)

    singsmooth[singsmooth<1.0e-16]=1.0e-16
    
    t1=np.sum(singsmooth)

    singsmooth2=np.multiply(sing,exlambda2)

    singsmooth2[singsmooth2<1.0e-16]=1.0e-16
    
    t2=np.sum(singsmooth2)

    A=(np.dot(np.power(rat, (- 1 / alpha)),t1) - t2) / (np.power(rat, (- 1 / alpha)) - 1)

    ff=np.exp(- scal / 3)

    ff[ff<1.0e-16]=1.0e-16
    
    singsmooth=np.multiply(sing,ff)

    singsmooth[singsmooth<1.0e-16]=1.e-16
    
    G=np.sum(singsmooth)

    a=- A / G

    csing=(np.multiply(sing,(1 + np.multiply(a,ff))))

    csing[csing<1.0e-16]=1.0e-16
    
    fragged=np.power(csing, (1 / (alpha - 1)))
    return fragged

def Levy(alpha=None,size=None,*args,**kwargs):

    #### NEEDS CHECKING - it's different to formula in
    #### Computer Simulation of Levy Stable Variables and Processes
    #### Aleksander Weron and Rafal Weron (1995)
    #### Also cf. scipy.stats.levy_stable
    W=np.random.exponential(size=size)
    U=np.random.uniform(size=size)
    V=-0.5*np.pi+np.pi*U 
    V0=-0.5*np.pi*(1-np.abs(1-alpha))/alpha
    tmp1=alpha*(V-V0)
    tmp2=np.sign(alpha-1)*np.sin(tmp1)*np.power(np.cos(V)*np.abs(alpha-1),(-1/alpha)) 
    tmp3=np.power(np.cos(V-tmp1)/W,((1-alpha)/alpha))
    return tmp2*tmp3

def eps1D(lambda_=None,alpha=None,C1=None,Switch=None,*args,**kwargs):

    lambda_=np.floor(lambda_)

#    xx=np.arange((1 - lambda_),(lambda_ - 1),2)
    xx=np.arange((1 - lambda_),(lambda_ - 0),2)

    if Switch == 0:
        NDf=1

        Heavi=1

    else:
        NDf=0.5

        Heavi=np.heaviside(sym(xx))

    
    scal=np.abs(xx)

#    ggen1=np.power((C1 / NDf), (1 / alpha))*Levy(alpha,size=size)

    # Use SciPy implementation of Levy alpha-stable variate generator
    ggen1=np.power((C1 / NDf), (1 / alpha))*scipy.stats.levy_stable(alpha=alpha,beta=-1,size=size)
    
    sing=Frac(scal,alpha)

    Hs=np.multiply(Heavi,sing).astype(np.float64)

#    print("Warning using convolve method that is not verified to be the same as in original.")
#    ggen1alpha=np.fftconvolve(ggen1,Hs)
    ggen1alpha=np.real(np.fft.ifft(np.multiply(np.fft.fft(ggen1),np.fft.fft(Hs))))
    ggen1alpha=np.exp(ggen1alpha)

    ff=np.mean(ggen1alpha)

    epsed=ggen1alpha / ff

    return epsed
   
def eps2D(lambdat=None,lambday=None,alpha=None,C1=None,Switch=None,*args,**kwargs):

    lambdat=np.floor(lambdat)

    lambday=np.floor(lambday)

    tt=np.arange((1 - lambdat),(lambdat - 0),2)

    yy=np.arange((1 - lambday),(lambday - 0),2)

    if Switch == 0:
        NDf=np.pi / 2

        Heavi=1

    else:
        raise("Error: causal not supported here")
        NDf=np.pi / 4

        Heavit=np.heaviside(tt)

        Heavi=repmat(Heavit.T,concat([1,length(yy)]))
   
    ttt,yyy=np.meshgrid(tt,yy)
    scal=np.square(ttt)+np.square(yyy)

    ggen1=np.power(C1/NDf,1./alpha)*Levy(alpha,size=(int(lambdat),int(lambday)))

    
    sing=Frac(scal,alpha)

    Hs=np.multiply(Heavi,sing)

    ggen1alpha=np.real(np.fft.ifft2(np.multiply(np.fft.fft2(ggen1),np.fft.fft2(Hs))))

    ggen1alpha=np.exp(ggen1alpha)

    ff=np.mean(ggen1alpha)

    epsed=ggen1alpha / ff

    return epsed
    
if __name__ == '__main__':
    
    ny=int(sys.argv[1])
    nx=int(sys.argv[2])
    alpha=float(sys.argv[3])
    C1=float(sys.argv[4])
    H=float(sys.argv[5])
    outFN=sys.argv[6]
    
    consUMF=eps2D(lambdat=ny,lambday=nx,alpha=alpha,C1=C1,Switch=0)
     
    # Fractionally integrate again to get non-conservative field
    if np.abs(H)>1.0e-4:
        nonConsFIF=fracInt.fractionalIntegration(consUMF,H)
        nonConsFIF/=np.mean(nonConsFIF)
        nonConsFIF[nonConsFIF<1.0e-16]=1.0e-16
    else:
        nonConsFIF=consUMF

    np.savetxt(outFN,nonConsFIF,delimiter=",",fmt="%g")

    debug=True
    if debug==True:
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.imshow(np.log2(consUMF),vmin=-3.)
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(np.log2(nonConsFIF),vmin=-3.)
        plt.colorbar()
        plt.show()
     
