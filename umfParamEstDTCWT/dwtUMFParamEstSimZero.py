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

# Please quote the following paper, if using this code:
# Scovell, R. W. (2020) Applications of Directional Wavelets, Universal Multifractals and Anisotropic Scaling in Ensemble Nowcasting; A Review of Methods with Case Studies. Quarterly Journal of the Royal Meteorological Society. In Press. URL: http://dx.doi.org/abs/10.1002/qj.3780

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from computeDOfH import wlsZeros,umfModelFit,dOfHCF

print (os.path.dirname(__file__)+"../stochasticNoise")
sys.path.insert(0,os.path.dirname(__file__)+"../stochasticNoise")
from fifGenLS2010 import eps2D

if __name__ == '__main__':

    debug = True

    # Load data
    noise=np.genfromtxt(sys.argv[1],delimiter=",")
    # Load threshold value
    thresh0=float(sys.argv[2])

    # Define min / max scales (first element of dtcwt coeff array is scale 1)
    minSc=1
    maxSc=5

    # Normalize the data
    noise[noise<thresh0]=0.
    dataMean0=np.mean(noise)
    print("Mean of image:",dataMean0)
    noise/=dataMean0
    thresh=thresh0/dataMean0

    # Optional differentiation
    diff = False
    if diff == True:
        noised0,noised1=np.gradient(noise)
        noise=np.sqrt(np.square(noised0)+np.square(noised1))

    firstGuess=[1.4,0.15,-2.]
    nextGuess=firstGuess

    while True:
        zeroNoise=eps2D(lambdat=noise.shape[0],lambday=noise.shape[1],alpha=nextGuess[0],C1=nextGuess[1],Switch=0)
        zeroNoise[~np.isfinite(zeroNoise)]=1.0e-16
        zeroNoiseMeanLow=np.mean(zeroNoise[zeroNoise<thresh])
        zeroNoise[zeroNoise>thresh]=1.0e-16
        noisePlusZeroField=noise+zeroNoise

        hfit,dfit=wlsZeros(noisePlusZeroField,thresh0,maxSc,qmin=0.0,qmax=6.0,decompmode='dtcwt',nq=81,minZeroDist=1,mode='once',zeroMask=False)
        popt,pcov=umfModelFit(hfit,dfit,nextGuess)
        print(popt)
        if debug == True:
            plt.plot(np.array(hfit[:,0]),dOfHCF(hfit[:,0],popt[0],popt[1],popt[2]))
            plt.scatter(np.array(hfit[:,0]),dfit[:,0]+2,marker='o',s=1.)
            plt.xlabel("Hölder Exponent h")
            plt.tight_layout()
            plt.show() 
    
