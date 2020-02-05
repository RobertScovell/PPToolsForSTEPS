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


import sys
import numpy as np
import numpy.fft
import scipy.stats
import scipy.signal
import fracInt

ny=int(sys.argv[1])
nx=int(sys.argv[2])
alpha=float(sys.argv[3])
C1=float(sys.argv[4])
H=float(sys.argv[5])
outFN=sys.argv[6]

if alpha<1.0001:
    print("This code requires alpha>1")
    quit()
# Generate rvs twice as large as needed, so that convolution can be extracted from centre window
beta=-1. # max asymmetry
rvs=scipy.stats.levy_stable.rvs(alpha,beta,size=(ny,nx))
print("RV max:",np.max(rvs))
print("RV min:",np.min(rvs))
# Fractionally integrate to get conservative field
# L+S convolve with |r|^(-D/alpha). Here the FT is multiplied by |k_r|^(-(D-D/alpha))
Hp=2.-2./alpha
fIntRvs=fracInt.fractionalIntegration(rvs,Hp)

print("FIRV max:",np.max(fIntRvs))
print("FIRV min:",np.min(fIntRvs))
nd=2.*np.pi
# This prefactor is needed to ensure correct normalization (confirmation of reason for this factor (around 256) needed) 
prefac=np.power(C1,1./alpha)*np.power(nd,-1./alpha)
consUMFExp=0.25*rvs.shape[0]*prefac*fIntRvs
consUMF=np.exp(consUMFExp)
consUMF/=np.mean(consUMF)

# Fractionally integrate again to get non-conservative field
if np.abs(H)>1.0e-4:
    nonConsFIF=fracInt.fractionalIntegration(consUMF,H)
    nonConsFIF[nonConsFIF<1.0e-8]=1.0e-8
else:
    nonConsFIF=consUMF
# Output
import matplotlib.pyplot as plt
plt.subplot(2,2,1)
plt.imshow(rvs,vmin=-1.)
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(fIntRvs,vmin=-1.)
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(np.log2(consUMF),vmin=-3.)
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(np.log2(nonConsFIF),vmin=-3.)
plt.colorbar()
plt.show()


np.savetxt(outFN,nonConsFIF,delimiter=",",fmt="%g")

