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
# uses : DTCWT from https://pypi.python.org/pypi/dtcwt ( for comparison to UD-DTCWT )

# Please cite the following paper, if using this code:
# Scovell, R. W. (2020) Applications of Directional Wavelets, Universal Multifractals and Anisotropic Scaling in Ensemble Nowcasting; A Review of Methods with Case Studies. Quarterly Journal of the Royal Meteorological Society. In Press. URL: http://dx.doi.org/abs/10.1002/qj.3780

# This is a fairly naive implementation of the Undecimated DTCWT, using Pywavelets to compute the single-tree SWT transforms and DTCWT to provide the Q-shift filters.
# Better methods may be available, e.g. based on:
# Hill, P. R., N. Anantrasirichai, A. Achim, M. E. Al-Mualla, and D. R. Bull. “Undecimated Dual-Tree Complex Wavelet Transforms.” Signal Processing: Image Communication 35 (2015): 61-70.
# The following site links to Matlab code: https://vilab.blogs.bristol.ac.uk/?p=1156

import numpy as np
import matplotlib as mpl
mpl.rcParams['image.interpolation']='nearest'
mpl.rcParams['font.size']=14
import matplotlib.pyplot as plt
import sys
import scipy.ndimage

import pywt
import dtcwt

def upscaleFilter(h):
    hUp=np.zeros(2*h.shape[0])
    hUp[0::2]=h
    return hUp

def udDTCWTLvl1(image,lvl1Filters):

    # pad the lvl 0 filters
    lvl1Len=np.max((lvl1Filters[0].shape[0],lvl1Filters[1].shape[0],lvl1Filters[2].shape[0],lvl1Filters[3].shape[0]))
    fac=1.0
    h0o=np.roll(np.pad(fac*lvl1Filters[0][::1,0],[3,3],mode='constant')[::1],1).tolist()
    g0o=np.roll(np.pad(fac*lvl1Filters[1][::1,0],[0,0],mode='constant')[::1],0).tolist()
    h1o=np.roll(np.pad(fac*lvl1Filters[2][::1,0],[0,0],mode='constant')[::1],1).tolist()
    g1o=np.roll(np.pad(fac*lvl1Filters[3][::1,0],[3,3],mode='constant')[::1],0).tolist()
    
    lvl1WvtA=pywt.Wavelet('lvl1',filter_bank=[np.roll(h0o,0),np.roll(h1o,0),np.roll(g0o,0),np.roll(g1o,0)])#g0o,g1o])
    lvl1WvtB=pywt.Wavelet('lvl1',filter_bank=[np.roll(h0o,1),np.roll(h1o,1),np.roll(g0o,1),np.roll(g1o,1)])
    
    # Undecimated "a trous" transform tree AA
    swtCoeffs=pywt.swt2(image,wavelet=(lvl1WvtA,lvl1WvtA),axes=(0,1),level=1)[0]
    AA_LL,(AA_LH,AA_HL,AA_HH)=swtCoeffs
    # Undecimated "a trous" transform tree AB
    swtCoeffs=pywt.swt2(image,wavelet=(lvl1WvtA,lvl1WvtB),axes=(0,1),level=1)[0]
    AB_LL,(AB_LH,AB_HL,AB_HH)=swtCoeffs
    # Undecimated "a trous" transform tree BA
    swtCoeffs=pywt.swt2(image,wavelet=(lvl1WvtB,lvl1WvtA),axes=(0,1),level=1)[0]
    BA_LL,(BA_LH,BA_HL,BA_HH)=swtCoeffs
    # Undecimated "a trous" transform tree BB
    swtCoeffs=pywt.swt2(image,wavelet=(lvl1WvtB,lvl1WvtB),axes=(0,1),level=1)[0]
    BB_LL,(BB_LH,BB_HL,BB_HH)=swtCoeffs
    
    fac=np.sqrt(0.5)
    coeffs1=fac*(AA_HH-BB_HH)+fac*1.j*(BA_HH+AB_HH)
    coeffs4=fac*(AA_HH+BB_HH)+fac*1.j*(-BA_HH+AB_HH)
    coeffs5=fac*(AA_LH-BB_LH)+fac*1.j*(BA_LH+AB_LH)
    coeffs0=fac*(AA_LH+BB_LH)+fac*1.j*(-BA_LH+AB_LH)
    coeffs3=fac*(AA_HL-BB_HL)+fac*1.j*(BA_HL+AB_HL)
    coeffs2=fac*(AA_HL+BB_HL)+fac*1.j*(-BA_HL+AB_HL)
    
    coeffs=[]
    coeffs.append((coeffs0,coeffs1,coeffs2,coeffs3,coeffs4,coeffs5))
    return coeffs,(AA_LL,AB_LL,BA_LL,BB_LL)

def udDTCWTLvl2(lvl1Coeffs,lvl2Filters,nLevels):
   
    coeffs=lvl1Coeffs[0]
    AA_LL=lvl1Coeffs[1][0]
    AB_LL=lvl1Coeffs[1][1]
    BA_LL=lvl1Coeffs[1][2]
    BB_LL=lvl1Coeffs[1][3]

    # phi (0)
    h0a=lvl2Filters[0][:,0]
    h0b=lvl2Filters[1][:,0]
    g0a=lvl2Filters[2][:,0]
    g0b=lvl2Filters[3][:,0]
    
    # psi (1)
    h1a=lvl2Filters[4][:,0]
    h1b=lvl2Filters[5][:,0]
    g1a=lvl2Filters[6][:,0]
    g1b=lvl2Filters[7][:,0]  
    
    for iLev in range(1,nLevels):
    
        # On the first and future iterations, insert extra '0' between filter coefficients
        # This is to account for lack of decimation, a la "a trous".
        # The pywavelets swt2 already does this but this accounts for previous iterations.
        h0a=upscaleFilter(h0a)
        h0b=upscaleFilter(h0b)
        h1a=upscaleFilter(h1a)
        h1b=upscaleFilter(h1b)
        g0a=upscaleFilter(g0a)
        g0b=upscaleFilter(g0b)
        g1a=upscaleFilter(g1a)
        g1b=upscaleFilter(g1b)
    
        # What is the half-sample delay offset?
        hsd=2**iLev
    
        if iLev % 2 == 0:
            lvl2WvtA=pywt.Wavelet('lvl2a',filter_bank=[np.roll(h0a,0),np.roll(h1a,0),np.roll(g0a,hsd),np.roll(g1a,hsd)])
            lvl2WvtB=pywt.Wavelet('lvl2b',filter_bank=[np.roll(h0b,hsd),np.roll(h1b,hsd),np.roll(g0b,0),np.roll(g1b,0)])
        else:
            lvl2WvtB=pywt.Wavelet('lvl2b',filter_bank=[np.roll(h0a,0),np.roll(h1a,0),np.roll(g0a,hsd),np.roll(g1a,hsd)])
            lvl2WvtA=pywt.Wavelet('lvl2a',filter_bank=[np.roll(h0b,hsd),np.roll(h1b,hsd),np.roll(g0b,0),np.roll(g1b,0)])
   
        # Undecimated "a trous" transform tree AA
        swtCoeffs=pywt.swt2(AA_LL,wavelet=(lvl2WvtA,lvl2WvtA),axes=(0,1),level=1)[0]
        AA_LL,(AA_LH,AA_HL,AA_HH)=swtCoeffs
        # Undecimated "a trous" transform tree AB
        swtCoeffs=pywt.swt2(AB_LL,wavelet=(lvl2WvtA,lvl2WvtB),axes=(0,1),level=1)[0]
        AB_LL,(AB_LH,AB_HL,AB_HH)=swtCoeffs
        # Undecimated "a trous" transform tree BA
        swtCoeffs=pywt.swt2(BA_LL,wavelet=(lvl2WvtB,lvl2WvtA),axes=(0,1),level=1)[0]
        BA_LL,(BA_LH,BA_HL,BA_HH)=swtCoeffs
        # Undecimated "a trous" transform tree BB
        swtCoeffs=pywt.swt2(BB_LL,wavelet=(lvl2WvtB,lvl2WvtB),axes=(0,1),level=1)[0]
        BB_LL,(BB_LH,BB_HL,BB_HH)=swtCoeffs
    
        fac=np.sqrt(0.5)
        coeffs1=fac*(AA_HH-BB_HH)+fac*1.j*(BA_HH+AB_HH)
        coeffs4=fac*(AA_HH+BB_HH)+fac*1.j*(-BA_HH+AB_HH)
        coeffs5=fac*(AA_LH-BB_LH)+fac*1.j*(BA_LH+AB_LH)
        coeffs0=fac*(AA_LH+BB_LH)+fac*1.j*(-BA_LH+AB_LH)
        coeffs3=fac*(AA_HL-BB_HL)+fac*1.j*(BA_HL+AB_HL)
        coeffs2=fac*(AA_HL+BB_HL)+fac*1.j*(-BA_HL+AB_HL)
        coeffs.append((coeffs0,coeffs1,coeffs2,coeffs3,coeffs4,coeffs5))

    return coeffs,(AA_LL,AB_LL,BA_LL,BB_LL)

def udDTCWTLvl1Inv(coeffsAll,lvl1Filters):

    AA_LL=coeffsAll[1][0]
    AB_LL=coeffsAll[1][1]
    BA_LL=coeffsAll[1][2]
    BB_LL=coeffsAll[1][3]

    (coeffs0,coeffs1,coeffs2,coeffs3,coeffs4,coeffs5)=coeffsAll[0]

    fac=np.sqrt(0.5)
    AA_HH=fac*(np.real(coeffs4)+np.real(coeffs1))
    AA_LH=fac*(np.real(coeffs0)+np.real(coeffs5))
    AA_HL=fac*(np.real(coeffs2)+np.real(coeffs3))

    BB_HH=fac*(np.real(coeffs4)-np.real(coeffs1))
    BB_LH=fac*(np.real(coeffs0)-np.real(coeffs5))
    BB_HL=fac*(np.real(coeffs2)-np.real(coeffs3))

    AB_HH=fac*(np.imag(coeffs1)+np.imag(coeffs4))
    AB_LH=fac*(np.imag(coeffs5)+np.imag(coeffs0))
    AB_HL=fac*(np.imag(coeffs3)+np.imag(coeffs2))

    BA_HH=fac*(np.imag(coeffs1)-np.imag(coeffs4))
    BA_LH=fac*(np.imag(coeffs5)-np.imag(coeffs0))
    BA_HL=fac*(np.imag(coeffs3)-np.imag(coeffs2))

    # pad the lvl 0 filters
    lvl1Len=np.max((lvl1Filters[0].shape[0],lvl1Filters[1].shape[0],lvl1Filters[2].shape[0],lvl1Filters[3].shape[0]))
    fac=1.0
    h0o=np.roll(np.pad(fac*lvl1Filters[0][::1,0],[3,3],mode='constant')[::1],1).tolist()
    g0o=np.roll(np.pad(fac*lvl1Filters[1][::1,0],[0,0],mode='constant')[::1],0).tolist()
    h1o=np.roll(np.pad(fac*lvl1Filters[2][::1,0],[0,0],mode='constant')[::1],1).tolist()
    g1o=np.roll(np.pad(fac*lvl1Filters[3][::1,0],[3,3],mode='constant')[::1],0).tolist()

    hsd=1

    lvl1WvtA=pywt.Wavelet('lvl1',filter_bank=[np.roll(h0o,0),np.roll(h1o,0),np.roll(g0o,0),np.roll(g1o,0)])#g0o,g1o])
    lvl1WvtB=pywt.Wavelet('lvl1',filter_bank=[np.roll(h0o,1),np.roll(h1o,1),np.roll(g0o,1),np.roll(g1o,1)])
    
    # Undecimated "a trous" transform tree AA
    swtCoeffs=(AA_LL,(AA_LH,AA_HL,AA_HH))
    AA_LL=pywt.iswt2((swtCoeffs,),wavelet=(lvl1WvtA,lvl1WvtA))
    # Undecimated "a trous" transform tree AB
    swtCoeffs=(AB_LL,(AB_LH,AB_HL,AB_HH))
    AB_LL=pywt.iswt2((swtCoeffs,),wavelet=(lvl1WvtA,lvl1WvtB))
    # Undecimated "a trous" transform tree BA
    swtCoeffs=(BA_LL,(BA_LH,BA_HL,BA_HH))
    BA_LL=pywt.iswt2((swtCoeffs,),wavelet=(lvl1WvtB,lvl1WvtA))
    # Undecimated "a trous" transform tree BB
    swtCoeffs=(BB_LL,(BB_LH,BB_HL,BB_HH))
    BB_LL=pywt.iswt2((swtCoeffs,),wavelet=(lvl1WvtB,lvl1WvtB))
  
    return (AA_LL,AB_LL,BA_LL,BB_LL)

def udDTCWTLvl2Inv(coeffsAll,lvl2Filters):
   
    AA_LL=coeffsAll[1][0]
    AB_LL=coeffsAll[1][1]
    BA_LL=coeffsAll[1][2]
    BB_LL=coeffsAll[1][3]

    nLevels=len(coeffsAll[0])

    # Index zero is the lvl1 part of the transform
    # So iterate down to index 1 and no further.
    for iLev in range(nLevels-1,0,-1): 

        (coeffs0,coeffs1,coeffs2,coeffs3,coeffs4,coeffs5)=coeffsAll[0][iLev]

        fac=np.sqrt(0.5)
        AA_HH=fac*(np.real(coeffs4)+np.real(coeffs1))
        AA_LH=fac*(np.real(coeffs0)+np.real(coeffs5))
        AA_HL=fac*(np.real(coeffs2)+np.real(coeffs3))
    
        BB_HH=fac*(np.real(coeffs4)-np.real(coeffs1))
        BB_LH=fac*(np.real(coeffs0)-np.real(coeffs5))
        BB_HL=fac*(np.real(coeffs2)-np.real(coeffs3))
    
        AB_HH=fac*(np.imag(coeffs1)+np.imag(coeffs4))
        AB_LH=fac*(np.imag(coeffs5)+np.imag(coeffs0))
        AB_HL=fac*(np.imag(coeffs3)+np.imag(coeffs2))
    
        BA_HH=fac*(np.imag(coeffs1)-np.imag(coeffs4))
        BA_LH=fac*(np.imag(coeffs5)-np.imag(coeffs0))
        BA_HL=fac*(np.imag(coeffs3)-np.imag(coeffs2))
    
        # phi (0)
        h0a=lvl2Filters[0][:,0]
        h0b=lvl2Filters[1][:,0]
        g0a=lvl2Filters[2][:,0]
        g0b=lvl2Filters[3][:,0]
        
        # psi (1)
        h1a=lvl2Filters[4][:,0]
        h1b=lvl2Filters[5][:,0]
        g1a=lvl2Filters[6][:,0]
        g1b=lvl2Filters[7][:,0]  
        # On the first and future iterations, insert extra '0' between filter coefficients
        # This is to account for lack of decimation, a la "a trous".
        # The pywavelets swt2 already does this but this accounts for previous iterations.
        for iiLev in range(0,iLev): #upscale once for lvl 1, twice for lvl 2, e.t.c.
            h0a=upscaleFilter(h0a)
            h0b=upscaleFilter(h0b)
            h1a=upscaleFilter(h1a)
            h1b=upscaleFilter(h1b)
            g0a=upscaleFilter(g0a)
            g0b=upscaleFilter(g0b)
            g1a=upscaleFilter(g1a)
            g1b=upscaleFilter(g1b)
    
        # What is the half-sample delay offset?
        hsd=2**iLev
    
        if iLev % 2 == 0:
            lvl2WvtA=pywt.Wavelet('lvl2a',filter_bank=[np.roll(h0a,0),np.roll(h1a,0),np.roll(g0a,hsd),np.roll(g1a,hsd)])
            lvl2WvtB=pywt.Wavelet('lvl2b',filter_bank=[np.roll(h0b,hsd),np.roll(h1b,hsd),np.roll(g0b,0),np.roll(g1b,0)])
        else:
            lvl2WvtB=pywt.Wavelet('lvl2b',filter_bank=[np.roll(h0a,0),np.roll(h1a,0),np.roll(g0a,hsd),np.roll(g1a,hsd)])
            lvl2WvtA=pywt.Wavelet('lvl2a',filter_bank=[np.roll(h0b,hsd),np.roll(h1b,hsd),np.roll(g0b,0),np.roll(g1b,0)])
    
        # Undecimated "a trous" transform tree AA
        swtCoeffs=(AA_LL,(AA_LH,AA_HL,AA_HH))
        AA_LL=np.roll(np.roll(pywt.iswt2((swtCoeffs,),wavelet=(lvl2WvtA,lvl2WvtA)),-1,axis=0),-1,axis=1)
        # Undecimated "a trous" transform tree AB
        swtCoeffs=(AB_LL,(AB_LH,AB_HL,AB_HH))
        AB_LL=np.roll(np.roll(pywt.iswt2((swtCoeffs,),wavelet=(lvl2WvtA,lvl2WvtB)),-1,axis=0),-1,axis=1)
        # Undecimated "a trous" transform tree BA
        swtCoeffs=(BA_LL,(BA_LH,BA_HL,BA_HH))
        BA_LL=np.roll(np.roll(pywt.iswt2((swtCoeffs,),wavelet=(lvl2WvtB,lvl2WvtA)),-1,axis=0),-1,axis=1)
        # Undecimated "a trous" transform tree BB
        swtCoeffs=(BB_LL,(BB_LH,BB_HL,BB_HH))
        BB_LL=np.roll(np.roll(pywt.iswt2((swtCoeffs,),wavelet=(lvl2WvtB,lvl2WvtB)),-1,axis=0),-1,axis=1)
    
    return (AA_LL,AB_LL,BA_LL,BB_LL)

# Load image
image=np.genfromtxt(sys.argv[1],delimiter=",")
image=image[::-1,:]
nLevels=6

if image.shape[0] != image.shape[1]:
    raise Exception("Image is not square!")
if image.shape[0] % 2**nLevels != 0 or image.shape[1] % 2**nLevels != 0:
    raise Exception("Image dimensions must be multiples of %d." % 2**nLevels)

plt.imshow(image)
plt.show()
transform = dtcwt.Transform2d(biort='near_sym_b', qshift='qshift_b')
dataT = transform.forward(image,nlevels=nLevels,include_scale=True)

# h0o, g0o, h1o and g1o (according to docs and DTCWT code) 
lvl1Filters=dtcwt.coeffs.biort("near_sym_b")
lvl2Filters=dtcwt.coeffs.qshift("qshift_b")

fac=1.0
h0o=np.roll(np.pad(fac*lvl1Filters[0][::1,0],[3,3],mode='constant')[::1],1).tolist()
g0o=np.roll(np.pad(fac*lvl1Filters[1][::1,0],[0,0],mode='constant')[::1],0).tolist()
h1o=np.roll(np.pad(fac*lvl1Filters[2][::1,0],[0,0],mode='constant')[::1],1).tolist()
g1o=np.roll(np.pad(fac*lvl1Filters[3][::1,0],[3,3],mode='constant')[::1],0).tolist()

# phi (0)
h0a=lvl2Filters[0][:,0]
h0b=lvl2Filters[1][:,0]
g0a=lvl2Filters[2][:,0]
g0b=lvl2Filters[3][:,0]

# psi (1)
h1a=lvl2Filters[4][:,0]
h1b=lvl2Filters[5][:,0]
g1a=lvl2Filters[6][:,0]
g1b=lvl2Filters[7][:,0]  

lvl1Coeffs=udDTCWTLvl1(image,lvl1Filters)
lvl2Coeffs=udDTCWTLvl2(lvl1Coeffs,lvl2Filters,nLevels)

lowPass=udDTCWTLvl2Inv(lvl2Coeffs,lvl2Filters)
lowPass=udDTCWTLvl1Inv((lvl2Coeffs[0][0],lowPass),lvl1Filters)

lowPass0=lowPass[0]
lowPass1=np.roll(lowPass[1],-2,axis=1)
lowPass2=np.roll(lowPass[2],-2,axis=0)
lowPass3=np.roll(np.roll(lowPass[3],-2,axis=0),-2,axis=1)
plt.imshow(lowPass0+lowPass1+lowPass2+lowPass3)
plt.colorbar()
plt.show()
plt.subplot(2,3,1)
plt.imshow(lowPass0)
plt.colorbar()
plt.subplot(2,3,2)
plt.imshow(lowPass1)
plt.colorbar()
plt.subplot(2,3,3)
plt.imshow(lowPass2)
plt.colorbar()
plt.subplot(2,3,4)
plt.imshow(lowPass3)
plt.colorbar()
plt.subplot(2,3,5)
plt.imshow(image)
plt.colorbar()
plt.subplot(2,3,6)
plt.imshow(lowPass0+lowPass1+lowPass2+lowPass3)
plt.colorbar()
plt.show()

coeffs=lvl2Coeffs[0]
for iLev in range(nLevels-1,nLevels):

    maxr=-1.0e10
    minr=1.0e10
    maxi=-1.0e10
    mini=1.0e10
    maxa=-1.0e10
    mina=1.0e10
    for iOri in range(6):
        print(coeffs[iLev][iOri].shape)
        maxrp=np.max(np.real(coeffs[iLev][iOri]))
        minrp=np.min(np.real(coeffs[iLev][iOri]))
        maxip=np.max(np.imag(coeffs[iLev][iOri]))
        minip=np.min(np.imag(coeffs[iLev][iOri]))
        maxap=np.max(np.absolute(coeffs[iLev][iOri]))
        minap=np.min(np.absolute(coeffs[iLev][iOri]))
        print(maxrp,minrp,maxip,minip,maxap,minap)
        if maxrp > maxr:
            maxr=maxrp
        if minrp < minr:
            minr=minrp
        if maxip > maxi:
            maxi=maxip
        if minrp < mini:
            mini=minip
        if maxap > maxa:
            maxa=maxap
        if minap < mina:
            mina=minap

    plt.clf()
    angles=["$-75^{\circ}$","$-45^{\circ}$","$-15^{\circ}$","$+15^{\circ}$","$+45^{\circ}$","$+75^{\circ}$"]
    for iOri in range(6):
        plt.subplot(4,6,iOri+1)
        plt.title(angles[iOri])
        im=plt.imshow(np.real(coeffs[iLev][iOri]),vmin=minr,vmax=maxr,cmap=plt.get_cmap('RdBu_r'))
        plt.xticks([])
        plt.yticks([])
        if iOri==0:
            plt.ylabel("Real UD")
    for iOri in range(6):
        plt.subplot(4,6,6+iOri+1)
        im=plt.imshow(np.real(dataT.highpasses[iLev][:,:,iOri]),vmin=minr,vmax=maxr,cmap=plt.get_cmap('RdBu_r'))
        plt.xticks([])
        plt.yticks([])
        if iOri==0:
            plt.ylabel("Real D")
    for iOri in range(6):
        plt.subplot(4,6,12+iOri+1)
        im=plt.imshow(np.absolute(coeffs[iLev][iOri]),vmin=mina,vmax=maxa,cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        if iOri==0:
            plt.ylabel("Abs UD")
    for iOri in range(6):
        plt.subplot(4,6,18+iOri+1)
        im=plt.imshow(np.absolute(dataT.highpasses[iLev][:,:,iOri]),vmin=mina,vmax=maxa,cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        if iOri==0:
            plt.ylabel("Abs D")
    plt.show()

