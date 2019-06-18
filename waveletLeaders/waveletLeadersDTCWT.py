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

# uses : DTDWT from https://pypi.python.org/pypi/dtcwt

import h5py
import numpy as np
import matplotlib as mpl
mpl.rcParams['image.interpolation']='nearest'
mpl.rcParams['font.size'] = 14
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
import scipy.ndimage

import dtcwt

import legendreTransformation
import fracInt

import colormaps # from dir ../pylib

batchFlag=os.getenv("BATCHMODE")
if batchFlag == None or batchFlag=="0":
    batchMode=False
    wlThresh=float(sys.argv[2])
else:
    # Batch processing selected if BATCHMODE defined and set to 1
    batchMode=True   
    wlThresh=1.0e-8

# Load noise
noise=np.genfromtxt(sys.argv[1],delimiter=",")
ny=noise.shape[0]
nx=noise.shape[1]

# compute DTDWT 
nLevels = int(np.floor(max(np.log2(ny),np.log2(nx))))

# Uncomment this to apply fractional integration before WaveletLeaders.
# This might help to resolve singularity spectrum in some cases (possibly due to presence of negative Holder exponents).
noise=fracInt.fractionalIntegration(noise,-0.5)

# Do not use rotationally-invariant transform. Adantage is much shorter filters. Disadvantage is that maximum mod over orientations may not be meaningful?
# Use length 10 'a' filters here, rather than the length 14 'b' (or length 18 'b_bp'). 
transform = dtcwt.Transform2d(biort='near_sym_a', qshift='qshift_a')
dataT = transform.forward(noise,nlevels=nLevels)
lvl1Filters=dtcwt.coeffs.biort('near_sym_a')
lvl2Filters=dtcwt.coeffs.qshift('qshift_a')

## Use rotationally-invariant transform. No inversion is required and need to ensure that maximum modulus over orientations is meaningful.
#transform = dtcwt.Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
#dataT = transform.forward(noise,nlevels=nLevels)
#lvl1Filters=dtcwt.coeffs.biort('near_sym_b_bp')
#lvl2Filters=dtcwt.coeffs.qshift('qshift_b_bp')

# Routine to calculate effective filters at required level. Used to calculate length of filter.
def iterConvFilters(h0a,h1a,level,dloFirst=None):
    if type(dloFirst) is np.ndarray:
        dloLast=dloFirst
    else:
        dloLast=h0a

    scal=[]
    wav=[]

    for iLev in range(level):
        dloCurr=np.convolve(dloLast,np.repeat(h0a,2**(iLev)))
        dhiCurr=np.convolve(dloLast,np.repeat(h1a,2**(iLev)))
        scal.append(dloCurr)
        wav.append(dhiCurr)
        dloLast=dloCurr
    return scal,wav

# Construct the effective filters on each of the nLevels levels. This is needed to compute the support.
lvl1Len=np.max((lvl1Filters[0].shape[0],lvl1Filters[1].shape[0],lvl1Filters[2].shape[0],lvl1Filters[3].shape[0]))
h0o=np.pad(lvl1Filters[0][:,0],[0,(lvl1Len-lvl1Filters[0].shape[0])],mode='constant')
h0a=np.array(lvl2Filters[0])
h1a=np.array(lvl2Filters[4])
h0a5,h1a5=iterConvFilters(h0a[:,0],h1a[:,0],nLevels,dloFirst=h0o)

# Regrid each sub-band of hp coefficients to the finest grid size and renormalize L2 -> L1.
# Need to use L1 norm for Wavelet Leaders.
coeffsRescaled=np.empty((nLevels,6,dataT.highpasses[0].shape[0],dataT.highpasses[0].shape[1]))
ln2Scales=[]
for iLev in range(nLevels):
    hp=dataT.highpasses[iLev]
    print hp.shape
    # Create index arrays that label block-subdivisions of the hp coefficients at this level
    nny=(ny/2)/hp.shape[0]
    nnx=(nx/2)/hp.shape[1]
    newArrIndsY=np.floor_divide(np.arange(ny/2),nny)
    newArrIndsX=np.floor_divide(np.arange(nx/2),nnx)
    yy,xx=np.meshgrid(newArrIndsY,newArrIndsX)
    ln2Scales.append(iLev+1)
    for iOri in range(6):
        # Put the hp coefficients on highest resolution grid and take absolute value
        coeffsRescaled[iLev,iOri]=np.absolute(hp[xx.flatten(),yy.flatten(),iOri].reshape((nx/2,ny/2)))
        # Renormalize from L2 to L1-norm by multiplying by a factor of 1/sqrt(2) per level per dimension.
        coeffsRescaled[iLev,iOri]*=np.power(2.0,-1.0*float(iLev))

# Convert ln2Scales to np.array
ln2Scales=np.array(ln2Scales)

# If True, use Wavelet Leaders method to determine partition function summands, otherwise just use the maximum over orientations at each i,j.
useLeaders=True
#useLeaders=False

# Compute maxima over orientations for each dyadic cube 
oriMax=np.max(coeffsRescaled,axis=1)

# If not in batch mode, display the orientational maxima ( on first of two rows )
if batchMode==False:
    imCount=0
    fig,axes=plt.subplots(nrows=2,ncols=4)
    axesFlat=axes.flat
    for iLev in range(1,5):
        im=axesFlat[imCount].imshow(oriMax[iLev],cmap='viridis',vmin=0.0,vmax=0.5)
        axesFlat[imCount].invert_yaxis()
        axesFlat[imCount].set_title("Level %d"%iLev)
        if iLev==1:
            axesFlat[imCount].set_ylabel("$|W_{ij}|}$",rotation=0,labelpad=25,fontsize=18)# % (iLev+1))
        if iLev>1:
            axesFlat[imCount].set_yticks([])
        for tick in axesFlat[imCount].get_xticklabels():
            tick.set_rotation(-90.0)
        imCount+=1


# Compute neighbourhood maxima columns for each scale.
minScale=2
maxScale=6
waveletLeaders=np.empty_like(oriMax)
for iLev in range(nLevels):
    if useLeaders==True:
        # Extended WLs
        if iLev == 0:
            nbSz=3*h0o.shape[0]
        else:
            nbSz=3*h0a5[iLev-1].shape[0]
        iScale=ln2Scales[iLev] 
        print "Scale: " + str(iLev-1) + " nbSz: " + str(nbSz)

        thisScaleMaxima=np.empty_like(oriMax[0:iLev+1,:,:])
        # Create an array of maxima along columns up to scale iLev
        if iLev==0:
            thisScaleMaxima[0,:,:]=scipy.ndimage.maximum_filter(oriMax[0,:,:],size=nbSz)
        else:
            for iiLev in range(iLev+1):
                thisScaleMaxima[iiLev,:,:] = scipy.ndimage.maximum_filter(oriMax[iiLev,:,:],size=nbSz)
        waveletLeaders[iLev,:,:]=np.max(thisScaleMaxima,axis=0)
    else:
        waveletLeaders[iLev,:,:] = oriMax[iLev,:,:]

    # If not in batch mode, display the wavelet leaders ( on the second of two rows )
    if batchMode==False:
        if iLev >=0 and iLev < 4:
            im=axesFlat[imCount].imshow(waveletLeaders[iLev],cmap='viridis',vmin=0.0,vmax=0.5)
            axesFlat[imCount].invert_yaxis()
            for tick in axesFlat[imCount].get_xticklabels():
               tick.set_rotation(-90.0)
            if iLev==0:
                axesFlat[imCount].set_ylabel("$L_{X}^{(ext)}$",rotation=0,labelpad=25,fontsize=18)# % (iLev+1))
            if iLev>0:
                axesFlat[imCount].set_yticks([])
            imCount+=1

# If not in batch mode, display orientational maxima and wavelet leaders.
if batchMode==False:
    fig.subplots_adjust(bottom=0.2,wspace=0.2)#,top=0.6)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
    cb=fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cb.ax.set_title("L1 modulus of coefficient / leader")
    plt.show()

# Compute qth order structure functions
nq=120
qmin=-4.0
qmax=8.0
qstep=(qmax-qmin)/float(nq)
qvals=np.linspace(qmin,qmax,num=nq+1,endpoint=True)

# Compute qth order structure functions
S=np.zeros((nLevels,nq))
SNN=np.zeros((nLevels,nq))
for iLev in range(maxScale):
    minWLs=np.min(waveletLeaders[iLev])
    maxWLs=np.max(waveletLeaders[iLev])
    print minWLs,maxWLs
    if ( np.absolute(maxWLs-minWLs) > wlThresh ) and ( wlThresh > 0.0 ):
        nzWLs=np.where(waveletLeaders[iLev]>minWLs+wlThresh*(maxWLs-minWLs))
    else:
        nzWLs=np.where(waveletLeaders[iLev]<np.inf)
    nj=len(nzWLs[0])
    f=1
    print "Number of non-zero WLs at level " + str(iLev) + ": " + str(nj)
    for iq in range(nq):
        S[iLev,iq]=(1.0/float(nj))*np.sum(np.power(f*waveletLeaders[iLev][nzWLs],qvals[iq]))
        if np.isnan(S[iLev,iq]) or np.isinf(S[iLev,iq]):
            print nj
            print waveletLeaders[iLev][nzWLs]
            print qvals[iq]
        SNN[iLev,iq]=np.sum(np.power(f*waveletLeaders[iLev][nzWLs],qvals[iq]))
        if np.isnan(SNN[iLev,iq]) or np.isinf(SNN[iLev,iq]):
            print waveletLeaders[iLev][nzWLs]
            print qvals[iq]

# Display S(a,q) vs a at selected scales
if batchMode==False:
    cmap=plt.get_cmap('jet')
    for iq in range(0,nq,10):
        ln2S=np.log2(S[:,iq])
        m,b=np.polyfit(ln2Scales[minScale:maxScale],ln2S[minScale:maxScale],1)#/np.log(2),1)
        plt.scatter(ln2Scales[minScale:maxScale],ln2S[minScale:maxScale],color='k',marker='o')
        fitvals=m*ln2Scales[minScale:maxScale]+b
        plt.plot(ln2Scales[minScale:maxScale],fitvals,color='k',label="$q=%3.2f\,(m=%3.2f)$"%(qvals[iq],m))
        plt.gca().text(ln2Scales[maxScale-1]-0.8,
                       fitvals[-1],
                       "$q=%3.2f$"%(qvals[iq]),
                       backgroundcolor=[1.0,1.0,1.0,0.8],
                       size=16,
                       rotation=(180.0/np.pi)*np.arctan((1.0/20.0)*m))
    plt.ylabel("$\mathrm{log}_{2}(S(q,a))$")
    plt.xlabel("$\mathrm{log}_{2}(a)$")
    plt.ylim([-70.0,110.0])
    plt.show()

    # Compute and display tau(q) using slopes of structure functions vs. scale
    tauq=np.empty((nq))
    for iq in range(nq):
        ln2S=np.log2(S[:,iq])
        m,b=np.polyfit(ln2Scales[minScale:maxScale],ln2S[minScale:maxScale],1)
        tauq[iq]=m-2
    plt.scatter(qvals[:nq],tauq,marker='o')
    plt.xlabel("Moment $q$")
    plt.ylabel("$\\tau(q)$")
    plt.show()
    
    # Compute D(h) using direct Legendre Transformation of tau(q)
    hvals,dvals=legendreTransformation.transform(tauq,qvals[:nq],xstep=qstep)
    plt.scatter(np.array(hvals),dvals,marker='^')
    plt.xlabel(u'Hölder exponent $\\alpha$')
    plt.ylabel("Dimension $D(\\alpha)$")
    plt.show()

# Compute D(h) using thermodynamical approach (Partition Function, Free Energy, Entropy) 
U=np.zeros((nLevels,nq))
V=np.zeros((nLevels,nq))
tmpu1=np.zeros((nLevels,nq))
tmpu2=np.zeros((nLevels,nq))
tmpv=np.zeros((nLevels,nq))
nj=np.zeros((nLevels))
for iLev in range(maxScale):
    maxWLs=np.max(waveletLeaders[iLev])
    minWLs=np.min(waveletLeaders[iLev])
    if ( np.absolute(maxWLs-minWLs)>1.0e-6 ):
        nzWLs=np.where(waveletLeaders[iLev]>minWLs+wlThresh*(maxWLs-minWLs))
    else:
        nzWLs=np.where(waveletLeaders[iLev]<np.inf)
    nj[iLev]=len(nzWLs[0])
    f=1
    for iq in range(nq):
        lx=f*waveletLeaders[iLev][nzWLs]
        lxq=np.power(lx,qvals[iq])
        s=SNN[iLev,iq] # unnormed PF
        tmpv[iLev,iq]=np.sum(lxq*np.log2(lx))
        tmpu1[iLev,iq]=np.sum(lxq*np.log2(lxq))
        tmpu2[iLev,iq]=np.sum(lxq)
        tmpu=tmpu1[iLev,iq]-np.log2(s)*tmpu2[iLev,iq]
        U[iLev,iq]=tmpu/s + np.log2(nj[iLev])
        V[iLev,iq]=tmpv[iLev,iq]/s

if batchMode==True:
    np.savetxt(sys.argv[2],tmpu1,delimiter=",")
    np.savetxt(sys.argv[3],tmpu2,delimiter=",")
    np.savetxt(sys.argv[4],tmpv,delimiter=",")
    np.savetxt(sys.argv[5],nj,delimiter=",")
    np.savetxt(sys.argv[6],SNN,delimiter=",")
else:
    # Display h(q,a) vs log2(a) function at selected scales
    order=1
    hfit=np.empty((nq,order+1))
    dfit=np.empty((nq,order+1))
    for iq in range(nq):
        hfit[iq,:]=np.polyfit(ln2Scales[minScale:maxScale],V[minScale:maxScale,iq],order)
        dfit[iq,:]=np.polyfit(ln2Scales[minScale:maxScale],U[minScale:maxScale,iq],order)
    plt.scatter(np.array(hfit[:,0]),dfit[:,0]+2,marker='o')
    plt.xlabel(u'Hölder exponent $\\alpha$')
    plt.ylabel("Dimension $D(\\alpha)$")
    plt.show()
    
