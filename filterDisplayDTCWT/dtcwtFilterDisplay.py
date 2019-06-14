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

import numpy as np
import matplotlib as mpl
mpl.rcParams['image.interpolation']='nearest'
import matplotlib.pyplot as plt
import colormaps

import dtcwt

lvl1Filters=dtcwt.coeffs.biort("near_sym_b")#_bp")
#lvl1Filters=dtcwt.coeffs.biort("antonini")
lvl2Filters=dtcwt.coeffs.qshift("qshift_b")#_bp")
#lvl2Filters=dtcwt.coeffs.qshift("qshift_06")

# index a/b is for odd/even
# index h/g is for tree 1 / tree 2
# index 0/1 is for low-pass / hi-pass

# phi (0)
h0a=lvl2Filters[0]
h0b=lvl2Filters[1]
g0a=lvl2Filters[2]
g0b=lvl2Filters[3]

# psi (1)
h1a=lvl2Filters[4]
h1b=lvl2Filters[5]
g1a=lvl2Filters[6]
g1b=lvl2Filters[7]

# iterative convolution to construct filters at longer length scales
def scaledFilt(f,level):
    cur=f
    for i in range(level):
        nxt=np.convolve(cur,f)
        cur=nxt
    return cur

def iterConvFiltersAOnly(h0a,h1a,level,dloFirst=None,roll=False):

    outScal=[]
    outWav=[]

    h0aCurr=h0a
    h1aCurr=h1a

    for i in range(1,level):
        if i == 1 and type(dloFirst) is np.ndarray:
            dloLast=dloFirst
        else:
            dloLast=h0a

        dloCurr=np.convolve(dloLast,h0aCurr)
        dhiCurr=np.convolve(dloLast,h1aCurr)
        outScal.append(dloCurr)
        outWav.append(dhiCurr)

        dloLast=dloCurr
        h0aCurr=h0a
        h1aCurr=h0b
    return outScal,outWav

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

# pad the lvl 0 filters
lvl1Len=np.max((lvl1Filters[0].shape[0],lvl1Filters[1].shape[0],lvl1Filters[2].shape[0],lvl1Filters[3].shape[0]))
h0o=np.pad(lvl1Filters[0][:,0],[0,(lvl1Len-lvl1Filters[0].shape[0])],mode='constant')
g0o=np.pad(lvl1Filters[1][:,0],[0,(lvl1Len-lvl1Filters[1].shape[0])],mode='constant')
h1o=np.pad(lvl1Filters[2][:,0],[0,(lvl1Len-lvl1Filters[2].shape[0])],mode='constant')
g1o=np.pad(lvl1Filters[3][:,0],[0,(lvl1Len-lvl1Filters[3].shape[0])],mode='constant')

plt.subplot(2,2,1)
plt.plot(h0o)
plt.subplot(2,2,2)
plt.plot(g0o)
plt.subplot(2,2,3)
plt.plot(h1o)
plt.subplot(2,2,4)
plt.plot(g1o)
plt.show()

h0a5,h1a5=iterConvFilters(h0a[:,0],h1a[:,0],5,dloFirst=h0o)
g0a5,g1a5=iterConvFilters(g0a[:,0],g1a[:,0],5,dloFirst=g0o)
h0b5,h1b5=iterConvFilters(h0b[:,0],h1b[:,0],5,dloFirst=h0o)
g0b5,g1b5=iterConvFilters(g0b[:,0],g1b[:,0],5,dloFirst=g0o)
phih=h0a5[-1]
psih=h1a5[-1]
phig=h0b5[-1]
psig=h1b5[-1]

plt.plot(psih)
plt.plot(psig)
plt.plot(np.absolute(psih+1.0j*psig))
plt.show()
plt.plot(np.fft.fftshift(np.fft.fft(psih-1.0j*psig)))
plt.show()

# From Selesnik eqns 43,44 
psi11=np.outer(phih,psih)
psi12=np.outer(psih,phih)
psi13=np.outer(psih,psih)
psi21=np.outer(phig,psig)
psi22=np.outer(psig,phig)
psi23=np.outer(psig,psig)

# From Selesnik eqns 49,50
psi31=np.outer(phig,psih)
psi32=np.outer(psig,phih)
psi33=np.outer(psig,psih)
psi41=np.outer(phih,psig)
psi42=np.outer(psih,phig)
psi43=np.outer(psih,psig)

fac=1.0/np.sqrt(2.0)

# Real parts of complex directional wavelets
d1r=fac*(psi11-psi21)
d2r=fac*(psi12-psi22)
d3r=fac*(psi13-psi23)
d4r=fac*(psi11+psi21)
d5r=fac*(psi12+psi22)
d6r=fac*(psi13+psi23)

# Imag parts of complex directional wavelets
d1i=fac*(psi31+psi41)
d2i=fac*(psi32+psi42)
d3i=fac*(psi33+psi43)
d4i=fac*(psi31-psi41)
d5i=fac*(psi32-psi42)
d6i=fac*(psi33-psi43)

d1=d1r+1.0j*d1i
d2=d2r+1.0j*d2i
d3=d3r+1.0j*d3i
d4=d4r+1.0j*d4i
d5=d5r+1.0j*d5i
d6=d6r+1.0j*d6i

maxrval=np.max((np.max(d1r),np.max(d2r),np.max(d3r),np.max(d4r),np.max(d5r),np.max(d6r)))
maxival=np.max((np.max(d1i),np.max(d2i),np.max(d3i),np.max(d4i),np.max(d5i),np.max(d6i)))
minrval=-1*maxrval
minival=-1*maxival

plt.subplot(3,6,1)
im=plt.imshow(np.real(d1[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('RdBu_r'),vmin=minrval,vmax=maxrval)
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$-75^{\circ}$")
plt.ylabel("Real",size=16)
plt.subplot(3,6,2)
im=plt.imshow(np.real(d6[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('RdBu_r'),vmin=minrval,vmax=maxrval)
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$-45^{\circ}$")
plt.subplot(3,6,3)
im=plt.imshow(np.real(d2[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('RdBu_r'),vmin=minrval,vmax=maxrval)
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$-15^{\circ}$")
plt.subplot(3,6,4)
im=plt.imshow(np.real(d5[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('RdBu_r'),vmin=minrval,vmax=maxrval)
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$+15^{\circ}$")
plt.subplot(3,6,5)
im=plt.imshow(np.real(d3[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('RdBu_r'),vmin=minrval,vmax=maxrval)
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$+45^{\circ}$")
plt.subplot(3,6,6)
im=plt.imshow(np.real(d4[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('RdBu_r'),vmin=minrval,vmax=maxrval)
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$+75^{\circ}$")
plt.subplot(3,6,6+1)
im=plt.imshow(np.imag(d1[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('RdBu_r'),vmin=minival,vmax=maxival)
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$-75^{\circ}$")
plt.ylabel("Imag",size=16)
plt.subplot(3,6,6+2)
im=plt.imshow(np.imag(d6[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('RdBu_r'),vmin=minival,vmax=maxival)
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$-45^{\circ}$")
plt.subplot(3,6,6+3)
im=plt.imshow(np.imag(d2[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('RdBu_r'),vmin=minival,vmax=maxival)
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$-15^{\circ}$")
plt.subplot(3,6,6+4)
im=plt.imshow(np.imag(d5[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('RdBu_r'),vmin=minival,vmax=maxival)
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$+15^{\circ}$")
plt.subplot(3,6,6+5)
im=plt.imshow(np.imag(d3[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('RdBu_r'),vmin=minival,vmax=maxival)
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$+45^{\circ}$")
plt.subplot(3,6,6+6)
im=plt.imshow(np.imag(d4[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('RdBu_r'),vmin=minival,vmax=maxival)
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$+75^{\circ}$")
plt.subplot(3,6,12+1)
im=plt.imshow(np.absolute(d1[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('viridis'))
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$-75^{\circ}$")
plt.ylabel("Abs",size=16)
plt.subplot(3,6,12+2)
im=plt.imshow(np.absolute(d6[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('viridis'))
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$-45^{\circ}$")
plt.subplot(3,6,12+3)
im=plt.imshow(np.absolute(d2[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('viridis'))
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$-15^{\circ}$")
plt.subplot(3,6,12+4)
im=plt.imshow(np.absolute(d5[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('viridis'))
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$+15^{\circ}$")
plt.subplot(3,6,12+5)
im=plt.imshow(np.absolute(d3[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('viridis'))
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$+45^{\circ}$")
plt.subplot(3,6,12+6)
im=plt.imshow(np.absolute(d4[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4]),cmap=plt.get_cmap('viridis'))
#im.axes.get_xaxis().set_visible(False)
#im.axes.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
plt.title("$+75^{\circ}$")
plt.show()
print d4[d1.shape[0]/4:-d1.shape[0]/4,d1.shape[0]/4:-d1.shape[0]/4].shape

d1fft=np.fft.fft2(d1)
d1fftsh=np.fft.fftshift(d1fft)
d2fft=np.fft.fft2(d2)
d2fftsh=np.fft.fftshift(d2fft)
d3fft=np.fft.fft2(d3)
d3fftsh=np.fft.fftshift(d3fft)
d4fft=np.fft.fft2(d4)
d4fftsh=np.fft.fftshift(d4fft)
d5fft=np.fft.fft2(d5)
d5fftsh=np.fft.fftshift(d5fft)
d6fft=np.fft.fft2(d6)
d6fftsh=np.fft.fftshift(d6fft)

plt.subplot(2,6,1)
plt.imshow((np.absolute(np.real(d1fftsh))))
plt.subplot(2,6,7)
plt.imshow((np.absolute(np.imag(d1fftsh))))
plt.subplot(2,6,2)
plt.imshow((np.absolute(np.real(d2fftsh))))
plt.subplot(2,6,8)
plt.imshow((np.absolute(np.imag(d2fftsh))))
plt.subplot(2,6,3)
plt.imshow((np.absolute(np.real(d3fftsh))))
plt.subplot(2,6,9)
plt.imshow((np.absolute(np.imag(d3fftsh))))
plt.subplot(2,6,4)
plt.imshow((np.absolute(np.real(d4fftsh))))
plt.subplot(2,6,10)
plt.imshow((np.absolute(np.imag(d4fftsh))))
plt.subplot(2,6,5)
plt.imshow((np.absolute(np.real(d5fftsh))))
plt.subplot(2,6,11)
plt.imshow((np.absolute(np.imag(d5fftsh))))
plt.subplot(2,6,6)
plt.imshow((np.absolute(np.real(d6fftsh))))
plt.subplot(2,6,12)
plt.imshow((np.absolute(np.imag(d6fftsh))))
plt.show()

plt.subplot(1,6,1)
plt.imshow(np.absolute(d1fftsh)[162-16:162+16,162-16:162+16],cmap=plt.get_cmap("viridis"))
plt.xticks([])
plt.yticks([])
plt.title("Abs,$-75^{\circ}$")
plt.subplot(1,6,2)
plt.imshow(np.absolute(d6fftsh)[162-16:162+16,162-16:162+16],cmap=plt.get_cmap("viridis"))
plt.xticks([])
plt.yticks([])
plt.title("Abs,$-45^{\circ}$")
plt.subplot(1,6,3)
plt.imshow(np.absolute(d2fftsh)[162-16:162+16,162-16:162+16],cmap=plt.get_cmap("viridis"))
plt.xticks([])
plt.yticks([])
plt.title("Abs,$-15^{\circ}$")
plt.subplot(1,6,4)
plt.imshow(np.absolute(d5fftsh)[162-16:162+16,162-16:162+16],cmap=plt.get_cmap("viridis"))
plt.xticks([])
plt.yticks([])
plt.title("Abs,$+15^{\circ}$")
plt.subplot(1,6,5)
plt.imshow(np.absolute(d3fftsh)[162-16:162+16,162-16:162+16],cmap=plt.get_cmap("viridis"))
plt.xticks([])
plt.yticks([])
plt.title("Abs,$+45^{\circ}$")
plt.subplot(1,6,6)
plt.imshow(np.absolute(d4fftsh)[162-16:162+16,162-16:162+16],cmap=plt.get_cmap("viridis"))
plt.xticks([])
plt.yticks([])
plt.title("Abs,$+75^{\circ}$")
plt.show()

