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

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path
import glob
import legendreTransformation

# Load all partition functions
pfDir=sys.argv[1]

cm=plt.get_cmap('jet')
for H in ["0.2","0.3","0.4","0.5","0.6","0.7","0.8"]:
    files=glob.glob(os.path.join(pfDir,'fbm_1024_1024_'+H+'_snn*.txt'))
    files=sorted(files)
    filesS1=glob.glob(os.path.join(pfDir,'fbm_1024_1024_'+H+'_tmpu1*.txt'))
    filesS1=sorted(filesS1)
    filesS2=glob.glob(os.path.join(pfDir,'fbm_1024_1024_'+H+'_tmpu2*.txt'))
    filesS2=sorted(filesS2)
    filesS3=glob.glob(os.path.join(pfDir,'fbm_1024_1024_'+H+'_tmpv*.txt'))
    filesS3=sorted(filesS3)
    filesS4=glob.glob(os.path.join(pfDir,'fbm_1024_1024_'+H+'_nj*.txt'))
    filesS4=sorted(filesS4)
    SdataTmp=np.genfromtxt(files[0],delimiter=",")
    nq=SdataTmp.shape[1]
    ns=SdataTmp.shape[0]
    SNN=np.zeros((ns,nq))
    sumU1=np.zeros((ns,nq))
    sumU2=np.zeros((ns,nq))
    sumV=np.zeros((ns,nq))
    nj=np.zeros((ns))
    
    for i in range(100):
        try:
            Sdata=np.genfromtxt(files[i],delimiter=",")
            SNN+=Sdata
            sumU1data=np.genfromtxt(filesS1[i],delimiter=",")
            sumU1+=sumU1data
            sumU2data=np.genfromtxt(filesS2[i],delimiter=",")
            sumU2+=sumU2data
            sumVdata=np.genfromtxt(filesS3[i],delimiter=",")
            sumV+=sumVdata
            njData=np.genfromtxt(filesS4[i],delimiter="\n")
            nj+=njData
        except:
            "Stopped after " + str(i) + "files."
    ln2Scales=np.arange(0,ns)
    nq=120
    qmin=-4.0
    qmax=8.0
    qstep=(qmax-qmin)/float(nq)
    qvals=np.linspace(qmin,qmax,num=nq+1,endpoint=True)
    
    minScale=2
    maxScale=5
    
    # Compute and display tau(q) using slopes of structure functions vs. scale
    tauq=np.empty((nq))
    for iq in range(nq):
        ln2S=np.log2((1.0/nj[:])*SNN[:,iq])
        m,b=np.polyfit(ln2Scales[minScale:maxScale],ln2S[minScale:maxScale],1)
        tauq[iq]=m-2
    m,b=np.polyfit(qvals[:nq],tauq,1)
    beta=4.0+tauq[60]
    plt.scatter(qvals[:nq],tauq,marker='o',label="H=%s (m=%3.2f)" % (H,m),color=cm(float(H)))
plt.legend(loc=2)
plt.xlabel("Moment $q$")
plt.ylabel("$\\tau(q)$")
plt.title("$\\tau(q)$ vs. $q$ for various $H$")
plt.show()

