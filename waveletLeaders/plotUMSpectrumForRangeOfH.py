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
import matplotlib.pyplot as plt
import sys
import os.path
import glob

# Load all partition functions
pfDir=sys.argv[1]
alpha=sys.argv[2]
colCount=0
cm=plt.get_cmap('jet')
for H in ["0.2","0.3","0.4","0.5","0.6","0.7","0.8"]:
    files=glob.glob(os.path.join(pfDir,'s_'+H+'_'+alpha+'*.txt'))
    files=sorted(files)
    filesS1=glob.glob(os.path.join(pfDir,'u1_'+H+'_'+alpha+'*.txt'))
    filesS1=sorted(filesS1)
    filesS2=glob.glob(os.path.join(pfDir,'u2_'+H+'_'+alpha+'*.txt'))
    filesS2=sorted(filesS2)
    filesS3=glob.glob(os.path.join(pfDir,'v_'+H+'_'+alpha+'*.txt'))
    filesS3=sorted(filesS3)
    filesS4=glob.glob(os.path.join(pfDir,'nj_'+H+'_'+alpha+'*.txt'))
    filesS4=sorted(filesS4)
    SdataTmp=np.genfromtxt(files[0],delimiter=",")
    nq=SdataTmp.shape[1]
    ns=SdataTmp.shape[0]
    SNN=np.zeros((ns,nq))
    sumU1=np.zeros((ns,nq))
    sumU2=np.zeros((ns,nq))
    sumV=np.zeros((ns,nq))
    nj=np.zeros((ns))
    
    for i in range(0,50):
        try:
            print(files[i])
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
    
    # Sum the contributions from each realization   
    U=np.empty((ns,nq))
    V=np.empty((ns,nq))
    
    for iLev in range(2,ns):
        for iq in range(nq):
            s=SNN[iLev,iq]
            sumU=sumU1[iLev,iq]-np.log2(s)*sumU2[iLev,iq]
            U[iLev,iq]=sumU/s + np.log2(nj[iLev])
            V[iLev,iq]=sumV[iLev,iq]/s
    
    # Display D(h) vs. h 
    order=1
    hfit=np.empty((nq,order+1))
    dfit=np.empty((nq,order+1))
    minScale=2
    for iq in range(nq):
        hfit[iq,:]=np.polyfit(ln2Scales[minScale:],V[minScale:,iq],order)
        dfit[iq,:]=np.polyfit(ln2Scales[minScale:],U[minScale:,iq],order)
    plt.scatter(hfit[0:,0],dfit[0:,0]+2,marker='o',color=cm(float(H)),label=("%s" % H))
plt.legend()
plt.ylim(0.0,2.2)
plt.show()
 
