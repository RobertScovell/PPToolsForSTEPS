#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import h5py

import dtcwt

mpl.rcParams['image.interpolation']='nearest'
mpl.rcParams['font.size'] = 20

inFn1=sys.argv[1]
inFn2=sys.argv[2]
inData1=np.genfromtxt(inFn1,delimiter=",")[::-1,:]
inData2=np.genfromtxt(inFn2,delimiter=",")[::-1,:]
data1=inData1
data2=inData2

imageLn1=np.log2(data1)
imageLn2=np.log2(data2)

transform = dtcwt.Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
nLevelsT=int(np.ceil(np.log2(data1.shape[0])))
dataT1 = transform.forward(imageLn1,nlevels=nLevelsT)
dataT2 = transform.forward(imageLn2,nlevels=nLevelsT)


inDataFFT1=np.fft.fftshift(np.fft.fft2(imageLn1))
inDataFFT2=np.fft.fftshift(np.fft.fft2(imageLn2))
inDataFFTAR1=np.copy(inDataFFT1)
stepsAutoCorr=[]
dtcwtAutoCorr=[]

w=[]
wls=[]
wcompl=[]
for ik in range(inDataFFT1.shape[0]*2):
    # time step in minutes
    deltat=5
    # lifetime at wavelength=1 in minutes
    tau0=240
    # lifetime at wavelength in minutes
    if ik==0:
        wavelength=0.
    else:
        wavelength=1./float(ik)
    tauk=np.power(wavelength,1.75)*tau0
    alphak=np.exp(-deltat/tauk)
    w.append(alphak)
    wls.append(wavelength)
    wcompl.append(np.sqrt(1.0-alphak*alphak))

for iky in range(inDataFFT1.shape[0]):
    for ikx in range(inDataFFT1.shape[1]):
        ikyy=iky-inDataFFT1.shape[0]/2
        ikxx=ikx-inDataFFT1.shape[1]/2
        if ikyy==0 and ikxx==0:
            inDataFFTAR1[iky,ikx]=0
            continue
        k=np.sqrt(ikyy*ikyy+ikxx*ikxx)
        ik=int(np.round(k))
        inDataFFTAR1[iky,ikx]=w[ik]*inDataFFT1[iky,ikx]+wcompl[ik]*inDataFFT2[iky,ikx]

plt.semilogx(1024*np.array(wls[1:512]),w[1:512],label="$r$")
plt.semilogx(1024*np.array(wls[1:512]),wcompl[1:512],label="$\sqrt{1-r^2}$")
plt.legend()
plt.show()
plt.subplot(1,3,1)
plt.imshow(imageLn1)
plt.subplot(1,3,2)
plt.imshow(imageLn2)
ar1Blend=np.fft.ifft2(np.fft.ifftshift(inDataFFTAR1))
plt.subplot(1,3,3)
plt.imshow(np.real(ar1Blend))
plt.show()

np.savetxt(sys.argv[3],np.power(2.,np.real(ar1Blend)),delimiter=",")

