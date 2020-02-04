#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import h5py

mpl.rcParams['image.interpolation']='nearest'
mpl.rcParams['font.size'] = 20

#font = {'family' : 'normal',
#        'weight' : 'normal',
#        'size'   : 20}

#mpl.rc('font', **font)

r4Width = 2.0
i4FFTSize = 1024
i4Nyquest = 768

nLevels = 4
r4ScaleRatio = 0.1
while r4ScaleRatio < 0.42:
    nLevels += 1
    r4ScaleRatio = np.power( 2.0 / float(i4FFTSize), 1.0 / float( nLevels - 1 ) );

print("Scale ratio: " + str(r4ScaleRatio))
print("Num levels: " + str(nLevels))

r4BandpassFilter=np.zeros((nLevels,i4Nyquest+1))
r4FilterSum=np.zeros(i4Nyquest+1)
r4CentreWaveLength = float(i4FFTSize)
centWls=[]
for i4ScaleLevel in range(nLevels):
    r4CentreFreq = 1.0 / r4CentreWaveLength
    for i4WaveNumber in range(1,i4Nyquest+1):
        r4Freq = float(i4WaveNumber)/float(i4FFTSize);
        if ( r4Freq > r4CentreFreq ):
            r4RelFrequency = r4Freq/r4CentreFreq;
        else:
            r4RelFrequency = r4CentreFreq/r4Freq;
        r4Filter = np.exp(-r4Width*r4RelFrequency);
        r4BandpassFilter[i4ScaleLevel,i4WaveNumber] = r4Filter;
        r4FilterSum[i4WaveNumber] += r4Filter;
    print(r4CentreWaveLength)
    centWls.append(r4CentreWaveLength)
    r4CentreWaveLength *= r4ScaleRatio


for i4ScaleLevel in range(nLevels):
    r4BandpassFilter[i4ScaleLevel,:]/=r4FilterSum[:]
    r4BandpassFilter[i4ScaleLevel,0]=0.0

inFn1=sys.argv[1]
inFn2=sys.argv[2]
inData1=np.genfromtxt(inFn1,delimiter=",")[::-1,:]
inData2=np.genfromtxt(inFn2,delimiter=",")[::-1,:]
dataLn1=np.log2(inData1)
dataLn2=np.log2(inData2)

## compute width of filter
#iFFTFilt=np.absolute(np.fft.fftshift(np.fft.fft(r4BandpassFilter[i4ScaleLevel,:])))
##plt.plot(iFFTFilt)
##plt.show()
#totFiltWgt=np.sum(np.absolute(np.square(iFFTFilt)))
#iFFTFilt/=np.sqrt(totFiltWgt) # 2-norm = 1
#filtDisp=np.linspace(0,iFFTFilt.shape[0],num=iFFTFilt.shape[0],endpoint=False)/iFFTFilt.shape[0]
#filtMean=np.sum(filtDisp*np.absolute(np.square(iFFTFilt)))
#print("Filter mean (real space):",filtMean)
##filtMeanPix=int(np.round(iFFTFilt.shape[0]*filtMean))
##print("Filter mean pix:",filtMeanPix)
#filtDisp-=filtMean
#plt.plot(filtDisp)
#plt.show()
##totFiltWgt=np.sum(np.absolute(iFFTFilt))
#wgtFilt=np.sum(np.square(filtDisp)*np.absolute(np.square(iFFTFilt)))
##filtStd=np.sqrt(wgtFilt/totFiltWgt)
#filtStd=np.sqrt(wgtFilt)
#print("Filter width (real space): ",filtStd*iFFTFilt.shape[0])

wccs=[]
lambds=[]

for iSc in range(r4BandpassFilter.shape[0]):

    inData1FFT=np.fft.fftshift(np.fft.fft2(dataLn1))
    inData2FFT=np.fft.fftshift(np.fft.fft2(dataLn2))
    for iky in range(inData1FFT.shape[0]):
        for ikx in range(inData1FFT.shape[1]):
            ikyy=iky-inData1FFT.shape[0]/2
            ikxx=ikx-inData1FFT.shape[1]/2
            if ikyy==0 and ikxx==0:
                inData1FFT[iky,ikx]=0
                continue
            k=np.sqrt(ikyy*ikyy+ikxx*ikxx)
            fac=r4BandpassFilter[iSc,int(k)]
            inData1FFT[iky,ikx]*=fac
            inData2FFT[iky,ikx]*=fac
    inData1Filt=np.real(np.fft.ifft2(np.fft.ifftshift(inData1FFT)))
    inData2Filt=np.real(np.fft.ifft2(np.fft.ifftshift(inData2FFT)))
#    plt.subplot(1,2,1)
#    plt.imshow(inData1Filt)
#    plt.subplot(1,2,2)
#    plt.imshow(inData2Filt)
#    plt.show()

#    r=np.corrcoef(inData1Filt.flat,inData2Filt.flat)
#    print("R:",r[0,1])

    cSub1=inData1Filt
    cSub2=inData2Filt
    # Compute wavelet cross-correlation over all sub-bands
    # (For general WCC explanation, Addison: Introduction to redundancy rules, 2018, A 29)
#    crossTerm=np.absolute(np.sum(np.conj(cSub1)*cSub2))
    crossTerm=np.sum(np.conj(cSub1)*cSub2)
    cSub1Term=np.sum(np.square(np.absolute(cSub1)))
    cSub2Term=np.sum(np.square(np.absolute(cSub2)))
    wcc=crossTerm/(np.sqrt(cSub1Term*cSub2Term))

    print(wcc,centWls[iSc])
    wccs.append(wcc)
    lambds.append(centWls[iSc])
np.savetxt("out_steps_wccs.txt",wccs,delimiter=",")
np.savetxt("out_steps_lambds.txt",lambds,delimiter=",")


