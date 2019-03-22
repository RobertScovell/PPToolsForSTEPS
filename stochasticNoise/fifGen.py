#!/usr/bin/env python2.7
# encoding: utf-8
# (C) Crown Copyright 2019, the Met Office.

# This code generates 2D realisations of the Fractionally Integrated Flux model, based on an algorithm by Lovejoy and Schertzer (2010).
# Parameters are:
# 1. Number of rows in image (not tested with unequal number of rows x columns )
# 2. Number of columns in image
# 3. Multifractality parameter ( alpha = 2.0 => lognormal cascade, alpha = 1.0 => Cauchy (not tested) ). Typically alpha = [1.5,1.8] for rain.
# 4. Codimension of the mean (see Lovejoy and Schertzer, 2013). This is the sparseness of the mean value. C1=0 => homogeneous.
# 5. Hurst parameter ( H=0.0 -> 1.0 ) used for fractional integration. Typically H=0.33 is reasonable for rain.
# 6. Output CSV file.

# Lovejoy, S. and Schertzer, D., 2010. On the simulation of continuous in scale universal multifractals, part I: spatially continuous processes. Computers & Geosciences, 36(11), pp.1393-1403.
# Lovejoy, S. and Schertzer, D., 2010. On the simulation of continuous in scale universal multifractals, Part II: Space–time processes and finite size corrections. Computers & Geosciences, 36(11), pp.1404-1413.
# Lovejoy, S. and Schertzer, D., 2013. The weather and climate: emergent laws and multifractal cascades. Cambridge University Press.

# http://www.physics.mcgill.ca/~gang/eprints/eprintLovejoy/neweprint/Continuous.multifractals.partI.3.9.10.pdf
# http://www.physics.mcgill.ca/~gang/eprints/eprintLovejoy/neweprint/Continuous.multifractals.partII.3.9.10.pdf


import sys
import numpy as np
import numpy.fft
import scipy.stats
import scipy.signal

ny=int(sys.argv[1])
nx=int(sys.argv[2])
alpha=float(sys.argv[3])
C1=float(sys.argv[4])
H=float(sys.argv[5])
outFN=sys.argv[6]

def fracEpsilon(scal,alpha):
    lambd=min(scal.shape)#/2.
    rat=2.
    alphaP=1./(1-1/alpha)
    lcut1=lambd/2.
    lcut2=lcut1/rat
    scal4lcut1=np.power(scal/lcut1,4.)
    scal4lcut1[scal4lcut1>200.0]=200.0
    scal4lcut2=np.power(scal/lcut2,4.)
    scal4lcut2[scal4lcut2>200.0]=200.0
    exLambda1=np.exp(-scal4lcut1)
    exLambda2=np.exp(-scal4lcut2)
    sing=np.power(scal,-1.0/alphaP)
    singSmooth1=sing*exLambda1
    singSmooth2=sing*exLambda2
    t1=np.sum(singSmooth1)
    t2=np.sum(singSmooth2)
    A=(np.power(rat,-1./alpha)*t1-t2)/(np.power(rat,-1./alpha)-1.)
    scal3=scal/3.
    scal3[scal3>200.0]=200.0
    ff=np.exp(-scal3)
    singsmooth=sing*ff
    G=np.sum(singsmooth)
    a=-A/G
    sing=np.power(sing*(1.+a*ff),1./(alpha-1.))
    return sing

def eps2D(ny,nx,alpha,C1):
    MDf=np.pi/2.
    x=np.linspace(-(nx-1),nx-1,num=nx,endpoint=True) # steps of 2
    y=np.linspace(-(ny-1),ny-1,num=ny,endpoint=True) # steps of 2
    xx,yy=np.meshgrid(x,y)
    beta=-1
    scal=np.square(xx)+np.square(yy)
    sing=fracEpsilon(scal,alpha)
    rvs=scipy.stats.levy_stable.rvs(alpha=alpha,beta=beta,loc=0.0,scale=1.0,size=(ny,nx))

    #### THIS IS NOT PART OF THE ORIGINAL MATHEMATICA IMPLEMENTATION
    maxrv=np.max(rvs)
    rvs[rvs<-1000.0*maxrv]=-1000.0*maxrv
    #### END

    ggen1=np.power((C1/MDf),1./alpha)*rvs
    ggen1alpha=scipy.signal.fftconvolve(ggen1,sing)
    ggen1alpha[ggen1alpha<-200.]=-200.
    ggen1alpha=np.exp(ggen1alpha)
    ff=np.mean(ggen1alpha)
    eps1alpha=ggen1alpha/ff
    return eps1alpha

#### THIS IS NOT PART OF THE ORIGINAL MATHEMATICA IMPLEMENTATION
umf=eps2D(ny,nx,alpha,C1)
nanMask=np.isnan(umf)
if np.sum(nanMask) != 0:
    print "Warning: converting NaN values to 0.0"
    umf[nanMask]=0.0
#### END

# This part does fractional integration on the pure cascade, leading to the Fractionally Integrated Flux (FIF).
# A value of H=0.0 means that this step is omitted
if ( np.absolute(H) > 1.0e-3 ):
    umfFFT=np.fft.fftshift(np.fft.fft2(umf))
    for iky in range(umfFFT.shape[0]):
        for ikx in range(umfFFT.shape[1]):
            ikyy=iky-umfFFT.shape[0]/2
            ikxx=ikx-umfFFT.shape[1]/2
            if ikyy==0 and ikxx==0:
                continue
            k=np.sqrt(ikyy*ikyy+ikxx*ikxx)
            fac=np.power(k,-H)
            umfFFT[iky,ikx]*=fac
    umfFrac=np.real(np.fft.ifft2(np.fft.ifftshift(umfFFT)))
else:
    umfFrac=umf

# Take only the central window, to avoid edge effects
umfFrac=umfFrac[umfFrac.shape[0]/4:-umfFrac.shape[0]/4,umfFrac.shape[1]/4:-umfFrac.shape[1]/4]
np.savetxt(outFN,umfFrac,delimiter=",",fmt="%f")

