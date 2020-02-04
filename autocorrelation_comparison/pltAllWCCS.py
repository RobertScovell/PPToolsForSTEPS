#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

w=[]
wls=[]
#wcompl=[]
#for iLambd in range(2,1024):
for ik in range(1,512):#inDataFFT1.shape[0]*2):

    # time step in minutes
    deltat=5
    # lifetime at wavelength=1 in minutes
    tau0=240
    # lifetime at wavelength in minutes
    wavelength=1./float(ik)
#    wavelength=float(iLambd)
    tauk=np.power(wavelength,1.75)*tau0
    alphak=np.exp(-deltat/tauk)
    w.append(alphak)
    wls.append(wavelength)
#    wcompl.append(np.sqrt(1.0-alphak*alphak))


stepsWCCs=np.genfromtxt("out_steps_wccs.txt",delimiter=",")
stepsLambdas=np.genfromtxt("out_steps_lambds.txt",delimiter=",")
mexWCCs=np.genfromtxt("out_mex_wccs.txt",delimiter=",")
mexLambdas=np.genfromtxt("out_mex_lambds.txt",delimiter=",")
dtcwtWCCs=np.genfromtxt("out_dtcwt_wccs.txt",delimiter=",")
dtcwtLambdas=np.genfromtxt("out_dtcwt_lambds.txt",delimiter=",")
uddtcwtWCCs=np.genfromtxt("out_uddtcwt_wccs.txt",delimiter=",")
uddtcwtLambdas=np.genfromtxt("out_uddtcwt_lambds.txt",delimiter=",")

#print(wls)
plt.semilogx(1024.*np.array(wls),np.array(w),label="Ref.")
plt.semilogx(2*uddtcwtLambdas,uddtcwtWCCs,label="UD-DTCWT")
plt.semilogx(4*dtcwtLambdas/14.,dtcwtWCCs,label="DTCWT")
plt.semilogx(stepsLambdas,stepsWCCs,label="STEPS")
plt.semilogx(mexLambdas,mexWCCs,label="Mex. Hat")
plt.legend()
plt.show()

