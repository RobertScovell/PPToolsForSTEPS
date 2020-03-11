This repository contains various bits of code in C++ and Python that were used to derive the results (images / statistics) for the following paper:
Scovell, R. W. (2020) Applications of Directional Wavelets, Universal Multifractals and Anisotropic Scaling in Ensemble Nowcasting; A Review of Methods with Case Studies. Quarterly Journal of the Royal Meteorological Society. In Press. URL: http://dx.doi.org/abs/10.1002/qj.3780
Please cite this paper if you use the code in your project.


/CWT
===
The CWT folder contains C++ code to compute the Continuous Wavelet Transform, for wavelets defined in the header wavelets.h.
The script comp.sh can be used to compile on your GNU Linux system - some modifications to library paths may be needed. It requires FFTW3, BOOST and HDF5 libraries.
The files mexicanHatTransform.cc morletTransform.cc cauchyTransform.cc are driver programmes for specific types of transform. They use CSV data as input and they output HDF5 files of coefficients, stored in an array of dimension (numScales,numOrientations,numRows,numCols,2). The last dimension is used for the real / imaginary parts.

/data
====
This folder contains two CSV files of case data (as in Scovell(2020)) that can be used with the various programmes / scripts.

/doubleTraceMoment
=================
This folder contains the Python3 scripts dtm.py and dtmCompare.py. The dtm.py script ingests a two-dimensional CSV file and computes the alpha,C1 multifractal parameters using the Double Trace Moment technique: Lavallée, D. (1991) Multifractal analysis and simulation techniques and turbulent fields. McGill University, Montreal.
The usage is:
```bash
    python3 dtm.py "csv-file"
```
to compute using default values for eta range [-2,1].
```bash
    python3 dtm.py "csv-file" "eta-min" "eta-max"
```
to compute using user specified eta range.
```bash
    python3 dtm.py "csv-file" "eta-min" "eta-max" "H"
```
to compute using user specified eta range and apply fractional integration order H to image prior to DTM. This hasn't been properly tested.
dtmCompare.py is similar to dtm.py except that it can compute DTM for two images and plot them side-by-side.

/filterDisplayDTCWT
==================
This folder contains a Python3 script to display the real, imaginary and absolute values of the two-dimensional complex-oriented DTCWT filters, for each orientation, at a given level of decomposition (level 5 used by default). The script requires no commend-line arguments.

/imageAnalysisDTCWT
==================
This folder contains a Python3 script to analyse both the global and the local anisotropy in a given image. It displays the fraction of power in each sub-band/scale and also the local orientation for a range of scales. It uses just the CSV image as the input.

/stochasticNoise
===============
This contains a Python3 implementation (fifGenLS2010.py) of the FIF realization generator that is available as Matlab code from Lovejoy and Schertzer's web page: http://www.physics.mcgill.ca/~gang/software/index.html 
, as described in Lovejoy and Schertzer (2010): On the simulation of continuous in scale universal multifractals, part i: spatially continuous processes. Computers & Geosciences, 36, 1393–1403.

It also contains a naive implementation that is based on power-law filtering alone.

The command-line arguments are documented in the scripts.

/synthMatchedNoiseDTCWT
======================
This contains a script to extract the global anisotropic scaling properties of a prior image and to apply them to an isotropic realisation (dtcwtApplyGlobalAnisotropyOfImage.py). Usage is:
```bash
    python3 dtcwtApplyGlobalAnisotropyOfImage.py "csv-of-prior" "csv-of-noise"
```
Also there is a script to demonstrate applying anisotropy using a test pattern (dtcwtApplyTestPatternAnisotropy.py). Usage is:
```bash
    python3 dtcwtApplyTestPatternAnisotropy.py "isotropic-realisation"
```
The other script (dtcwtNoiseBlend.py) does scale-selective blending of a prior image with a supplied isotropic realisation. The realisation is adjusted to match the anisotropy in the prior before blending. The default setting is to use the realisation at all scales but this can be adjusted in the code.

/umfParamEstDTCWT
================
The Python3 script dwtUMFParamEst.py takes a CSV file and estimates the UMF parameters using the structure function approach, as described in Scovell (2020). Usage is:
python3 dwtUMFParamEst.py "csv file" "detection threshold", where the detection threshold is the minimum data value above zero. For fields containing negative values, the modulus can be used first.
For CSV files containing zeros, it is better to simulate small singularities (that would otherwise be seen below the detection threshold), to replace the actual zeros, otherwise the zeros introduce artificial regularity in the field. This script aims to do that, using the FIF realization generator above. This is based on the approach of Gires, A., Tchiguirinskaia, I., Schertzer, D. and Lovejoy, S., 2012. Influence of the zero-rainfall on the assessment of the multifractal parameters. Advances in water resources, 45, pp.13-25. The usage is the same as above.

/undecimatedDTCWT
================
This is a fairly naive implementation of the Undecimated DTCWT, using Pywavelets to compute the single-tree SWT transforms and DTCWT to provide the Q-shift filters. Better methods may be available, e.g. based on: Hill, P. R., N. Anantrasirichai, A. Achim, M. E. Al-Mualla, and D. R. Bull. “Undecimated Dual-Tree Complex Wavelet Transforms.” Signal Processing: Image Communication 35 (2015): 61-70. The following site links to Matlab code: https://vilab.blogs.bristol.ac.uk/?p=1156
Usage is:
```bash
    python3 undecimatedDTCWT.py "csv-file"
```

/waveletLeaders
==============
This contains a naive implementation of the Wavelet Leaders method, based on Wendt, H., Roux, S.G., Jaffard, S. and Abry, P., 2009. Wavelet leaders and bootstrap for multifractal analysis of images. Signal Processing, 89(6), pp.1100-1114.
 Usage is:
```bash
    python3 waveletLeadersDTCWT.py "csv-file" "threshold"
```
Here the threshold is a fraction of the range of wavelet coefficients on each scale level. There are some obsolete scripts to compute ensemble-averaged wavelet leaders spectra and the script above can be run in a batch mode, requiring more arguments, to support this.

/autocorrelation_comparison
==========================
These files are not currently intended for wider use.


