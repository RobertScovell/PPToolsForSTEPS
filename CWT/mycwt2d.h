// (C) British Crown Copyright 2017-2019 Met Office.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/*
 * Referneces:
 * Torrence, C. and G. Compo, 1998: A Practical Guide to Wavelet Analysis. Bull. Amer. Meteor. Soc., 79, 61-78,  doi: 10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2. 
 * Wang, N. and C. Lu, 2010: Two-Dimensional Continuous Wavelet Analysis and Its Application to Meteorological Data. J. Atmos. Oceanic Technol., 27, 652-666,  doi: 10.1175/2009JTECHA1338.1. 
 * Liu, Y., X. San Liang, and R. Weisberg, 2007: Rectification of the Bias in the Wavelet Power Spectrum. J. Atmos. Oceanic Technol., 24, 2093-2102,  doi: 10.1175/2007JTECHO511.1. 
 * Farge, M., 1992: Wavelet Transforms and their Applications to Turbulence. Annual Review of Fluid Mechanics, Vol.24, 395-458), doi: 10.1146/annurev.fl.24.010192.002143 
 * Daubechies, I., 1992: Ten Lectures on Wavelets. Society for Industrial and Applied Mathematics Philadelphia, PA, USA (C)1992. ISBN:0-89871-274-2
 *
*/

#ifndef __MYCWT2D_H_
#define __MYCWT2D_H_

#include <boost/multi_array.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <string>

#include <cmath>
#include <complex>

#include <hdf5.h>
#include <fftw3.h>
#include <omp.h>

#include "wavelets.h"

class h5DCloser {
  hid_t _dset;
  public:
  h5DCloser(hid_t dset) : _dset(dset) {};
  ~h5DCloser() { H5Dclose(_dset); }
};

class h5FCloser {
  hid_t _file;
  public:
  h5FCloser(hid_t file) : _file(file) {};
  ~h5FCloser() { H5Fclose(_file); }
};

void outputTxtArray(boost::multi_array<std::complex<double>,2ul>& arr,std::string fn,std::string delimiter = std::string(" ") ) {
  std::ofstream ofs(fn.c_str());
  for ( int i = 0 ; i < arr.shape()[0] ; i ++ ) {
    for ( int j = 0 ; j < arr.shape()[1] ; j ++ ) {
      ofs << arr[i][j].real() << delimiter;
    }
    ofs << std::endl;
  }
  ofs.close();
}

void outputTxtArray(boost::multi_array<double,2ul>& arr,std::string fn,std::string delimiter = std::string(" ") ) {
  std::ofstream ofs(fn.c_str());
  for ( int i = 0 ; i < arr.shape()[0] ; i ++ ) {
    for ( int j = 0 ; j < arr.shape()[1] ; j ++ ) {
      ofs << arr[i][j] << delimiter;
    }
    ofs << std::endl;
  }
  ofs.close();
}

void outputTxtArray(boost::multi_array<bool,2ul>& arr,std::string fn,std::string delimiter = std::string(" ") ) {
  std::ofstream ofs(fn.c_str());
  for ( int i = 0 ; i < arr.shape()[0] ; i ++ ) {
    for ( int j = 0 ; j < arr.shape()[1] ; j ++ ) {
      ofs << arr[i][j] << delimiter;
    }
    ofs << std::endl;
  }
  ofs.close();
}

//////////////////////////////////////////////////////////////////
// Continuous Wavelet Transform class

class myCWT2d {

  boost::multi_array<std::complex<double>,2ul> image;
  boost::multi_array<std::complex<double>,2ul> fftImage;
  boost::multi_array<std::complex<double>,2ul> fftImageShifted;
  boost::multi_array<std::complex<double>,4ul> coefficients;
  boost::multi_array<std::complex<double>,2ul> residualCoefficients;
  std::vector<std::vector<boost::multi_array<std::complex<double> ,2ul> > > kernels;
  boost::multi_array<std::complex<double>,2ul> residualKernel;
  boost::multi_array<std::complex<double>,2ul> residualField;
  std::vector<std::vector<double> > kernelFracInGrid;
  boost::multi_array<int,4ul> coefficients_eval_count;
  myWavelet2d *wavelet;
  double theta0;
  std::complex<double> fieldMean;
  std::complex<double> cDelta;
  int numScales;
  double sSmall;
  double sLarge;
  double deltaS;
  int voicesPerOctave;
  int K;
  std::vector<double> cosltheta0;
  std::vector<double> sinltheta0;
  std::vector<std::vector<std::complex<double> > > wDelta;
  std::vector<std::vector<double> > kernSqNorm;
  std::vector<double> scales;
  omp_lock_t writelock;

  void computeKernels() {

    kernels.resize(numScales);
    wDelta.resize(numScales);
    kernSqNorm.resize(numScales);
    kernelFracInGrid.resize(numScales);
    for ( size_t i = 0 ; i < numScales ; i ++ ) {
      wDelta[i].resize(K);
      kernSqNorm[i].resize(K);
      kernelFracInGrid[i].resize(K);
      for ( size_t j = 0 ; j < K ; j ++ ) {
        wDelta[i][j] = 0.0;
        kernSqNorm[i][j] = 0.0;
        kernelFracInGrid[i][j] = 0.0;
      }
    }

    int nSteps = fftImage.shape()[0];
    for ( size_t i = 0 ; i < kernels.size() ; i ++ ) kernels[i].resize(K);

    // loop over the discrete indexes m,n,k,l.
    std::cout << "myCWT2d: computing kernels: " << std::endl;
    for ( int im = 0 ; im < numScales ; im ++ ) { // octave number ( a0^m is scale )
      std::cout << "myCWT2d: octave: " << im / voicesPerOctave << ", voice: "
        << im % voicesPerOctave << ", dilation: " << scales[im] << "." << std::endl;
#pragma omp parallel for
      for ( int il = 0 ; il < K ; il ++ ) { // rotation

        boost::multi_array<std::complex<double>,2ul> kernelEvalTmp(boost::extents[nSteps+nSteps/2][nSteps+nSteps/2]);
        std::fill(kernelEvalTmp.origin(),kernelEvalTmp.origin()+kernelEvalTmp.num_elements(),0.0);
        // Compute wavelet kernel on the Fourier-domain grid
        kernels[im][il].resize(boost::extents[nSteps][nSteps]);

        // Loop over frequencies defined by the Fourier domain grid
        // Use a 50% larger frequency domain, so that wavelets around the Nyquist frequency can be fully summed
        // (even though they are out of grid)
        double kernelValid = 0.0;
        double kernelInvalid = 0.0;
        for ( int i = -nSteps/4 ; i < nSteps + nSteps/4 ; i ++ ) {
          double f1 = static_cast<double>( i - nSteps / 2 ) / static_cast<double>(nSteps); // units of cycles / grid spacing ( [-0.5,0.5] )
          for ( int j = -nSteps/4 ; j < nSteps + nSteps/4 ; j ++ ) {
            double f2 = static_cast<double>( j - nSteps / 2 ) / static_cast<double>(nSteps); // units of cycles / grid spacing ( [-0.5,0.5] )
            double omega1 = 2.0 * M_PI * f1 / 1.0; // TC98, eqn 5 
            double omega2 = 2.0 * M_PI * f2 / 1.0;
            // Compute wavelet at omega1,omega2
            std::complex<double> kernEval = computePsiFThetaL(il,scales[im]*omega1,scales[im]*omega2);
            kernelEvalTmp[i+nSteps/4][j+nSteps/4] = kernEval;
            // Compute 2-norm of kernel - this is not used in the dec / rec but in other apps. 
            kernSqNorm[im][il] += std::norm(kernEval);
          }
        }

        // Initialize pre-computed discretized kernel, based on discretized version of continuous wavelet.
        // The discretized version will not have the same 2-norm as the continuous in all cases.
        // Also compute fraction of discrete power in the grid vs. total discrete power.
        for ( int i = -nSteps/4 ; i < nSteps + nSteps/4 ; i ++ ) {
          for ( int j = -nSteps/4 ; j < nSteps + nSteps/4 ; j ++ ) {
            if ( i >= 0 && i < nSteps && j >= 0 && j < nSteps ) {            
              kernels[im][il][i][j] = ( 2.0 * M_PI * scales[im] / 1.0 ) * std::conj(kernelEvalTmp[i+nSteps/4][j+nSteps/4]);
              // Add 2-norm to total in-grid
              kernelValid += std::norm(kernelEvalTmp[i+nSteps/4][j+nSteps/4]);
            }
            else {
              // Add 2-norm to total out-of-grid
              kernelInvalid += std::norm(kernelEvalTmp[i+nSteps/4][j+nSteps/4]);
            }
          }
        }

        // Compute fraction of kernel power in grid - this is useful when
        // injecting stochastic noise because the high frequency components
        // can be supressed at certain scales / rotations, due to being
        // out-of-grid, whereas stochastic noise does not know about this.
        kernelFracInGrid[im][il] = kernelValid / ( kernelValid + kernelInvalid );
      }
    }
  }

  void computeReconstructionFactor() {
    for ( int im = 0 ; im < numScales ; im ++ ) {
      for ( int il = 0 ; il < K ; il ++ ) { // rotation

        // For reconstruction, need to have wDelta according to TC98, eqn 12.
        // This is also based on Farge (1992) reconstruction formula.
        // The wDelta is the wavelet transform of a kroneker delta
        // at x=0,y=0 for each scale and rotation.
        // Using this information, it becomes possible to calculate cDelta,
        // the reconstruction factor.
        // This will also pick up any scale-independent factor in the
        // supplied wavelet function (which cancels on reconstruction).
        // Here wDelta is computed for each scale/rot separately.
        for ( int i = 0 ; i < image.shape()[0] ; i ++ ) {
          for ( int j = 0 ; j < image.shape()[1] ; j ++ ) {
            wDelta[im][il] += 1.0 * 1.0 * kernels[im][il][i][j]; 
          }
        }
        wDelta[im][il] /= image.shape()[0] * image.shape()[1];
      }
    }
    // Compute the normalization cDelta for this wavelet (TC98 eqn 13).
    cDelta = 0.0;
    for ( int im = 0 ; im < numScales ; im ++ ) {
      for ( int il = 0 ; il < K ; il ++ ) {
        cDelta += wDelta[im][il];
      }
    }
    std::cout << "Reconstruction factor: " << cDelta << std::endl;
  }

  void computeRCoeff(int im,int il) {
    throw;
  }

  void computeFCoeff(int im,int il) {

    boost::multi_array<std::complex<double>,2ul> coeffF;

    // Compute wavelet kernel on the Fourier-domain grid
    int nSteps = fftImage.shape()[0];
    coeffF.resize(boost::extents[nSteps][nSteps]);

    // Multiply conjugate of Fourier domain wavelet kernel with Fourier transform of image    
    // Also shift back to FFTW-ordering
    for ( size_t i = 0 ; i < nSteps ; i ++ ) {
      for ( size_t j = 0 ; j < nSteps ; j ++ ) {
        int newi = i + fftImage.shape()[0] / 2;
        int newj = j + fftImage.shape()[1] / 2;
        if ( newi >= fftImage.shape()[0] ) newi -= fftImage.shape()[0];
        if ( newj >= fftImage.shape()[1] ) newj -= fftImage.shape()[1];
        coeffF[newi][newj] = std::conj(kernels[im][il][i][j]) * fftImageShifted[i][j];
        coeffF[newi][newj] /= static_cast<double>(nSteps * nSteps); // introduced by FFT
      }
    }

    // Inverse Fourier transform
    int dims[2];
    for ( int i = 0 ; i < 2 ; i ++ ) dims[i] = image.shape()[i];
      omp_set_lock(&writelock);
      fftw_plan p_bwd = fftw_plan_dft(2,dims,
          reinterpret_cast<fftw_complex*>(coeffF.origin()),
          reinterpret_cast<fftw_complex*>(coefficients[im][il].origin()),
          FFTW_BACKWARD,FFTW_ESTIMATE);
      omp_unset_lock(&writelock);
      fftw_execute(p_bwd);   
  }

  std::complex<double> computePsiRThetaL(int l,double x1,double x2) {

    // Frequencies rotated by l * theta0
    double rotatedX1 = x1 * cosltheta0[l] + x2 * sinltheta0[l];
    double rotatedX2 = -1.0 * x1 * sinltheta0[l] + x2 * cosltheta0[l];

    // Wavelet 
    std::complex<double> psiRThetaL = wavelet->evalR(rotatedX1,rotatedX2);

    return psiRThetaL;
  }
  std::complex<double> computePsiFThetaL(int l,double f1,double f2) {

    // Frequencies rotated by l * theta0
    double rotatedF1 = f1 * cosltheta0[l] + f2 * sinltheta0[l];
    double rotatedF2 = -1.0 * f1 * sinltheta0[l] + f2 * cosltheta0[l];

    // Wavelet 
    std::complex<double> psiFThetaL = wavelet->evalF(rotatedF1,rotatedF2);

    return psiFThetaL;
  }

  public:
  boost::multi_array<std::complex<double>,2ul>& getKernel(int im,int il) { return kernels[im][il]; } 
  const boost::multi_array<std::complex<double>,2ul>& getKernel(int im,int il) const { return kernels[im][il]; } 
  int getNumScales() { return numScales; };
  std::vector<double>& getScales() { return scales; }
  const std::vector<double>& getScales() const { return scales; }
  const boost::multi_array<std::complex<double>,4ul>& getCoefficients() const { return coefficients; }
  const std::vector<std::vector<double> >& getKernelFracInGrid() const { return kernelFracInGrid; }
  boost::multi_array<std::complex<double>,4ul>& getCoefficients() { return coefficients; }

  // Constructor.
  // Subtract field mean from image and compute FFT of image.
  // Initialise wavelet kernels based on supplied image and wavelet function.
  // Compute reconstruction factor.
  myCWT2d(
      boost::multi_array<std::complex<double>,2ul>& _image,
      myWavelet2d *_wvt,
      double _deltaS,
      int _K,
      double _sSmall = 1.0,
      double _sLarge = 1024.0) : image(_image),wavelet(_wvt),deltaS(_deltaS),K(_K),sSmall(_sSmall),sLarge(_sLarge) {

    // compute field mean
    fieldMean = 0.0;
    int fieldMeanCount = 0;
    for ( int i = 0 ; i < image.num_elements() ; i ++ ) {
      fieldMean += image.origin()[i];
      fieldMeanCount ++;
    }
    fieldMean /= static_cast<double>(fieldMeanCount);
    std::cout << "myCWT2d: image field mean: " << fieldMean << std::endl;

    // subtract field mean
    for ( int i = 0 ; i < image.num_elements() ; i ++ ) {
      image.origin()[i] -= fieldMean;
    }

    // compute FFT
    int dims[2];
    for ( int i = 0 ; i < 2 ; i ++ ) dims[i] = image.shape()[i];
    fftImage.resize(boost::extents[image.shape()[0]][image.shape()[1]]);
    fftImageShifted.resize(boost::extents[image.shape()[0]][image.shape()[1]]);
    fftw_plan p_fwd = fftw_plan_dft(2,dims,
        reinterpret_cast<fftw_complex*>(image.origin()),
        reinterpret_cast<fftw_complex*>(fftImage.origin()),
        FFTW_FORWARD,FFTW_ESTIMATE);
    fftw_execute(p_fwd);

    // To compute the coefficients in the Fourier domain, the image is better represented as having the DC component at the centre
    // First half shift to the right half. Second half shift to the left half.
    for ( int i = 0 ; i < fftImage.shape()[0] ; i ++ ) {
      for ( int j = 0 ; j < fftImage.shape()[1] ; j ++ ) {
        int newi = i + fftImage.shape()[0] / 2;
        if ( newi >= fftImage.shape()[0] ) newi -= fftImage.shape()[0];
        int newj = j + fftImage.shape()[1] / 2;
        if ( newj >= fftImage.shape()[1] ) newj -= fftImage.shape()[1];
        fftImageShifted[newi][newj] = fftImage[i][j];
      }
    }

    theta0 = 2.0 * M_PI / ( static_cast<double>(K) );

    // Get number of scales (deltaS is the number of voices per octave)
    int nSteps = image.shape()[0];
    voicesPerOctave = static_cast<int>( round( 1.0 / deltaS ) );
    numScales = static_cast<int> ( ceil ( ( 1.0 / deltaS ) * log ( sLarge / sSmall ) / log ( 2.0 ) ) );
    std::cout << "myCWT2d: number of scales: " << numScales << std::endl;

    // Compute scales
    scales.resize(numScales);
    for ( size_t i = 0 ; i < numScales ; i ++ ) scales[i] = sSmall * pow ( 2.0 , static_cast<double>(i) * deltaS );

    // Precompute sin / cos of rotation angles
    cosltheta0.resize(K);
    sinltheta0.resize(K);
    std::cout << "myCWT2d: orientations (deg): ";
    for ( size_t l = 0 ; l < K ; l ++ ) {
      std::cout << (180.0/M_PI) * ( l * theta0 ) << " ";
      cosltheta0[l] = cos(l*theta0);
      sinltheta0[l] = sin(l*theta0);
    }
    std::cout << std::endl;

    // Precompute kernels
    computeKernels();
    // Compute reconstruction factor
    computeReconstructionFactor();
  }

  // Deallocate memory associated with kernels (but keep any computed coefficients)
  void dumpKernels() {
    for ( int im = 0 ; im < numScales ; im ++ ) { // octave number ( a0^m is scale )
      for ( int il = 0 ; il < K ; il ++ ) { // rotation
        kernels[im][il].resize(boost::extents[0][0]);
      }
    }
  }

  void computeCoeffs() {
    coefficients.resize(boost::extents[numScales][K][fftImage.shape()[0]][fftImage.shape()[1]]);
    std::fill(coefficients.origin(),coefficients.origin()+coefficients.num_elements(),0.0);
    // loop over the discrete indexes of scale (im) and rotation (il)
    std::cout << "myCWT2d: computing coeffs: " << std::endl;
    for ( int im = 0 ; im < numScales ; im ++ ) { // octave number ( a0^m is scale )
      std::cout << "myCWT2d: octave: " << im / voicesPerOctave << ", voice: "
        << im % voicesPerOctave << ", dilation: " << scales[im] << "." << std::endl;
      omp_init_lock(&writelock);
#pragma omp parallel for
      for ( int il = 0 ; il < K ; il ++ ) { // rotation
        computeFCoeff(im,il);
      }
      omp_destroy_lock(&writelock);
    }
  }

  // Reconstruct image using delta-function reconstruction (Farge, 1992).
  void inverseTransformDeltaFunc(boost::multi_array<std::complex<double>,2ul>& imageOut) {


    imageOut.resize(boost::extents[image.shape()[0]][image.shape()[1]]);
    for ( int im = 0 ; im < numScales ; im ++ ) { 
#pragma omp parallel for
      for ( int il = 0 ; il < K ; il ++ ) { // rotation
        for ( int in = 0 ; in < image.shape()[0] ; in ++ ) { // number of shifts in x-direction
          for ( int ik = 0 ; ik < image.shape()[1] ; ik ++ ) { // number of shifts in y-direction
            imageOut[in][ik] += coefficients[im][il][in][ik] / cDelta;
          }
        }
      }
    }

    // add back field mean
    for ( int i = 0 ; i < image.num_elements() ; i ++ ) {
      imageOut.origin()[i] += fieldMean;
    }
  }

  //////////////////////////////////////////////////////////////////
  // Utility functions

  void calcFilterFreqBarycentres(std::vector<double>& freqCentres,std::vector<double>& fWidthsSq,std::vector<double>& rWidthsSq) {
    freqCentres.resize(numScales);
    std::fill(freqCentres.begin(),freqCentres.end(),0.0);
    fWidthsSq.resize(numScales);
    std::fill(fWidthsSq.begin(),fWidthsSq.end(),0.0);
    rWidthsSq.resize(numScales);
    std::fill(rWidthsSq.begin(),rWidthsSq.end(),0.0);

    for ( int im = 0 ; im < numScales ; im ++ ) { // voice number ( a0^m is scale )
      double integral = 0.0;
      double norm2 = 0.0;
      // Integrate over positive frequencies
      for ( size_t i = 0 ; i < fftImage.shape()[0] / 2 ; i ++ ) {
        for ( size_t j = 0 ; j < fftImage.shape()[1] / 2; j ++ ) {
          double ki = static_cast<double>(i);
          double kj = static_cast<double>(j);
          double kr = sqrt(ki*ki+kj*kj);
          integral += kr * std::norm(kernels[im][0][fftImage.shape()[0]/2+i][fftImage.shape()[1]/2+j]) * 1.0;
          norm2 += std::norm(kernels[im][0][fftImage.shape()[0]/2+i][fftImage.shape()[1]/2+j]) * 1.0;
        }
      }
      freqCentres[im] = integral / norm2;

      integral = 0.0;
      norm2 = 0.0;
      for ( size_t i = 0 ; i < fftImage.shape()[0] / 2 ; i ++ ) {
        for ( size_t j = 0 ; j < fftImage.shape()[1] / 2; j ++ ) {
          double ki = static_cast<double>(i);
          double kj = static_cast<double>(j);
          double kr = sqrt(ki*ki+kj*kj);
          integral += pow( kr - freqCentres[im] , 2.0 ) * std::norm(kernels[im][0][fftImage.shape()[0]/2+i][fftImage.shape()[1]/2+j]) * 1.0;
          norm2 += std::norm(kernels[im][0][fftImage.shape()[0]/2+i][fftImage.shape()[1]/2+j]) * 1.0;
        }
      }
      fWidthsSq[im] = integral / norm2;

      // Inverse Fourier transform the f-space kernel
      int dims[2];
      for ( int i = 0 ; i < 2 ; i ++ ) dims[i] = fftImage.shape()[i];
      boost::multi_array<std::complex<double>,2ul> kernR(boost::extents[fftImage.shape()[0]][fftImage.shape()[1]]);
      fftw_plan p_bwd = fftw_plan_dft(2,dims,
          reinterpret_cast<fftw_complex*>(kernels[im][0].origin()),
          reinterpret_cast<fftw_complex*>(kernR.origin()),
          FFTW_BACKWARD,FFTW_ESTIMATE);
      fftw_execute(p_bwd);   

      integral = 0.0;
      norm2 = 0.0;
      for ( size_t i = 0 ; i < fftImage.shape()[0] ; i ++ ) {
        for ( size_t j = 0 ; j < fftImage.shape()[1] ; j ++ ) {
          kernR[i][j] /= sqrt ( fftImage.shape()[0] * fftImage.shape()[1] );
          double ii = static_cast<double>(i);
          if ( ii > fftImage.shape()[0] / 2 ) ii = fftImage.shape()[0] - ii;
          double jj = static_cast<double>(j);
          if ( jj > fftImage.shape()[1] / 2 ) jj = fftImage.shape()[1] - jj;
          double iijj = sqrt(ii*ii+jj*jj); // distance from centre in real space
          integral += pow( iijj , 2.0 ) * std::norm(kernR[i][j]) * 1.0;
          norm2 += std::norm(kernR[i][j]) * 1.0;
        }
      }
      rWidthsSq[im] = integral / norm2;
    }
  }

  // Calculate variance on each scale level, including negative ("zero") pixels. On each scale level, the mean should be zero.
  void calcScaleLevelVariances(std::vector<double>& meanRes,std::vector<double>& vars,std::vector<double>& globalWaveletSpectrum,
      bool excludeCOI=false,bool excludeZeros=false) {

    meanRes.resize(numScales);
    vars.resize(numScales);
    globalWaveletSpectrum.resize(numScales);
    for ( size_t i = 0 ; i < numScales ; i ++ ) {
      vars[i] = 0.0;
      globalWaveletSpectrum[i] = 0.0;
    }

    // Computing variance on scale levels
    for ( size_t is = 0 ; is < numScales ; is ++ ) {
      double var = 0.0;
      double en = 0.0;
      double meanRe = 0.0;
      int count = 0;
      for ( size_t ir = 0 ; ir < K ; ir ++ ) {
        // Determine Cone Of Influence if needed
        double coi;
        if ( excludeCOI == true ) coi = wavelet->getCOI(scales[is]);
        for ( size_t i = 0 ; i < image.shape()[0] ; i ++ ) {
          if ( excludeCOI == true && ( i - 0 < coi || image.shape()[0] - i < coi ) ) continue;
          for ( size_t j = 0 ; j < image.shape()[1] ; j ++ ) {
            if ( excludeCOI == true && ( j - 0 < coi || image.shape()[1] - j < coi ) ) continue;
            if ( excludeZeros == true && ( std::abs ( coefficients[is][ir][i][j] ) < 1.0e-4 ) ) continue; // less than zero magnitude
            meanRe += coefficients[is][ir][i][j].real();
            var += std::norm( coefficients[is][ir][i][j] );
            count++;
          }
        }
      }
      meanRe /= static_cast<double>(count);
      meanRes[is] = meanRe;
      // Here the sqrt(s) is applied to ensure the wavelet response has the same energy on all scales.
      // It's not necessarily a unit energy response, unless the wavelet is properly normalized.
      // However, according to Liu et al. (2007) this needs to be divided by the scale, otherwise it is biased.
      // This effectively increases the slope of the power spectrum by 1.
      vars[is] = var * pow ( scales[is] , 1.5 ); // the "scale" here is the Fourier space scale, i.e. 1/scale
    }
  }

  // Calculate variance on each scale level and rotation, including negative ("zero") pixels. On each scale level, the mean should be zero.
  void calcScaleAndRotationalVariances(
    std::vector<std::vector<double> >& meanRes,
    std::vector<std::vector<double> >& vars,
    std::vector<std::vector<double> >& globalWaveletSpectrum,
      bool excludeCOI=false,bool excludeZeros=false) {

    meanRes.resize(numScales);
    vars.resize(numScales);
    for ( size_t i = 0 ; i < vars.size() ; i ++ ) {
      meanRes[i].resize(K);
      vars[i].resize(K);
      for ( size_t j = 0 ; j < K ; j ++ ) {
        meanRes[i][j] = 0.0;
        vars[i][j] = 0.0;
      }
    }

    for ( size_t im = 0 ; im < numScales ; im ++ ) {
      for ( size_t il = 0 ; il < K ; il ++ ) {
        double var = 0.0;
        double en = 0.0;
        double meanRe = 0.0;
        int count = 0;
        double coi;
        if ( excludeCOI == true ) coi = wavelet->getCOI(scales[im]);
        for ( size_t i = 0 ; i < image.shape()[0] ; i ++ ) {
          if ( excludeCOI == true && ( i - 0 < coi || image.shape()[0] - i < coi ) ) continue;
          for ( size_t j = 0 ; j < image.shape()[1] ; j ++ ) {
            if ( excludeCOI == true && ( j - 0 < coi || image.shape()[1] - j < coi ) ) continue;
            if ( excludeZeros == true && ( std::abs ( coefficients[im][il][i][j] ) < 1.0e-4 ) ) continue; // less than zero magnitude
            meanRe += coefficients[im][il][i][j].real();
            var += std::norm( coefficients[im][il][i][j] );
            count++;
          }
        }
        meanRe /= static_cast<double>(count);
        meanRes[im][il] = meanRe;
        // Here the sqrt(s) is applied to ensure the wavelet response has the same energy on all scales.
        // It's not necessarily a unit energy response, unless the wavelet is properly normalized.
        // However, according to Liu et al. (2007) this needs to be divided by the scale, otherwise it is biased.
        // This effectively increases the slope of the power spectrum by 1.
        vars[im][il] = var * pow ( scales[im] , 1.5 );
      }
    }
  }

  void writeHDF5(const std::string& outFileName) {

    hid_t file,space,dset,dcpl;
    herr_t status;

    /* Save old error handler */
    herr_t (*old_func)(hid_t,void*);
    void *old_client_data;

    H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);

    /* Turn off error handling */
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    // Create file
    file = H5Fcreate(outFileName.c_str(),H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
    if ( file < 0 ) throw ( std::runtime_error(std::string("Unable to open HDF5 file: ") + outFileName) );
    h5FCloser fsetc(file);
    hsize_t dims[5];
    for ( size_t i = 0 ; i < 4 ; i ++ ) {
      dims[i] = coefficients.shape()[i]; std::cout << "dims[" << i << "]: " << dims[i] << std::endl;
    }
    dims[4]=2;

    space = H5Screate_simple (5, dims, NULL);
    if ( space < 0 ) {
      throw ( std::runtime_error(std::string("Unable to create space for coefficients dataset in file: ") + outFileName) );
    }
    dcpl = H5Pcreate (H5P_DATASET_CREATE);
    status = H5Pset_deflate(dcpl,6);
    hsize_t chunks[5];
    chunks[0] = 1;
    chunks[1] = 1;
    chunks[2] = image.shape()[0] / 8;
    chunks[3] = image.shape()[1] / 8;
    chunks[4] = 1;
    if ( chunks[0] == 0 ) chunks[0] = 1;
    if ( chunks[1] == 0 ) chunks[1] = 1;
    if ( chunks[2] == 0 ) chunks[2] = 1;
    if ( chunks[3] == 0 ) chunks[3] = 1;
    if ( chunks[4] == 0 ) chunks[4] = 1;
    status = H5Pset_chunk(dcpl,5,chunks);

    // Create the dataset
    dset = H5Dcreate1(file,"coefficients",H5T_NATIVE_DOUBLE,space,dcpl);
    if ( dset < 0 )
      throw ( std::runtime_error(std::string("Unable to create dataset in file: ") + outFileName) );
    h5DCloser dsetc(dset);

    // Create temporary array to store output data
    status = H5Dwrite(dset,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,reinterpret_cast<void*>(coefficients.origin()));
    if ( status < 0 )
      throw ( std::runtime_error(std::string("Unable to write HDF5 file: ") + outFileName) );

    /* Restore previous error handler */
    H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);

    (void)H5Pclose(dcpl);
    (void)H5Sclose(space);
  }
};



#endif // __MYCWT2D_H_
