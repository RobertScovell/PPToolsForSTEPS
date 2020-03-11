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

// Please quote the following paper, if using this code:
// Scovell, R. W. (2020) Applications of Directional Wavelets, Universal Multifractals and Anisotropic Scaling in Ensemble Nowcasting; A Review of Methods with Case Studies. Quarterly Journal of the Royal Meteorological Society. In Press. URL: http://dx.doi.org/abs/10.1002/qj.3780

#ifndef __WAVELETS_H_
#define __WAVELETS_H_

const double sqrt8pi = sqrt(8.0*M_PI);
const double invsqrt2pi = 1.0 / sqrt(2.0*M_PI);
const double omega0 = 6.0; // different choices exist - need omega0 > 5.5 to ensure admissability.

int fakt(int n) {

  int fact=1;
  int i;

  for (i=1; i<=n; i++)
    fact*=i;

  return fact;
}


class myWavelet2d {

  public:
    virtual std::complex<double> evalR(double x1,double x2) = 0;
    virtual std::complex<double> evalF(double f1,double f2) = 0;
    virtual double scaleToWavelength(double s) = 0;
    virtual double wavelengthToScale(double lambda) = 0;
    virtual double getCOI(double s) = 0;
};

class sombreroMarrHildreth80 : public myWavelet2d {
  // Based on Mexican Hat wavelet of Marr and Hildreth 1980

  double threeQuaterSquareRootPi;

  public:
  sombreroMarrHildreth80() { threeQuaterSquareRootPi = 0.75 * sqrt(M_PI); }

  // return wavelet in real space according to torrence and compo 1998 (and extention to 2d, according to wang;2009) 
  virtual std::complex<double> evalR(double x1,double x2) { // dimensionless x1 and x2

    double modSqX = x1 * x1 + x2 * x2; // x1 and x2 are real
    return std::complex<double>( (1.0/threeQuaterSquareRootPi) * ( 2.0 - modSqX ) * exp ( -0.5 * modSqX ) , 0.0 ); // real valued
  }

  // return wavelet in Fourier space according to torrence and compo 1998 (and extention to 2d, according to wang;2009) 
  virtual std::complex<double> evalF(double omega1,double omega2) { // dimensionless omega1 and omega2
    double modSqOmega = omega1 * omega1 + omega2 * omega2; // omega1 and omega2 are real
    return std::complex<double>( (1.0/threeQuaterSquareRootPi) * modSqOmega * exp ( -0.5 * modSqOmega ) , 0.0 ); // real valued    
  }
  virtual double scaleToWavelength(double s) {
    // From TC98.
    return 2.0 * M_PI * s / sqrt ( 2.0 + 0.5 ); // for 2nd deriv. of Gaussian
  }
  virtual double wavelengthToScale(double lambda) {
    // From TC98.
    return lambda * sqrt ( 2.0 + 0.5 ) / ( 2.0 * M_PI ); // for 2nd deriv. of Gaussian
  }
  virtual double getCOI(double s) {
    // From TC98.
    return sqrt(2.0) * s; 
  }
};

class cauchyAntoine99 : public myWavelet2d {
  int il;
  int im;
  // Rotation angle
  double eta1;
  double eta2;
  // Angles in cone
  double alpha;
  double beta;
  // Components of angles in cone
  double ea1;
  double ea2;
  double eb1;
  double eb2;
  // Angles in dual cone
  double alphap;
  double betap;
  // Components of angles in dual cone
  double eap1;
  double eap2;
  double ebp1;
  double ebp2;
  // Axis of cone (same in dual)
  double axis1;
  double axis2;
  // Constant needed to evaluate real wavelet
  std::complex<double> c;
  inline bool inCone(double k1,double k2) {
    // Return 0 if not in cone
    double arg = atan2(k2,k1); // +ve going anticlockwise from +ve x-axis
    if ( arg > beta ) return false;
    if ( arg < alpha ) return false;
    return true;
  }
  inline bool inDualCone(double x1,double x2) {
    // Return 0 if not in cone - REALLY NEEDS CHECKING
    double arg = atan2(x1,x2);
    if ( arg > betap ) return false;
    if ( arg < alphap ) return false;
 
    return true;
  }
  public:
  cauchyAntoine99(int _il,int _im,double _alpha,double _beta,double _eta1,double _eta2) :
    il(_il),im(_im),alpha(_alpha),beta(_beta),eta1(_eta1),eta2(_eta2) {

      // TODO: ETA MUST BE IN CONE

      if ( alpha > beta ) throw;
      if ( beta - alpha > M_PI ) throw;
      // Angles in dual cone (Ant99;eqn 2.4)
      betap = alpha + M_PI / 2.0; 
      alphap = beta - M_PI / 2.0;
      // Components
      ea1 = cos(alpha);
      ea2 = sin(alpha);
      eb1 = cos(beta);
      eb2 = sin(beta);
      eap1 = cos(alphap);
      eap2 = sin(alphap);
      ebp1 = cos(betap);
      ebp2 = sin(betap);
      axis1 = cos(0.5*(alpha+beta));
      axis2 = sin(0.5*(alpha+beta));
      // Constant needed (Ant99; eqn A.1)
      std::complex<double> sqrtminus1 = std::complex<double>(0.0,1.0);
      std::complex<double> c1 = std::pow(sqrtminus1,static_cast<double>(il+im+2));
      double c2 = fakt(il)*fakt(im);
      double c3 = pow(sin(beta-alpha),static_cast<double>(il+im+1));
      std::cout << "c1: " << c1 << std::endl;
      std::cout << "c2: " << c2 << std::endl;
      std::cout << "c3: " << c3 << std::endl;
      c = c1 * c2 * c3 / ( 2.0 * M_PI );
    }

  virtual std::complex<double> evalR(double x1,double x2) {
// THIS CHECK SHOULD BE RE-ENABLED IF USING REAL SPACE CONVOLUTION
//    if ( ! inDualCone(x1,x2) ) return 0.0;
    // Compute wavelet value at x1,x2, if in cone
    std::complex<double> z1 = std::complex<double>(x1,eta1);
    std::complex<double> z2 = std::complex<double>(x2,eta2);
    std::complex<double> lFactor = pow ( ( z1 * ea1 + z2 * ea2 ) , -1.0 * static_cast<double>(il) - 1.0 );
    std::complex<double> mFactor = pow ( ( z1 * eb1 + z2 * eb2 ) , -1.0 * static_cast<double>(im) - 1.0 );
    return c * lFactor * mFactor;
  }
  virtual std::complex<double> evalF(double f1,double f2) {
    // Return 0 if not in cone
    if ( ! inCone(f1,f2) ) return 0.0;
    // Compute wavelet value at f1,f2, if in cone
    std::complex<double> oscFactor = exp ( - ( f1 * eta1 + f2 * eta2 ) ); //exp(-k.eta)
    double mFactor = pow ( ( f1 * ebp1 + f2 * ebp2 ) , static_cast<double>(im) );
    double lFactor = pow ( ( f1 * eap1 + f2 * eap2 ) , static_cast<double>(il) );
    return mFactor * lFactor * oscFactor;
  }
  virtual double scaleToWavelength(double s) {
    // From TC98.
    if ( im != il ) throw(std::runtime_error("Not implemented for im != il yet."));
    return 4.0 * M_PI * s / ( 2.0 * static_cast<double>(im) + 1.0 );
  }
  virtual double wavelengthToScale(double lambda) {
    // From TC98.
    if ( im != il ) throw(std::runtime_error("Not implemented for im != il yet."));
    return lambda * ( 2.0 * static_cast<double>(im) + 1.0 ) / ( 4.0 * M_PI );
  }
  virtual double getCOI(double s) {
    // From TC98.
    return s / sqrt(2.0);
  }
};

class morletWangLu2010 : public myWavelet2d {

  // Dimensionless frequencies, omega01, omega02
  double omega01;
  double omega02;
  double axisRatio;
  double sqrtAxisRatio;
  double invFourthRootOfPi;
  public:
    morletWangLu2010(double _omega01 = 6.0,double _omega02 = 0.0,double _axisRatio = 2.0) { 
      omega01 = _omega01;
      omega02 = _omega02;
      axisRatio = _axisRatio;
      sqrtAxisRatio = sqrt(axisRatio);
      invFourthRootOfPi = pow ( M_PI , - 1.0 / 4.0 );
      // Assumed that the correction factor is negligeable - this is only reasonable if omega0 >= 6.0
      if ( sqrt(omega01*omega01+omega02*omega02) < 6.0 - 1.0e-5 )
        throw(std::runtime_error("Error: can't initialize morlet wavelet with omega0 < 6.0"));
  }
  // return Morlet wavelet in real space according to Torrence and Compo 1998 (and extention to 2d, according to Wang;2009) 
  virtual std::complex<double> evalR(double x1,double x2) { // dimensionless x1 and x2

    // assumed omega0 points along x1 only
    std::complex<double> phasefactor( cos ( omega01 * x1 + omega02 * x2 ) , sin ( omega0 * x1 + omega02 * x2 ) );
    std::complex<double> envelopefactor( exp ( -0.5 * ( x1 * x1 / axisRatio + x2 * x2 ) ) , 0.0 );
    std::complex<double> psir = invFourthRootOfPi * invFourthRootOfPi * phasefactor * envelopefactor;
    return psir;
  }
  // return Morlet wavelet in Fourier space according to Torrence and Compo 1998 (and extention to 2d, according to Wang;2009) 
  virtual std::complex<double> evalF(double omega1,double omega2) { // dimensionless omega1 and omega2
    // if ( omega1 < 0.0 || omega2 < 0.0 ) return 0.0; // this is used, e.g. in Torrence and Compo 1998
    double exponent = -0.5 * ( axisRatio * ( omega1 - omega01 ) * ( omega1 - omega01 ) + ( omega2 - omega02 ) * ( omega2 - omega02 ) );
    std::complex<double> psif( sqrtAxisRatio * exp ( exponent ) , 0.0 );
    return invFourthRootOfPi * invFourthRootOfPi * psif;
  }
  virtual double scaleToWavelength(double s) {
    // From TC98.
    if ( omega02 > 1.0e-4 ) throw (std::runtime_error("Not implemented for non-zero omega02"));
    return 4.0 * M_PI * s / ( omega0 + sqrt ( 2.0 + omega0 * omega0 ) );
  }
  virtual double wavelengthToScale(double lambda) {
    // From TC98.
    if ( omega02 > 1.0e-4 ) throw (std::runtime_error("Not implemented for non-zero omega02"));
    return lambda * ( omega0 + sqrt ( 2.0 + omega0 * omega0 ) ) / ( 4.0 * M_PI );
  }
  virtual double getCOI(double s) {
    // From TC98.
    return sqrt(2.0) * s;
  }
};

#endif // __WAVELETS_H_

