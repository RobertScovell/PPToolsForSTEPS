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

#include <boost/multi_array.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>

#include "mycwt2d.h"

void readTxtArray(boost::multi_array<std::complex<double>,2ul>& arr,std::string fn) {
  std::ifstream ifs(fn.c_str());
  for ( int i = 0 ; i < arr.shape()[0] ; i ++ ) {
    std::string buf;
    std::getline(ifs,buf);
    std::vector<std::string> elems;
    boost::split(elems,buf,boost::is_any_of(","));
    for ( int j = 0 ; j < elems.size() ; j ++ ) {
      arr[i][j] = boost::lexical_cast<double>(elems[j]);
    }
  }
  ifs.close();
}


int main(int argc,char *argv[]) {

  int nx = 512;
  int ny = 512;
  std::string inputDataFn(argv[1]);
  std::string outputDataFn(argv[2]);
  if ( argc > 3 ) {
    ny = boost::lexical_cast<int>(argv[3]);
    nx = boost::lexical_cast<int>(argv[4]);
  }

  double lambdaBot=2.0;
  double lambdaTop=float(nx);
  int nK = 1;
  double deltaS = 1.0;//0.05;
  sombreroMarrHildreth80 wvt;
  std::cout << wvt.wavelengthToScale(lambdaTop) << std::endl;

  boost::multi_array<std::complex<double>,2ul> image(boost::extents[ny][nx]);
  readTxtArray(image,inputDataFn.c_str());
//  myCWT2d cwt(image,&wvt,deltaS,nK,wvt.wavelengthToScale(lambdaBot)*1.4985,wvt.wavelengthToScale(lambdaTop));
  myCWT2d cwt(image,&wvt,deltaS,nK,wvt.wavelengthToScale(lambdaBot),wvt.wavelengthToScale(lambdaTop));
  std::cout << "Computing coeffs" << std::endl;
//  std::cout << "WARNING: fudge factor present." << std::endl;
  cwt.computeCoeffs();
  std::vector<double> freqCentres;
  std::vector<double> fWidthsSq;
  std::vector<double> rWidthsSq;
  cwt.calcFilterFreqBarycentres(freqCentres,fWidthsSq,rWidthsSq);
  for ( int i = 0 ; i < freqCentres.size() ; i ++ ) {
      std::cout << freqCentres[i] << "," << image.shape()[0]*std::sqrt(fWidthsSq[i]) << "," << image.shape()[0]*std::sqrt(rWidthsSq[i]) << std::endl;
  }
  std::cout << "Writing coeffs" << std::endl;
//  exit(1);
  cwt.writeHDF5(outputDataFn.c_str());

  return 0;
}


