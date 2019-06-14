#!/bin/bash

LIBFFTW3DIR=~/local_install

g++ -O3 -fopenmp -DBOOST_DISABLE_ASSERTS mexicanHatTransform.cc -o mexicanHatTransform -I"${LIBFFTW3DIR}/include" -L"${LIBFFTW3DIR}/lib" -lfftw3 -lhdf5 
g++ -O3 -fopenmp -DBOOST_DISABLE_ASSERTS morletTransform.cc -o morletTransform  -I"${LIBFFTW3DIR}/include" -L"${LIBFFTW3DIR}/lib" -lfftw3 -lhdf5
 
