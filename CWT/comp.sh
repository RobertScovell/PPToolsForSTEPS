#!/bin/bash

LIBFFTW3DIR=/usr
BOOSTDIR=/usr
HDF5DIR=/usr

CPPFLAGS="-O3 -fopenmp -DBOOST_DISABLE_ASSERTS -I${BOOSTDIR}/include -I${LIBFFTW3DIR}/include -I${HDF5DIR}/include"
LDFLAGS="-fopenmp -L${BOOSTDIR}/lib -L${LIBFFTW3DIR}/lib -lfftw3 -lhdf5"

g++ ${CPPFLAGS} mexicanHatTransform.cc -o mexicanHatTransform ${LDFLAGS}
g++ ${CPPFLAGS} morletTransform.cc -o morletTransform ${LDFLAGS} 
 
