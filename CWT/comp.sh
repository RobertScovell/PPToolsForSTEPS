#!/bin/bash

LIBFFTW3DIR=/usr/local
BOOSTDIR=/usr/local

#CPP=cc
CPP=g++

CPPFLAGS="-O3 -fopenmp -DBOOST_DISABLE_ASSERTS -I${LIBFFTW3DIR}/include -I${BOOSTDIR}/include"

#LDFLAGS="-L${LIBFFTW3DIR}/lib -L${BOOSTDIR}/lib ${LDFLAGS} -lc++ -lfftw3 -lhdf5 -lm"
LDFLAGS=-L"${LIBFFTW3DIR}/lib" -L"${BOOSTDIR}/lib" ${LDFLAGS} -lfftw3 -lhdf5 -lm 

${CPP} ${CPPFLAGS} mexicanHatTransform.cc -o mexicanHatTransform ${LDFLAGS}
${CPP} ${CPPFLAGS} morletTransform.cc -o morletTransform ${LDFLAGS}

