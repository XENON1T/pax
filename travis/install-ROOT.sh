#!/bin/bash


if [ -z "$ROOT" ]; then
    echo "ROOT version not set.  Making guess"
    export ROOT=6.03.04
fi  

if [ "${TRAVIS_OS_NAME}" == "linux" ]
then
  conda remove system 
fi
 
time wget -nv --no-check-certificate http://root.cern.ch/download/root_v${ROOT}.source.tar.gz
tar xfz root_v${ROOT}.source.tar.gz
cd root*
python3.4-config --configdir --includes --cflags --ldflags --exec-prefix

if [ -e `python3.4-config --exec-prefix`/lib/libpython3.4m.so ]
then
 echo so
 ln -s `python3.4-config --exec-prefix`/lib/libpython3.4m.so `python3.4-config --exec-prefix`/lib/libpython3.4.so
fi

if [ -e `python3.4-config --exec-prefix`/lib/libpython3.4m.dylib ]
then
 echo dylib
 ln -s `python3.4-config --exec-prefix`/lib/libpython3.4m.dylib `python3.4-config --exec-prefix`/lib/libpython3.4.dylib
fi

./configure --minimal --enable-python --with-python-incdir=`python3.4-config --exec-prefix`/include/python3.4m --with-python-libdir=`python3.4-config --exec-prefix`/lib

cat config.log

echo making ROOT...

make -j2

echo source ROOT environment...

source bin/thisroot.sh

export LD_LIBRARY_PATH=`python3.4-config --exec-prefix`/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$LD_LIBRARY_PATH
export PYTHONPATH=$LD_LIBRARY_PATH

cd ..

if [ "${TRAVIS_OS_NAME}" == "linux" ]
then
  conda install system
fi
