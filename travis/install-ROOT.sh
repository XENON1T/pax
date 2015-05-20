#!/bin/bash


if [ -z "$ROOT" ]; then
    echo "ROOT version not set.  Making guess"
    export ROOT=6.03.04
fi  

#conda remove system
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

./configure --minimal --enable-python --with-python-incdir=`python3.4-config --exec-prefix`/include/python3.4m --with-python-libdir=`python3.4-config --exec-prefix`/lib --prefix=`python3.4-config --exec-prefix` #--enable-builtin-zlib --enable-builtin-lzma --enable-builtin-pcre --disable-shared

cat config.log

echo making ROOT...

make -j2 install

echo source ROOT environment...

source `python3.4-config --exec-prefix`/bin/thisroot.sh

cd ..
