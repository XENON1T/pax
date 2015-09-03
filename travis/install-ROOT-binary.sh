#!/bin/bash


if [ -z "$ROOT" ]; then
    echo "ROOT version not set. Making guess"
    export ROOT=6.03.04
fi

if [ "${TRAVIS_OS_NAME}" == "linux" ]
then
    conda remove system
fi

time wget -nv --no-check-certificate http://www.nikhef.nl/~ctunnell/root_v${ROOT}_linux_binaries.tar.gz
tar xfz root_v${ROOT}_linux_binaries.tar.gz
cd root*

source bin/thisroot.sh

export LD_LIBRARY_PATH=`python3.4-config --exec-prefix`/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$LD_LIBRARY_PATH

cd ..

if [ "${TRAVIS_OS_NAME}" == "linux" ]
then
    conda install system
fi
