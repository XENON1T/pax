#!/bin/bash

conda remove system

time wget -nv --no-check-certificate http://www.nikhef.nl/~bartp/root_v5.34.25_binaries.tar.gz
tar xfz root_v${ROOT}_binaries.tar.gz
cd root*

source bin/thisroot.sh

cd ..
