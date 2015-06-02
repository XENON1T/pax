#!/bin/bash

echo zipping entire ROOT installation
echo $PWD
echo $PATH

echo remove current source

rm root_v${ROOT}.source.tar.gz

echo make new archive of current installation

tar cfz root_v${ROOT}_${TRAVIS_OS_NAME}_binaries.tar.gz root*

echo print size of created file

du -h root_v${ROOT}_${TRAVIS_OS_NAME}_binaries.tar.gz

echo send file to temporary server, hopefully one time will do

# make sure putty-tools in installed when calling pscp
sudo apt-get install -y putty-tools
yes | pscp -q -pw speakfriendandenter root_v${ROOT}_${TRAVIS_OS_NAME}_binaries.tar.gz dropoff@electro.dyndns-server.com:/home/dropoff/

echo done travis-save-ROOT
