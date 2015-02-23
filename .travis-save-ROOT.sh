#!/bin/bash

echo zipping entire ROOT installation
echo $PWD

echo remove current source

rm root_v${ROOT}.source.tar.gz

echo make new archive of current installation

tar -cfz root_v${ROOT}_binaries.tar.gz root*

echo print size of created file

du -h root_v${ROOT}_binaries.tar.gz

echo send file to temporary server, hopefully one time will do

yes | pscp -q -pw speakfriendandenter root_v${ROOT}_binaries.tar.gz dropoff@electro.dyndns-server.com:/home/dropoff/

echo done travis-save-ROOT
