sudo apt-get update -qq
sudo apt-get install -y
sudo apt-get install -y build-essential git dpkg-dev make binutils libx11-dev libxpm-dev libxft-dev libxext-dev gcc-multilib g++ gcc libsnappy1 libsnappy-dev libsnappy-dev python-software-properties

sudo apt-get build-dep python-snappy

# update gcc to 4.8 via toolchain
yes "" | sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y gcc-4.8 g++-4.8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 20
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 20
sudo update-alternatives --config gcc
sudo update-alternatives --config g++