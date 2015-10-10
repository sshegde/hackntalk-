#! /bin/bash

sudo apt-get install -y  build-essential python-dev python-setuptools \
                     python-numpy python-scipy \
                     libatlas-dev libatlas3gf-base

sudo apt-get install -y autotools-dev
sudo apt-get install -y automake autoconf libtool
sudo apt-get instal -y python-sklearn
sudo apt-get install -y python-pip
sudo apt-get install -y mpg123
sudo pip install  arff

wget http://liquidtelecom.dl.sourceforge.net/project/openart/openEAR-0.1.0.tar.gz 
tar -zxvf openEAR-0.1.0.tar.gz 
cd openEAR-0.1.0/
chmod +x autogen.sh
./autogen.sh
./configure
sudo ./configure
make -j4;make 
make install

cd ..
python data_processing_hat.py
