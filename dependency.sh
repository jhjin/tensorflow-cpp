#!/usr/bin/env bash

# apt-get packages
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y build-essential python-pip python-dev git python-numpy swig python-dev default-jdk zip zlib1g-dev

# java8 on ubuntu 14.04
UBUNTU_VERSION=$(lsb_release -a 2> /dev/null | grep "Release" | awk '{print $2}')
if [ $UBUNTU_VERSION == "14.04" ]; then
  sudo add-apt-repository ppa:webupd8team/java
  sudo apt-get update
  sudo apt-get install -y oracle-java8-installer
fi
