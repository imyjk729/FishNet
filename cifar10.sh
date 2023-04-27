#!/bin/bash

# Make and go to data folder 
mkdir cifar10
cd ./cifar10

# download cifar10 python version
# cite link : https://www.cs.toronto.edu/~kriz/cifar.html
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -zxvf cifar-10-python.tar.gz