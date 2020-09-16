#!/bin/bash

docker run -it --rm --name transformers -v $PWD:/opt -w /opt $@ transformers bash


docker run -it --rm --name transformers -v $PWD:/opt -v ~/vtdeep/saved/pretrained:/opt/pretrained -w /opt $@ registry.dl.vsmart00.com:5000/petclassify:downstream --traindata notebooks/files/train3.csv --testdata notebooks/files/test3.csv
