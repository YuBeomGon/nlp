#!/bin/bash

docker run -it --rm --name transformers -v $PWD:/opt -w /opt -p 8080:8080 $@ transformers