#!/bin/bash

log_dir=$1
echo $log_dir
rm -rf log_pre/$log_dir
mc cp -r minio/petcharts/$log_dir log_pre/
tensorboard --logdir log_pre/$log_dir
