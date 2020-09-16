#!/bin/bash

S3_ENDPOINT=http://minio.cleverai.com
S3_USE_HTTPS=0
S3_VERIFY_SSL=0
AWS_ACCESS_KEY_ID=haruband
AWS_SECRET_ACCESS_KEY=haru1004

docker run -it --rm --name tensorboard -p 6006:6006 -e S3_ENDPOINT=$S3_ENDPOINT -e S3_USE_HTTPS=$S3_USE_HTTPS -e S3_VERIFY_SSL=$S3_VERIFY_SSL -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY tensorflow/tensorflow tensorboard --host 0.0.0.0 --logdir $1