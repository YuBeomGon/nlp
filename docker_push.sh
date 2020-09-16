#/bin/bash

#docker build --file dockerfiles/dockerfile.huggingface-pytorch-gpu -t 056936070848.dkr.ecr.ap-northeast-2.amazonaws.com/petclassify:huggingface-pytorch-gpu .
#
#docker build --file dockerfiles/dockerfile.tokenizer -t 056936070848.dkr.ecr.ap-northeast-2.amazonaws.com/petclassify:tokenizer .
#
#docker build --file dockerfiles/dockerfile.pretrained -t 056936070848.dkr.ecr.ap-northeast-2.amazonaws.com/petclassify:pretrained .
#
#docker build --file dockerfiles/dockerfile.downstream -t 056936070848.dkr.ecr.ap-northeast-2.amazonaws.com/petclassify:downstream .
#
#
#docker push 056936070848.dkr.ecr.ap-northeast-2.amazonaws.com/petclassify:huggingface-pytorch-gpu
#docker push 056936070848.dkr.ecr.ap-northeast-2.amazonaws.com/petclassify:tokenizer
#docker push 056936070848.dkr.ecr.ap-northeast-2.amazonaws.com/petclassify:pretrained
#docker push 056936070848.dkr.ecr.ap-northeast-2.amazonaws.com/petclassify:downstream


#docker build --file dockerfiles/dockerfile.huggingface-pytorch-gpu -t registry.dl.vsmart00.com:5000/petclassify:huggingface-pytorch-gpu .

#docker build --file dockerfiles/dockerfile.huggingface-pytorch-gpu-from-source -t registry.dl.vsmart00.com:5000/petclassify:huggingface-pytorch-gpu-from-source .

#
#docker build --file dockerfiles/dockerfile.tokenizer -t registry.dl.vsmart00.com:5000/petclassify:tokenizer .
#
docker build --file dockerfiles/dockerfile.pretrained -t registry.dl.vsmart00.com:5000/petclassify:pretrained .

docker build --file dockerfiles/dockerfile.downstream -t registry.dl.vsmart00.com:5000/petclassify:downstream .

docker build --file dockerfiles/dockerfile.contra-downstream -t registry.dl.vsmart00.com:5000/petclassify:contra-downstream .
#
#docker build --file dockerfiles/dockerfile.reptile -t registry.dl.vsmart00.com:5000/petclassify:reptile .


#docker push registry.dl.vsmart00.com:5000/petclassify:huggingface-pytorch-gpu
#docker push registry.dl.vsmart00.com:5000/petclassify:huggingface-pytorch-gpu-from-source
#docker push registry.dl.vsmart00.com:5000/petclassify:tokenizer
docker push registry.dl.vsmart00.com:5000/petclassify:pretrained
docker push registry.dl.vsmart00.com:5000/petclassify:downstream
docker push registry.dl.vsmart00.com:5000/petclassify:contra-downstream
#docker push registry.dl.vsmart00.com:5000/petclassify:reptile

#python3 run-pipelines.py --testname reptile --epochs0 1
