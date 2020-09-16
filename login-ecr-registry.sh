#!/bin/bash
ECR_REPO=${ECR_REPO:=056936070848.dkr.ecr.ap-northeast-2.amazonaws.com}
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin $ECR_REPO

