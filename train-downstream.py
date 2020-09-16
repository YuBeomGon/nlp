import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
#import sys
#sys.path.append('../')
from torch.utils.tensorboard import SummaryWriter
from minioset import (
    connect_server,
    load_object,
    save_object,
    compress_object,
    uncompress_object,
)

import os
import argparse
import json
import torch.nn as nn

from util import *
from losses import LabelSmoothingCrossEntropy, SupConLoss, FocalLoss
from augment import *
from feed import * #PetDataset

from torch.utils.data.dataset import ConcatDataset
# from torch_model import SupConRobertaNet, SupConMultiRobertaNet
from torch.utils.data.sampler import RandomSampler
from torch_model import *
from preprop import *
from transformers import PreTrainedModel, BertPreTrainedModel
from transformers import (
        BertPreTrainedModel, BertModel, BertForPreTraining, BertForMaskedLM, BertLMHeadModel,
        BertForNextSentencePrediction, BertForSequenceClassification, BertForMultipleChoice,
        BertForTokenClassification, BertForQuestionAnswering,
        load_tf_weights_in_bert, BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        BertLayer, BertSelfAttention, BertAttention, BertLayer, BertEncoder
)

from torch_model import YubertForSequenceClassification

def model_eval(test_df) :
    model.eval()

    test_dataset = PetDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    total_loss = 0
    total_len = 0
    total_correct = 0

    for text, label in test_loader:
    #   encoded_list = [tokenizer.encode(t, add_special_token=True) for t in text]
      encoded_list = [tokenizer.encode(t, max_length=512, truncation=True) for t in text]
      padded_list = [e[:512] + [0] * (512-len(e[:512])) for e in encoded_list]
      sample = torch.tensor(padded_list)
      sample, label = sample.to(device), label.to(device)
      labels = torch.tensor(label)
      outputs = model(sample, labels=labels)
      _, logits = outputs

      pred = torch.argmax(F.softmax(logits), dim=1)
      correct = pred.eq(labels)
      total_correct += correct.sum().item()
      total_len += len(labels)

    print('Test accuracy: ', total_correct / total_len)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--bucket", help="bucket name", default="petcharts")
    parser.add_argument("-t", "--traindata", help="train file", default="train.csv")
    parser.add_argument("-f", "--testdata", help="test file", default="test.csv")
    parser.add_argument(
        "-p", "--pretrained", help="pretrained model zip file", default="roberta.zip"
    )
    parser.add_argument(
        "--transfer", help="pretrained model zip file", default="roberta.zip"
    )
    parser.add_argument(
        "-o",
        "--downstream",
        help="downstream model zip file",
        default="classifier.zip",
    )
    parser.add_argument("-c", "--classes", help="classes", type=int, default=26)
    parser.add_argument("-e", "--epochs", help="epochs", type=int, default=24)
    parser.add_argument("-b", "--batchsize", help="batchsize", type=int, default=32)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-H", "--host", help="object server")
    parser.add_argument("-A", "--accesskey", help="access key")
    parser.add_argument("-K", "--secretkey", help="secret key")
    parser.add_argument("--logdir", help="tensorboard logdir", default="./logs")
    parser.add_argument("--weightdecay", help="weight decay", type=float, default=0.01)
    parser.add_argument("--scheduler", help="scheduler type", default="linear")
    args = parser.parse_args()
    print(args.classes)

    try:
        client = connect_server(args.host, args.accesskey, args.secretkey)
        load_object(client, args.bucket, args.traindata)
        load_object(client, args.bucket, args.testdata)
        load_object(client, args.bucket, args.pretrained)
        #load_object(client, args.bucket, args.transfer)
    except:
        pass

    uncompress_object(args.pretrained, ".")
    #uncompress_object(args.transfer, ".")

    train_df = pd.read_csv(args.traindata)
    test_df = pd.read_csv(args.testdata)

    device = torch.device(args.device)
    tokenizer = RobertaTokenizer.from_pretrained("./pretrained", do_lower_case=False)
    if os.path.exists("./transfer") :
        print("transfer learnng")
        model = YubertForSequenceClassification.from_pretrained(
            "./transfer", num_labels=args.classes
        )
    else :
        print("model is init from pretrained model")
        model = YubertForSequenceClassification.from_pretrained(
            "./pretrained", num_labels=args.classes
        )

    print(model.config)
    #print(tokenizer.tokenize("내 강아지가 아파서 병원에 갔어, hi my pet is sick"))
    model.to(device)

#    optimizer = Adam(
#        model.parameters(), lr=0.000005, betas=(0.9, 0.999), weight_decay=args.weightdecay
#    )
    optimizer =Adam(
        model.parameters(), lr=0.000004
    )
    if args.scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=10, eta_min=0
        )
    else:
        scheduler = lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=lambda epoch: 1 / (int(epoch/4) + 1)
        )

    writer = SummaryWriter(args.logdir)

    train_dataset = PetDataset(train_df)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2
    )

    model.train()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_len = 0
        total_correct = 0
        total_count = 0
        for text, label in train_loader:
            optimizer.zero_grad()

            encoded_list = [tokenizer.encode(t, add_special_tokens=True, max_length=512, truncation=True) for t in text]
            padded_list = [e[:512] + [0] * (512 - len(e[:512])) for e in encoded_list]
            sample = torch.tensor(padded_list)
            sample, label = sample.to(device), label.to(device)
            labels = torch.tensor(label)
            outputs = model(sample, labels=labels)
            loss, logits = outputs

            pred = torch.argmax(F.softmax(logits), dim=1)
            correct = pred.eq(labels)
            total_correct += correct.sum().item()
            total_len += len(labels)
            total_loss += loss.item()
            total_count += 1
            loss.backward()
            optimizer.step()

        writer.add_scalar("Loss/Train", total_loss / total_count, epoch + 1)
        writer.add_scalar("Accuracy/Train", total_correct / total_len, epoch + 1)
        writer.add_scalar("LearningRate/Train", scheduler.get_last_lr()[0], epoch + 1)

        print(
            "[Epoch {}/{}] Train Loss: {:.4f}, Accuracy: {:.3f}, Learning Rate: {:.7f}".format(
                epoch + 1,
                args.epochs,
                total_loss / total_count,
                total_correct / total_len,
                scheduler.get_last_lr()[0],
            )
        )

        scheduler.step()
        model_eval(test_df)

    test_dataset = PetDataset(test_df)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=2
    )

    model.eval()

    total_loss = 0
    total_len = 0
    total_correct = 0

    for text, label in test_loader:
        encoded_list = [tokenizer.encode(t,  max_length=512, truncation=True) for t in text]
        padded_list = [e[:512] + [0] * (512 - len(e[:512])) for e in encoded_list]
        sample = torch.tensor(padded_list)
        sample, label = sample.to(device), label.to(device)
        labels = torch.tensor(label)
        outputs = model(sample, labels=labels)
        _, logits = outputs

        pred = torch.argmax(F.softmax(logits), dim=1)
        correct = pred.eq(labels)
        total_correct += correct.sum().item()
        total_len += len(labels)

    print("Test accuracy: ", total_correct / total_len)

    model.save_pretrained("./pretrained")

    compress_object(args.downstream, "./pretrained")

    try:
        save_object(client, args.bucket, args.downstream)
    except:
        pass

    metadata = {"outputs": [{"type": "tensorboard", "source": args.logdir}]}
    with open("/opt/mlpipeline-ui-metadata.json", "w") as fd:
        json.dump(metadata, fd)
