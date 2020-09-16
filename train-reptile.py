import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
from minioset import (
    connect_server,
    load_object,
    save_object,
    compress_object,
    uncompress_object,
)
import time
import torch.nn as nn
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataset import ConcatDataset
#from torchsampler import ImbalancedDatasetSampler

from torch_model import TransferRobertaNet, MLPRobertaNet, CNNRobertaNet, SIMRobertaNet, ContraRobertaNet
from feed import * #PetDataset
from eval import model_eval
from util import *
from augment import *
from losses import LabelSmoothingCrossEntropy, SupConLoss, FocalLoss
from variables import *

def reset_model(model, model_state=None) :
    if model_state is not None :
        model.load_state_dict(model_state)

def model_eval(test_df, model, istransfer=True) :
    model.eval()

    test_dataset = PetDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

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
          outputs = model(sample=sample, istransfer=istransfer)

          pred = torch.argmax(F.softmax(outputs), dim=1)
          correct = pred.eq(labels)
          total_correct += correct.sum().item()
          total_len += len(labels)

    print('Test accuracy: ', total_correct / total_len)
    return total_correct / total_len

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--bucket", help="bucket name", default="petcharts")
#    parser.add_argument("-t", "--traindata", help="train file", default="unlabel_train1.csv")
#    parser.add_argument("-f", "--testdata", help="test file", default="unlabel_test1.csv")
    parser.add_argument("-t", "--traindata", help="train file", default="disease_train.csv")
    parser.add_argument("-f", "--testdata", help="test file", default="disease_test.csv")
    parser.add_argument(
        "-p", "--pretrained", help="pretrained model zip file", default="roberta.zip"
    )
    parser.add_argument(
        "--transfer", help="pretrained model for transfer learning", default="roberta.transfer.zip"
    )
    parser.add_argument(
        "-o",
        "--downstream",
        help="downstream model zip file",
        default="classifier.zip",
    )
    parser.add_argument("-c", "--classes", help="classes", type=int, default=20)
    parser.add_argument("-e", "--epochs", help="epochs", type=int, default=24)
    parser.add_argument("--contra_epochs", help="contrasitve epochs", type=int, default=40)
    parser.add_argument("-b", "--batchsize", help="batchsize", type=int, default=32)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-H", "--host", help="object server")
    parser.add_argument("-A", "--accesskey", help="access key")
    parser.add_argument("-K", "--secretkey", help="secret key")
    parser.add_argument("--logdir", help="tensorboard logdir", default="./logs")
    parser.add_argument("--weightdecay", help="weight decay", type=float, default=0.01)
    parser.add_argument("--scheduler", help="scheduler type", default="linear")
    args = parser.parse_args()
    cluster_flag = True

    try:
        client = connect_server(args.host, args.accesskey, args.secretkey)
        load_object(client, args.bucket, args.traindata)
        load_object(client, args.bucket, args.testdata)
        load_object(client, args.bucket, args.pretrained)
    except:
        print("minio connection fails")
        cluster_flag = False
        pass

    if cluster_flag :
        uncompress_object(args.pretrained, ".")
        train_df = pd.read_csv(args.traindata)
        test_df = pd.read_csv(args.testdata)
    else :
        print("local file reading")
        train_df = pd.read_csv('notebooks/files/unlabel_train1.csv')
        test_df = pd.read_csv('notebooks/files/unlabel_test1.csv')

    Num_label = len(train_df.label_id.value_counts())

    device = torch.device(args.device)
    tokenizer = RobertaTokenizer.from_pretrained("./pretrained", do_lower_case=False)
    model = TransferRobertaNet(path="./pretrained",                       
                                  embedding_dim=768,
                                  num_class=Num_label,
                                  num_class1=args.classes)

    criterion = FocalLoss(alpha=0.97, reduce=True)
    model.to(device)
    criterion.to(device)

    optimizer =Adam(
        model.parameters(), lr=0.00008
    )
    if args.scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=10, eta_min=0
        )
    else:
        scheduler = lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=lambda epoch: 1 / (int(epoch/4) + 1)
        )

    train_dataset = PetDataset(train_df)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2
    )

    df_dict ={}
    for label in range(Num_label) :
        df = train_df[train_df['label_id'] == label]
        df_dict[label] = df    

    writer = SummaryWriter(args.logdir)

    epochs = 10
    model.train()
    high_acc = 0
    task_batch_size = 10    
    MAX_SEQ_LEN = 512
    
    for epoch in range(epochs):
        losses = AverageMeter()
        total_loss = 0
        total_len = 0
        total_correct = 0
        total_count = 0
        model.train()
        updates = []
        model_backup = model.state_dict()
        op_state = optimizer.state_dict()
        for text, labels in train_loader:
            reset_model(model, model_backup)
            optimizer.load_state_dict(op_state)
            
            text1 = get_text(labels, df_dict )
            #text2 = get_text(labels, df_dict )
            
            #for text in [text, text1, text2] :
            for text in [text, text1] :
                encoded_list = [tokenizer.encode(t, add_special_tokens=True, max_length=MAX_SEQ_LEN, truncation=True) for t in text]
                padded_list = [e[:MAX_SEQ_LEN] + [0] * (MAX_SEQ_LEN-len(e[:MAX_SEQ_LEN])) for e in encoded_list]
                sample = torch.tensor(padded_list)
                sample, labels = sample.to(device), labels.to(device)
                labels = torch.tensor(labels)
                outputs = model(sample=sample, istransfer=True)
    
                loss = criterion(outputs, labels)
#                losses.update(loss.item(), args.batchsize)
    
                pred = torch.argmax(F.softmax(outputs), dim=1)
                correct = pred.eq(labels)
                total_correct += correct.sum().item()
                total_len += len(labels)
                total_loss += loss.item()
                total_count += 1
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            updates.append(subtract_vars(model.state_dict(), model_backup))
            op_state = optimizer.state_dict()
            if total_count % task_batch_size == 0 :
                update = average_vars(updates)
                updates = []
                model_backup = add_vars(model_backup, scale_vars(update, epsilon=0.99))
           
        scheduler.step()
        accr = model_eval(test_df, model, istransfer=True)
        if accr > high_acc :
            high_acc = accr
            best_model = model.state_dict()
#            torch.save(model.state_dict(), 'maml/transfer')
            print('model is saved')
    
        writer.add_scalar("Loss/Train", total_loss / total_count, epoch + 1)
        writer.add_scalar("LearningRate/Train", scheduler.get_last_lr()[0], epoch + 1)

        print(
            "[Epoch {}/{}] Train Loss: {:.4f}, Learning Rate: {:.7f}".format(
                epoch + 1,
                epochs,
                total_loss / total_count,
                scheduler.get_last_lr()[0],
            )
	)

    torch.save(best_model, './reptile')
#    compress_object('reptile.zip', './reptile')
    compress_object(args.transfer, './reptile')

    try:
        save_object(client, args.bucket, args.transfer)
    except:
        print("model  save error to minio") 

    metadata = {"outputs": [{"type": "tensorboard", "source": args.logdir}]}
    with open("/opt/mlpipeline-ui-metadata.json", "w") as fd:
        json.dump(metadata, fd)

