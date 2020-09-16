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

from losses import LabelSmoothingCrossEntropy, SupConLoss, FocalLoss
from feed import * #PetDataset
from torch.utils.data.sampler import RandomSampler
from torch_model import MLPRobertaNet, CNNRobertaNet, SIMRobertaNet, ContraRobertaNet
#from eval import model_eval
from util import *
from augment import *
import time
MAX_SEQ_LEN = 512

def model_eval(test_df, tokenizer, model) :
#     device = torch.device("cuda")
#     model.to(device)       
    model.eval()

    test_dataset = PetDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    total_loss = 0
    total_len = 0
    total_correct = 0

    for text, label in test_loader:
        #   encoded_list = [tokenizer.encode(t, add_special_token=True) for t in text]
        encoded_list = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in text]
        padded_list = [e[:MAX_SEQ_LEN] + [0] * (MAX_SEQ_LEN-len(e[:MAX_SEQ_LEN])) for e in encoded_list]
        sample = torch.tensor(padded_list)
        sample, label = sample.to(device), label.to(device)
        label = torch.tensor(label)
        outputs = model(sample=sample, iscontra=False)
        logits = outputs

        pred = torch.argmax(F.softmax(logits), dim=1)
        correct = pred.eq(label)
        total_correct += correct.sum().item()
        total_len += len(label)

    print('Test accuracy: ', total_correct / total_len)
    return total_correct / total_len

def contrasitve_training_base(epochs=20, learning_rate=0.00001, denum=40) :
    optimizer =Adam(
        model.parameters(), lr=learning_rate
    )
    if args.scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=10, eta_min=0
        )
    else:
        scheduler = lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=lambda epoch: 1 / ((epoch/denum) + 1)
        )

    model.train()
    
    start = time.time()
    for epoch in range(epochs):
        losses = AverageMeter()
        total_loss = 0
        total_len = 0
        total_correct = 0
        total_count = 0
        for text, label in train_loader:
            encoded_list = [tokenizer.encode(t, add_special_tokens=True, max_length=MAX_SEQ_LEN, truncation=True) for t in text]
            padded_list = [e[:MAX_SEQ_LEN] + [0] * (MAX_SEQ_LEN-len(e[:MAX_SEQ_LEN])) for e in encoded_list]

            sample = torch.tensor(padded_list)
            sample, label = sample.to(device), label.to(device)
            label = torch.tensor(label)
            outputs = model(sample=sample, iscontra=True)
            outputs = torch.unsqueeze(outputs, dim=1)        

            loss = criterion(outputs, label)
            losses.update(loss.item(), args.batchsize)
    #         print(loss)
            
            total_len += len(label)
            total_loss += loss.item()
            total_count += 1
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
        
    #            if (total_count + 1) % 1 == 0:
    #                contra_accuracy(test_df=test_df, tokenizer=tokenizer, model=model)

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
        scheduler.step()
    print("contrastive learning time :", time.time() - start)

# batch size should be divided by 2, because one more batchs are sampled in below code.
def contrasitve_training_tune(epochs=20, learning_rate=0.00001, denum=40) :
    optimizer =Adam(
        model.parameters(), lr=learning_rate
    )
    if args.scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=10, eta_min=0
        )
    else:
        scheduler = lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=lambda epoch: 1 / ((epoch/denum) + 1)
        )

    model.train()
    
    start = time.time()
    for epoch in range(epochs):
        losses = AverageMeter()
        total_loss = 0
        total_len = 0
        total_correct = 0
        total_count = 0
        for text, label in train_loader:
            text1 = get_text(label, df_dict)
            padded_lists = []
            bsz = label.shape[0]
            for text in [text, text1] :
                encoded_list = [tokenizer.encode(t, add_special_tokens=True, max_length=MAX_SEQ_LEN, truncation=True) for t in text]
                padded_list = [e[:MAX_SEQ_LEN] + [0] * (MAX_SEQ_LEN-len(e[:MAX_SEQ_LEN])) for e in encoded_list]
                padded_lists.append(padded_list)

            sample = torch.cat([torch.tensor(padded_lists[0]), torch.tensor(padded_lists[1])], dim=0)
            sample, label = sample.to(device), label.to(device)
            label = torch.tensor(label)
            output = model(sample=sample, iscontra=True)
            o1, o2 = torch.split(output, [bsz, bsz], dim=0)
            outputs = torch.cat([o1.unsqueeze(1), o2.unsqueeze(1)], dim=1)
        
            loss = criterion(outputs, label)
            losses.update(loss.item(), args.batchsize)
    #         print(loss)
            
            total_len += len(label)
            total_loss += loss.item()
            total_count += 1
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
        
    #            if (total_count + 1) % 1 == 0:
    #                contra_accuracy(test_df=test_df, tokenizer=tokenizer, model=model)

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
        scheduler.step()
    print("contrastive learning time :", time.time() - start)

#def downstream_training(epochs=20, learning_rate=0.00001, denum=4, criterion=torch.nn.CrossEntropyLoss()) :
def downstream_training(epochs=20, learning_rate=0.00001, denum=4) :
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.LambdaLR(
        optimizer=optimizer, lr_lambda=lambda epoch: 1 / ((epoch/denum) + 1)
    )

    for epoch in range(epochs):
        losses = AverageMeter()
        total_loss = 0
        total_len = 0
        total_correct = 0
        total_count = 0
        model.train()
        for text, label in train_loader:
    #         print(label)
            encoded_list = [tokenizer.encode(t, add_special_tokens=True, max_length=MAX_SEQ_LEN, truncation=True) for t in text]
            padded_list = [e[:MAX_SEQ_LEN] + [0] * (MAX_SEQ_LEN-len(e[:MAX_SEQ_LEN])) for e in encoded_list]
            sample = torch.tensor(padded_list)
            sample, label = sample.to(device), label.to(device)
            label = torch.tensor(label)
            outputs = model(sample=sample, iscontra=False)
    
            loss = criterion(outputs, label)
            losses.update(loss.item(), args.batchsize)
    #         print(loss)
            pred = torch.argmax(F.softmax(outputs), dim=1)
            correct = pred.eq(label)
            total_correct += correct.sum().item()            
            total_len += len(label)
            total_loss += loss.item()
            total_count += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

        writer.add_scalar("Loss/Train", total_loss / total_count, epoch + 1)
        writer.add_scalar("Accuracy/Train", total_correct / total_len, epoch + 1)
        writer.add_scalar("LearningRate/Train", scheduler.get_last_lr()[0], epoch + 1)
    
        print(
            "[Epoch {}/{}] Train Loss: {:.4f}, Accuracy: {:.3f}, Learning Rate: {:.7f}".format(
                epoch + 1,
                epochs,
                total_loss / total_count,
                total_correct / total_len,
                scheduler.get_last_lr()[0],
            )
        )
        scheduler.step()
        model_eval(test_df, tokenizer, model)

#def change_criterion(lossft=torch.nn.CrossEntropyLoss()) :
#    criterion = lossft
#    criterion.to(torch.device('cuda'))
#    return criterion
#
#def contra_accuracy() :
#    criterion = change_criterion()
#    for param, state in zip(model.parameters(), model.state_dict()) :
#        if 'fc.' not in state :
#            param.requires_grad = False
#    model.eval()
#    downstream_training(epochs=2, learning_rate=0.00001, denum=4, criterion=criterion)
#
#    change_criterion(lossft=SupConLoss(temperature=1))
#    for param, state in zip(model.parameters(), model.state_dict()) :
#        if 'fc.' not in state :
#            param.requires_grad = True
#    model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--bucket", help="bucket name", default="petcharts")
    parser.add_argument("-t", "--traindata", help="train file", default="train.csv")
    parser.add_argument("-f", "--testdata", help="test file", default="test.csv")
    parser.add_argument(
        "-p", "--pretrained", help="pretrained model zip file", default="roberta.zip"
    )
    parser.add_argument(
        "-o",
        "--downstream",
        help="downstream model zip file",
        default="classifier.zip",
    )
    parser.add_argument("-c", "--classes", help="classes", type=int, default=18)
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
        train_df = pd.read_csv('notebooks/files/train3.csv')
        test_df = pd.read_csv('notebooks/files/test3.csv')

    Num_label = len(train_df.label_id.value_counts())
    print('#label ', Num_label)

    device = torch.device(args.device)
    tokenizer = RobertaTokenizer.from_pretrained("./pretrained", do_lower_case=False)
    model = ContraRobertaNet(path=
        "./pretrained", embedding_dim=768, num_class=Num_label
    )

    criterion = SupConLoss(temperature=1)
    model.to(device)
    criterion.to(device)

    train_dataset = PetDataset(train_df)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2
    )

    df_dict ={}
    for label in range(Num_label) :
        df = train_df[train_df['label_id'] == label]
        df_dict[label] = df    

    writer = SummaryWriter(args.logdir)

    contrasitve_training_base(epochs=30, learning_rate=0.00008, denum=40)
    contrasitve_training_base(epochs=args.contra_epochs, learning_rate=0.00001, denum=5)

    criterion = torch.nn.CrossEntropyLoss()
#    criterion = FocalLoss(alpha=0.97, reduce=True)
    criterion.to(device)

#    downstream_training(epochs=15, learning_rate=0.00008, denum=4)

    for test_num in range(2,3):
        LEARN_RATE = 0.00004/(test_num/2)
        #freezing the model except classifier for linear evaluation protocol
        print("***********************freezing classifier******************************")
        for param, state in zip(model.parameters(), model.state_dict()) :
            if 'fc.' not in state :
                param.requires_grad = False
            else :
                param.requires_grad = True
    
        downstream_training(epochs=20, learning_rate=LEARN_RATE/2, denum=4)
    
        #unfreezing the classifier except model for fine tuning
        print("**********************freezing model**************************")
        for param, state in zip(model.parameters(), model.state_dict()) :
            if 'fc.' in state :
                param.requires_grad = False
            else :
                param.requires_grad = True
            
        downstream_training(epochs=10, learning_rate=LEARN_RATE, denum=4)

    for param, state in zip(model.parameters(), model.state_dict()) :
        param.requires_grad = True

    torch.save(model.state_dict(), './contra-downstream')
    compress_object('contrastive.zip', './contra-downstream')

    try:
        save_object(client, args.bucket, 'contrastive.zip')
    except:
        print("model  save error to minio") 


    metadata = {"outputs": [{"type": "tensorboard", "source": args.logdir}]}
    with open("/opt/mlpipeline-ui-metadata.json", "w") as fd:
        json.dump(metadata, fd)

