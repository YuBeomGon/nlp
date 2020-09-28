import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertTokenizer,  AlbertForSequenceClassification, AlbertModel

from transformers import RobertaTokenizer,  RobertaForSequenceClassification, RobertaModel, BertPreTrainedModel, RobertaConfig
import sys
sys.path.append('../')
from modeling_yubert import YubertModel, YubertClassificationHead

#for contrastive learning
class SupConRobertaNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, path, embedding_dim=768, feat_dim=64, num_class=20):
        super(SupConRobertaNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.path = path
        self.encoder = RobertaModel.from_pretrained(self.path)
#         self.encoder = model_fun()
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.feat_dim)
        )
        self.fc =  nn.Linear(self.embedding_dim, self.num_class)

    def forward(self, iscontra, x ):
        if iscontra == True :
            r = self.encoder(x)
            r = r[0][:,0,:]
            z = F.normalize(self.projection(r), dim=1)
            return z
        else :
            r = self.encoder(x)
            r = r[0][:,0,:]
            r = self.fc(r)
            return r

#for contrastive + multi task learning
class SupConMultiRobertaNet(SupConRobertaNet):
    def __init__(self, path, embedding_dim=768, feat_dim=64, task_label_dict=None, num_class=20):
        super(SupConMultiRobertaNet, self).__init__(path, embedding_dim, feat_dim, num_class)
        self.Multi = nn.ModuleDict({})
        for task_id in task_label_dict.keys():
            self.Multi[task_id] = nn.Linear(self.embedding_dim,
                                           task_label_dict[task_id])         

    def forward(self,  task_id , sample, ContraMutliFlag=2):
        r = self.encoder(sample)
        r = r[0][:,0,:]   

        if ContraMutliFlag == 1 : #contrastive learning
            z = F.normalize(self.projection(r), dim=1)
            return z

        elif ContraMutliFlag == 2: #multi task learning
            r = self.Multi[task_id](r)
            return r

        else : # downstream
            r = self.fc(r)
            return r

#for roberta + CNN        
class CNNRobertaNet(nn.Module):        
    """backbone + projection head"""
    def __init__(self, path, embedding_dim=768, max_seq_length=512, num_class=20):
        super(CNNRobertaNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_class = num_class
        self.path = path
        self.encoder = RobertaModel.from_pretrained(self.path)
#         self.encoder = model_fun()
        self.cnn = nn.Sequential(
            nn.Conv1d(embedding_dim,embedding_dim,(10), stride=4),

            
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.AvgPool1d(7)
#             nn.Conv1d(embedding_dim,embedding_dim,(10), stride=5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(2, stride=2)              
        )
        self.fc =  nn.Linear(self.embedding_dim, self.num_class)

    def forward(self, sample):
        r = self.encoder(sample)
        r = r[0][:,1:-1,:]
        r = torch.transpose(r, 1, 2)
        r = self.cnn(r)
        r = torch.transpose(r, 1, 2)
        r = torch.squeeze(r)
        r = self.fc(r)
        return r

#for roberta + MLP        
class MLPRobertaNet(nn.Module):        
    """backbone + projection head"""
    def __init__(self, path, embedding_dim=768, max_seq_length=512, num_class=20):
        super(MLPRobertaNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.num_class = num_class
        self.path = path
        self.encoder = RobertaModel.from_pretrained(self.path)
#         self.encoder = model_fun()
        self.mlp = nn.Sequential(
            nn.Linear(self.max_seq_length-2, (self.max_seq_length-2)),
            nn.ReLU(inplace=True),
            nn.Linear(self.max_seq_length-2, 1)
        )
        self.fc =  nn.Linear(self.embedding_dim, self.num_class)

    def forward(self, sample):
        r = self.encoder(sample)
        r = r[0][:,1:-1,:]
        r = torch.transpose(r, 1, 2)
        r = self.mlp(r)
        r = torch.squeeze(r)
        r = self.fc(r)
#         r = self.fc(F.normalize(r, dim=1))
        return r    


#for inner product roberta output and label tensor with shifted
class SIMRobertaNet(nn.Module):        
    """backbone + projection head"""
    def __init__(self, path, embedding_dim=768, max_seq_length=512, num_class=20):
        super(SIMRobertaNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.num_class = num_class
        self.path = path
        self.encoder = RobertaModel.from_pretrained(self.path)
#         self.encoder = model_fun()
        self.pool = nn.AvgPool1d(100, stride=20)
        self.fc =  nn.Linear(self.embedding_dim, self.num_class)

#     def forward(self, sample, isLabel=True):
#         r = self.encoder(sample)
#         r = r[0][:,1:-1,:]
#         if isLabel == False: # for training
#             r = torch.transpose(r, 1, 2)
#             r = self.pool(r)
#             r = torch.transpose(r, 1, 2)
#         else : # for label tensor
#             r = torch.transpose(r, 1, 2)
#             r = nn.AvgPool1d(50, stride=50)(r)
#             r = nn.AdaptiveMaxPool1d(1)(r)
#             r = torch.transpose(r, 1, 2)
#             r = torch.squeeze(r)

#         return r    
    
    def forward(self, sample, isLabel=True):
        r = self.encoder(sample)
    
        if isLabel == False: # for training
            r = r[0][:,0,:]

        else : # for label tensor
            r = r[0][:,1:-1,:]
            r = torch.transpose(r, 1, 2)
            r = nn.AvgPool1d(50, stride=50)(r)
            r = nn.AdaptiveMaxPool1d(1)(r)
            r = torch.transpose(r, 1, 2)
            r = torch.squeeze(r)

        return r        
    
    
#for inner product roberta output and label tensor with shifted
class CNNInnerRobertaNet(nn.Module):        
    """backbone + projection head"""
    def __init__(self, path, embedding_dim=768, max_seq_length=512, num_class=20):
        super(CNNInnerRobertaNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.num_class = num_class
        self.path = path
        self.encoder = RobertaModel.from_pretrained(self.path)
#         self.cnn = nn.Sequential(
#             nn.Conv1d(embedding_dim,embedding_dim,(50), stride=20),
# #             nn.ReLU(inplace=True),
#             nn.Conv1d(embedding_dim,embedding_dim,(3), stride=1),
# #             nn.ReLU(inplace=True),                
#         )    
        self.fcn = nn.Linear(self.max_seq_length-2, 10)
        self.fc =  nn.Linear(self.embedding_dim, self.num_class)
    
    def forward(self, sample, isLabel=True):
        r = self.encoder(sample)
        r = r[0][:,1:-1,:]
        if isLabel == False: # for training
#             print(r.size())
            r = torch.transpose(r, 1, 2)
            r = self.fcn(r)
        else : # for label tensor
            r = torch.transpose(r, 1, 2)
            r = nn.AvgPool1d(50, stride=50)(r)
            r = nn.AdaptiveMaxPool1d(1)(r)
            r = torch.transpose(r, 1, 2)
            r = torch.squeeze(r)
        return r   
    
    
#for contrastive learning
class ContraRobertaNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, path, embedding_dim=768, feat_dim=128, num_class=20):
        super(ContraRobertaNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.path = path
        self.encoder = RobertaModel.from_pretrained(self.path)
#         self.encoder = model_fun()
#        self.projection = nn.Sequential(
#            nn.Linear(self.embedding_dim, self.embedding_dim),
#            nn.ReLU(inplace=True),
#            nn.Linear(self.embedding_dim, self.feat_dim)
#        )
        self.projection = nn.Linear(self.embedding_dim, self.feat_dim)
        self.fc =  nn.Linear(self.embedding_dim, self.num_class)

    def forward(self, iscontra, sample ):
        r = self.encoder(sample)
        r = r[0][:,0,:]        
        if iscontra == True :
            # below line could be removed
            #r = F.normalize(self.projection(r), dim=1)
#             r = self.projection(r)
            return r
        else :
            r = self.fc(r)
            return r    
        
#for Transfer leraning from unlabeled data with label then fine tune.
# use focal loss for unbalancing class
class TransferRobertaNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, path, embedding_dim=768, num_class=20, num_class1=20):
        super(TransferRobertaNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_class = num_class
        self.num_class1 = num_class1
        self.path = path
        self.encoder = RobertaModel.from_pretrained(self.path)
#         self.encoder = model_fun()
        self.transfer_fc = nn.Linear(self.embedding_dim, self.num_class)
        self.down_fc =  nn.Linear(self.embedding_dim, self.num_class1)

    def forward(self, istransfer, sample ):
        r = self.encoder(sample)
        r = r[0][:,0,:]        
        if istransfer == True :
#             z = F.normalize(self.projection(r), dim=1)
            return self.transfer_fc(r)
        else :
            return self.down_fc(r)
        
# class YuBertNet(nn.Module):
#     """backbone + projection head"""
#     def __init__(self, path, embedding_dim=768, num_class=20):
# #         print('YuBertNet ',config)
#         super(YuBertNet, self).__init__()
#         print('YuBertNet is called ')
#         self.embedding_dim = embedding_dim
#         self.num_class = num_class
#         self.path = path
#         self.encoder = YubertModel.from_pretrained(self.path)
#         print('self.encoder')
# #         self.transfer_fc = nn.Linear(self.embedding_dim, self.num_class)
#         self.down_fc =  nn.Linear(self.embedding_dim, self.num_class)

#     def forward(self, sample):
#         r = self.encoder(sample)
#         r = r[0][:,0,:]   
#         return self.down_fc(r) 
# #         if istransfer == True :
# # #             z = F.normalize(self.projection(r), dim=1)
# #             return self.transfer_fc(r)
# #         else :
# #             return self.down_fc(r)        
                      
class YuBertNet(BertPreTrainedModel):
    """backbone + projection head"""
    config_class = RobertaConfig
    base_model_prefix = "roberta"
    
    def __init__(self, config):
#         super(YuBertNet, self).__init__(config)
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = YubertModel(config)
#         self.down_fc =  nn.Linear(config.hidden_size, config.num_labels)
        self.down_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*4),
            nn.Tanh(),
#             nn.ReLU(),
            nn.Linear(config.hidden_size*4, config.num_labels)
        )           

    def forward(self, sample):
        r = self.roberta(sample)
        r = r[0][:,0,:]   
        return self.down_fc(r) 
    
class YubertForSequenceClassification(RobertaForSequenceClassification):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = YubertModel(config)
        self.classifier = YubertClassificationHead(config)

        self.init_weights()        