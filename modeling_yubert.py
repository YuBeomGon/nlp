# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch YUBERT model. """


import logging
import math
import os
import warnings

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

# from .activations import gelu, gelu_new, swish
# from .configuration_bert import BertConfig
# from .file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable
# from .modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
# from .modeling_roberta import *
# from .modeling_bert import *
# from transformers import gelu, gelu_new, swish
from transformers import BertConfig
# from transformers import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable
from transformers import PreTrainedModel, RobertaModel, RobertaConfig, RobertaForMaskedLM, RobertaLMHead, RobertaForSequenceClassification
from transformers import (
        BertPreTrainedModel,
        BertModel,
        BertForPreTraining,
        BertForMaskedLM,
        BertLMHeadModel,
        BertForNextSentencePrediction,
        BertForSequenceClassification,
        BertForMultipleChoice,
        BertForTokenClassification,
        BertForQuestionAnswering,
        load_tf_weights_in_bert,
        BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        BertLayer,
        BertSelfAttention,
        BertAttention,
        BertLayer,
        BertEncoder
) 

class YuBertSelfAttentionJupyter(BertSelfAttention) :
    def __init__(self, config):
        super().__init__(config)        
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
#         self.key_pro = nn.Linear(config.seq_len, self.med_seq_len)
#         self.value_pro = nn.Linear(config.seq_len, self.med_seq_len)        

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)        

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)        

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         attention_scores = torch.nn.ELU()(attention_scores)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)

#         attention_probs = nn.Tanh()(attention_scores)
#         print(attention_scores.size())
        #Fast Transforemer, use Linear attention instead of softmax for reducing complexity of time
#         attention_scores = max()
#         print(attention_scores)
#         attention_scores = torch.nn.ReLU()(attention_scores-0.1) - torch.nn.ReLU()(-(attention_scores+0.1))
#         print(attention_scores)
        attention_probs = attention_scores/torch.norm(attention_scores, dim=3, keepdim=True) + 1e-6

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs      

class YuBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)        
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)        

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = torch.nn.ELU()(attention_scores)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         print(attention_scores.size())
#         attention_scores = torch.nn.ReLU()(attention_scores-0.1) - torch.nn.ReLU()(-(attention_scores+0.1))
        
        attention_probs = attention_scores/torch.norm(attention_scores, dim=3, keepdim=True) + 1e-6
#         attention_probs = nn.Tanh()(attention_scores)

#         denum = torch.max(torch.abs(attention_scores), dim=3, keepdim=True).values
#         attention_probs = nn.Tanh()(attention_scores/denum)
#         attention_probs = attention_probs/torch.norm(attention_probs, dim=3, keepdim=True)
#         denum = torch.max(torch.abs(input), dim=2, keepdim=True).values
#         attention_probs = nn.Tanh()(input/denum)
#         attention_probs/torch.norm(attention_probs, dim=2, keepdim=True)
#         print(attention_probs1.size())
#         print(torch.sum(attention_probs1, dim=3))
#         print(torch.sum(attention_probs1, dim=3).size())
#         print(attention_probs1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs      

class YuBertAttention(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        if (config.isjupyter == True) :
            self.self = YuBertSelfAttentionJupyter(config) 
        else :
            self.self = YuBertSelfAttention(config)

class YuBertLayer(BertLayer):        
    def __init__(self, config):
        super().__init__(config)
        self.attention = YuBertAttention(config)

class YuBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([YuBertLayer(config) for _ in range(config.num_hidden_layers)])

class YubertModel(RobertaModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = RobertaConfig
    base_model_prefix = "roberta"    

    def __init__(self, config):
        super().__init__(config)
        self.encoder = YuBertEncoder(config)
        
class YubertForMaskedLM(RobertaForMaskedLM):        
    config_class = RobertaConfig
    base_model_prefix = "roberta"   
    
    def __init__(self, config):
        super().__init__(config)
        self.roberta = YubertModel(config)    
    
class YubertLMHead(RobertaLMHead):    
    def __init__(self, config):
        super().__init__(config)        
        
class YubertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.num_labels)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
        x = self.dense(x)
        return x        