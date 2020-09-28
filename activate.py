import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.utils.data
# import torchvision

def swish(x):
    return x * torch.nn.Sigmoid()(x)

def ymish(x) :
    return x * torch.tanh(torch.abs(x))