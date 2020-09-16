import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.utils.data
import torchvision

def get_text(labels, df_dict) :    
    df_list = []
    seed = np.random.randint(0, 100000, size=None)
    for label in labels.numpy() :
        df_train = df_dict[label].sample(n=1, random_state=seed, replace=True )
        df_list.append(df_train)
    df = pd.concat(df_list)
    return df.SE

class PetDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.SE_index = [ i for i, c in enumerate(df.columns) if "SE" in c][0]
        self.label_index = [ i for i, c in enumerate(df.columns) if "label_id" in c][0]
        self.Num_class = len(df[df.columns[self.label_index]].value_counts())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, self.SE_index]
        label = self.df.iloc[idx, self.label_index]
        return text, label

# # https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py    
# class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
#     """Samples elements randomly from a given list of indices for imbalanced dataset
#     Arguments:
#         indices (list, optional): a list of indices
#         num_samples (int, optional): number of samples to draw
#         callback_get_label func: a callback-like function which takes two arguments - dataset and index
#     """

#     def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
#         # if indices is not provided, 
#         # all elements in the dataset will be considered
#         self.indices = list(range(len(dataset))) \
#             if indices is None else indices

#         # define custom callback
#         self.callback_get_label = callback_get_label

#         # if num_samples is not provided, 
#         # draw `len(indices)` samples in each iteration
#         self.num_samples = len(self.indices) \
#             if num_samples is None else num_samples
            
#         # distribution of classes in the dataset 
#         label_to_count = {}
#         for idx in self.indices:
#             label = self._get_label(dataset, idx)
#             if label in label_to_count:
#                 label_to_count[label] += 1
#             else:
#                 label_to_count[label] = 1
                
#         # weight for each sample
#         weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
#                    for idx in self.indices]
#         self.weights = torch.DoubleTensor(weights)

#     def _get_label(self, dataset, idx):
#         if isinstance(dataset, torchvision.datasets.MNIST):
#             return dataset.train_labels[idx].item()
#         elif isinstance(dataset, torchvision.datasets.ImageFolder):
#             return dataset.imgs[idx][1]
#         elif isinstance(dataset, torch.utils.data.Subset):
#             return dataset.dataset.imgs[idx][1]
#         elif self.callback_get_label:
#             return self.callback_get_label(dataset, idx)
#         else:
#             raise NotImplementedError
                
#     def __iter__(self):
#         return (self.indices[i] for i in torch.multinomial(
#             self.weights, self.num_samples, replacement=True))

#     def __len__(self):
#         return self.num_samples
