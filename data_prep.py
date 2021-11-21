'''
Prepare the CIFAR-100 dataset as tokenized torch tensors
Functionality to quantize the image data to desired quantization
'''

import torch
import torch.nn as nn
from transformers import ElectraTokenizer, BertTokenizer, RobertaTokenizer
from datasets import load_dataset
import numpy as np


class DataTensorLoader():
    def __init__(self):
        self.dataset = load_dataset('cifar_100')
    
    def _get_data(self, data):

        imgs = data['img']
        labels = data['label']

        labels = torch.LongTensor(labels)
        imgs = torch.FloatTensor(imgs)

        return  imgs, labels

    def get_train(self):
        return self._get_data(self.dataset['train'])

    def get_test(self):
        return self._get_data(self.dataset['test'])
    
    def quantize(self, X, quantization=256):
        '''
        Quantize data
        Input X: torch tensor [num_samples x W X H X C]
        '''
        num_bin_edges = quantization + 1
        bins = np.linspace(0, 255, num=num_bin_edges)
        X = X.cpu().detach().numpy()
        inds = np.digitize(X, bins)
        X_quantized = torch.FloatTensor(np.vectorize(bins.tolist().__getitem__)(inds.astype(int)))
        return X_quantized


