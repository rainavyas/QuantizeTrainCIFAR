'''
Prepare the CIFAR-100 dataset as tokenized torch tensors
Functionality to quantize the image data to desired quantization
'''

import torch
import torch.nn as nn
from transformers import ElectraTokenizer, BertTokenizer, RobertaTokenizer
from datasets import load_dataset


class DataTensorLoader():
    def __init__(self):
        self.dataset = load_dataset('cifar_100')
    
    def _get_data(self, data):


        labels = torch.LongTensor(labels)

        return  labels

    def get_train(self):
        return self._get_data(self.dataset['train'])

    def get_test(self):
        return self._get_data(self.dataset['test'])
    
    def quantize(self, X, quantization=256):
        '''
        Quantize data
        Input X: [num_samples x W X H X C]
        '''
