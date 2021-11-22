import torch
import torch.nn as nn
from cnn_finetune import make_model
import torchvision.transforms as transforms

class Classifier(nn.Module):
    def __init__(self, arch, num_classes, size=32) -> None:
        super().__init__()
        # self.normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2009, 0.1984, 0.2023])
        self.main_model =  make_model(arch, num_classes=num_classes, pretrained=True, input_size=(size, size))
    
    def forward(self, X):
        '''
        X: torch Tensor
                [B x C x H x W]
        '''
        return self.main_model(X)
        # return self.main_model(self.normalize(X))

