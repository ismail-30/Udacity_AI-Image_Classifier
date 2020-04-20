import torch
from torchvision import models
from collections import OrderedDict
import torch.nn as nn

def load_checkpoint(filepath, cuda=False):
    if cuda:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location='cpu')
    
    arch = checkpoint['structure']
    model = eval("models.{}(pretrained=True)".format(arch))
    
    # Freeze the feature parameters
    for params in model.parameters():
        params.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])    
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    
    return model