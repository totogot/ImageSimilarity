import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import itertools

import torchvision.models as models
import torchvision.transforms as transforms


def set_device():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    return device


def initiate_model(model_name: str):

    model = models.model_name(weights='DEFAULT')
    device = set_device()

    return model.to(device)


def obtain_children(model):
     
     model_embed = nn.Sequential(*list(model.children())[:-1])

     return model_embed


def obtain_classifier(model):

    model.classifier = model.classifier[:-1]
    
    return model


def assign_layer(model):

    embed_dict = {
        "resnet50": obtain_children(model),
        "vgg19":  obtain_classifier(model),
        "efficientnet_b0": obtain_classifier(model)
    }

    model_embed = embed_dict(model)


m = initiate_model("resnet50")





