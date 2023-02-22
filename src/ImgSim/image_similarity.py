#import numpy as np
#import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
#import itertools
import os

import torchvision.models as models
import torchvision.transforms as transforms


class Img2Vec():

    def __init__(self, model_name, weights='DEFAULT'):

        # dictionary defining the supported embeddings
        self.embed_dict = {
            "resnet50": self.obtain_children,
            "vgg19":  self.obtain_classifier,
            "efficientnet_b0": self.obtain_classifier
            }
        
        # assign class parameters
        self.architecture = self.validate_model(model_name)
        self.weights = weights
        self.transform = self.assign_transform(weights)
        self.device = self.set_device()
        self.model = self.initiate_model()
        self.embed = self.assign_layer()
        self.dataset = {}

    
    def validate_model(self, model_name):
        if model_name not in self.embed_dict.keys():
            raise ValueError(f'The model {model_name} is not supported')
        else:
            return model_name

    
    def assign_transform(self, weights):
       
        weights_dict = {
            "resnet50": models.ResNet50_Weights,
            "vgg19": models.VGG19_Weights,
            "efficientnet_b0":models.EfficientNet_B0_Weights
        }

        try:
            w = weights_dict[self.architecture]
            weights = getattr(w, weights)
            preprocess =  weights.transforms()
        except:
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
                ])

        return preprocess

    
    def set_device(self):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        
        return device
    
    
    def initiate_model(self):
        
        m = getattr(models, self.architecture) # equ to assigning m as models.resnet50() 
        model = m(weights = self.weights) # equ to models.resnet50(weights = ...)
        model.to(self.device)
        
        return model.eval()
    

    def assign_layer(self):
        model_embed = self.embed_dict[self.architecture]()

        return model_embed
    

    def obtain_children(self):
        model_embed = nn.Sequential(*list(self.model.children())[:-1])
        
        return model_embed


    def obtain_classifier(self):
        self.model.classifier = self.model.classifier[:-1]
        
        return self.model
      

    def directory_to_list(self, source):

        types = (
            '.png',
            '.jpg',
            '.jpeg'
        )
        
        dir_list = os.listdir(source)
        source_list = [
            os.path.join(source, f) for f in dir_list 
            if os.path.splitext(f)[1] in types
            ]

        return source_list
        
    
    def validate_source(self, source):

        if isinstance(source, list):
            source_list = [f for f in source if os.path.isfile(f)]
        elif os.path.isdir(source):
            source_list = self.directory_to_list(source)
        elif os.path.isfile(source):
            source_list = [source]
        else:
            raise ValueError('"source" expected as file, list or directory.')
        
        return source_list

    
    def embed_image(self, img):
        
        img = Image.open(img)
        img_trans = self.transform(img)

        if self.device == "cuda:0":
            img_trans = img_trans.cuda()
        
        img_trans = img_trans.unsqueeze(0)

        return self.embed(img_trans)

    
    def embed_dataset(self, source):
        
        self.files = self.validate_source(source)    

        for file in self.files:
            vector = self.embed_image(file)
            self.dataset[str(file)] = vector
        
        return
            
    
    def similar_images(self, target_file, n=None):

        target_vector = self.embed_image(target_file)
        cosine = nn.CosineSimilarity(dim=1)

        sim_dict = {}
        for k, v in self.dataset.items():
            sim = cosine(v, target_vector)[0].item()
            sim_dict[k] = sim
        
        sim_dict = {k: v for k, v in sorted(sim_dict.items(), 
                                            key=lambda item: item[1], 
                                            reverse=True)}

        if n is not None:
            sim_dict = dict(list(sim_dict.items())[:n])

        self.output_images(sim_dict, target_file)

        return sim_dict
    

    def output_images(self, similar, target):

        self.display_img(target, 'original')

        for k, v in similar.items():
            self.display_img(k, 'similarity:' + str(v))

        return
    

    def display_img(self, path, title):

        plt.imshow(Image.open(path))
        plt.axis('off')
        plt.title(title)
        plt.show()

        return


    def save_dataset(self, path):
        
        data = {
            "model": self.architecture,
            "embeddings": self.dataset
            }
        
        torch.save(data, os.path.join(path,'tensors.pt')) #need to update functionality for naming convention

        return
    

    def load_dataset(self, source):

        data = torch.load(source)

        if data["model"] == self.architecture:
            self.dataset = data["embeddings"]
        else:
            raise AttributeError(
                f'NN architecture "{self.architecture}" does not match the '
                +f'"{data["model"]}" model used to generate saved embeddings. '
                +'Re-initiate Img2Vec with correct architecture and reload.'
                )
        
        return
        