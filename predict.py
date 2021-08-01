import argparse
parser = argparse.ArgumentParser(description ='predict')
#parser.add_argument('filepath')
parser.add_argument('image_path')
parser.add_argument('model')
#parser.add_argument('topk')
args = parser.parse_args()

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models

from collections import OrderedDict
from PIL import Image
import numpy as np

import json
#image_path = './flowers/test/99/image_07833.jpg'
filepath = 'checkpoint.pth'
topk = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.vgg16_bn(pretrained=True)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])
    learning_rate = checkpoint['learning_rate']
    return model

model = load_checkpoint(filepath)

def process_image(image_path):
    pic = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406]
                             [0.229, 0.224, 0.225])])
    pic_trans = transform(pic)
    pic_to_array = np.array(pic_trans)
    return pic_to_array

def predict(image_path, model, device, topk=5):
    pic = process_image(image_path)
    pic = torch.from_numpy(pic).type(torch.FloatTensor)
    pic=pic.unsqueeze_(0)
    model = load_checkpoint(model)
    model.to(device)
    model.eval()
    with torch.no_grad():
        log_ps = model.forward(pic)
        ps = torch.exp(log_ps)
        top_ps,top_idx = ps.topk(topk,dim=1)
        list_ps = top_ps.tolist()[0]
        list_idx = top_idx.tolist()[0]
        index_mapping=dict(map(reversed, model.class_to_idx.items()))
        classes = []
        for i in list_idx:
            classes.append(index_mapping[i])
        model.train()
    return list_ps, classes

probs, classes = predict (image_path, model)
print(probs)
print(classes)