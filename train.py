import argparse
parser = argparse.ArgumentParser(description ='train')
parser.add_argument("data_dir")
parser.add_argument("learning_rate", type=float)
parser.add_argument("epochs", type=int)
parser.add_argument("model_arch", choices=['vgg16_bn', 'densnet121'])
parser.add_argument("hidden_layer1", type=int)
parser.add_argument("device", choices=['cuda', 'cpu'])

args = parser.parse_args()
lr=args.learning_rate
epochs=args.epochs
structure=args.model_arch
hidden_layer1=args.hidden_layer1
device=args.device

# cd ImageClassifier/
# python train.py data_dir 0.001 1 'vgg16_bn' 4096 'gpu'

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models

from collections import OrderedDict
import json


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'


transforms_train = transforms.Compose([transforms.Resize(224),
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


transforms_test = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


image_datasets_train = datasets.ImageFolder(train_dir,transforms_train)
image_datasets_valid = datasets.ImageFolder(valid_dir,transforms_test)

train_dataloaders = torch.utils.data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)
valid_dataloaders = torch.utils.data.DataLoader(image_datasets_valid, batch_size=64)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_arch = {'vgg16_bn':25088, 'densnet121':1024}
if structure == 'vgg16_bn':
    model = models.vgg16_bn(pretrained=True)
    model.name = 'vgg16_bn'
elif structure == 'densnet121':
    model = models.densnet121(pretrained=True)
    model.name = 'densnet121'

for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(model_arch[structure],hidden_layer1)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('fc2', nn.Linear(hidden_layer1,102)),
            ('output', nn.LogSoftmax(dim=1))
             ]))

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)

if torch.cuda.is_available() and device =='cuda':
    model.to(device) # to device
   
train_losses, valid_losses = [], []
for epoch in range(epochs):
    running_loss = 0
    for images, labels in train_dataloaders:
        if torch.cuda.is_available() and device =='cuda':                             
            images, labels = images.to(device), labels.to(device) # to device
        optimizer.zero_grad()
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
        
            model.eval()
            
            for images, labels in valid_dataloaders:
                if torch.cuda.is_available() and device =='cuda':                      
                    images, labels = images.to(device), labels.to(device) # to device
                logps = model.forward(images)
                test_loss += criterion(logps, labels)
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
             
        model.train() 
        train_losses.append(running_loss/len(train_dataloaders))
        valid_losses.append(test_loss/len(valid_dataloaders))
        
        print(f"Epoch {epoch+1}/{epochs}.. "
                # average of training loss over the last print_every batches
                f"Train loss: {running_loss/len(train_dataloaders):.3f}.. "
                # validation loss & accuracy across the entire dataset
                f"Test loss: {test_loss/len(valid_dataloaders):.3f}.. "
                f"Test accuracy: {accuracy/len(valid_dataloaders):.3f}")

def save_checkpoint(model, image_datasets_train, path='checkpoint.pth'):
    checkpoint = {'hidden_layer1':hidden_layer1,
            'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'classifier': model.classifier,
             'epochs': epochs,
             'learning_rate': lr, 
             'class_to_idx':image_datasets_train.class_to_idx}
    return torch.save(checkpoint, path)  

checkpoint = save_checkpoint(model, image_datasets_train)
