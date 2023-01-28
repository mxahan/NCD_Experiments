#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 15:31:12 2023

@author: zahid
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import sklearn
import sklearn.cluster

import numpy as np
import random 

import idx2numpy
from torch.utils.data import Dataset, DataLoader
#%%

transform=transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
        ])


train_transform = transforms.Compose(
                    [
#                    transforms.ToPILImage(),
#                     transforms.RandomRotation(30),
                    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    ])

# train_transform = transforms.Compose(
#                     [
#                     transforms.ToPILImage(),
# #                     transforms.RandomRotation(30),
#                     transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
#                     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[train_mean], std=[train_std]),
#                     ])
 

def image_file_labe(train= True):
    
    if train: 
        imagefile = 'MNIST/raw/train-images-idx3-ubyte'
        labelfile = 'MNIST/raw/train-labels-idx1-ubyte'
    else: 
        imagefile = 'MNIST/raw/t10k-images-idx3-ubyte'
        labelfile = 'MNIST/raw/t10k-labels-idx1-ubyte'        
    
    imagearray = idx2numpy.convert_from_file(imagefile)
    
    imagelabel = idx2numpy.convert_from_file(labelfile)
    
    return imagearray, imagelabel



class CustomDataset(Dataset):
    def __init__(self, imagearray, imagelabel, transform = None, 
                 label_S = list(range(0,9)), unlabel_S = None):
        self.file_list = imagearray
        self.label_list = imagelabel
        self.transform = transform
        self.label_S = label_S
        if unlabel_S == None:
            self.unlabel_S = list(set(list(range(0,9))) - set(label_S))
        else:
            self.unlabel_S = unlabel_S
        
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        return self.img_L_ret(idx)
        
    
    def img_L_ret(self, idx):
        class_id = self.label_list[idx]
        
        while  class_id not in self.label_S:
            idx = random.randint(1, self.__len__())-1
            class_id = self.label_list[idx]
            
        img = self.file_list[idx:idx+1]
        img_tensor = torch.from_numpy(np.array(img).astype('float')/255.0)
        
        if self.transform:
            img_tensor = transform(img_tensor)
        
        class_id = torch.tensor([class_id])
        return img_tensor, class_id


def Fdata_loader(label_s = [0,1,2,3,4,5,6,7,8,9]):
    tr_data =  CustomDataset(*image_file_labe(), transform= train_transform,
                              label_S = label_s)
    
    val_data =  CustomDataset(*image_file_labe(train = False), 
                              transform=transform,
                              label_S =  label_s)
    
    loader = DataLoader(tr_data, batch_size = 32, shuffle = True)
    
    val_loader = DataLoader(val_data, batch_size = 32, shuffle = True)
    
    return loader, val_loader

loader, val_loader = Fdata_loader(labels = [0,1,2,3,4,5,6])


#%% Model 

from networks_mnist import Model, Net

#%% Define Model

model = Model()

#%% loss and optimizer 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.NLLLoss()   # with log_softmax() as the last layer, this is equivalent to cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#%% Training Loop

# Training Time!
import time
import copy

# Some initialization work first...
epochs = 100
train_losses, val_losses = [], []
train_accu, val_accu = [], []
start_time = time.time()
early_stop_counter = 10   # stop when the validation loss does not improve for 10 iterations to prevent overfitting
counter = 0
best_val_loss = float('Inf')

for e in range(epochs):
    epoch_start_time = time.time()
    running_loss = 0
    accuracy=0
    # training step
    model.train()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        log_ps, _ = model(images.float())
        
        ps = torch.exp(log_ps)                
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        loss = criterion(log_ps, labels[:,0])
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # record training loss and error, then evaluate using validation data
    train_losses.append(running_loss/len(loader))
    train_accu.append(accuracy/len(loader))
    val_loss = 0
    accuracy=0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            log_ps,_ = model(images.float())
            val_loss += criterion(log_ps, labels[:,0])

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    val_losses.append(val_loss/len(val_loader))
    val_accu.append(accuracy/len(val_loader))

    print("Epoch: {}/{}.. ".format(e+1, epochs),
          "Time: {:.2f}s..".format(time.time()-epoch_start_time),
          "Training Loss: {:.3f}.. ".format(train_losses[-1]),
          "Training Accu: {:.3f}.. ".format(train_accu[-1]),
          "Val Loss: {:.3f}.. ".format(val_losses[-1]),
          "Val Accu: {:.3f}".format(val_accu[-1]))

#     print('Epoch %d / %d took %6.2f seconds' % (e+1, epochs, time.time()-epoch_start_time))
#     print('Total training time till this epoch was %8.2f seconds' % (time.time()-start_time))
    
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        counter=0
        best_model_wts = copy.deepcopy(model.state_dict())
    else:
        counter+=1
        print('Validation loss has not improved since: {:.3f}..'.format(best_val_loss), 'Count: ', str(counter))
        if counter >= early_stop_counter:
            print('Early Stopping Now!!!!')
            model.load_state_dict(best_model_wts)
            break
#%% Save and Load Model

# torch.save(model.state_dict(), "Saved_models/model_all.pth")

model.load_state_dict(torch.load("Saved_models/model_0_5.pth"))

#%% Visualization
#https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5

#%% Preparing test dataset (change for every new dataset)
# test result
# a = np.int16([0, 1650 , 3240, 4530, 6240,7980,9180,10980, 11580,12030])

val_data =  CustomDataset(*image_file_labe(train = False), 
                          transform=transform,
                          label_S = [0,1,2,3,4,5,6,7,8,9])

val_loader = DataLoader(val_data, batch_size = 32, shuffle = False)

import pdb
def embed_test(val_loader, my_net):
    data_t =[] 
    data_lab = []
    my_net.eval()
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            log_ps,output = model(images.float())

            data_t.append(output.cpu().numpy())
            data_lab.append(labels.cpu())
    return data_t, data_lab


data_t, data_lab = embed_test(val_loader, model)
# rmsev=  data_t/np.sqrt(np.sum(data_t**2, axis = 1))[:, np.newaxis] # 



#%% Ploting TSNE

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
View_po = ['C0', 'C1', 'C2']

# if 'X_embedded' in locals():
#     None
# else:
#     X_embedded = TSNE(n_components=2, verbose=1).fit_transform(data_t)
 

def tsne_plot(data = data_t, n_comp = 3, label1 = data_lab, Lol = None, LoL = 1):
    if Lol== None:
        X_embedded = TSNE(n_components=n_comp, verbose=1).fit_transform(data)
    else:
        X_embedded = LoL
    
    print(sklearn.metrics.silhouette_score(data, label1))
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal', adjustable='box')
    if n_comp == 3:ax = fig.add_subplot(projection ='3d')
    
    # cdict = {0: 'red', 1: 'blue', 2: 'green'}
    
    markers = ['v', 'x', 'o', '.', '>', '<', '1', '2', '3', '4']
    
    for i, g in enumerate(np.unique(label1)):
        ix = np.where(label1 == g)
        if n_comp==3:
            ax.scatter(X_embedded[ix,0], X_embedded[ix,1], X_embedded[ix,2], marker = markers[i], label = g, alpha = 0.8)
        else:
            ax.scatter(X_embedded[ix,0], X_embedded[ix,1], marker = markers[i], label = g, alpha = 0.8)
    
    ax.set_xlabel('Embedding dim 1')
    ax.set_ylabel('Embedding dim 2')
    if n_comp==3:ax.set_zlabel('Z Label')
    if n_comp==3:ax.set_zlabel('Z Label')
    ax.legend(fontsize='small', markerscale=2, loc = "upper left", ncol = 1)
    # fig.savefig("camera_info_sf_adv.svg", dpi=500, format='svg', metadata=None)
    plt.show()
    #plt 2
  
    
tsne_plot(np.vstack(data_t), 2, np.vstack(data_lab)[:,0], Lol =None,  LoL =  1)
#%% shillohette coefficient!

print(sklearn.metrics.silhouette_score(np.vstack(data_t), np.vstack(data_lab)[:,0]))

# Optimal cluster Number

for i in range(3,11):
    new_data = sklearn.cluster.KMeans(n_clusters=i).fit(np.vstack(data_t))
    print(sklearn.metrics.silhouette_score(np.vstack(data_t), new_data.labels_))