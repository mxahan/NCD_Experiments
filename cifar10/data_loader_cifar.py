#%%
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader
import torch.nn as nn
#%%

transform = transforms.Compose(
    [transforms.Resize((32,32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def pdb_checking():
    import pdb
    pdb.set_trace()
    return 0
#%%

from torchvision import datasets
from collections import defaultdict, deque
import itertools

import numpy as np

import random
#%% Method 1
class Cifar5000(datasets.CIFAR10):
    def __init__(self, path, transforms, train=True, label_S = list(range(0,9))):
        super().__init__(path, train, download=False)
        self.transforms = transforms
        self.n_images_per_class = 5000
        self.n_classes = 10
        self.labels = label_S
        self.cifar10 =  datasets.CIFAR10(path, train=train, download=True)




    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        index = random.randint(0, self.__len__()-1)
        im, label = self.cifar10[index]

        while label not in self.labels: 
            index = random.randint(0, self.__len__()-1)
            im, label = self.cifar10[index]

        if self.transforms:    
            im = self.transforms(im)

        return im, label


def Fdata_loader_C10(label_s = [0,1,2,3,4,8,9],unlabel_s = [5,6,7], batch_size = 32):
    tr_data_l =  Cifar5000(path = "../data", transforms= transform,
                              label_S = label_s)
    tr_data_u =  Cifar5000(path = "../data", transforms= transform,
                              label_S = unlabel_s)
    test_data =  Cifar5000(path = "../data", train= False, transforms= transform,
                              label_S = label_s)
    loader_l = DataLoader(tr_data_l, batch_size=batch_size,
                                               shuffle=True, num_workers=0)
    loader_u = DataLoader(tr_data_u, batch_size=batch_size,
                                               shuffle=True, num_workers=0)
    val_loader = DataLoader(test_data, batch_size=batch_size,
                                               shuffle=True, num_workers=0,
                                               drop_last=True)    
    return loader_l, loader_u, val_loader






classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
#%% Data Visualization


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

batch_size = 32
tr_loader_l, _, _ = Fdata_loader_C10(label_s= [0,1,2,3, 7,8,9],unlabel_s=[6,7], batch_size=batch_size,)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(tr_loader_l)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


#%% Augmentation Set

Aug_Set = [transforms.RandomHorizontalFlip(p = 0),
    transforms.RandomHorizontalFlip(p = 1),# FLips the image w.r.t horizontal axis
    transforms.RandomVerticalFlip(p = 1),# FLips the image w.r.t vertical axis
    transforms.RandomRotation(10),     #Rotates the image to a specified angel
    transforms.GaussianBlur(kernel_size= 3, sigma=(0.2, 1.5)),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
    transforms.Compose([transforms.CenterCrop(25), transforms.Resize((32,32)) ])]

#### getting all the augmented images in single line.
# imf_long = list(map(lambda Aug_Set: Aug_Set(images), Aug_Set))

#### Getting all images togehter

# start_idx = 0
# idx_crops =  len(Aug_set)
# for end_idx in idx_crops:
#     _out = torch.cat(imf_long[start_idx: end_idx])
#     if start_idx == 0:
#         output = _out
#     else:
#         output = torch.cat((output, _out))
#     start_idx = end_idx


#%% Import Model

from cifar10_model import Net, MobileNetV2, resnet18

#%%

net = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

import torch.optim as optim

#%% Loss functions

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)



sm = nn.Softmax(dim= 1)
import pdb
def neg_loss(ip, SS, tau = 0.1):
    yn = sm(ip/tau)
    # nl = torch.zeros(1,ip.shape[1], dtype= torch.float32)
    # nl[0][SS] = torch.tensor(1.0)
    # return -torch.sum(torch.mm(nl.cuda(),torch.log(1.0 - yn +1*10**-5).t()))
    return -torch.log(1 - yn[:,SS] + 1*10**-10).mean()


def entropy_loss(ip, US):
    yn = sm(ip[:, US]).cuda()
    return -torch.mean(torch.sum(yn*torch.log(yn + 1*10**-8), axis = 1))

import torch.nn.functional as F
class HLoss(nn.Module):
    def __init__(self, tau = 0.1):
        super(HLoss, self).__init__()
        self.tau = tau

    def forward(self, x):
        b = F.softmax(x/self.tau, dim=1) * F.log_softmax(x/self.tau, dim=1)
        b = -1.0 * b.sum(dim = 1)
        return b.mean()


def var_loss(ip, US):
    yn = sm(ip[:, US]/1).cuda()
    obj_var = (yn.shape[1] -1)/yn.shape[1]**2
    var_v= yn.var(0)
    return torch.mean((obj_var - var_v)**2)

#     return torch.sum(torch.square(var_v - obj_var)

KLD = torch.nn.KLDivLoss(log_target = True).cuda()

def kl_loss_ul(ip, tau =0.1, US = None):
    yn = sm(ip/tau)
    pp =  yn.mean(dim = 0)
    targ_ =  torch.zeros(pp.shape)
    targ_[US] = 1/len(US)
    return -torch.mean(targ_[US].cuda()*torch.log((pp[US] + 1*10**-10)/(targ_[US]+1*10**-10).cuda()))
    # yn = sm(ip[:, US])
    # return KLD(torch.log(targ_).cuda(), torch.log(pp).cuda())

    # return criterion(F.relu(ip).sum(dim =0), targ_.cuda()) # a relu to avoid any negative helping
    # return torch.sum((targ_.cuda()- pp)**2)
    # return  F.relu(targ_[US].cuda() -pp[US]).sum()*5
# Distance wise loss  F.relu(targ_ -pp).mean()


def kl_loss_uf_full(ip, US):
    yn = ip
    pp = yn.mean(0)
    targ_ =  torch.zeros(pp.shape)
    targ_[US] = 1/len(US)
    # return KLD(torch.log(targ_).cuda(), torch.log(pp).cuda())
    return None # use proper KLD loss / JS loss

def norm_loss(m1, m2):
    return torch.norm(m1 -m2)

l_mse = nn.MSELoss()
def norm_loss2(m1, target):
    return l_mse(m1, target)

def mul_var_loss(ip, tau = 0.1, US = None):
    yn = sm(ip/tau)
    pp =  yn.var(dim = 0)
    targ_ =  torch.zeros(pp.shape)
    targ_[US] = 1/len(US)
    targ_ = targ_*(1-targ_)
    return l_mse(pp.cuda(), targ_.cuda())

def neg_loss2(ip, US):
    yn= sm(ip)
    v_sum = torch.relu(1-torch.sum(yn[:,US], axis=1)).cuda()
    return torch.mean(v_sum)

#%% Training Loop
entropy_c = HLoss(tau=0.2)
#  https://discuss.pytorch.org/t/two-dataloaders-from-two-different-datasets-within-the-same-loop/87766/7
def train_loop(net, trainloader_l, trainloader_u, epoch_range = 2, SS = None, US = None):
    for epoch in range(epoch_range):  # loop over the dataset multiple times
        # pdb.set_trace()
        running_loss = 0.0
        for i, (data_l, data_u) in enumerate(zip(trainloader_l, trainloader_u), 0):
            
            # get the inputs; data is a list of [inputs, labels]
            images_, labels = data_l
            images_u_, _ = data_u
            
            aug_num =  random.sample([_ for _ in range(len(Aug_Set))], 2)
            
            if random.random()<0.80:
                images = Aug_Set[aug_num[0]](images_)
                images_u = Aug_Set[aug_num[0]](images_u_)
            else:
                images =  images_
                images_u = images_u_
            
                
            
            images = images.to(device)
            labels = labels.to(device)
            images_u = images_u.to(device)
            
            # pdb.set_trace()
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(images)
            loss_s1 = criterion(outputs/0.1, labels)
            n_loss_s = neg_loss(outputs,US)
            loss_s = loss_s1+n_loss_s
            if torch.sum(torch.isnan(loss_s))>0: #loss_s.isnan().any()
                pdb.set_trace()
            loss_s.backward()
            optimizer.step()
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs_u= net(images_u)
            # stored_op = outputs_u.detach()
            n_loss = neg_loss(outputs_u,SS)
            # e_loss = entropy_loss(outputs_u, US)
            e_loss = entropy_c(outputs_u[:, US])
            v_loss = mul_var_loss(outputs_u, tau= 0.1, US=US)
            dist_loss = 3*kl_loss_ul(outputs_u, tau=0.1,US = US)
            # dist_loss = kl_loss_uf_full(outputs_u, US)
            loss = dist_loss + e_loss + n_loss
            if torch.sum(torch.isnan(loss))>0: #loss.isnan().any()
                pdb.set_trace()
            loss.backward()
            optimizer.step()

            # optimizer.zero_grad()
            
            # images_u1 = Aug_Set[aug_num[1]](images_u_).cuda()
            # outputs_u1= net(images_u1)

            # full_loss =0.01* norm_loss(outputs_u1, stored_op)
            # # dist_loss = kl_loss_uf_full(outputs_u, US)
            # loss = full_loss
            
            # if torch.sum(torch.isnan(loss))>0: #loss.isnan().any()
            #     pdb.set_trace()
            # loss.backward()
            # optimizer.step()
            running_loss += loss_s.item()
            if i % 500 == 19:    # print every 500 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                print(f'sup: {loss_s} entropy loss: {e_loss} dist: {dist_loss} {n_loss} {n_loss_s}')  #
                # print(f'sup: {loss_s}')
                running_loss = 0.0
        scheduler.step()
        
    print('Finished Training')
#%% Data Load
label_s = [0,1,2,3,4]
unlabel_s = [5,6,7,8,9]
loader_l, loader_u, val_loader = Fdata_loader_C10(label_s=label_s, unlabel_s= unlabel_s, batch_size = 64)

#%% main train
# careful about nan value due to negative in torch.log input. keep a good distance from 0.
# stupid torch.log goes to inf then nan follows
train_loop(net, loader_l, loader_u, epoch_range = 30, SS = label_s, US = unlabel_s)

#%% Preparing test dataset (change for every new dataset)
# test result
# a = np.int16([0, 1650 , 3240, 4530, 6240,7980,9180,10980, 11580,12030])


batch_size = 32
_, _,val_loader = Fdata_loader_C10(label_s= [0,1,2,3,4,5,6,7,8,9], batch_size=batch_size,)

#%% 

device = 'cpu'
model = net

import pdb
def embed_test(val_loader, model):
    data_t =[] 
    data_lab = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            output= model(images)
            # output = activation['fc2']

            data_t.append(output.cpu().numpy())
            data_lab.append(labels.cpu())
    return data_t, data_lab


data_t, data_lab = embed_test(val_loader, net)

#%% Ploting TSNE

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

import sklearn
import sklearn.cluster as clusters

def tsne_plot(data = data_t, label1 = data_lab, n_comp = 2 ):
    
    
    X_embedded = TSNE(n_components=n_comp, verbose=1).fit_transform(data)

    
    new_data = clusters.KMeans(n_clusters=10).fit(data)
    print(sklearn.metrics.silhouette_score(data, new_data.labels_))

    # print(sklearn.metrics.silhouette_score(data, label1))
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal', adjustable='box')
    if n_comp == 3:ax = fig.add_subplot(projection ='3d')
    
    
    markers = ['v', 'x', 'o', '.', '>', '<', '1', '2', '3', '4']
    
    for i, g in enumerate([0,1,2,3,4]):#enumerate(np.unique(label1)):
        ix = np.where(label1 == g)
        if n_comp==3:
            ax.scatter(X_embedded[ix,0], X_embedded[ix,1], X_embedded[ix,2], marker = markers[i], label = g, alpha = 0.8)
        else:
            ax.scatter(X_embedded[ix,0], X_embedded[ix,1], marker = markers[i], label = classes[g], alpha = 0.8)
    
    ax.set_xlabel('Embedding dim 1')
    ax.set_ylabel('Embedding dim 2')
    if n_comp==3:ax.set_zlabel('Z Label')
    if n_comp==3:ax.set_zlabel('Z Label')
    ax.legend(fontsize='small', markerscale=2, loc = "upper left", ncol = 1)
    # fig.savefig("camera_info_sf_adv.svg", dpi=500, format='svg', metadata=None)
    plt.show()
    #plt 2
  
    
tsne_plot(np.vstack(data_t)[:,:,0,0],  np.vstack(data_lab).ravel(),2)


#%% Supervised setting Training Alternate training loop from dataloader
SupL =  nn.CrossEntropyLoss()

# my_net = net
net.cuda()
def sup_testing_res(criterion, my_net, data_loader):
    j = 0

    my_net.eval()
    r_test, r_test_full = [], []
    r_true = []
    
    for s1, s2 in data_loader: 
        s1 =  s1.cuda()
        s2 =  s2.cuda()

        
        with torch.no_grad():
            output = my_net(s1)
            outut=  activation['fc2']
            
        r_test_full.append(output.cpu())
        
        r_test.append(output.argmax(dim = 1).cpu())
        r_true.append(s2.cpu())

        if j == 500:
            return r_test, r_true, r_test_full
            break
            
        j = j+1
            
            
        
        
def sup_testing(net):
    data_loader = []; del(data_loader); torch.cuda.empty_cache()
    
    data_loader ,_ , _= Fdata_loader_C10(label_s= [0,1,2,3,4,5,6,7,8,9], batch_size=16,)
    r_ts, r_tr, r_ts_f = sup_testing_res(SupL, net, data_loader)
    
    
    r_ts_f = np.array(r_ts_f)
    r_tsnp =  np.array(r_ts)
    r_trnp =  np.array(r_tr)
    for i in range(len(r_ts)):r_trnp[i] = r_trnp[i].numpy();r_tsnp[i] = r_tsnp[i].numpy(); r_ts_f[i] = r_ts_f[i].numpy()
    del(data_loader); torch.cuda.empty_cache()
    r_tsc = np.concatenate(r_tsnp)
    r_trc = np.concatenate(r_trnp)
    
    r_tfs = np.concatenate(r_ts_f)

    return r_tsc, r_trc, r_tfs
        
r_ts, r_tr, r_tfs = sup_testing(net)

#%%

import seaborn as sn
from sklearn.metrics import confusion_matrix
# import pandas as pd

def conf_plot(r_ts, r_tr):
    # labs = list(LabelDict.keys())
    cf_matrix = confusion_matrix(r_ts, r_tr)
    print(cf_matrix)
    # df_cm = pd.DataFrame(cf_matrix/((np.sum(cf_matrix, axis = 0))), index = [i for i in [0,1,2,3,4,5,6,7,8,9]],
    #                        columns = [i for i in [0,1,2,3,4,5,6,7,8,9]])
    # plt.figure(figsize = (12,7))
    # sn.heatmap(df_cm, annot=True)

conf_plot(r_ts, r_tr)

#%% extra graph theory

import numpy as np

def sym_laps_fc(k = 5):
    
    l=[]
    for i in range(k):
        for j in range(k):
            if i==j:
                l.append(k-1)
            else:
                l.append(-1)
    return np.array(l).reshape(k,k)
            

def diag_block_mat_slicing(L):
    shp = L[0].shape
    N = len(L)
    r = range(N)
    out = np.zeros((N,shp[0],N,shp[1]),dtype=int)
    out[r,:,r,:] = L
    return out.reshape(np.asarray(shp)*N)


def diag_mat(rem=[], result=np.empty((0, 0))):
    if not rem:
        return result
    m = rem.pop(0)
    result = np.block(
        [
            [result, np.zeros((result.shape[0], m.shape[1]))],
            [np.zeros((m.shape[0], result.shape[1])), m],
        ]
    )
    return diag_mat(rem, result)

l_5= sym_laps_fc(3)
l_3= sym_laps_fc(5)
l_6 = sym_laps_fc(12)
l_7 = sym_laps_fc(6)

ll = diag_mat([l_5, l_3, l_6, l_7])

def r_c_swap_sym(ll, i, j):
    ll_s =ll
    ll_s[[i,j],:] = ll[[j,i],:]
    ll_s[:,[i,j]] = ll[:,[j,i]]
    return ll_s

from numpy.linalg import eigh

w, v= eigh(ll)


#%% graph Clustering 

from sklearn.cluster import KMeans

v_3 = v[:, :4]
kmeans = KMeans(n_clusters=4)
kmeans.fit(v_3)

#%% ResNet with pretrained weight

from torchvision.models import resnet50, ResNet50_Weights
import torch 
rn50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
rn50
img=  torch.randn(5,3,32,32)
rn50(img)
rn50.requires_grad_
rn50.fc
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
rn50.avgpool.register_forward_hook(get_activation('fc2'))