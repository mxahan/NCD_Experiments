#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 21:15:36 2023

@author: zahid
"""
import torch

import torch.nn as nn

from torchvision import transforms
from torchvision.datasets import UCF101

#%%

ucf_data_dir = "../data"
ucf_label_dir = "/data"
frames_per_clip = 5
step_between_clips = 1
batch_size = 32

tfs = transforms.Compose([
            # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
            # scale in [0, 1] of type float
            transforms.Lambda(lambda x: x / 255.),
            # reshape into (T, C, H, W) for easier convolutions
            transforms.Lambda(lambda x: x.permute(0, 1, 2, 3)),
            # rescale to the most common size
            transforms.Lambda(lambda x: nn.functional.interpolate(x, (240, 320))),
])

#%% Masud Version
from glob import glob
import yaml
import math
import torch
import random
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx import all as vfx
from torch.utils.data import Dataset, DataLoader

# we can change the label.yaml  to feed the required data class
class UCF101dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, label_file = "data/UCF-101/label.yaml",clip_len=16, 
                 resize=(224, 224), random_clip = False, label_s = list(range(0,101))):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.resize = resize 
        self.random_clip = random_clip
        with open(label_file, 'r') as stream:
            self.label = yaml.safe_load(stream)['label']
        self.ind = self.find_dir()
        np.random.shuffle(self.ind)
        self.label_s = label_s

    def __getitem__(self, index):
        clip, label = self.ind[index]
        
        while label not in self.label_s: 
            index = random.randint(0, self.__len__()-1)
            clip, label = self.ind[index]
        
        clip = VideoFileClip(clip)
        clip_duration = clip.duration
        start_time = np.random.uniform(0, clip_duration - self.clip_len) if self.random_clip else 0
        if self.resize:
            clip = vfx.resize(clip, self.resize)
        clip = np.array(list(clip.iter_frames()))
        start_time = random.randint(0, len(clip) - self.clip_len -1) if self.random_clip else 0
#         return clip[:self.clip_len], self.ind[index][0]
                
        vid_clips = torch.as_tensor(clip[start_time:start_time+self.clip_len]/255.0).float().contiguous()
        vid_clips = vid_clips.permute(3,0,1,2)
        return  vid_clips,torch.as_tensor(label).long().contiguous(), self.ind[index][0]

    def __len__(self):
        return len(self.ind)
        
    def find_dir(self):
        file_list = []
        label_file = []
        for k, v in self.label.items():
            if v == 255:
                pass
            files = glob(self.root_dir+k+"/*.avi")
            file_list += files
            label_file += [v] * len(files)
        files = list(zip(file_list, label_file))
        return files
    
#%% Helper YAML Creation

def yaml_creation():
        
    c = glob("../data/UCF101/UCF-101/*")
    dirt = []
    for a in c:
        b = a.split('/')[-1]
        dirt.append(b)
    dirt.sort()
    # print(dirt)
    for i in range(len(dirt)):
        b = ' ' + dirt[i] + ' : ' +repr(i)
        print(b)
    # copy it to the yaml file
    return

def lab_ulab(l_set = list(range(0,90))):
    Comp_set = list(range(0, 101))
    u_set = list(set(Comp_set) - set(l_set))
    return l_set, u_set

#%% 
dataset = UCF101dataset(root_dir = '../data/UCF101/UCF-101/', label_file = '../data/UCF101/label.yaml',
                        label_s=list(range(0,51)))
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

dataiter = iter(data_loader)
images, labels, df = next(dataiter)

#%%
from glob import glob
import yaml
import math
import torch
import pickle
import random
import pdb
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx import all as vfx
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as nn

# we can change the label.yaml  to feed the required data class

# the following class can be updated later on for better results. 
class UCF101dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, label_file = "../Data/UCF101/UCF-101/all_class.yaml",clip_len=16, 
                 resize=(224, 224), SS = list(range(0,96)), US = list(range(96,101)), bs = 8):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.resize = resize 
        with open(label_file, 'r') as stream:
            self.label = yaml.safe_load(stream)['label']
        self.ind = self.find_dir()
        np.random.shuffle(self.ind)
        self.SS = SS
        self.US = US
        self.bs = bs
        self.mini = len(self.ind)-1

    def __getitem__(self, index):
        return self.get_data_neg_learn()

    def __len__(self):
        return self.mini +1
        
    def find_dir(self):
        file_list = []
        label_file = []
        for k, v in self.label.items():
            if v == 255:
                pass
            files = glob(self.root_dir+k+"/*.avi")
            file_list += files
            label_file += [v] * len(files)
        files = list(zip(file_list, label_file))
        return files
    
    def get_anchor(self, index =  None):
        if index ==None:
            index = random.randint(0, self.mini)
        clip, label = self.ind[index]
        clip = pickle.load(open(clip + '.pkl', 'rb'))[0]
#         clip1 = VideoFileClip(clip)
# #         clip_duration = clip1.duration
# #         if self.resize: clip1 = vfx.resize(clip1, self.resize)
#         clip = np.array(list(clip1.iter_frames())).copy()
#         clip1.close()
        start_time = random.randint(0, clip.shape[0] - self.clip_len -1)
        vid_clips = clip[start_time:start_time+self.clip_len].copy()
        del(clip)
        return vid_clips, label, self.ind[index][0]
    
    
    
    def get_data_neg_learn(self):    
        
        sample, gt, FN = [], [], []
        if np.random.uniform()>(0.1-np.exp(-10)):
            for _ in range(self.bs):
                index = random.randint(0, self.mini)    
                clip, label = self.ind[index]
                while label not in self.SS: 
                    index = random.randint(0, self.mini)
                    clip, label = self.ind[index]
                anchor, label, info_v = self.get_anchor(index)
                gt.append(label)
                sample.append(anchor)
                FN.append(info_v)
                
        else:
            for _ in range(self.bs):
                index = random.randint(0, self.mini)    
                clip, label = self.ind[index]
                while label not in self.US: 
                    index = random.randint(0, self.mini)
                    clip, label = self.ind[index]
                anchor, label, info_v = self.get_anchor(index)
                label = 1000
                gt.append(label)
                sample.append(anchor)
                FN.append(info_v)
        try:        
            ret = np.moveaxis(np.stack(sample), -1, -4)
        except:
            for i in range(self.bs):
                sample[i]= nn.interpolate(torch.tensor(sample[i]).permute(0, 3,1,2).type(torch.float32), size = (224, 224), mode = "bilinear").permute(0,2,3,1).numpy()
            ret = np.moveaxis(np.stack(sample), -1, -4)
    
        try:
            gt = np.stack(gt)
        except ValueError:
            pdb.set_trace()

        return ret.astype(np.float32)/255.0, gt, FN

