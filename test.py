from __future__ import division, print_function

import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

import transforms
from dataset import cardCutDataset
from losses import card_loss, get_accum_card_error
from models import MyUNet
from transforms import transforms_train, transforms_val
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Testing on: ',device)

PATH = 'model/cardcut_cnn.pt'
path_CSV = "data/cardpoints.csv"
path_images = "data/images"
path_masks = "data/masks"
dataset = cardCutDataset(path_CSV, path_images, path_masks, transform=None)
dataset_val = cardCutDataset(path_CSV, path_images, path_masks, transform=None)

dataset.set_image_shape_manual(60, 80)
dataset_val.set_image_shape_manual(60, 80)

batch_size = 32
validation_split = .2
shuffle_dataset = True
random_seed= 420

train_indices, val_indices = get_train_val_indices(dataset, random_seed, validation_split)

dataset.set_transform(transforms_train)
dataset_val.set_transform(transforms_val)

# Creating data samplers/loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, sampler=valid_sampler, shuffle=False) # unshuffled, so we can always track the first sample of the first batch

model = MyUNet(temperature=0.25)
model.to(device)

model.load_state_dict(torch.load(PATH))
model.eval()
model.to('cpu')

show_model_validation(model, validation_loader, 0, 1, waitTime=300)