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
from dataset import CardPoseDataset
from losses import card_loss, get_accum_card_error
from models import CardPoseCNN
from transforms import transforms_train, transforms_val
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on: ', device)

PATH = 'model/CardPose_cnn.pt'
path_CSV = "data/cardpoints.csv"
path_images = "data/images"
path_masks = "data/masks"
path_figures = "figures"
dataset = CardPoseDataset(path_CSV, path_images, path_masks, transform=None)
dataset_val = CardPoseDataset(path_CSV, path_images, path_masks, transform=None)

# dataset.set_image_shape_auto()
# dataset.create_mask_dataset()
# dataset.show_dataset_contours()

dataset.set_image_shape_manual(60, 80)
dataset_val.set_image_shape_manual(60, 80)
# dataset.create_mask_dataset()
# dataset.show_dataset_contours()

num_epochs = 25
batch_size = 32
validation_split = .2
shuffle_dataset = True
random_seed = 420

train_indices, val_indices = get_train_val_indices(dataset, random_seed, validation_split)

# Dataset normalisation
print('Computing the mean and std of the training dataset')
start = time.process_time()
mean, std = dataset.compute_mean_std(train_indices)
elapsed = time.process_time() - start
print('Done in {} minutes {} seconds.'.format(int(elapsed // 60), int(elapsed % 60)))
# mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

dataset.set_transform(transforms_train)
dataset_val.set_transform(transforms_val)

# Creating data samplers/loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           sampler=train_sampler)
# Unshuffled dataloader, so we can always track the first sample of the first batch
validation_loader = torch.utils.data.DataLoader(dataset_val,
                                                batch_size=batch_size,
                                                sampler=valid_sampler,
                                                shuffle=False)

stage_loaders = {'train': train_loader,
                 'validation': validation_loader}

n_cards_bins = 52  # Use -1 to predict a single scalar instead of N logits
model = CardPoseCNN(temperature=0.5, n_bins=n_cards_bins)
model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=0.001)
# For point detection (like binary classification)
criterion_visible = torch.nn.BCELoss()
# For number of cards: value regression or probability
criterion_ncards = torch.nn.CrossEntropyLoss() if n_cards_bins > 0 else torch.nn.MSELoss(reduction='mean')
# For keypoint heatmaps
criterion_mask = torch.nn.MSELoss(reduction='none')

running_loss_history = {'train': [], 'validation': []}
running_loss_history_separate = {'train': {'loss_cards': [],
                                           'loss_mask': [],
                                           'loss_visibility': []},
                                 'validation': {'loss_cards': [],
                                                'loss_mask': [],
                                                'loss_visibility': []}}
start_training = time.process_time()

for epoch in range(num_epochs):
    running_loss = {'train': 0.0, 'validation': 0.0}
    running_loss_separate = {'train': {'loss_cards': 0.0,
                                       'loss_mask': 0.0,
                                       'loss_visibility': 0.0},
                             'validation': {'loss_cards': 0.0,
                                            'loss_mask': 0.0,
                                            'loss_visibility': 0.0}}
    running_card_error = 0.0
    for stage in stage_loaders.keys():
        start = time.process_time()
        for batch_index, batch in enumerate(stage_loaders[stage]):
            inputs = batch['image'].to(device)
            value = batch['value'].to(device)
            points = batch['points'].squeeze().to(device)
            detections = batch['detections'].squeeze().to(device)
            masks = batch['mask_stack'].to(device)
            cards_counted = batch['cards_counted'].to(device)
            n_cards = cards_counted if n_cards_bins > 0 else value
            if stage == 'train':  # training
                model.train()
                optim.zero_grad()
                out_mask, out_detections, out_value = model(inputs)
                loss_visibility = criterion_visible(out_detections, detections)
                loss_ncards = criterion_ncards(out_value, n_cards)
                loss_mask = criterion_mask(out_mask, masks)
                loss_mask = torch.sum(loss_mask, dim=[2, 3]).mean()
                loss = loss_visibility + loss_ncards + loss_mask
                loss.backward()
                optim.step()
            else:  # validation
                model.eval()
                with torch.no_grad():
                    out_mask, out_detections, out_value = model(inputs)
                    loss_visibility = criterion_visible(out_detections, detections)
                    loss_ncards = criterion_ncards(out_value, n_cards)
                    loss_mask = criterion_mask(out_mask, masks)
                    loss_mask = torch.sum(loss_mask, dim=[2, 3]).mean()
                    loss = loss_visibility + loss_ncards + loss_mask
                    card_num = torch.argmax(out_value, axis=1).unsqueeze(1) / n_cards_bins if n_cards_bins > 0 else out_value
                    running_card_error += get_accum_card_error(value, card_num).item()

            running_loss[stage] += loss.item()
            running_loss_separate[stage]['loss_cards'] += loss_ncards.item()
            running_loss_separate[stage]['loss_mask'] += loss_mask.item()
            running_loss_separate[stage]['loss_visibility'] += loss_visibility.item()

        end = time.process_time()
        epoch_loss = round(running_loss[stage] / len(stage_loaders[stage]), 6)
        running_loss_history[stage].append(epoch_loss)

        running_loss_history_separate[stage]['loss_cards'].append(round(running_loss_separate[stage]['loss_cards'] / len(stage_loaders[stage]), 6))
        running_loss_history_separate[stage]['loss_mask'].append(round(running_loss_separate[stage]['loss_mask'] / len(stage_loaders[stage]), 6))
        running_loss_history_separate[stage]['loss_visibility'].append(round(running_loss_separate[stage]['loss_visibility'] / len(stage_loaders[stage]), 6))

        if stage == 'validation':
            card_error_epoch = int(round(running_card_error / (len(stage_loaders[stage]) * batch_size), 1))
            print('{} - Card error:  {} cards.'.format(stage, card_error_epoch))
            show_mask_only_sample(out_mask, 0, save=False)

        print('{} - Epoch [{}/{}] ({} minutes {} seconds) loss: '.format(stage, epoch, num_epochs - 1, (end - start) // 60, int((end - start) % 60)), epoch_loss)

cv2.destroyAllWindows()
torch.save(model.state_dict(), PATH)

end_training = time.process_time()
total_time = end_training - start_training
total_time_hrs = int(total_time // 3600)
total_time_mins = int((total_time % 3600) // 60)
total_time_secs = int((total_time % 3600) % 60)
print('Time elapsed: {} hours {} minutes {} seconds'.format(total_time_hrs, total_time_mins, total_time_secs))

plt.plot(range(num_epochs), running_loss_history['train'])
plt.plot(range(num_epochs), running_loss_history['validation'], '--')
plt.legend(['Train', 'Validation'])
plt.title('Training: {} epochs ({} hrs {} min {} sec)'.format(num_epochs, total_time_hrs, total_time_mins, total_time_secs))
plt.xlabel('Epoch')
plt.ylabel('Combined Loss')
plt.savefig(os.path.join(path_figures, 'training_loss.png'))
plt.show(block=False)
plt.pause(10)
plt.close()

plt.subplot(1, 3, 1)
plt.plot(range(num_epochs), running_loss_history_separate['train']['loss_cards'])
plt.plot(range(num_epochs), running_loss_history_separate['validation']['loss_cards'], '--')
plt.legend(['Train', 'Validation'])
plt.title('N_Card loss (card count)')
plt.xlabel('Epoch')
plt.ylabel('CE Loss')

plt.subplot(1, 3, 2)
plt.plot(range(num_epochs), running_loss_history_separate['train']['loss_mask'])
plt.plot(range(num_epochs), running_loss_history_separate['validation']['loss_mask'], '--')
plt.legend(['Train', 'Validation'])
plt.title('MSE (Heatmaps)')
plt.xlabel('Epoch')
plt.ylabel('Mask Loss')

plt.subplot(1, 3, 3)
plt.plot(range(num_epochs), running_loss_history_separate['train']['loss_visibility'])
plt.plot(range(num_epochs), running_loss_history_separate['validation']['loss_visibility'], '--')
plt.legend(['Train', 'Validation'])
plt.title('MSE (Visibility)')
plt.xlabel('Epoch')
plt.ylabel('Visibility Loss')

plt.suptitle('Training: {} epochs ({} hrs {} min {} sec)'.format(num_epochs, total_time_hrs, total_time_mins, total_time_secs))
plt.tight_layout()
plt.savefig(os.path.join(path_figures, 'training_loss_all.png'))
plt.show(block=False)
plt.pause(10)
plt.close()
