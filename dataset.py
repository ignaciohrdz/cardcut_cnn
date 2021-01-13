from __future__ import print_function, division
import os

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd
import math
import numpy as np
import cv2
from skimage import io, transform

from PIL import Image

def show_points(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

class cardCutDataset(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir=None, transform=None):
        self.csv_file = csv_file
        self.data = pd.read_csv(csv_file, delimiter=",")
        self.data = self.data[self.data['annotated']==1] # keep only the annotated rows
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.points_visible = [k for k in list(self.data.columns.values) if '_visible' in k]
        self.points_coordinates = [k for k in list(self.data.columns.values) if ('_x' in k) or ('_y' in k)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.image_dir,self.data.iloc[idx,0])
        image = Image.open(img_path)
        image = image.resize((image.width // 4, image.height // 4))
        value = np.array([(self.data.iloc[idx,2]/self.data.iloc[idx,3])]).astype('float')
        points = self.data.loc[idx,self.points_coordinates]
        # points = np.array([points]).astype('float').reshape(-1,2)
        points = np.array([points]).astype('float')
        points_visible = np.array([self.data.loc[idx,self.points_visible]]).astype('float')

        # mask_path = os.path.join(self.mask_dir,self.data.loc[idx,'mask'])
        # mask = Image.open(mask_path)

        # sample = {'image': image, 'value': value, 'points': points, 'detections': points_visible, 'mask': mask}
        sample = {'image': image, 'value': value, 'points': points, 'detections': points_visible}

        if self.transform:
            sample = self.transform(sample)

        sample = self.generate_points_masks(sample, sigma=3)

        return sample

    # Setting the image size from the first image of the dataset
    def set_image_shape_auto(self):
        image_files = self.data.loc[0,:]
        img_path = os.path.join(self.image_dir,image_files[0])
        image = Image.open(img_path)
        self.image_shape = [image.size[1], image.size[0]]
        total_rows = self.image_shape[0]
        total_cols = self.image_shape[1]
        self.X_mask, self.Y_mask = np.ogrid[:total_rows, :total_cols]

    # Setting the image shape manually (useful for setting the mask size)
    def set_image_shape_manual(self, rows, cols):
        self.image_shape = [rows, cols]
        total_rows = self.image_shape[0]
        total_cols = self.image_shape[1]
        self.X_mask, self.Y_mask = np.ogrid[:total_rows, :total_cols]

    def set_transform(self, transform):
        self.transform = transform

    def compute_mean_std(self, idxes):
        # Remember: PIL is RGB, openCV is BGR
        image_files = self.data.loc[idxes,:]
        mean = np.zeros(3)
        std = np.zeros(3)
        w = 0
        h = 0

        # Mean
        for _, row in image_files.iterrows():
            img_path = os.path.join(self.image_dir,row[0])
            image = Image.open(img_path)
            # image = image.resize((image.width//4, image.height//4))
            w = image.width
            h = image.height
            image = np.array(image)/255
            mean += image.sum(axis=(0,1))
        mean = mean/(h*w*len(image_files))

        # Standard deviation
        for _, row in image_files.iterrows():
            img_path = os.path.join(self.image_dir,row[0])
            image = Image.open(img_path)
            # image = image.resize((image.width//4, image.height//4))
            image = np.array(image)/255
            image = (image - mean)**2
            std += image.sum(axis=(0,1))

        std = np.sqrt(std/(h*w*len(image_files)-1))

        return mean.tolist(), std.tolist()

    def generate_points_masks(self, sample, sigma=1):
        if torch.is_tensor(sample['points']):
            points = sample['points'].squeeze().numpy()
            detections = sample['detections'].squeeze().numpy()
        else:
            points = sample['points']
            detections = sample['detections']

        masks = []

        for i in range(len(detections)):
            if detections[i] == 1:
                center_row = int(points[2*i]*self.image_shape[0])
                center_col = int(points[2*i + 1]*self.image_shape[1])
                mask_i = (1/(sigma*np.sqrt(2*math.pi))) * np.exp(-((self.X_mask - center_row) ** 2 + (self.Y_mask - center_col) ** 2) / (2 * sigma ** 2))
                mask_i = mask_i/mask_i.max()
                masks.append(mask_i)
            else:
                mask_i = np.zeros(self.image_shape)
                masks.append(mask_i)

        mask_stack = np.stack(masks, axis=2).transpose((2, 0, 1))
        if torch.is_tensor(sample['points']):
            sample['mask_stack'] = torch.from_numpy(mask_stack).type(torch.float)
        else:
            sample['mask_stack'] = mask_stack

        return sample

    def create_list_points_by_index(self, idx):
        row = self.data.loc[idx,:]
        points_x = [int(row[c] * self.image_shape[0]) for c in self.data.columns.values if "_x" in c and row[c] > 0]
        points_y = [int(row[c] * self.image_shape[1]) for c in self.data.columns.values if "_y" in c and row[c] > 0]
        contour = list(map(list,zip(points_y, points_x)))
        contour = [np.array(contour)]
        return contour
    
    def create_list_points_by_row(self, row):
        points_x = [int(row[c] * self.image_shape[1]) for c in self.data.columns.values if "_x" in c and row[c] > 0]
        points_y = [int(row[c] * self.image_shape[0]) for c in self.data.columns.values if "_y" in c and row[c] > 0]
        contour = zip(points_y, points_x)
        contour = [np.array(contour)]
        return contour

    def show_dataset_contours(self):
        for idx, _ in self.data.iterrows():
            self.show_contour_by_index(idx)
        cv2.destroyAllWindows()

    def show_contour_by_index(self, idx):
        row = self.data.loc[idx,:]
        img_path = os.path.join(self.image_dir,row[0])
        img = cv2.imread(img_path)
        contour = self.create_list_points_by_index(idx)
        draw = np.zeros(self.image_shape + [3], np.uint8)
        hull = cv2.convexHull(contour[0])
        cv2.drawContours(draw, [hull], 0, (255,255,255), thickness=-1)
        cv2.imshow('Image', img)
        cv2.imshow('Contour (hull)', draw)
        cv2.waitKey()

    def show_contour_by_row(self, row):
        img_path = os.path.join(self.image_dir,row[0])
        img = cv2.imread(img_path)
        contour = self.create_list_points_by_row(row)
        draw = np.zeros(self.image_shape + [3], np.uint8)
        hull = cv2.convexHull(contour[0])
        cv2.drawContours(draw, [hull], 0, (255,255,255), thickness=-1)
        cv2.imshow('Image', img)
        cv2.imshow('Contour (hull)', draw)
        cv2.waitKey()

    def make_contour_by_index(self, idx):
        contour = self.create_list_points_by_index(idx)
        draw = np.zeros(self.image_shape + [3], np.uint8)
        hull = cv2.convexHull(contour[0])
        cv2.drawContours(draw, [hull], 0, (255,255,255), thickness=-1)
        mask = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
        return mask

    def make_contour_by_row(self, row):
        contour = self.create_list_points_by_row(row)
        draw = np.zeros(self.image_shape + [3], np.uint8)
        hull = cv2.convexHull(contour[0])
        cv2.drawContours(draw, [hull], 0, (255,255,255), thickness=-1)
        mask = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
        return mask

    def create_mask_dataset(self):
        for idx, row in self.data.iterrows():
            img_path = os.path.join(self.image_dir,row[0])
            mask_name = row[0].replace('.jpg', '_mask.jpg')
            self.data.loc[idx,'mask'] = mask_name
            mask_path = img_path.replace('.jpg', '_mask.jpg').replace('images','masks')
            mask = self.make_contour_by_index(idx)
            cv2.imwrite(mask_path, mask)
        
        self.data.to_csv(self.csv_file, index=False)