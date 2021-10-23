from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from torchvision import transforms

from PIL import Image


class Normalise(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = mean
        self.t = transforms.Normalize(self.mean, self.std)

    def __call__(self, sample):
        sample['image'] = self.t(sample['image'])
        return sample


class ColorJitter(object):
    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        self.p = p
        self.tf = transforms.ColorJitter(brightness=brightness,
                                         contrast=contrast,
                                         saturation=saturation,
                                         hue=hue)

    def __call__(self, sample):
        r = np.random.random_sample()
        if r > self.p:
            sample['image'] = self.tf(sample['image'])
        return sample


class ColorToGrayscale(object):
    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, sample):
        r = np.random.random_sample()
        if r > self.p:
            image = np.array(sample['image'].convert('LA'))
            image = np.stack([image[:, :, 0], image[:, :, 0], image[:, :, 0]], axis=2)
            sample['image'] = Image.fromarray(image)
        return sample


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        r = np.random.random_sample()
        if r > self.p:
            image = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
            points = sample['points']
            points[0, 1::2] = (1 - points[0, 1::2]) * (points[0, 1::2] > 0).astype('float')
            new_points = np.zeros_like(points)
            for k in range(4):
                new_points[0, k * 4] = points[0, (k * 4) + 2]  # Swap X
                new_points[0, k * 4 + 1] = points[0, (k * 4) + 3]  # Swap Y
                new_points[0, (k * 4) + 2] = points[0, (k * 4)]  # Swap X
                new_points[0, (k * 4) + 3] = points[0, (k * 4) + 1]  # Swap Y

            detections = sample['detections']
            new_detections = np.zeros_like(detections)
            for k in range(4):  # Swap left for right detections and viceversa
                new_detections[0, k * 2] = detections[0, k * 2 + 1]
                new_detections[0, k * 2 + 1] = detections[0, k * 2]

            sample['image'] = image
            sample['points'] = new_points
            sample['detections'] = new_detections

        return sample


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        r = np.random.random_sample()
        if r > self.p:
            image = sample['image'].transpose(Image.FLIP_TOP_BOTTOM)
            points = sample['points']
            points[0, ::2] = (1 - points[0, ::2]) * (points[0, ::2] > 0).astype('float')
            new_points = np.zeros_like(points)
            new_points[0, :8] = points[0, 8:]  # Swap top and bottom
            new_points[0, 8:] = points[0, :8]  # Swap top and bottom

            detections = sample['detections']
            new_detections = np.zeros_like(detections)
            new_detections[0, :4] = detections[0, 4:]  # Swap top and bottom
            new_detections[0, 4:] = detections[0, :4]  # Swap top and bottom

            sample['image'] = image
            sample['points'] = new_points
            sample['detections'] = new_detections

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, value, points, detections = sample['image'], sample['value'], sample['points'], sample['detections']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(image).transpose((2, 0, 1)) / 255
        sample['image'] = torch.from_numpy(image).type(torch.float)
        sample['value'] = torch.from_numpy(value).type(torch.float)
        sample['points'] = torch.from_numpy(points).type(torch.float)
        sample['detections'] = torch.from_numpy(detections).type(torch.float)

        return sample


transforms_train = transforms.Compose([
    ColorJitter(p=0.2, brightness=0.25, saturation=0.5, contrast=0.5, hue=0.25),
    # RandomHorizontalFlip(p=0.5),
    transforms.RandomChoice([
        RandomHorizontalFlip(),
        RandomVerticalFlip()]),
    ColorToGrayscale(),
    ToTensor(),
    # Normalise(mean, std)
])

transforms_val = transforms.Compose([
    ToTensor(),
    # Normalise(mean, std)
])


# UNUSED
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, points = sample['image'], sample['points']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for points because for images,
        # x and y axes are axis 1 and 0 respectively
        points = points * [new_w / w, new_h / h]
        sample['image'] = img
        sample['points'] = points
        return sample


# UNUSED
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, points = sample['image'], sample['points']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        points = points - [left, top]

        return {'image': image, 'points': points}
