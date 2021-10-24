import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


# Read this paper? https://arxiv.org/pdf/1804.06208
class MyUNet(nn.Module):
    def __init__(self, filters=[64, 128, 256, 384], temperature=0.5, n_bins=52):
        super(MyUNet, self).__init__()

        self.n_bins = n_bins

        # Down path (normal convolutions)
        self.conv1 = nn.Conv2d(3, filters[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)

        # Up path (upsampling + convolutions)
        self.upsample = nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True)
        self.up_conv1 = nn.Conv2d(filters[3] + filters[1], filters[2], kernel_size=1)
        self.up_conv2 = nn.Conv2d(filters[2] + filters[0], filters[1], kernel_size=1)
        self.up_conv3 = nn.Conv2d(filters[1], 8, kernel_size=1)

        # Pooling and activations
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptAvgPool = nn.AdaptiveAvgPool2d(1)
        self.adaptMaxPool = nn.AdaptiveMaxPool2d((15, 20))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(2400 + filters[3], 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc_out_detect = nn.Linear(512, 8)
        self.fc_out_ncards = nn.Linear(512, self.n_bins)

        self.temperature = temperature

    def forward(self, x):
        x = self.conv1(x)
        x_conv1 = self.relu(x)
        x_conv1 = self.maxpool(x_conv1)

        x = self.conv2(x_conv1)
        x_conv2 = self.relu(x)
        x_conv2 = self.maxpool(x_conv2)

        x = self.conv3(x_conv2)
        x_conv3 = self.relu(x)
        x_conv3 = self.maxpool(x_conv3)

        x = self.conv4(x_conv3)
        x_conv4 = self.relu(x)

        # Predicting the mask of the cards pack (this is generating a 32 x 48 map)
        x_mask = self.upsample(x_conv4)
        x_mask = torch.cat([x_mask, x_conv2], dim=1)
        x_mask = self.up_conv1(x_mask)

        x_mask = self.upsample(x_mask)
        x_mask = torch.cat([x_mask, x_conv1], dim=1)
        x_mask = self.up_conv2(x_mask)

        x_mask = self.up_conv3(x_mask)
        x_mask = x_mask / self.temperature
        # x_mask = self.sigmoid(x_mask) # comment this depending on the loss you use?
        x_mask = self.softmax(x_mask.view(*x_mask.size()[:2], -1)).view_as(x_mask)

        # Predicting the coordinates of the points
        # combine encoder + detached decoder outputs... will it work?
        x_reg = torch.cat([self.flat(self.adaptAvgPool(x_conv4)), self.flat(self.adaptMaxPool(x_mask.detach()))], dim=1)
        x_reg = self.fc1(x_reg)
        x_reg = self.fc2(x_reg)

        # Point visibility (detected: yes/no)
        out_visibility = self.sigmoid(self.fc_out_detect(x_reg))

        # Number of cards: a probability distribution instead of a single scalar?)
        # Based on these sources:
        #   - https://indatalabs.com/blog/head-pose-estimation-with-cv
        #   - THIS paper: https://arxiv.org/pdf/1710.00925.pdf
        out_ncards_distrib_logits = self.fc_out_ncards(x_reg) / self.temperature

        return x_mask, out_visibility, out_ncards_distrib_logits
