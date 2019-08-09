# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from scipy import misc
import torch.utils.data as data
from torchvision import datasets, transforms

data_root = "../data/SlidePatches/"
num_class = 3

train_dir = os.path.join(data_root, 'Train')
val_dir = os.path.join(data_root, 'Val')

def find_ext_files(dir_name, ext):
    assert os.path.isdir(dir_name), "{} is not a valid directory".format(dir_name)

    file_list = []
    for root, _, files in os.walk(dir_name):
        for cur_file in files:
            if cur_file.endswith(ext):
                 file_list.append(os.path.join(root, cur_file))

    return file_list

def get_mean_and_std(img_dir, suffix):
    mean, std = np.zeros(3), np.zeros(3)
    filelist = find_ext_files(img_dir, suffix)

    for idx, filepath in enumerate(filelist):
        cur_img = misc.imread(filepath) / 255.0
        for i in range(3):
            mean[i] += cur_img[:,:,i].mean()
            std[i] += cur_img[:,:,i].std()
    mean = [ele * 1.0 / len(filelist) for ele in mean]
    std = [ele * 1.0 / len(filelist) for ele in std]
    return mean, std

# rgb_mean, rgb_std = get_mean_and_std(train_dir, suffix=".png")
# print("mean rgb: {}".format(rgb_mean))
# print("std rgb: {}".format(rgb_std))
rgb_mean, rgb_std = (0.790, 0.609, 0.806), (0.169, 0.189, 0.108)
batch_size = 32

def train_loader():
    kwargs = {"num_workers": 4, "pin_memory": True}

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)])
        )

    loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return loader


def val_loader():
    kwargs = {"num_workers": 4, "pin_memory": True}

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)])
        )

    loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return loader


class PatchDataset(data.Dataset):
    """
    Dataset for thyroid slide testing. Thyroid slides would be splitted into multiple patches.
    Prediction is made on these splitted patches.
    """

    def __init__(self, slide_patches):
        self.patches = slide_patches
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)])

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        sample = self.patches[idx,...]
        if self.transform:
            sample = self.transform(sample)

        return sample
