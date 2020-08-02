#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torchvision
import argparse
import time
from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
from folder import ImageFolder
from PIL import ImageFile  # Python：IOError: image file is truncated 的解决办法
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def create_bags(config, train=True):
    r = np.random.RandomState(config.seed)

    # data transform
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=224),
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # dataset
    if train:
        dataset = ImageFolder(os.path.join(config.data_base, 'train'), transform=train_transform)
        num_in_train = len(dataset.imgs)
        # loader = DataLoader(dataset, batch_size=num_in_train, shuffle=False)
        loader = DataLoader(dataset, batch_size=1000, shuffle=False)
        num_bag = config.num_bag_train
    else:
        dataset = ImageFolder(os.path.join(config.data_base, 'val'), transform=test_transform)
        num_in_test = len(dataset.imgs)
        # loader = DataLoader(dataset, batch_size=num_in_test, shuffle=False)
        loader = DataLoader(dataset, batch_size=1000, shuffle=False)
        num_bag = config.num_bag_test

    all_images_list, all_labels_list, all_paths_list = [], [], []
    for batch_data, batch_labels, batch_paths in loader:
        all_images_list.append(batch_data)
        all_labels_list.append(batch_labels)
        all_paths_list.append(np.array(batch_paths))
    all_images = torch.cat(all_images_list)
    all_labels = torch.cat(all_labels_list)
    all_paths = np.concatenate(all_paths_list)

    # print('--->', all_images.shape, '\n--->', all_labels.shape, '\n--->', all_paths.shape)

    for i in range(num_bag):
        bag_length = np.int(r.normal(config.mean_bag_length, config.var_bag_length, 1))
        if bag_length < 1:
            bag_length = 1

        if train:
            indices = r.randint(0, num_in_train, bag_length)
            bag_set = (all_images[indices], all_labels[indices], all_paths[indices])
            torch.save(bag_set, os.path.join('bags/train', '{}.pt'.format(i)))
        else:
            indices = r.randint(0, num_in_test, bag_length)
            bag_set = (all_images[indices], all_labels[indices], all_paths[indices])
            torch.save(bag_set, os.path.join('bags/val', '{}.pt'.format(i)))

        # print('Bag {} Generated!'.format(i))


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description='Generate bags')
    parser.add_argument('--data_base', dest='data_base', type=str, required=True,
                        help='Data Base Folder Name')
    parser.add_argument('--mean_bag_length', dest='mean_bag_length', type=int,
                        default=10, metavar='ML', help='Average bag length')
    parser.add_argument('--var_bag_length', dest='var_bag_length', type=int,
                        default=2, metavar='VL', help='variance of bag length')
    parser.add_argument('--num_bag_train', dest='num_bag_train', type=int,
                        default=200, metavar='NTrain', help='Number of bags in training set')
    parser.add_argument('--num_bag_test', dest='num_bag_test', type=int,
                        default=50, metavar='NTest', help='Number of bags in test set')
    parser.add_argument('--seed', dest='seed', type=int,
                        default=1, metavar='S', help='Random seed (default:1)')
    args = parser.parse_args()

    os.popen('mkdir -p bags/train bags/val')

    create_bags(args, train=True)
    create_bags(args, train=False)

    end = time.time()
    print('Bags have been generated and saved into bags/')
    print('------ Total Runtime {} ------'.format(end - start))
