#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import torch
from torch.utils.data import Dataset
from PIL import ImageFile  # Python：IOError: image file is truncated 的解决办法
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(1)
torch.cuda.manual_seed(1)


class CUB200Bags(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self._train = train

        if self._train:
            self._bag_dir = 'bags/train'
        else:
            self._bag_dir = 'bags/val'
        self._files = [
            d for d in os.listdir(self._bag_dir) if (os.path.isfile(os.path.join(self._bag_dir, d))) and
                                                    (d.endswith('.pt'))
        ]
        if len(self._files) <= 0:
            raise AssertionError('Bags have not been generated!')
        self._files.sort()
    
    def __getitem__(self, index):
        bag, labels, paths = torch.load(os.path.join(self._bag_dir, self._files[index]))
        label = [max(labels), labels]
        path = list(paths)
        return bag, label, path

    def __len__(self):
        return len(self._files)
