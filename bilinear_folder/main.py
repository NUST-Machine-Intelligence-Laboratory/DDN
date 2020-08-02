#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import time
import torch
import torchvision
import config1 as cf1
import config2 as cf2
from manager import Manager

from PIL import ImageFile # Python：IOError: image file is truncated 的解决办法
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(path):

    parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
    parser.add_argument('--step', dest='step', type=int, required=True,
                        help='Step 1 is training fc only; step 2 is training the entire network')
    args = parser.parse_args()
    if args.step == 1:
        options = {
            'base_lr': cf1.base_lr,
            'batch_size': cf1.batch_size,
            'epochs': cf1.epochs,
            'weight_decay': cf1.weight_decay,
            'step': args.step,
            'path': path,
            'data_base': cf1.data_base
        }
    else:
        options = {
            'base_lr': cf2.base_lr,
            'batch_size': cf2.batch_size,
            'epochs': cf2.epochs,
            'weight_decay': cf2.weight_decay,
            'step': args.step,
            'path': path,
            'data_base': cf2.data_base
        }

    manager = Manager(options)
    manager.train()


if __name__ == '__main__':

    print(os.path.join(os.popen('pwd').read().strip(), 'model'))

    if not os.path.isdir(os.path.join(os.popen('pwd').read().strip(), 'model')):
        print('>>>>>> Creating directory \'model\' ... ')
        os.mkdir(os.path.join(os.popen('pwd').read().strip(), 'model'))

    path = os.path.join(os.popen('pwd').read().strip(), 'model')

    main(path)
