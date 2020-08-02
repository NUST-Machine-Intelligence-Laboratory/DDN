#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torchvision
from torch.utils.data import DataLoader
from bcnn import BCNN
from PIL import ImageFile # Python：IOError: image file is truncated 的解决办法
ImageFile.LOAD_TRUNCATED_IMAGES = True

import time

torch.manual_seed(0)
torch.cuda.manual_seed(0)


class Manager(object):
    def __init__(self, options):
        """
        Prepare the network, criterion, Optimizer and data
        Arguments:
            options [dict]  Hyperparameter
            path    [dict]  path of the dataset and model
        """
        print('------------------------------------------------------------------------------')
        print('Preparing the network and data ... ')
        self._options = options
        self._path = options['path']
        self._data_base = options['data_base']
        # Step
        self._step = options['step']
        print('This training is training over', 'fc-only' if self._step == 1 else 'all layers')
        # Network

        if self._step == 1:
            net = BCNN(pretrained=True)
        else:
            net = BCNN(pretrained=False)

        if torch.cuda.device_count() >= 1:
            self._net = torch.nn.DataParallel(net).cuda()
            print('cuda device : ', torch.cuda.device_count())
        # elif torch.cuda.device_count() == 1:
        #     self._net = net.cuda()
        else:
            raise EnvironmentError('This is designed to run on GPU but no GPU is found')

        if self._step == 2:
            self._net.load_state_dict(torch.load(os.path.join(self._path, 'step1_vgg_16_epoch_best.pth')))
        # print(self._net)
        # Criterion
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Optimizer
        if self._step == 1:
            params_to_optimize = self._net.module.fc.parameters()
        else:
            params_to_optimize = self._net.parameters()
        self._optimizer = torch.optim.SGD(params_to_optimize, lr=self._options['base_lr'],
                                          momentum=0.9, weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='max', factor=0.1,
                                                                     patience=3, verbose=True, threshold=1e-4)

        train_transform = torchvision.transforms.Compose([
            # torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        test_transform = torchvision.transforms.Compose([
            # torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        # Load data
        data_dir = self._data_base
        train_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        test_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)
        self._train_loader = DataLoader(train_data, batch_size=self._options['batch_size'],
                                        shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = DataLoader(test_data, batch_size=16,
                                       shuffle=False, num_workers=4, pin_memory=True)

    def train(self):
        """
        Train the network
        """
        print('Training ... ')
        best_accuracy = 0.0
        best_epoch = None
        print('Epoch\tTrain Loss\tTrain Accuracy\tTest Accuracy\tEpoch Runtime')
        for t in range(self._options['epochs']):
            epoch_start = time.time()

            epoch_loss = []
            num_correct = 0
            num_total = 0
            for X, y in self._train_loader:
                # Enable training mode
                self._net.train(True)
                # Data
                X = X.cuda()
                y = y.cuda(async=True)
                # Clear the existing gradients
                self._optimizer.zero_grad()
                # Forward pass
                score = self._net(X)  # score is in shape (N, 200)
                # pytorch only takes label as [0, num_classes) to calculate loss
                loss = self._criterion(score, y)
                epoch_loss.append(loss.item())
                # Prediction
                _, prediction = torch.max(score.data, 1)
                # prediction is the index location of the maximum value found,
                num_total += y.size(0)  # y.size(0) is the batch size
                num_correct += torch.sum(prediction == y.data).item()
                # Backward
                loss.backward()
                self._optimizer.step()
            # Record the train accuracy of each epoch
            train_accuracy = 100 * num_correct / num_total
            test_accuracy = self.test(self._test_loader)
            self._scheduler.step(test_accuracy)  # the scheduler adjust lr based on test_accuracy

            epoch_end = time.time()

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = t + 1  # t starts from 0
                print('*', end='')
                # Save mode onto disk for the usage of the second step of fine-tune over all layer
                torch.save(self._net.state_dict(), os.path.join(self._path, 'step{}_vgg_16_epoch_best.pth'.format(self._step)))
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f' % (t + 1, sum(epoch_loss) / len(epoch_loss),
                                                            train_accuracy, test_accuracy,
                                                            epoch_end - epoch_start))

        print('-----------------------------------------------------------------')
        print('Best at epoch %d, test accuracy %f' % (best_epoch, best_accuracy))
        print('-----------------------------------------------------------------')

    def test(self, dataloader):
        """
        Compute the test accuracy

        Argument:
            dataloader  Test dataloader
        Return:
            Test accuracy in percentage
        """
        self._net.train(False) # set the mode to evaluation phase
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for X, y in dataloader:
                # Data
                X = X.cuda()
                y = y.cuda(async=True)
                # Prediction
                score = self._net(X)
                _, prediction = torch.max(score, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data).item()
        self._net.train(True)  # set the mode to training phase
        return 100 * num_correct / num_total
