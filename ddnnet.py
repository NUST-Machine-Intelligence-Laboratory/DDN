#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn.functional as F


class DDNNet(torch.nn.Module):
    """
    DDNNet
    The basis is VGG-16
    The input is (3, 224, 224)
    The structure of VGG-16 is as follows:
        conv1_1 (64) -> relu -> conv1_2 (64) -> relu -> pool1(64*112*112)
    ->  conv2_1(128) -> relu -> conv2_2(128) -> relu -> pool2(128*56*56)
    ->  conv3_1(256) -> relu -> conv3_2(256) -> relu -> conv3_3(256) -> relu -> pool3(256*28*28)
    ->  conv4_1(512) -> relu -> conv4_2(512) -> relu -> conv4_3(512) -> relu -> pool4(512*14*14)
    ->  conv5_1(512) -> relu -> conv5_2(512) -> relu -> conv5_3(512) -> relu -> pool5(512*7*7)
    ->  fc(L)        -> relu
    ->  fc(D)        -> tanh -> fc(K)
    ->  fc(1)        -> sigmoid
    """
    def __init__(self, pretrained=True, gated=False):
        super().__init__()
        self._pretrained = pretrained
        self._gated = gated
        self.L = 4096
        self.D = 1024
        self.K = 1

        vgg16 = torchvision.models.vgg16(pretrained=self._pretrained)
        # vgg19 = torchvision.models.vgg19(pretrained=self._pretrained)
        # resnet50 = torchvision.models.resnet50(pretrained=self._pretrained)
        self.feature_extractor_part1 = vgg16.features  # in 3*224*224, out 512*7*7
        # self.feature_extractor_part1 = vgg19.features  # in 3*224*224, out 512*7*7
        # self.feature_extractor_part1 = torch.nn.Sequential(*list(resnet50.children())[:-2])  # in 3*224*224, out 2048*7*7
        self.feature_extractor_part2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=512*7*7, out_features=self.L),
            torch.nn.ReLU(),
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.L, out_features=self.D),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=self.D, out_features=self.K),
        )
        self.attention_gated_part1_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.L, out_features=self.D),
            torch.nn.Tanh(),
        )
        self.attention_gated_part1_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.L, out_features=self.D),
            torch.nn.Sigmoid(),
        )
        self.attention_gated_part2 = torch.nn.Linear(in_features=self.D, out_features=self.K)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.L * self.K, out_features=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 512*7*7)
        H = self.feature_extractor_part2(H)        # N * L

        if not self._gated:
            A = self.attention(H)                  # N * K
        else:
            V = self.attention_gated_part1_1(H)    # N * D
            U = self.attention_gated_part1_2(H)    # N * D
            A = self.attention_gated_part2(V * U)  # N * K
        A = torch.transpose(A, 1, 0)               # K * N
        A = F.softmax(A, dim=1)                    # softmax over N
        M = torch.mm(A, H)                         # K * L

        A = torch.transpose(A, 1, 0)               # N * K
        A = 2 * torch.sigmoid(5 * A) - 1           # N * K
        # A = 1/math.pi * torch.atan(A - 0.3) + 0.5
        # A = torch.sqrt(A)
        A = torch.transpose(A, 1, 0)

        Y_prob = self.classifier(M)                # shape : 1
        Y_hat = torch.ge(Y_prob, 0.5).float()      # shape : 1

        return Y_prob, Y_hat, A
