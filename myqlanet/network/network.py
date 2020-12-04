import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from . import train
from . import predict
import os
from ..preprocessing import MaculaDataset
from ..utils import dataset_util
from ..utils import *


class MyQLaNet(nn.Module):
    """
    Deep Learning Model for MyQLaNet
    """

    def __init__(self):
        super(MyQLaNet, self).__init__()

        self.iscuda = torch.cuda.is_available()

        self.num_output = 4

        ## [(W−K+2P)/S]+1, W -> input, K -> kernel_size, P -> padding, S -> stride
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        
        self.drop1 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, self.num_output)

        self.loss_fn = nn.MSELoss()

        if self.iscuda:
            self.cuda()
            self.loss_fn.cuda()

        self.train_loader = None
        self.test_loader = None

        self.batch_size = 1
        self.learning_rate = 4e-4
        self.optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0)

        self.best_loss = 9.9999999999e9
        self.start_epoch = 0

        self.num_epochs = 100

        self.train_dataset = None
        self.test_dataset = None

        self.loss_now = 9e6
        self.iou_now = 0
        self.epoch_now = 0

    def forward(self, x):        
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)

        for _ in range(2):
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = F.max_pool2d(x, 2, stride=1)

        x = x.view(-1, 192)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        return x

    def optimizer(self):
        return self.optim

    def max_epoch(self):
        return self.num_epochs

    def loss(self):
        return self.loss_fn

    def isCudaAvailable(self):
        return self.iscuda

    def train_utility_dataset(self):
        return self.train_dataset, self.train_loader, self.test_loader

    def train_utility_parameters(self):
        return self.batch_size, self.start_epoch, self.num_output, self.best_loss

    def update_best_loss(self, loss):
        self.best_loss = loss

    def update_loss(self):
        return self.epoch_now, self.loss_now

    def update_iou(self):
        return self.epoch_now, self.iou_now

    def compile(self, dataset=None, _loss_fn=None, _optimizer=None):

        if _loss_fn is not None:
            self.loss_fn = _loss_fn
            if self.iscuda:
                self.loss_fn.cuda()

        if _optimizer is not None:
            self.optim = _optimizer

        if dataset is None:
            print("Please insert a valid dataset format.")
            return

        train_dataset = None
        test_dataset = None

        train_dataset, test_dataset = dataset_util.split_train_test(dataset)

        if (len(test_dataset) == 0) :
            print("Train Dataset size is 0!")
            test_dataset = train_dataset
        self.epoch_now = 0

        self.train_dataset = train_dataset # train_dataset
        self.test_dataset = test_dataset  # test_dataset
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def fit(self, path=''):
        success = False
        if path == '':
            print("Please Insert Path!")
            return success
        if os.path.isfile(path):
            try:
                print("=> loading checkpoint '{}' ...".format(path))
                if self.iscuda:
                    checkpoint = torch.load(path)
                else:
                    # Load GPU model on CPU
                    checkpoint = torch.load(
                        path, map_location=lambda storage, loc: storage)
                self.start_epoch = checkpoint['epoch']
                self.best_loss = checkpoint['best_loss']
                self.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (trained for {} epochs)".format(
                    path, checkpoint['epoch']))
            except:
                print("Training Failed!")
                return success
        for epoch in range(self.num_epochs):
            self.loss_now, self.iou_now = train.train(self, epoch, path)
            self.epoch_now = epoch
            success = True
        return success

    def predict(self, weight_path='', path=''):
        if path == '' and weight_path == '':
            print("Please Insert Path!")
            return None
        if os.path.isfile(weight_path):
            try:
                print("=> loading checkpoint '{}' ...".format(weight_path))
                if self.iscuda:
                    checkpoint = torch.load(weight_path)
                else:
                    # Load GPU model on CPU
                    checkpoint = torch.load(
                        weight_path, map_location=lambda storage, loc: storage)
                self.start_epoch = checkpoint['epoch']
                self.best_loss = checkpoint['best_loss']
                self.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (trained for {} epochs)".format(
                    weight_path, checkpoint['epoch']))
            except:
                print("Please Train your Network First!")
                return None
        else:
            print("Please Train your Network First!")
            return None
        ret = predict.predict(self, path)
        return ret
