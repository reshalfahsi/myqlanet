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

    def __init__(self, legacy_model = True):
        super(MyQLaNet, self).__init__()

        self.iscuda = torch.cuda.is_available()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.num_output = 4

        self.legacy = legacy_model

        # [(W−K+2P)/S]+1, W -> input, K -> kernel_size, P -> padding, S -> stride

        if not self.legacy:
            self.encoder_conv = nn.ModuleList([nn.Sequential(nn.Conv2d(3, 1, kernel_size=prop[0], stride=prop[1], padding=prop[2]), nn.BatchNorm2d(1), nn.ReLU()) for prop in [(1, 1, 0), (3, 1, 1), (5, 1, 2)]])
            self.conv_blocks = []
            for channel in [(3, 27), (27, 81), (81, 27), (27, 3)]:
                self.conv_blocks.append(self.conv_block(channel[0], channel[1]).to(self.device))
            self.drop = nn.Dropout(p=0.5)
            self.fc1 = nn.Linear(72, 16)
            self.fc2 = nn.Linear(16, self.num_output)
        else:        
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride = 2, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride = 2, padding=1)
            self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride = 2, padding=1)
            self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride = 2, padding=1)
            self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride = 2, padding=1)
            self.conv6 = nn.Conv2d(32, 16, kernel_size=3, stride = 2, padding=1)
            self.conv7 = nn.Conv2d(16, 8, kernel_size=3, stride = 2, padding=1)
            self.drop1 = nn.Dropout2d(p=0.25)
            self.fc1 = nn.Linear(704, 128)
            self.drop2 = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(128, self.num_output)

        self.loss_fn = nn.MSELoss()

        if self.iscuda:
            self.cuda()
            self.loss_fn.cuda()

        self.train_loader = None
        self.test_loader = None

        self.batch_size = 1
        self.learning_rate = 1e-3
        self.optim = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=0.0)

        self.best_loss = 9.9999999999e9
        self.start_epoch = 0

        self.num_epochs = 256

        self.train_dataset = None
        self.test_dataset = None

        self.loss_now = 9e6
        self.iou_now = 0
        self.epoch_now = 0

    def forward(self, x):

        if not self.legacy:
            for conv in self.conv_blocks:
                x = conv(x)
            out = [conv(x) for conv in self.encoder_conv]
            out = torch.cat(out, 1)

            x = x + out
            x = F.max_pool2d(x, 2)

            for conv in self.conv_blocks:
                x = conv(x)

            out = [conv(x) for conv in self.encoder_conv]
            out = torch.cat(out, 1)

            x = x + out

            x = x.view(-1, 72)
            x = F.relu(self.fc1(x))
            x = self.drop(x)
            x = F.relu(self.fc2(x))
        else :
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = self.drop1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = x.view(-1, 704)
            x = F.relu(self.fc1(x))
            x = self.drop2(x)
            x = F.relu(self.fc2(x))
        
        return x

    def conv_block(self, in_channel, out_channel):
        ret = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                     stride=2, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU())
        return ret

    def optimizer(self):
        return self.optim

    def set_saved_training_parameters(self, _start_epoch, _best_loss):
        self.start_epoch = _start_epoch
        self.best_loss = _best_loss

    def max_epoch(self):
        return self.num_epochs

    def loss(self):
        return self.loss_fn

    def isCudaAvailable(self):
        return self.iscuda
    
    def getNumEpochs(self):
        return self.num_epochs
    
    def set_training_progress_params(self, loss, iou, epochs):
        self.loss_now = loss
        self.iou_now = iou
        self.epoch_now = epochs

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

    def compile(self, dataset=None, batch_size=1, _loss_fn=None, _optimizer=None):

        self.batch_size = batch_size

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

        if (len(test_dataset) == 0):
            print("Train Dataset size is 0!")
            test_dataset = train_dataset
        self.epoch_now = 0

        self.train_dataset = train_dataset  # train_dataset
        self.test_dataset = test_dataset  # test_dataset
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def fit(self, path=''):
        
        if path == '':
            print("Please Insert Path!")
            return False

        success = train.process(self, path)

        return success

    def predict(self, weight_path='', path=''):
        if path == '' and weight_path == '':
            print("Please Insert Path!")
            return None
        
        ret = predict.predict(self, weight_path, path)
        return ret
