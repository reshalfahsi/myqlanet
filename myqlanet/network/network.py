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

    def __init__(self, legacy_model=True):
        super(MyQLaNet, self).__init__()

        self.__network_parameters = {}

        self.__network_parameters['is_cuda'] = torch.cuda.is_available()
        self.__network_parameters['device'] = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.__network_parameters['num_output'] = 4
        self.__network_parameters['legacy'] = legacy_model

        # [(W−K+2P)/S]+1, W -> input, K -> kernel_size, P -> padding, S -> stride

        if not self.__network_parameters['legacy']:

            # '''
            self.encoder_conv1 = self.inception_block(
                27, 81).to(self.__network_parameters['device'])
            self.skip_conv1 = nn.Conv2d(
                27, 81, kernel_size=1, stride=1, padding=0)
            self.encoder_conv1_continuous = self.inception_block(
                81, 81).to(self.__network_parameters['device'])

            self.encoder_conv2 = self.inception_block(
                81, 81).to(self.__network_parameters['device'])
            self.skip_conv2 = nn.Conv2d(
                81, 81, kernel_size=1, stride=1, padding=0)
            self.encoder_conv2_continuous = self.inception_block(
                81, 81).to(self.__network_parameters['device'])
            # '''

            # '''
            self.encoder_conv3 = self.inception_block(
                81, 243).to(self.__network_parameters['device'])
            self.skip_conv3 = nn.Conv2d(
                81, 243, kernel_size=1, stride=1, padding=0)
            self.encoder_conv3_continuous = self.inception_block(
                243, 243).to(self.__network_parameters['device'])
            # '''

            self.conv_block1 = self.conv_block(3, 9, 3, 2, 0).to(
                self.__network_parameters['device'])
            self.conv_block2 = self.conv_block(9, 9, 3, 2, 0).to(
                self.__network_parameters['device'])
            self.conv_block3 = self.conv_block(9, 27, 3, 2, 0).to(
                self.__network_parameters['device'])

            # self.conv_blocks = []
            # for channel in [(3, 9), (9, 9), (9, 27)]:
            #     self.conv_blocks.append(self.conv_block(channel[0], channel[1], 3, 2, 0).to(
            #         self.__network_parameters['device']))

            self.conv_blocks_continuous_1 = self.conv_block(
                243, 256, 3, 1, 1).to(self.__network_parameters['device'])
            self.conv_blocks_continuous_2 = self.conv_block(
                256, 256, 3, 1, 1).to(self.__network_parameters['device'])
            self.conv_blocks_continuous_3 = self.conv_block(
                256, 512, 3, 1, 1).to(self.__network_parameters['device'])

            self.max_pool = nn.MaxPool2d(2)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

            # self.conv_blocks_continuous = []
            # for channel in [(243, 256), (256, 256), (256, 512)]:
            #     self.conv_blocks_continuous.append(self.conv_block(channel[0], channel[1], 3, 1, 1).to(
            #         self.__network_parameters['device']))

            self.fc1 = nn.Linear(512, 128)
            self.fc2 = nn.Linear(128, self.__network_parameters['num_output'])

        else:

            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
            self.conv6 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
            self.conv7 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
            self.drop1 = nn.Dropout2d(p=0.25)
            self.fc1 = nn.Linear(704, 128)
            self.drop2 = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(128, self.__network_parameters['num_output'])

        self.loss_fn = nn.MSELoss()
        self.__network_parameters['loss_function'] = nn.MSELoss()

        if self.__network_parameters['is_cuda']:
            self.cuda()
            self.__network_parameters['loss_function'].cuda()

        self.__network_parameters['train_loader'] = None
        self.__network_parameters['test_loader'] = None

        self.__network_parameters['batch_size'] = 1
        # if self.__network_parameters['legacy'] else 1e-1
        self.__network_parameters['learning_rate'] = 1e-3

        adam = torch.optim.Adam(self.parameters(
        ), lr=self.__network_parameters['learning_rate'], weight_decay=0.0)
        sgd = torch.optim.SGD(self.parameters(
        ), lr=self.__network_parameters['learning_rate'], momentum=0.9, nesterov=True)

        # if self.__network_parameters['legacy'] else sgd
        self.__network_parameters['optimizer'] = adam

        # if self.__network_parameters['legacy']:
        #     self.__network_parameters['optimizer'] = torch.optim.Adam(
        #         self.parameters(), lr=self.__network_parameters['learning_rate'], weight_decay=0.0)
        # else:
        #     self.__network_parameters['optimizer'] = torch.optim.SGD(self.parameters(
        #     ), lr=self.__network_parameters['learning_rate'], momentum=0.9, nesterov=True)

        self.__network_parameters['best_loss'] = 9.9999999999e9
        self.__network_parameters['start_epoch'] = 0

        # if self.__network_parameters['legacy'] else 1000
        self.__network_parameters['num_epochs'] = 256

        self.__network_parameters['train_dataset'] = None
        self.__network_parameters['test_dataset'] = None

        self.__network_parameters['loss_now'] = 9e6
        self.__network_parameters['iou_now'] = 0
        self.__network_parameters['epoch_now'] = 0

    def forward(self, x):

        if not self.__network_parameters['legacy']:
            ############################################
            # for conv in self.conv_blocks:
            #     x = conv(x)

            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)

            x = self.max_pool(x)
            ############################################
            #                                          #
            ############################################
            # '''
            res1 = x
            out = [conv(x) for conv in self.encoder_conv1]
            out = torch.cat(out, 1)
            out = [conv(out) for conv in self.encoder_conv1_continuous]
            out = torch.cat(out, 1)
            res1 = self.skip_conv1(res1)
            x = res1 + out

            res2 = x
            out = [conv(x) for conv in self.encoder_conv1_continuous]
            out = torch.cat(out, 1)
            out = [conv(out) for conv in self.encoder_conv1_continuous]
            out = torch.cat(out, 1)
            res2 = res2 + res1
            x = res2 + out

            res3 = x
            out = [conv(x) for conv in self.encoder_conv2]
            out = torch.cat(out, 1)
            out = [conv(out) for conv in self.encoder_conv2_continuous]
            out = torch.cat(out, 1)
            res3 = self.skip_conv2(res3)
            res2 = self.skip_conv2(res2)
            res1 = self.skip_conv2(res1)
            res3 = res3 + res2
            res3 = res3 + res1
            x = res3 + out

            res4 = x
            out = [conv(x) for conv in self.encoder_conv2_continuous]
            out = torch.cat(out, 1)
            out = [conv(out) for conv in self.encoder_conv2_continuous]
            out = torch.cat(out, 1)
            res4 = res4 + res3
            res4 = res4 + res2
            res4 = res4 + res1
            x = res4 + out

            x = self.max_pool(x)
            # '''
            ############################################
            #                                          #
            ############################################
            res1 = x
            out = [conv(x) for conv in self.encoder_conv2_continuous]
            out = torch.cat(out, 1)
            out = [conv(out) for conv in self.encoder_conv2_continuous]
            out = torch.cat(out, 1)
            x = res1 + out

            res2 = x
            out = [conv(x) for conv in self.encoder_conv2_continuous]
            out = torch.cat(out, 1)
            out = [conv(out) for conv in self.encoder_conv2_continuous]
            out = torch.cat(out, 1)
            res2 = res2 + res1
            x = res2 + out

            res3 = x
            out = [conv(x) for conv in self.encoder_conv2_continuous]
            out = torch.cat(out, 1)
            out = [conv(out) for conv in self.encoder_conv2_continuous]
            out = torch.cat(out, 1)
            res3 = res3 + res1
            res3 = res3 + res2
            x = res3 + out

            x = self.max_pool(x)
            ############################################
            #                                          #
            ############################################
            # '''
            out = [conv(x) for conv in self.encoder_conv3]
            out = torch.cat(out, 1)
            out = [conv(out) for conv in self.encoder_conv3_continuous]
            out = torch.cat(out, 1)
            x = self.skip_conv3(x)
            x = x + out

            out = [conv(x) for conv in self.encoder_conv3_continuous]
            out = torch.cat(out, 1)
            out = [conv(out) for conv in self.encoder_conv3_continuous]
            out = torch.cat(out, 1)
            x = x + out
            # '''

            # for conv in self.conv_blocks_continuous:
            #     x = conv(x)

            x = self.conv_blocks_continuous_1(x)
            x = self.conv_blocks_continuous_2(x)
            x = self.conv_blocks_continuous_3(x)

            # x = nn.AdaptiveAvgPool2d((1, 1))(x)
            x = self.avg_pool(x)
            ############################################
            #                                          #
            ############################################
            x = x.view(-1, 512)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            ############################################

        else:

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

    def inception_block(self, in_channel, out_channel):
        # ret = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channel, out_channel//3, kernel_size=prop[0], stride=prop[1], padding=prop[2]), nn.BatchNorm2d(
        # out_channel//3), nn.ReLU()) for prop in [(1, 1, 0), (3, 1, 1), (5, 1, 2)]])
        ret = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channel, out_channel//3, kernel_size=prop[0],
                                                     stride=prop[1], padding=prop[2]), nn.ReLU()) for prop in [(1, 1, 0), (3, 1, 1), (5, 1, 2)]])
        return ret

    def conv_block(self, in_channel, out_channel, kernel_size=3, stride=2, padding=0):
        # ret = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
        #   stride=stride, padding=padding), nn.BatchNorm2d(out_channel), nn.ReLU())
        ret = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                                      stride=stride, padding=padding), nn.ReLU())
        return ret

    def get_network_parameters(self, key=''):
        if key == '':
            print('Failed to get network parameters')
            return None

        return self.__network_parameters[key]

    def set_network_parameters(self, key='', value=None):
        if key == '' or value is None:
            print("Please insert the key or the value")
            return None

        self.__network_parameters[key] = value

    def compile(self, dataset=None, batch_size=1, _loss_fn=None, _optimizer=None):

        # self.__network_parameters['batch_size'] = batch_size

        if _loss_fn is not None:
            self.__network_parameters['loss_function'] = _loss_fn
            if self.__network_parameters['is_cuda']:
                self.__network_parameters['loss_function'].cuda()

        if _optimizer is not None:
            self.__network_parameters['optimizer'] = _optimizer

        if dataset is None:
            print("Please insert a valid dataset format.")
            return

        train_dataset = None
        test_dataset = None

        train_dataset, test_dataset = dataset_util.split_train_test(dataset)

        if (len(test_dataset) == 0):
            print("Train Dataset size is 0!")
            test_dataset = train_dataset
        self.__network_parameters['epoch_now'] = 0

        # train_dataset
        self.__network_parameters['train_dataset'] = train_dataset
        # test_dataset
        self.__network_parameters['test_dataset'] = test_dataset
        self.__network_parameters['train_loader'] = torch.utils.data.DataLoader(
            dataset=self.__network_parameters['train_dataset'], batch_size=self.__network_parameters['batch_size'], shuffle=True)
        self.__network_parameters['test_loader'] = torch.utils.data.DataLoader(
            dataset=self.__network_parameters['test_dataset'], batch_size=self.__network_parameters['batch_size'], shuffle=False)

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