import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from . import train_engine
from . import predict_engine
import os
from ..preprocessing import MaculaDataset

class MyQLaNet(nn.Module):
    """
    Deep Learning Model for MyQLaNet
    """
    def __init__(self, num_output):
        super(MyQLaNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride = 2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride = 2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride = 2, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride = 2, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride = 2, padding=1)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, stride = 2, padding=1)
        self.conv7 = nn.Conv2d(16, 8, kernel_size=3, stride = 2, padding=1)
        self.drop1 = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(704, 128)
        #self.fc1 = nn.Linear(1120, 128)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, num_output)
        self.iscuda = torch.cuda.is_available()
        if self.iscuda:
            self.cuda()
        self.train_loader = None
        self.test_loader = None
        self.loss_fn = nn.MSELoss()
        if self.iscuda:
            self.loss_fn.cuda()
        self.batch_size = 1
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #self.weight_path = ''
        self.best_loss = 9.9999999999e9
        self.num_epochs = 1000
        self.start_epoch = 0
        self.train_dataset = None
        self.test_dataset = None
        self.num_output = num_output
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(-1, 704)
        #x = x.view(-1, 1120)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return F.relu(x)

    def compile(self,train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) 


    def fit(self,path = ''):
        if path == '':
           print("Please Insert Path!")
           return None
        if os.path.isfile(path):
            print("=> loading checkpoint '{}' ...".format(path))
            if cuda:
                checkpoint = torch.load(path)
            else:
                # Load GPU model on CPU
                checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
            self.start_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (trained for {} epochs)".format(path, checkpoint['epoch']))
        for epoch in range(num_epochs):
            train_engine.train(self, self.train_dataset, self.optimizer, self.train_loader, self.test_loader, self.loss_fn, self.iscuda, self.batch_size, epoch, self.start_epoch, self.num_output, path, self.best_loss) 

    def predict(self, weight_path = '', path = ''):
        if path == '' and weight_path == '':
           print("Please Insert Path!")
           return None
        if os.path.isfile(weight_path):
            print("=> loading checkpoint '{}' ...".format(weight_path))
            if cuda:
                checkpoint = torch.load(weight_path)
            else:
                # Load GPU model on CPU
                checkpoint = torch.load(weight_path, map_location=lambda storage, loc: storage)
            self.start_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (trained for {} epochs)".format(weight_path, checkpoint['epoch']))
        else:
            print("Please Train your Network First!")
            return None
        ret = predict(self, path, self.iscuda)
        return ret
