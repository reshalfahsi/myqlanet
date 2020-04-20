import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dsets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import make_grid
import shutil
import os.path
import time
import numpy as np
import pandas as pd
from skimage import io
import math
import argparse
import cv2

# Hyperparameter
batch_size = 1
num_output = 4
learning_rate = 1e-3
num_epochs = 1000
print_every = 5
best_loss = 9.9999999999e9 
start_epoch = 0
model = None
loss_fn = None
optimizer = None

# Data Utility
train_dataset = None
test_dataset = None
train_loader = None
test_loader = None

# Path to saved model weights(as hdf5)
resume_weights = "model/checkpoint.pth.tar"

# CUDA?
cuda = torch.cuda.is_available()

# Seed for reproducibility
torch.manual_seed(1)
if cuda:
    torch.cuda.manual_seed(1)

class MaculaDataset(Dataset):
    """Macula dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
        on a sample.
        """
        self.macula_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.macula_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.macula_frame.iloc[idx, 0])

        image = io.imread(img_name)
        bbox = self.macula_frame.iloc[idx, 1:]
        bbox = np.array(bbox)
        bbox = bbox.astype('float').reshape(-1)
        sample = {'image': image, 'bbox': bbox}
        if self.transform:
            sample = self.transform(sample)
        sample = (sample['image'], sample['bbox'])
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).float().to('cuda'),
        'bbox': torch.from_numpy(bbox).float().to('cuda')}

def train(model, optimizer, train_loader, loss_fn):
    global train_dataset
    """Perform a full training over dataset"""
    average_time = 0
    # Model train mode
    model.train()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        batch_time = time.time()
        images = Variable(images).float()
        target = Variable(target).float()
        if cuda:
            images, target = images.cuda(), target.cuda()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.float()
        #print('Output Size: '+str(outputs.size(0)))
        #print(target)
        #print('Target Size: '+str(target.size(0)))
        loss = loss_fn(outputs, target)
        # Load loss on CPU
        if cuda:
            loss.cpu()
        loss.backward()
        optimizer.step()
        # Measure elapsed time
        batch_time = time.time() - batch_time
        # Accumulate over batch
        average_time += batch_time
        # ### Keep track of metric every batch
        # Accuracy Metric
        #prediction = outputs.data.max(1)[1]   # first column has actual prob.
        #accuracy = prediction.eq(target.data).sum() / batch_size * 100
        # Log
        if (i + 1) % print_every == 0:
            print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Batch time: %f'
            % (epoch + 1,
            num_epochs,
            i + 1,
            len(train_dataset) // batch_size,
            loss.data,
            #accuracy,
            average_time/print_every))  # Average


def eval(model, optimizer, test_loader):
    """Eval over test set"""
    model.eval()
    ret_loss = 0
    # Get Batch
    for (data, target) in test_loader:
        loss = 0
        data, target = Variable(data), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        # Evaluate
        output = model(data)
        # Load output on CPU
        if cuda:
            output.cpu()
        # Compute Loss
        for d in range(0,num_output):
          loss += math.sqrt(abs(output[0][d]**2 - target[0][d]**2))
        loss /=  num_output
        ret_loss += loss
    return ret_loss


def save_checkpoint(state, is_best, filename='model/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation did not improve")

# #### Model ####
# Convolutional Neural Network Model
class CNN(nn.Module):
    """Conv[ReLU] -> Conv[ReLU] -> MaxPool -> Dropout(0.25)-
    -> Flatten -> FC()[ReLU] -> Dropout(0.5) -> FC()[Softmax]
    """
    def __init__(self, num_output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride = 2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride = 2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride = 2, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride = 2, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride = 2, padding=1)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, stride = 2, padding=1)
        self.conv7 = nn.Conv2d(16, 8, kernel_size=3, stride = 2, padding=1)
        self.drop1 = nn.Dropout2d(p=0.25)
        #self.fc1 = nn.Linear(704, 128)
        self.fc1 = nn.Linear(1120, 128)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, num_output)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        #x = x.view(-1, 704)
        x = x.view(-1, 1120)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return F.relu(x)

def initialize():
    global train_dataset, test_dataset, train_loader, test_loader, model, loss_fn, optimizer
    train_dataset = MaculaDataset(csv_file='data/train/macula.csv',root_dir='data/train/', transform = transforms.Compose([ToTensor()]))
    test_dataset = MaculaDataset(csv_file='data/test/macula.csv',root_dir='data/test/', transform = transforms.Compose([ToTensor()]))

    # Training Dataset Loader (Input Pipline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Testing Dataset Loader (Input Pipline)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = CNN(num_output)
    print(model)

    # If you are running a GPU instance, load the model on GPU
    if cuda:
        model.cuda()

    # #### Loss and Optimizer ####
    # Softmax is internally computed.
    loss_fn = nn.MSELoss()
    # If you are running a GPU instance, compute the loss on GPU
    if cuda:
        loss_fn.cuda()

    # Set parameters to be updated.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # If exists a best model, load its weights!
    if os.path.isfile(resume_weights):
        print("=> loading checkpoint '{}' ...".format(resume_weights))
        if cuda:
            checkpoint = torch.load(resume_weights)
        else:
            # Load GPU model on CPU
            checkpoint = torch.load(resume_weights, map_location=lambda storage, loc: storage)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))

def train():
    global best_loss, model, loss_fn, optimizer

    # Training the Model
    for epoch in range(num_epochs):
        train(model, optimizer, train_loader, loss_fn)
        loss = eval(model, optimizer, test_loader)
        print('=> Test set: Loss: {:.2f}'.format(loss))
        is_best = bool(loss < best_loss)
        if is_best:
          best_loss = loss
        # Save checkpoint if is a new best
        save_checkpoint({'epoch': start_epoch + epoch + 1, 'state_dict': model.state_dict(), 'best_loss': best_loss}, is_best)

def result():
    model.eval()
    result_path = 'result/'
    source_path = 'source/macula.csv'
    root_source = 'source/data/'
    macula_frame = pd.read_csv(source_path)
    with open(result_path + "macula.csv", "a") as text_file:
        text_file.write( 'img_name, y_lower, x_lower, y_upper, x_upper' + '\n')
    # Get Batch
    for idx in range(len(test_dataset)+len(train_dataset)):
        img_name = macula_frame.iloc[idx, 0]
        img_path = os.path.join(root_source, img_name)
        image = cv2.imread(img_path)
        image_tensor = image.transpose((2, 0, 1))
        if cuda: 
            image_tensor = torch.from_numpy(image_tensor).float().to('cuda')
        else:
            image_tensor = torch.from_numpy(image_tensor).float()
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = Variable(image_tensor)
        #print(image_tensor)
        output = model(image_tensor)
        start_point = (int(output[0][3]), int(output[0][2]))
        end_point = (int(output[0][1]), int(output[0][0]))
        with open(result_path + "macula.csv", "a") as text_file:
            text_file.write(str(idx)+ ',' + img_name + ',' + str(int(output[0][0])) + ',' + str(int(output[0][1])) + ',' + str(int(output[0][2])) + ',' + str(int(output[0][3])) + '\n')
        color = (0, 0, 255) 
        thickness = 4
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        bbox = macula_frame.iloc[idx, 1:]
        start_point = (int(bbox[3]), int(bbox[2]))
        end_point = (int(bbox[1]), int(bbox[0]))
        color = (0, 255, 0) 
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        img_result = os.path.join(result_path, img_name)
        cv2.imwrite(img_result, image)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    initialize()
    if args.train:
        train()
    else:
        result()
