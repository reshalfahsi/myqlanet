import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import pandas as pd
from .tensor_util import ToTensor
from ..preprocessing import MaculaDataset
import torchvision.transforms as transforms 
import os
import random
import math

from .image_util import VALID_IMAGE_FORMATS

def split_train_test(dataset):
    train = None
    test = None

    root_dir, csv_dir = dataset.getDirectory()
 
    csv_file = pd.read_csv(csv_dir)

    train_csv = os.path.join(root_dir, "train.csv") 
    if(os.path.exists(train_csv)):
        os.remove(train_csv)
    csv_edit = open(train_csv, "w")
    csv_edit.write('img_name, y_lower, x_lower, y_upper, x_upper' + '\n')
    csv_edit.close()

    test_csv = os.path.join(root_dir, "test.csv") 
    if(os.path.exists(test_csv)):
        os.remove(test_csv)
    csv_edit = open(test_csv, "w")
    csv_edit.write('img_name, y_lower, x_lower, y_upper, x_upper' + '\n')
    csv_edit.close()
    
    image_list = []
    
    data_idx = 0

    for file in os.listdir(root_dir):
        if any(file.endswith(ext) for ext in VALID_IMAGE_FORMATS):
            bbox = csv_file.iloc[data_idx, 1:]
            bbox = np.array(bbox)
            bbox = bbox.astype('float').reshape(-1)
            temp = (os.path.join(root_dir, file), bbox) 
            image_list.append(temp)
            data_idx += 1

    total = int(math.ceil(0.8*(len(image_list))))
    train_image_list = []
    test_image_list = []
    while(len(image_list) > total):
        idx = random.randInt(0,(len(image_list)-1))
        train_image_list.append(image_list.pop(idx))
    test_image_list = image_list
    if(len(test_image_list)==0):
        test_image_list = train_image_list

    data_idx = 0
    for data in train_image_list:
       csv_edit = open(train_csv, "a")
       temp = str(str(data_idx) + ', ' + str(data[0]) + ', ' + str(data[1][0]) + ', ' + str(data[1][1]) + ', ' + str(data[1][2]) + ', ' + str(data[1][3]) + '\n')
       csv_edit.write(temp)
       csv_edit.close()
       data_idx += 1
   
    data_idx = 0
    for data in test_image_list:
       csv_edit = open(test_csv, "a")
       temp = str(str(data_idx) + ', ' + str(data[0]) + ', ' + str(data[1][0]) + ', ' + str(data[1][1]) + ', ' + str(data[1][2]) + ', ' + str(data[1][3]) + '\n')
       csv_edit.write(temp)
       csv_edit.close()
       data_idx += 1
    
    train = MaculaDataset(csv_file = train_csv,root_dir = root_dir, transform = transforms.Compose([ToTensor()]))
    test = MaculaDataset(csv_file = test_csv,root_dir = root_dir, transform = transforms.Compose([ToTensor()]))

    return train, test
