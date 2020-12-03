import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from .tensor_util import ToTensor
from ..preprocessing import MaculaDataset
import torchvision.transforms as transforms
import math
from .image_util import VALID_IMAGE_FORMATS

def split_train_test(dataset):
    
    train = None
    test = None

    train_data = {'img_name': [], 'y_lower': [], 'x_lower': [], 'y_upper': [], 'x_upper': []}
    test_data = {'img_name': [], 'y_lower': [], 'x_lower': [], 'y_upper': [], 'x_upper': []}

    root_dir, csv_dir = dataset.getDirectory()
    csv_file = pd.read_csv(csv_dir, skipinitialspace = True)
    size = csv_file.count()[0]

    ok = '1' * math.ceil(0.8 * size)
    not_ok = '0' * math.ceil(0.2 * size)
    p = np.random.permutation([int(o) for o in ok] + [ int(n) for n in not_ok]).tolist()
    ok_size = len(ok) + len(not_ok)
    if(ok_size > size):
        diff = ok_size - size
        for idx in range(diff):
            p.pop()
    elif(ok_size < size):
        diff = size - ok_size
        for idx in range(diff):
            prob = np.random.random()
            if(prob < 0.5):
                p.append(1)
            else:
                p.append(0)
                        
    for idx in range(size):
        if(p[idx]):
            train_data['img_name'].append(csv_file.loc[idx, 'img_name'])
            train_data['y_lower'].append(csv_file.loc[idx, 'y_lower'])
            train_data['x_lower'].append(csv_file.loc[idx, 'x_lower'])
            train_data['y_upper'].append(csv_file.loc[idx, 'y_upper'])
            train_data['x_upper'].append(csv_file.loc[idx, 'x_upper'])
        else:
            test_data['img_name'].append(csv_file.loc[idx, 'img_name'])
            test_data['y_lower'].append(csv_file.loc[idx, 'y_lower'])
            test_data['x_lower'].append(csv_file.loc[idx, 'x_lower'])
            test_data['y_upper'].append(csv_file.loc[idx, 'y_upper'])
            test_data['x_upper'].append(csv_file.loc[idx, 'x_upper'])

    train_csv = pd.DataFrame(train_data)
    test_csv = pd.DataFrame(test_data)
    
    train = MaculaDataset(csv_file = train_csv,root_dir = root_dir, transform = transforms.Compose([ToTensor()]))
    test = MaculaDataset(csv_file = test_csv,root_dir = root_dir, transform = transforms.Compose([ToTensor()]))

    return train, test
