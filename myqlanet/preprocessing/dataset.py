import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io
import numpy as np
import pandas as pd
import os
from .ggb import GGB
from ..utils import CropImage, ResizeImage, VALID_IMAGE_SIZE
from ..utils import ToTensor

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
        if(isinstance(csv_file, str)):
            self.macula_frame = pd.read_csv(csv_file, skipinitialspace = True)
        else:
            self.macula_frame = csv_file
        self.root_dir = root_dir
        if transform is None:
            self.transform = transforms.Compose([ToTensor()])
        else:
            self.transform = transform
        self.csv_file = csv_file
        self.crop = CropImage()
        self.resize = ResizeImage()
        self.ggb = GGB()

    def __len__(self):
        return len(self.macula_frame)

    def __getitem__(self, idx):

        # print("Constructing Dataset")

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.macula_frame.iloc[idx, 0])
        # print(self.macula_frame.iloc[idx, 0])

        image = io.imread(img_name)
        image = self.crop.run(image)
        image = self.resize.run(image, VALID_IMAGE_SIZE)
        image = self.ggb.run(image)

        bbox = self.macula_frame.iloc[idx, 1:]
        bbox = np.array(bbox)
        bbox = bbox.astype('float').reshape(-1)
        sample = {'image': image, 'bbox': bbox}
        if self.transform:
            sample = self.transform(sample)
        sample = (sample['image'], sample['bbox'])
        return sample

    def getDirectory(self):
        return self.root_dir, self.csv_file
