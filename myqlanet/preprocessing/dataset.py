import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import pandas as pd

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
        self.csv_file = csv_file

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

    def getDirectory(self):
        return self.root_dir, self.csv_file
