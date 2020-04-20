import cv2
import os
import torchvision.transforms as transforms
from .preprocessing import MaculaDataset
from .utils import ToTensor

class DatasetAdjustment():
    def __init__(self):
        self.path = ''
        self.dataset = None 
 
    def setPath(self, file_dir, root_dir):
        self.path = _path
        self.dataset = MaculaDataset(csv_file = file_dir,root_dir = root_dir, transform = transforms.Compose([ToTensor()]))

    def getResult(self):
        return self.dataset
