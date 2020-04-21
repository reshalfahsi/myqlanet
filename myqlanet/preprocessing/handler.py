import os
from ..utils import CropImage, ResizeImage

class DirectoryHandler():
    def __init__(self):
        self.fixed_size = (1799, 2699)
        self.images = []
        self.crop = CropImage()
        self.path = ''
        self.resize = ResizeImage()
        self.valid_image_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
        self.names = []

    def setPath(self,path):
        self.path = path

    def process(self):
        for file in os.listdir(self.path):
            if any(file.endswith(ext) for ext in self.valid_image_extensions):
                self.names.append(file)
                name = os.path.join(self.path, file)
                self.crop.setFilePath(name)
                self.crop.process()
                res = self.crop.getResult()
                self.resize.process(res, self.fixed_size)
                res = self.resize.getResult()
                self.images.append(res)
    
    def getResult(self):
        return self.images

    def getNames(self):
        return self.names

class FileHandler():
    def __init__(self):
        self.fixed_size = (1799, 2699)
        self.image = None
        self.crop = CropImage()
        self.path = ''
        self.resize = ResizeImage()
        self.valid_image_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']

    def setPath(self,path):
        self.path = path

    def process(self):
        temp = os.path.splitext(self.path)
        extension = temp[1]
        if(extension in self.valid_image_extensions):
            self.crop.setFilePath(self.path)
            self.crop.process()
            res = self.crop.getResult()
            self.resize.process(res, self.fixed_size)
            res = self.resize.getResult()
            self.image = res
            self.names = os.path.basename(self.path)
        else:
            print('File is not a Valid Image!')
    
    def getResult(self):
        return self.image

    def getNames(self):
        return self.names

