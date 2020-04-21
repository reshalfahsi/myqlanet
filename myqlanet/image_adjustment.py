import cv2
import os

from .preprocessing import FileHandler, DirectoryHandler

class ImageAdjustment():
    def __init__(self):
        self.path = ''
        self.isDirectory = False
        self.handler = None
        self.images_names = [] 
 
    def setPath(self, _path):
        self.path = _path
        if(os.path.isdir(self.path)):
            self.isDirectory = True
            self.handler = DirectoryHandler()
        else:
            self.handler = FileHandler()

        self.handler.setPath(self.path)

    def getResult(self):
        self.handler.process()
        self.images_names = self.handler.getNames()
        ret = self.handler.getResult()
        return ret

    def getNames(self):
        return self.images_names
