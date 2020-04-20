import cv2
import numpy as np

class GGB():
    def __init__(self):
        self.path = ''
        self.img = None

    def setPath(self, path):
        self.path = path

    def process(self):
        self.img = cv2.imread(self.path)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        b_,g_,_ = cv2.split(img_output)
        self.img = cv2.merge((b_,g_,g_))
    
    def getResult(self):
        return self.img
