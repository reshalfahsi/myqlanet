import cv2
import numpy as np

class GGB():
    def __init__(self):
        self.path = ''
        self.img = None

    def setPath(self, path):
        self.path = path

    def process(self):
        ''' opencv format '''
        self.img = cv2.imread(self.path)
        b_, g_, _ = cv2.split(self.img)
        g_ = cv2.equalizeHist(g_)
        b_ = b_.astype('float32')
        g_ = g_.astype('float32')
        mean = np.mean(b_)
        b_ /= (mean + 1.0)
        mean = np.mean(g_)
        g_ /= (mean + 1.0)
        b_ = np.clip(b_, 0, 1)
        g_ = np.clip(g_, 0, 1)
        b_ *= 255.0
        g_ *= 255.0
        self.img = cv2.merge((b_,g_,g_))
    
    def getResult(self):
        return self.img

    def run(self, img):
        ''' skimage format '''
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
        b_, g_, _ = cv2.split(img)
        g_ = cv2.equalizeHist(g_)
        b_ = b_.astype('float32')
        g_ = g_.astype('float32')
        mean = np.mean(b_)
        b_ /= (mean + 1.0)
        mean = np.mean(g_)
        g_ /= (mean + 1.0)
        b_ = np.clip(b_, 0, 1)
        g_ = np.clip(g_, 0, 1)
        b_ *= 255.0
        g_ *= 255.0
        ret = cv2.merge((g_, g_, b_))
        return ret
