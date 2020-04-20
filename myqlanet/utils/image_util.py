import cv2

class CropImage():
    def __init__(self):
        self.crop_percentage = 77
        self.file_path = ''
        self.crop_img = None

    def setFilePath(self, file_path):
        self.file_path = file_path

    def process(self):
        if(self.file_path != ''):
            img = cv2.imread(self.file_path)
            height = img.shape[0]
            width = img.shape[1]
            center = [height/2.0, width/2.0];
            startX = center[1]-float(self.crop_percentage/100.0)*center[1];
            startY = center[0]-float(self.crop_percentage/100.0)*center[0];
            lengthX = 2*float(self.crop_percentage/100.0)*center[1];
            lengthY = 2*float(self.crop_percentage/100.0)*center[0];
            self.crop_img = img[int(startY):int(startY+lengthY), int(startX):int(startX+lengthX)]

    def setCropPercentage(self, percentage):
        self.crop_percentage = percentage

    def getResult(self):
        return self.crop_img

class ResizeImage():
    def __init__(self):
        self.result = None

    def process(self, img, dim):
        self.result = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    def getResult(self):
        return self.result
