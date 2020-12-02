import cv2
from skimage import io
from skimage.transform import resize

VALID_IMAGE_FORMATS = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
VALID_IMAGE_SIZE = (2699, 1799)

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = float(interArea) / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou


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
            center = [height/2.0, width/2.0]
            startX = center[1]-float(self.crop_percentage/100.0)*center[1]
            startY = center[0]-float(self.crop_percentage/100.0)*center[0]
            lengthX = 2*float(self.crop_percentage/100.0)*center[1]
            lengthY = 2*float(self.crop_percentage/100.0)*center[0]
            self.crop_img = img[int(startY):int(startY+lengthY), int(startX):int(startX+lengthX)]

    def run(self,img):
        height = img.shape[0]
        width = img.shape[1]
        center = [height/2.0, width/2.0]
        startX = center[1]-float(self.crop_percentage/100.0)*center[1]
        startY = center[0]-float(self.crop_percentage/100.0)*center[0]
        lengthX = 2*float(self.crop_percentage/100.0)*center[1]
        lengthY = 2*float(self.crop_percentage/100.0)*center[0]
        self.crop_img = img[int(startY):int(startY+lengthY), int(startX):int(startX+lengthX)]
        return self.crop_img

    def setCropPercentage(self, percentage):
        self.crop_percentage = percentage

    def getResult(self):
        return self.crop_img

class ResizeImage():
    def __init__(self):
        self.result = None

    def process(self, img, dim): 
        ''' opencv format '''
        self.result = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    def run(self, img, dim): 
        ''' skimage format '''
        self.result = resize(img, dim)
        return self.result

    def getResult(self):
        return self.result
