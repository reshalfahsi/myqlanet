import torch
import pandas as pd
import os
import cv2
from torch.autograd import Variable
from ..utils import CropImage, ResizeImage, VALID_IMAGE_FORMATS, VALID_IMAGE_SIZE
from ..preprocessing import GGB
from skimage import io

def predict(model, path):

    model.eval()

    crop_img = CropImage()
    resize_img = ResizeImage()
    ggb = GGB()

    result = None
    image = io.imread(path)

    image = crop_img.run(image)
    image = resize_img.run(image, VALID_IMAGE_SIZE)
    image = ggb.run(image)
    image = image.astype('float32')
    # image /= 255.0
    image = (image - 127.5)/127.5
    image_tensor = image.transpose((2, 0, 1))
    if model.isCudaAvailable(): 
        image_tensor = torch.from_numpy(image_tensor).float().to('cuda')
    else:
        image_tensor = torch.from_numpy(image_tensor).float()
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = Variable(image_tensor)
    output = model(image_tensor)
    output = output.cpu().detach().numpy()
    result = output[0]
    
    return result
