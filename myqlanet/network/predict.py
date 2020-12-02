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
    
    # Get Batch
    # if(os.path.isdir(path)):
    #     result = []
    #     for file in os.listdir(path):
    #         if any(file.endswith(ext) for ext in VALID_IMAGE_FORMATS):
    #             img_name = file
    #             img_path = os.path.join(path, img_name)
    #             image = cv2.imread(img_path)
    #             image = crop_img.run(image)
    #             image = resize_img.run(image,VALID_IMAGE_SIZE)
    #             image = ggb.run(image)
    #             image_tensor = image.transpose((2, 0, 1))
    #             if model.isCudaAvailable(): 
    #                 image_tensor = torch.from_numpy(image_tensor).float().to('cuda')
    #             else:
    #                 image_tensor = torch.from_numpy(image_tensor).float()
    #             image_tensor = image_tensor.unsqueeze(0)
    #             image_tensor = Variable(image_tensor)
    #             # print(image_tensor.shape)
    #             output = model(image_tensor)
    #             output = output.cpu().detach().numpy()
    #             result.append(output)
    # else:
    #     temp = os.path.splitext(path)
    #     extension = temp[1]
    #     if(extension in VALID_IMAGE_FORMATS):
    #         image = cv2.imread(path)
    #         image = crop_img.run(image)
    #         image = resize_img.run(image,VALID_IMAGE_SIZE)
    #         image = ggb.run(image)
    #         image_tensor = image.transpose((2, 0, 1))
    #         if model.isCudaAvailable(): 
    #             image_tensor = torch.from_numpy(image_tensor).float().to('cuda')
    #         else:
    #             image_tensor = torch.from_numpy(image_tensor).float()
    #         image_tensor = image_tensor.unsqueeze(0)
    #         image_tensor = Variable(image_tensor)
    #         # print(image_tensor.shape)
    #         output = model(image_tensor)
    #         output = output.cpu().detach().numpy()
    #         result.append(output)
    # return result[0]
    return result
