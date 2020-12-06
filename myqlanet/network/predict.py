import torch
import pandas as pd
import os
import cv2
from torch.autograd import Variable
from ..utils import CropImage, ResizeImage, VALID_IMAGE_FORMATS, VALID_IMAGE_SIZE
from ..preprocessing import GGB
from skimage import io


def predict(model, weight_path, path):

    if os.path.isfile(weight_path):
        try:
            print("=> loading checkpoint '{}' ...".format(weight_path))
            if model.isCudaAvailable():
                checkpoint = torch.load(weight_path)
            else:
                # Load GPU model on CPU
                checkpoint = torch.load(weight_path, map_location=lambda storage, loc: storage)

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (trained for {} epochs)".format(weight_path, checkpoint['epoch']))
        except:
            print("Please Train your Network First!")
            return     
    else:
        print("Please Train your Network First!")
        return None

    model.eval()

    crop_img = CropImage()
    resize_img = ResizeImage()
    ggb = GGB()

    result = None
    image = io.imread(path)

    image = crop_img.run(image)
    image = resize_img.run(image, VALID_IMAGE_SIZE)
    image = ggb.run(image)
    # image = image.astype('float32')
    # image /= 255.0
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
