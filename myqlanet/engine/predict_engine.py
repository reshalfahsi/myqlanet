import torch
import pandas as pd
import os
import cv2
from torch.autograd import Variable

valid_image_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']

def predict(model, path, cuda):
    global valid_image_extension

    model.eval()

    result = None
    # Get Batch
    if(os.path.isdir(path)):
        result = []
        for file in os.listdir(path):
            if any(file.endswith(ext) for ext in valid_image_extensions):
                img_name = file
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path)
                image_tensor = image.transpose((2, 0, 1))
                if cuda: 
                    image_tensor = torch.from_numpy(image_tensor).float().to('cuda')
                else:
                    image_tensor = torch.from_numpy(image_tensor).float()
                image_tensor = image_tensor.unsqueeze(0)
                image_tensor = Variable(image_tensor)
                output = model(image_tensor)
                result.append(output)
    else:
        temp = os.path.splitext(self.filenames)
        extension = temp[1]
        if(extension in self.valid_image_extensions):
            image = cv2.imread(path)
            image_tensor = image.transpose((2, 0, 1))
            if cuda: 
                image_tensor = torch.from_numpy(image_tensor).float().to('cuda')
            else:
                image_tensor = torch.from_numpy(image_tensor).float()
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = Variable(image_tensor)
            output = model(image_tensor)
            result = output
    return result
