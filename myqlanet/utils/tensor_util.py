import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).float().to('cuda'),
        'bbox': torch.from_numpy(bbox).float().to('cuda')}
