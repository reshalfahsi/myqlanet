import torch
from torch.utils.data import DataLoader
from ..preprocessing import MaculaDataset

import time
import math

from torch.autograd import Variable
from ..utils import image_util

def eval(model, test_loader, cuda, num_output):
    """Eval over test set"""
    model.eval()
    ret_loss = 0
    ret_iou = 0
    idx = 0
    # Get Batch
    for (data, target) in test_loader:
        # print("evaluating...")
        loss = 0
        iou = 0
        idx += 1
        data, target = Variable(data), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        # Evaluate
        output = model(data)
        # Load output on CPU
        if cuda:
            output.cpu()
        # Compute Loss
        for d in range(0,num_output):
            loss += math.sqrt(abs(output[0][d]**2 - target[0][d]**2))
        gt_start_point = (target[0][3], target[0][2])
        gt_end_point = (target[0][1], target[0][0])
        start_point = (output[0][3], output[0][2])
        end_point = (output[0][1], output[0][0])
        loss /=  num_output
        ret_loss += loss
        iou = float(image_util.bb_intersection_over_union((gt_start_point[0], gt_start_point[1], gt_end_point[0], gt_end_point[1]),(start_point[0], start_point[1], end_point[0], end_point[1])))
        ret_iou += iou
    ret_iou /= (float(idx) + 1.0)
    return ret_loss, ret_iou

def save_checkpoint(state, is_best, filename=''):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        if (filename == '') :
           print('Path Empty!')
           return None
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation did not improve")

# def train(model, train_dataset, optimizer, train_loader, test_loader, loss_fn, cuda, batch_size, epoch, start_epoch, num_output, path, best_loss):
def train(model, epoch, path):
    """Perform a full training over dataset"""
    average_time = 0
    # Model train mode
    model.train()
    batch_size, start_epoch, num_output, best_loss = model.train_utility_parameters()
    train_dataset, train_loader, test_loader = model.train_utility_dataset()
    cuda = model.isCudaAvailable()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        batch_time = time.time()
        
        images = Variable(images).float()
        target = Variable(target).float()
        if cuda:
            images, target = images.cuda(), target.cuda()
        # Forward + Backward + Optimize
        model.optimizer().zero_grad()

        #print(images.shape)

        outputs = model(images)
        outputs = outputs.float()
        #print('Output Size: '+str(outputs.size(0)))
        #print(target)
        #print('Target Size: '+str(target.size(0)))
        loss = model.loss()(outputs, target)
        # Load loss on CPU
        if cuda:
            loss.cpu()
        loss.backward()
        model.optimizer().step()
        # Measure elapsed time
        batch_time = time.time() - batch_time
        # Accumulate over batch
        average_time += batch_time
        # ### Keep track of metric every batch
        # Accuracy Metric
        #prediction = outputs.data.max(1)[1]   # first column has actual prob.
        #accuracy = prediction.eq(target.data).sum() / batch_size * 100
        # Log
        print_every = 5
        if (i + 1) % print_every == 0:
            print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Batch time: %f'
            % (epoch + 1,
            model.max_epoch(),
            i + 1,
            len(train_dataset) // batch_size,
            loss.data,
            #accuracy,
            average_time/print_every))  # Average
    loss, iou = eval(model, test_loader, cuda, num_output)
    print('=> Test set: Loss: {:.2f}'.format(loss))
    is_best = bool(loss <= best_loss)
    if is_best:
        best_loss = loss
        # Save checkpoint if is a new best
        save_checkpoint({'epoch': start_epoch + epoch + 1, 'state_dict': model.state_dict(), 'best_loss': best_loss}, is_best, path)
    return loss, iou


