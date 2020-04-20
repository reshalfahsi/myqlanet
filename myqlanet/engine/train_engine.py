import torch
from torch.utils.data import DataLoader
from ..preprocessing import MaculaDataset
import time
from torch.autograd import Variable

def eval(model, optimizer, test_loader, cuda, num_output):
    """Eval over test set"""
    model.eval()
    ret_loss = 0
    # Get Batch
    for (data, target) in test_loader:
        loss = 0
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
        loss /=  num_output
        ret_loss += loss
    return ret_loss

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

def train(model, train_dataset, optimizer, train_loader, test_loader, loss_fn, cuda, batch_size, epoch, start_epoch, num_output, path, best_lost): 
    """Perform a full training over dataset"""
    average_time = 0
    # Model train mode
    model.train()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        batch_time = time.time()
        images = Variable(images).float()
        target = Variable(target).float()
        if cuda:
            images, target = images.cuda(), target.cuda()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.float()
        #print('Output Size: '+str(outputs.size(0)))
        #print(target)
        #print('Target Size: '+str(target.size(0)))
        loss = loss_fn(outputs, target)
        # Load loss on CPU
        if cuda:
            loss.cpu()
        loss.backward()
        optimizer.step()
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
            num_epochs,
            i + 1,
            len(train_dataset) // batch_size,
            loss.data,
            #accuracy,
            average_time/print_every))  # Average
    loss = eval(model, optimizer, test_loader, cuda, num_output)
    print('=> Test set: Loss: {:.2f}'.format(loss))
    is_best = bool(loss < best_loss)
    if is_best:
        best_loss = loss
        # Save checkpoint if is a new best
        train_engine.save_checkpoint({'epoch': start_epoch + epoch + 1, 'state_dict': self.state_dict(), 'best_loss': best_loss}, is_best, path)


