import numpy as np
import h5py 
from sys import platform
from scipy import ndimage 
import torch
import torch.nn.functional as F
from torch import nn
#from hip_unet import UNet
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import UNet
from UNet import UNetOriginal
from dataloader import *




class BCELoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)
    



def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        if use_cuda:
            data = Variable(data.cuda())
            target = Variable(target.cuda())
        else:
            data = Variable(data)
            target = Variable(target)
        
        #data =data.squeeze()
        #target =target.squeeze()
        #data = data.transpose(1,3).contiguous()
        #target = target.transpose(1,3).contiguous()


        print(data.size())
        print(target.size())
        
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100 * batch_idx / len(train_loader), loss.data[0]))


def evaluate():
    model.eval()
    val_loss = 0
    correct = 0
    for data, target in val_loader:

        if use_cuda:
            data = Variable(data.cuda(), volatile=True)
            target = Variable(target.cuda())
        else:
            data = Variable(data, volatile=True)
            target = Variable(target)

        #data =data.squeeze()
        #target =target.squeeze()
        #data = data.transpose(1,3).contiguous()
        #target = target.transpose(1,3).contiguous()
        
        output = model(data)

        val_loss += criterion(output, target).data[0] # sum up batch loss                                                               
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    val_loss /= len(val_loader.dataset) # mean loss
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100 * correct / len(val_loader.dataset)))




if __name__ == '__main__':

    train_loader, val_loader = build_loader()

    use_cuda = torch.cuda.is_available()

    model = UNetOriginal([3,512,512])
    criterion = BCELoss2d()
    
    optimizer = optim.SGD(model.parameters(),
                      weight_decay=1e-4,
                      lr=1e-4,
                      momentum=0.9,
                      nesterov=True)

    for epoch in range(1, 3):
        train(epoch)
        evaluate()


