from torch.utils.data.dataset import Dataset
import numpy as np
import h5py 
from sys import platform
from scipy import ndimage 
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from utils import *


class HIP(Dataset):
    def __init__(self, X, y, subset="train", transform = None):
        
        # initialize variables
        self.subset = subset
        self.X = np.array(X, dtype=">f")
        self.y = np.array(y, dtype=">f")
        self.transform = transform


    def __getitem__(self, index):
    
        img = torch.from_numpy(self.X[index])
        target = torch.from_numpy(self.y[index])
        
        #crop target which match with final-input(324)
        target = target[94:-94,94:-94]

        # apply transforms to both
        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)
            
        return img, target

    def __len__(self):

        return self.X.shape[0]
     


def load_data():
	# load dataset from h5
	# for data, i have total four cases data for train the deep convnets 
	noOfCases = 4
	X = np.zeros([noOfCases, 512, 512, 60],dtype='uint16')
	y = np.zeros([noOfCases, 512, 512, 60,2],dtype='uint16') # last slice is for one-hot-encoding
	filename = './data/HipSegmentation_2Sean.h5'
	f1 = h5py.File(filename,'r')
	for name, data in f1.items():
	    if name == 'image':
	        for i in range(noOfCases):
	            tmp = data['%d'%(i+1)].value
	            X[i] = np.rollaxis(tmp,0,3)
	    elif name =='mask':
	        for i in range(noOfCases):
	            tmp = data['%d'%(i+1)].value
	            y[i,...,1] = np.rollaxis(tmp,0,3)
	            y[i,...,0] = 1-np.rollaxis(tmp,0,3)

	# obtain 3 consecutive slices, treated as 3 channals 
	X_new, y_new = get2DMultipleSlices(X, y, noOfCases)
	X_new = zeroMeanUnitVariance(X_new)


	dataset = HIP(X = X_new,
	              y = y_new, 
	              subset="train", 
	              transform = None) 
	# I do not need transform, coz my input is ndarary not PIL image 

	return dataset 

def build_loader():

	dataset = load_data()

	train_loader = torch.utils.data.DataLoader(dataset=dataset,
	                                           batch_size=1,
	                                           shuffle=True,
	                                           pin_memory=False, # If True, the data loader will copy tensors into CUDA pinned memory before returning them.
	                                           num_workers=0) # change to 1 if run in the server

	val_loader = torch.utils.data.DataLoader(dataset=dataset,
	                                           batch_size=1,
	                                           shuffle=True,
	                                           pin_memory=False, # If True, the data loader will copy tensors into CUDA pinned memory before returning them.
	                                           num_workers=0) # change to 1 if run in the server

	return (train_loader, val_loader)


