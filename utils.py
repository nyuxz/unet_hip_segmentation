import numpy as np
import h5py 
from sys import platform
from scipy import ndimage 

def compute_weights(y):
    flat_y = y.reshape([-1, 2])
    weight = flat_y[:,0].sum() / flat_y.sum()
    return (weight, 1-weight)


def zeroMeanUnitVariance(input_image):
  # zero mean unit variance
    augmented_image = np.zeros(input_image.shape,dtype='float32')
    for ci in range(input_image.shape[0]):
        mn = np.mean(input_image[ci, ...])
        sd = np.std(input_image[ci, ...])
        augmented_image[ci, ...] = (input_image[ci, ...] - mn) / np.amax([sd, 1e-5])
    return augmented_image
  
def get2DMultipleSlices(X_train,y_train,noOfCases):
    '''
    for the case of hop image, there are 60 slices of each image
    '''
    ct_train = 0
    train_X = np.zeros([noOfCases * 60, 512, 512, 3],dtype= 'float32')
    train_y = np.zeros([noOfCases * 60, 512, 512],dtype='uint16')
    for ii in range(X_train.shape[0]):
        tmp_X = X_train[ii]
        tmp_y = y_train[ii]
        for ti in range(tmp_X.shape[2]):
            if np.sum(tmp_y[...,ti,1])!=0:
                train_X[ct_train,:,:,:] = tmp_X[...,ti-1:ti+2] 
                train_y[ct_train,:,:] = tmp_y[...,ti,1] 
                ct_train += 1
               
    train_X = train_X[:ct_train]
    train_y = train_y[:ct_train]
    
    train_X = train_X.transpose(0,3,1,2)

    return train_X, train_y
