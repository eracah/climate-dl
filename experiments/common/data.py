import netCDF4 as nc
from os import listdir, system
from os.path import isfile, join, isdir
import re
import numpy as np
from shutil import copyfile
import imp
import itertools
from sklearn.manifold import TSNE
import numpy as np
import cPickle as pickle
import gzip
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
%matplotlib inline
import os
import sys
from time import time
sys.path.append("..")
from pylab import rcParams
rcParams['figure.figsize'] = 15, 20
import pdb
import itertools


# -------------------------------------

#tropical depression are 0
# hurricanes are 1

def make_time_slice(dataset, time, variables, x=768, y=1152):
    '''Takes in a dataset, a time and variables and gets one time slice for all the variables and all x and y'''
    variables_at_time_slice = [dataset[k][time] for k in variables]
    tensor = np.vstack(variables_at_time_slice).reshape(len(variables), x,y)
    
    return tensor


def make_spatiotemporal_tensor(dataset,num_time_slices, variables, x=768, y=1152):
    '''takes in: dataset, num_time_slices
       returns: num_time_slices, num_variables,x,y'''
    time_slices = [ make_time_slice(dataset, time, variables) for time in range(num_time_slices) ]
    tensor = np.vstack(time_slices).reshape(num_time_slices, len(variables), x, y)

    return tensor


def make_labels_for_dataset(dataset, time_steps):
    mask = dataset['teca_mask'][:] == 1
    '''for a given dataset, for each time step, generates a list of boxes (each box is xmin, xmax, ymin,ymax, category)'''
    labels_list = [zip(*[list(dataset[k][time_step][mask[time_step]]) for k in [ u'teca_xmin',
                                                               u'teca_xmax',
                                                               u'teca_ymin',
                                                               u'teca_ymax',
                                                               u'teca_category']] ) for time_step in range(time_steps) ]
    labels_list = edit_classes(labels_list)
    return labels_list

def edit_classes(labels_list):
    for i in range(len(labels_list)):
        for j in range(len(labels_list[i])):
            tup = list(labels_list[i][j])
            if labels_list[i][j][4] == -1:
                tup[4] = 0
                labels_list[i][j] = tuple(tup)
            else:
                tup[4] = 1
                labels_list[i][j] = tuple(tup)
    return labels_list
    
def make_masks_for_dataset(dataset, time_steps, classes=2):
    labels_list = make_labels_for_dataset(dataset, time_steps)
    x, y = dataset['TMQ'][0].shape
    masks = np.zeros((time_steps, classes, x ,y))
    bg = np.ones((time_steps, 1, x ,y))
    for time_step, labels in enumerate(labels_list):
        
        for label in labels:
            cls = label[-1]
            x_slice = slice(label[0],label[1])
            y_slice = slice(label[2], label[3])
            masks[time_step, cls, y_slice,  x_slice] = 1
            bg[time_step, 0,y_slice, x_slice] = 0
    masks = np.concatenate((masks, bg), axis=1).reshape(time_steps, classes+1, x ,y)
    #masks is an 8,num_classes, 768, 1152 mask 0's everywhere except where class is
    return masks


                
                

import pdb

def _day_iterator(year=1979, data_dir="/project/projectdirs/dasrepo/gordon_bell/climate/data/big_images/", shuffle=False, start_day=1, end_day=365,
                  month1='01', day1='01', time_steps=8, classes=2):
    """
    This iterator will return tensors of dimension (8, 16, 768, 1152) 
    each tensor corresponding to one of the 365 days of the year
    """
    variables = [u'PRECT',
                 u'PS',
                 u'PSL',
                 u'QREFHT',
                 u'T200',
                 u'T500',
                 u'TMQ',
                 u'TREFHT',
                 u'TS',
                 u'U850',
                 u'UBOT',
                 u'V850',
                 u'VBOT',
                 u'Z1000',
                 u'Z200',
                 u'ZBOT']    
    # this directory can be accessed from cori
    maindir = data_dir + str(year) 
    lsdir=listdir(maindir)
    rpfile = re.compile(r"^cam5_.*\.nc$")
    camfiles = [f for f in lsdir if rpfile.match(f)]
    camfiles.sort()

    prefix = 'cam5_1_amip_run2.cam2.h2.1979'
    suffix = '00000.nc'
    #print camfiles
    ind= camfiles.index('-'.join([prefix, month1, day1, suffix]))
    camfiles = camfiles[ind+(start_day-1):ind+end_day] # [
    if shuffle:
        np.random.shuffle(camfiles)
    for camfile in camfiles:
        dataset = nc.Dataset(maindir+'/'+camfile, "r", format="NETCDF4")
        x=768
        y=1152
        day_slice = make_spatiotemporal_tensor(dataset,time_steps,variables) #one day slice per dataset
        tr_data = day_slice.reshape(time_steps,len(variables), x, y)
        masks = make_masks_for_dataset(dataset, time_steps, classes=classes)
        
        yield tr_data, masks


    
def data_iterator(batch_size,
                  data_dir="/project/projectdirs/dasrepo/gordon_bell/deep_learning/data/climate/big_images/",
                  time_chunks_per_example=8,
                  shuffle=False,
                  start_day=1,
                  end_day=365,
                  month1='01',
                  day1='01',
                  time_steps=8, classes=2):
    '''
    Args:
       batch_size: number of examples in a batch
       data_dir: base dir where data is
       time_chunks_per_example: how many time steps are in a given example (default is one, but when we do 3D conv -> move to >1)
                            - should divide evenly into 8
    '''
    # for each day (out of 365 days)
    day=0
    for tensor, masks in _day_iterator(data_dir=data_dir, shuffle=shuffle, start_day=start_day, end_day=end_day, month1=month1, day1=day1, time_steps=time_steps):  #tensor is 8,16,768,1152
        # preprocess for day
        tensor, min_, max_ = normalize(tensor)
        #TODO: preprocess over everything
        #TODO: split up into train,test, val
        time_chunks_per_day, variables, h, w = tensor.shape #time_chunks will be 8
        assert time_chunks_per_day % time_chunks_per_example == 0, "For convenience, \
        the time chunk size should divide evenly for the number of time chunks in a single day"
        
        #reshapes the tensor into multiple spatiotemporal chunks of (chunk_size, 16, 768,1152)
        spatiotemporal_tensor = tensor.reshape(time_chunks_per_day / time_chunks_per_example, 
                                               time_chunks_per_example, variables, h ,w)
        sp_mask = masks.reshape(time_chunks_per_day / time_chunks_per_example, 
                                               time_chunks_per_example,classes+1, h ,w)
        
        if shuffle:
            np.random.shuffle(spatiotemporal_tensor)

        b = 0
        while True:
            if b*batch_size >= spatiotemporal_tensor.shape[0]:
                break
            # todo: add labels
            yield spatiotemporal_tensor[b*batch_size:(b+1)*batch_size], sp_mask[b*batch_size:(b+1)*batch_size]
            b += 1

def consecutive_day_iterator_3d(start_day, end_day, data_dir="/project/projectdirs/dasrepo/gordon_bell/deep_learning/data/climate/big_images/", 
                                shuffle=False, classes=2, labels_only=True):
    def imerge(a,b):
        for i,j in itertools.izip(a,b):
            yield i,j
    # e.g. this will return days 1...364 (assuming start and end day is (1, 365))
    iter1 = data_iterator(batch_size=1, data_dir=data_dir, time_chunks_per_example=8, shuffle=shuffle, start_day=start_day, end_day=end_day-1)
    # e.g. this will return days 2..365 (assuming start and end day is (1, 365))
    iter2 = data_iterator(batch_size=1, data_dir=data_dir, time_chunks_per_example=8, shuffle=shuffle, start_day=start_day+1, end_day=end_day)
    # e.g. combined iterator will return: (day1, day2), (day2, day3), ..., (day 364, day 365)
    for tp1, tp2 in imerge(iter1, iter2):
        x1 = tp1[0]
        y1 = tp1[1]    
        x2 = tp2[0]
        y2 = tp2[1]
        if labels_only:
            x = np.concatenate((x1[:,[0,2,4,6]],x2[:, [0,2,4,6]]),axis=1)
            y = np.concatenate((y1[:, [0,2,4,6]],y2[:, [0,2,4,6]]), axis=1)
            x = np.swapaxes(x,1,2 )
            yield x,y
        else:
            yield (x1,y1), (x2,y2)
def segmentation_iterator(start_day, end_day, data_dir="/project/projectdirs/dasrepo/gordon_bell/deep_learning/data/climate/big_images/", 
                                shuffle=False, classes=2, labels_only=True ):
    for x,y in data_iterator(batch_size=1, data_dir=data_dir, time_chunks_per_example=8, 
                  shuffle=shuffle, start_day=start_day, end_day=end_day, classes=classes):
        if labels_only:
            yield x[:,[0,2,4,6]], y[:,[0,2,4,6]]
        else:
            yield x,y
        
            

                
def normalize(arr,min_=None, max_=None, axis=(0,2,3)):
        if min_ is None or max_ is None:
            min_ = arr.min(axis=(0,2,3), keepdims=True)

            max_ = arr.max(axis=(0,2,3), keepdims=True)

        midrange = (max_ + min_) / 2.

        range_ = max_ - min_

        arr = (arr - midrange) / (range_ /2.)
        return arr, min_, max_
    
def normalize_tmp(arr,min_=None, max_=None, axis=(0,2,3)):
        if min_ is None or max_ is None:
            min_ = arr.min(axis=(0,2,3), keepdims=True)

            max_ = arr.max(axis=(0,2,3), keepdims=True)

        midrange = (max_ + min_) / 2.

        range_ = max_ - min_

        arr = (arr - min_) / (max_ - min_)
        return arr, min_, max_

    

import pdb
  

  
# # little test/example
# if __name__ == "__main__":
#     """
#     for x,y in data_iterator(batch_size=1716, time_chunks_per_example=1):
        
#         # if any of the labels are not none
#         if y[y>-3].shape[0] > 0:
#             labelled_chunks = x[y>-3]
#         print x.shape, y.shape
#     """

#     """
#     i = 0
#     for x,y in data_iterator(batch_size=1, time_chunks_per_example=8, img_size=-1, data_dir="/storeSSD/cbeckham/nersc/big_images/"):
#         pdb.set_trace()
#         i += 1
#     print i
#     assert i == 365*8
#     """

#     i=0
#     for x1, x2 in consecutive_day_iterator_3d(start_day=1, end_day=10, data_dir="/storeSSD/cbeckham/nersc/big_images/", shuffle=False):
#         print x1.shape, x2.shape
#         i += 1
#     print i
# if __name__ == "__main__":
#     for i, (x,y) in enumerate(data_iterator(batch_size=1,
#                       data_dir="/project/projectdirs/dasrepo/gordon_bell/deep_learning/data/climate/big_images/",
#                       time_chunks_per_example=8,
#                       shuffle=False,
#                       start_day=1,
#                       end_day=2,
#                       month1='10',
#                       day1='06',
#                       time_steps=8)):
#         print x.shape, y.shape
#         plt.figure(i)
#         ax = plt.subplot(2,2,1)
#         ax.imshow(x[0,0,6])
#         for j in range(3):
#             ax = plt.subplot(2,2,j + 2)
#             ax.imshow(y[0,0,j], cmap='gray')
#         plt.show()
#         assert False
    

from pylab import rcParams
rcParams['figure.figsize'] = 15, 25
if __name__ == "__main__":
    for i,(x,  y) in enumerate(segmentation_iterator(1, 6, classes=2)):
        print x.shape, y.shape
        plt.figure(i)

        c = 0
        for k in range(4):
            ax = plt.subplot(4,4,c+1)
            ax.imshow(x[0,k,6])
            c+=1
            for m in range(3):
                
                ax = plt.subplot(4,4,c+1)
                ax.imshow(y[0,k,m], cmap='gray')
                c+=1
                
        assert False


#         plt.figure(2)
#         ax = plt.subplot(2,2,1)
#         ax.imshow(x2[0,0,6])
#         for j in range(3):
#             ax = plt.subplot(2,2,j + 2)
#             ax.imshow(y2[0,0,j], cmap='gray')
#         plt.show()
        





