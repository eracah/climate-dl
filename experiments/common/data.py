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
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import pyplot
import os
import sys
from time import time
sys.path.append("..")
from pylab import rcParams
rcParams['figure.figsize'] = 15, 20
import pdb



# -------------------------------------

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

    return labels_list
    

def calc_int_over_box_area(coords_1, box_coords):
    ''' calc intersection over box area'''
    xmin1,xmax1, ymin1, ymax1 = coords_1
    xmin2,xmax2, ymin2, ymax2 = box_coords
    inters = max(0,(min(xmax1,xmax2) - max(xmin1,xmin2)))   * \
                          max(0,(min(ymax1,ymax2) - max(ymin1,ymin2)) )
    def get_area(box_mm):
        xmin, xmax, ymin, ymax = box_mm
        return (xmax - xmin) * (ymax - ymin)
    box_area = get_area((xmin2, xmax2, ymin2, ymax2))                                                     
    
    return inters / float(box_area)


def get_chunk_class(boxes, chunk_coords):
    
    # for some reason there are classes that are less than 0, so we set
    # the none class to -3
    cls=-3
    for box in boxes:
        box_coords = box[:4]
        box_class = box[-1]
        ioba = calc_int_over_box_area(chunk_coords, box_coords)
        # if the intersection of the chunk with the box 
        #normalized by box area is over 0.75, then we label the chunk as that class
        if ioba >= 0.75:
            cls = box_class
            break

    return cls
    

import pdb

def _day_iterator(year=1979, data_dir="/project/projectdirs/dasrepo/gordon_bell/climate/data/big_images/", shuffle=False, start_day=1, end_day=365, month1='01', day1='01', time_steps=8):
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
        labels = make_labels_for_dataset(dataset, time_steps)
        yield tr_data, labels


def get_slices(img, labels, img_size=128, step_size=20):
    '''
    input: time_chunk_size, 16, 768,1152 tensor
    returns: iterator of overlapping time_chunk_size,16,128,128 tensors'''
    slices = []
    classes = []
    height, width = img.shape[-2], img.shape[-1]
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            chunk = img[:,:, y:y+img_size, x:x+img_size]
            cls = get_chunk_class(labels, (x, x+img_size, y, y+img_size))
            #classes.append(cls)
            if chunk.shape[2:] == (img_size,img_size):
                #slices.append(chunk)
                yield chunk, cls
    #pdb.set_trace()
    #return np.asarray(slices, dtype=img.dtype), np.asarray(classes)

def data_iterator(batch_size,
                  data_dir="/project/projectdirs/dasrepo/gordon_bell/climate/data/big_images/",
                  time_chunks_per_example=1,
                  shuffle=False,
                  img_size=128,
                  step_size=20,
                  start_day=1,
                  end_day=365,
                  month1='01',
                  day1='01',
                  time_steps=8):
    '''
    Args:
       batch_size: number of examples in a batch
       data_dir: base dir where data is
       time_chunks_per_example: how many time steps are in a given example (default is one, but when we do 3D conv -> move to >1)
                            - should divide evenly into 8
    '''
    # for each day (out of 365 days)
    day=0
    for tensor, labels in _day_iterator(data_dir=data_dir, shuffle=shuffle, start_day=start_day, end_day=end_day, month1=month1, day1=day1, time_steps=time_steps):  #tensor is 8,16,768,1152
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
        if shuffle:
            np.random.shuffle(spatiotemporal_tensor)
        
        #for each spt_chunk -> patches up chunk into (num_chunks, chunk_size, 16,128,128)
        #time slices is list of multiple (num_chunks, chunk_size, 16,128,128) tensors
        #old: memory intensive
        #time_slices_with_classes = [ get_slices(spt_chunk, labels=labels[ind]) for ind, spt_chunk in enumerate(spatiotemporal_tensor)]
        #for (time_slice, classes) in time_slices_with_classes: ...
        
        #print spatiotemporal_tensor.shape
        x_buf, y_buf = [], []
        # for each day
        
        for ind, spt_chunk in enumerate(spatiotemporal_tensor):
            # for each w*h px chunk
            for chunk, cls_ in get_slices(spt_chunk, labels=labels[ind], step_size=step_size, img_size=img_size):
                if len(x_buf) == batch_size:
                    x_buf = np.asarray(x_buf, dtype=x_buf[0].dtype)
                    y_buf = np.asarray(y_buf, dtype="int32")
                    yield x_buf, y_buf
                    x_buf = []
                    y_buf = []
                else:
                    x_buf.append(chunk)
                    y_buf.append(cls_)
            # if there is left over stuff in the buffer in the end, yield that too
            if len(x_buf) != 0:
                x_buf = np.asarray(x_buf, dtype=x_buf[0].dtype)
                y_buf = np.asarray(y_buf, dtype="int32")                
                yield x_buf, y_buf
                x_buf = []
                y_buf = []
            day = day + 1

def normalize(arr,min_=None, max_=None, axis=(0,2,3)):
        if min_ is None or max_ is None:
            min_ = arr.min(axis=(0,2,3), keepdims=True)

            max_ = arr.max(axis=(0,2,3), keepdims=True)

        midrange = (max_ + min_) / 2.

        range_ = max_ - min_

        arr = (arr - midrange) / (range_ /2.)
        return arr, min_, max_
    


def plot_learn_curve(tr_losses, val_losses, save_dir='.'):
    plt.clf()
    plt.plot(tr_losses)
    plt.plot(val_losses)
    plt.savefig(save_dir + '/learn_curve.png')
    plt.clf()
    

# little test/example
if __name__ == "__main__":
    for x,y in data_iterator(batch_size=1716, time_chunks_per_example=1):
        
        # if any of the labels are not none
        if y[y>-3].shape[0] > 0:
            labelled_chunks = x[y>-3]
        print x.shape, y.shape

