import netCDF4 as nc
from os import listdir, system
from os.path import isfile, join, isdir
import re
import numpy as np
from matplotlib import pyplot as plt

import itertools
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






def _day_iterator(year=1979, data_dir="/project/projectdirs/dasrepo/gordon_bell/climate/data/big_images/"):
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
    for camfile in camfiles:
        dataset = nc.Dataset(maindir+'/'+camfile, "r", format="NETCDF4")
        time_steps=8
        x=768
        y=1152
        day_slice = make_spatiotemporal_tensor(dataset,time_steps,variables) #one day slice per dataset
        tr_data = day_slice.reshape(time_steps,len(variables), x, y)
        yield tr_data


def get_slices(img, img_size=128, step_size=20):
    '''
    input: time_chunk_size, 16, 768,1152 tensor
    returns: array of time_chunk_size,16,128,128 tensors'''
    slices = []
    height, width = img.shape[-2], img.shape[-1]
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            chunk = img[:,:, y:y+img_size, x:x+img_size]
            if chunk.shape[2:] == (img_size,img_size):
                slices.append(chunk)
    return np.asarray(slices, dtype=img.dtype)

def data_iterator(batch_size, data_dir, time_chunks_per_example=1):
    '''
    Args:
       batch_size: number of examples in a batch
       data_dir: base dir where data is
       time_chunks_per_example: how many time steps are in a given example (default is one, but when we do 3D conv -> move to >1)
                            - should divide evenly into 8
    '''
    # for each day (out of 365 days)
    for tensor in _day_iterator(data_dir=data_dir):  #tensor is 8,16,768,1152
        
        time_chunks_per_day, variables, h, w = tensor.shape #time_chunks will be 8
        assert time_chunks_per_day % time_chunks_per_example == 0, "For convenience, \
        the time chunk size should divide evenly for the number of time chunks in a single day"
        
        #reshapes the tensor into multiple spatiotemporal chunks of (chunk_size, 16, 768,1152)
        spatiotemporal_tensor = tensor.reshape(time_chunks_per_day / time_chunks_per_example, time_chunks_per_example, variables, h ,w)
        
        #for each spt_chunk -> patches up chunk into (num_chunks, chunk_size, 16,128,128)
        #time slices is list of multiple (num_chunks, chunk_size, 16,128,128) tensors
        time_slices = [ get_slices(spt_chunk) for spt_chunk in spatiotemporal_tensor]
        # for each time step that day (there are 8 time steps
        # with 3 hr gaps, so 8*3 = 24 hrs)
        for time_slice in time_slices:
            # we'll get e.g. ~2000 slices (at 128px), and now we run these
            # through the batch size iterator to get these in 128-size batches
            # for the gpu
            num_examples = time_slice.shape[0]
            for b in range(0, num_examples, batch_size):
                res = time_slice[b:b+batch_size]
                yield res