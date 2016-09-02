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
    # the none class to -999
    cls=-999
    for box in boxes:
        box_coords = box[:4]
        box_class = box[-1]
        ioba = calc_int_over_box_area(chunk_coords, box_coords)
        # if the intersection of the chunk with the box 
        #normalized by box area is over 0.75, then we label the chunk as that class
        if ioba > 0.75:
            cls = box_class
            break

    return cls
    
    

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
        labels = make_labels_for_dataset(dataset, time_steps)
        yield tr_data, labels


def get_slices(img, labels, img_size=128, step_size=20):
    '''
    input: time_chunk_size, 16, 768,1152 tensor
    returns: array of overlapping time_chunk_size,16,128,128 tensors'''
    slices = []
    classes = []
    height, width = img.shape[-2], img.shape[-1]
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            chunk = img[:,:, y:y+img_size, x:x+img_size]
            cls = get_chunk_class(labels, (x, x+img_size, y, y+img_size))
            classes.append(cls)
            if chunk.shape[2:] == (img_size,img_size):
                slices.append(chunk)
    
    return np.asarray(slices, dtype=img.dtype), np.asarray(classes)

def data_iterator(batch_size, data_dir="/project/projectdirs/dasrepo/gordon_bell/climate/data/big_images/", time_chunks_per_example=1):
    '''
    Args:
       batch_size: number of examples in a batch
       data_dir: base dir where data is
       time_chunks_per_example: how many time steps are in a given example (default is one, but when we do 3D conv -> move to >1)
                            - should divide evenly into 8
    '''
    # for each day (out of 365 days)
    for tensor, labels in _day_iterator(data_dir=data_dir):  #tensor is 8,16,768,1152
        time_chunks_per_day, variables, h, w = tensor.shape #time_chunks will be 8
        assert time_chunks_per_day % time_chunks_per_example == 0, "For convenience, \
        the time chunk size should divide evenly for the number of time chunks in a single day"
        
        #reshapes the tensor into multiple spatiotemporal chunks of (chunk_size, 16, 768,1152)
        spatiotemporal_tensor = tensor.reshape(time_chunks_per_day / time_chunks_per_example, 
                                               time_chunks_per_example, variables, h ,w)
        
        #for each spt_chunk -> patches up chunk into (num_chunks, chunk_size, 16,128,128)
        #time slices is list of multiple (num_chunks, chunk_size, 16,128,128) tensors
        time_slices_with_classes = [ get_slices(spt_chunk, labels=labels[ind]) for ind, spt_chunk in enumerate(spatiotemporal_tensor)]
        
        # for each time step that day (there are 8 time steps
        # with 3 hr gaps, so 8*3 = 24 hrs)
        for (time_slice, classes) in time_slices_with_classes:
            # we'll get e.g. ~2000 slices (at 128px), and now we run these
            # through the batch size iterator to get these in 128-size batches
            # for the gpu
            num_examples = time_slice.shape[0]
            for b in range(0, num_examples, batch_size):
                res = time_slice[b:b+batch_size]
                cls = classes[b:b+batch_size]
                yield res, cls

# little test/example
if __name__ == "__main__":
    for x,y in data_iterator(batch_size=1716, time_chunks_per_example=1):
        
        # if any of the labels are not none
        if y[y>-999].shape[0] > 0:
            labelled_chunks = x[y>-999]
        print x.shape, y.shape

