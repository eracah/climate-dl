import netCDF4 as nc
from os import listdir, system
from os.path import isfile, join, isdir
import re
import numpy as np
from matplotlib import pyplot as plt

"""
This iterator will return tensors of dimension (8, 16, 1152, 864),
each tensor corresponding to one of the 365 days of the year
"""
def _day_iterator(year=1979, data_dir="/project/projectdirs/dasrepo/gordon_bell/climate/data/big_images/"):
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
        datasets = [ nc.Dataset(maindir+'/'+camfile, "r", format="NETCDF4") ]
        time_steps=8
        x=768
        y=1152
        day_slices = [make_spatiotemporal_tensor(dataset,time_steps,variables) for dataset in datasets]
        tr_data = np.vstack(day_slices).reshape(len(datasets), time_steps,len(variables), x, y)
        yield tr_data[0]


def get_slices(img, img_size=128, step_size=20):
    slices = []
    _, height, width = img.shape
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            chunk = img[:, y:y+img_size, x:x+img_size]
            if chunk.shape[1::] == (img_size,img_size):
                slices.append(chunk)
    return np.asarray(slices, dtype=img.dtype)

def data_iterator(batch_size, data_dir):
    # for each day (out of 365 days)
    for tensor in _day_iterator(data_dir=data_dir):
        hourly_slices = [ get_slices(hr) for hr in tensor ]
        # for each time step that day (there are 8 time steps
        # with 3 hr gaps, so 8*3 = 24 hrs)
        for hourly_slice in hourly_slices:
            # we'll get e.g. ~2000 slices (at 128px), and now we run these
            # through the batch size iterator to get these in 128-size batches
            # for the gpu
            num_examples = hourly_slice.shape[0]
            for b in range(0, num_examples, batch_size):
                res = hourly_slice[b:b+batch_size]
                yield res
