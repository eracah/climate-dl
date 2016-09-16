import netCDF4 as nc
from os import listdir, system
from os.path import isfile, join, isdir
import re
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from shutil import copyfile
import imp
from lasagne.layers import *
from sklearn.manifold import TSNE

import itertools

def make_dense_conv_encoder(args):
    conv_kwargs = dict(filter_size=3, pad=1, nonlinearity=args['nonlinearity'])
    inp = InputLayer(args["input_shape"])
    conc = Conv2DLayer(inp, num_filters=args['k0'], **conv_kwargs)
    conv_kwargs.update({'num_filters': args['k'], 'nonlinearity': None})
    for j in range(args['B']):
        conc = make_dense_block(conc, args, conv_kwargs=conv_kwargs)
        if j < args['B'] - 1:
            conc = make_trans_layer(conc, args)
    return conc

            
def make_dense_conv_classifier(args):  
    conc = make_dense_conv_encoder(args)
    conc = Pool2DLayer(conc, pool_size=2, stride=2,mode='average_exc_pad')
    sm = DenseLayer(conc, num_units=args['num_classes'], nonlinearity=softmax)
    for layer in get_all_layers(sm):
        print  layer, layer.output_shape
    print count_params(layer)
    print sm.output_shape
    return sm

    

def make_dense_block(inp, args, conv_kwargs={}):
        conc = inp
        block_layers = [conc]
        for i in range(args['L']):
            bn = BatchNormLayer(conc)
            bn_relu = NonlinearityLayer(bn ,nonlinearity=args['nonlinearity'])
            bn_relu_conv = Conv2DLayer(bn_relu, **conv_kwargs)
            block_layers.append(bn_relu_conv)
            conc = ConcatLayer(block_layers, axis=1)
        return conc

def make_trans_layer(inp,args):
    conc = inp
    conv = Conv2DLayer(conc, num_filters=conc.output_shape[1], filter_size=1)
    conc = Pool2DLayer(conv, pool_size=2,stride=2, mode='average_exc_pad')
    return conc

def make_inverse_trans_layer(inp, layer):
    conc = inp
    #because trans layers are 2 layerrs log
    for lay in get_all_layers(layer)[::-1][:2]:
        #print "*******************\n\n",lay, "\n\n********************\n\n"
        conc = make_inverse(conc, lay)
    
    #return whole network and next layer we are going to invert
    return conc, lay.input_layer

def make_inverse_dense_block(inp, layer, args):
    conc = inp
    block_layers = [conc]
    #3 layers per comp unit and args['L'] units per block
    for lay in get_all_layers(layer)[::-1][:4*args['L']]:
        if isinstance(lay, Conv2DLayer):
            conc = BatchNormLayer(conc)
            conc = make_inverse(conc, lay, filters=args['k'])
            conc = NonlinearityLayer(conc, nonlinearity=args['nonlinearity'])
            block_layers.append(conc)
            if args['concat_inver']:
                conc = ConcatLayer(block_layers,axis=1)
            
    return conc, lay.input_layer
            
        
    

def make_inverse(l_in, layer, filters=None):
    
    if isinstance(layer, Conv2DLayer):
        if filters is None:
            filters = layer.input_shape[1]
        return Deconv2DLayer(l_in, filters, layer.filter_size, stride=layer.stride, crop=layer.pad,
                             nonlinearity=layer.nonlinearity)
    elif isinstance(layer, Pool2DLayer) or isinstance(layer, DenseLayer):
        return InverseLayer(l_in, layer)
    else:
        return l_in

def make_dense_conv_autoencoder(args):
    conc = make_dense_conv_encoder(args)
    conc = make_trans_layer(conc, args)
    last_conv_shape = tuple([k if k is not None else [i] for i,k in enumerate(get_output_shape(conc,args['input_shape']))] )
    hid_lay = DenseLayer(conc, num_units=args['num_fc_units'])
    rec = DenseLayer(hid_lay,num_units=np.prod(last_conv_shape[1:]))
    conc = ReshapeLayer(rec, shape=last_conv_shape)
    
    
    for layer in get_all_layers(conc)[::-1]:
        if not isinstance(layer, DenseLayer) and not isinstance(layer, ReshapeLayer):
            break
        
    for i in range(args['B']):
        conc, layer = make_inverse_trans_layer(conc, layer)
        #print "lol: ", layer
        conc, layer = make_inverse_dense_block(conc, layer, args)
        

    for lay in get_all_layers(layer)[::-1]:
        if isinstance(lay, InputLayer):
            break
        conc = make_inverse(conc, lay)
    for layer_ in get_all_layers(conc):
            print  layer_, layer_.output_shape
    print count_params(layer_)
    
    
    return conc
            
            
    
    
    


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
        if ioba > 0.75:
            cls = box_class
            break

    return cls
    

import pdb

def _day_iterator(year=1979, data_dir="/project/projectdirs/dasrepo/gordon_bell/climate/data/big_images/", shuffle=False, days=365, month1='01', day1='01'):
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
    camfiles = camfiles[ind:ind+days]
    if shuffle:
        np.random.shuffle(camfiles)
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

def data_iterator(batch_size, data_dir="/project/projectdirs/dasrepo/gordon_bell/climate/data/big_images/", time_chunks_per_example=1, shuffle=False, step_size=20, days=365,month1='01', day1='01'):
    '''
    Args:
       batch_size: number of examples in a batch
       data_dir: base dir where data is
       time_chunks_per_example: how many time steps are in a given example (default is one, but when we do 3D conv -> move to >1)
                            - should divide evenly into 8
    '''
    # for each day (out of 365 days)
    day=0
    for tensor, labels in _day_iterator(data_dir=data_dir, shuffle=shuffle, days=days, month1=month1, day1=day1):  #tensor is 8,16,768,1152
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
        for ind, spt_chunk in enumerate(spatiotemporal_tensor):
            time_slices_with_class = get_slices(spt_chunk, labels=labels[ind], step_size=step_size)        
            # for each time step that day (there are 8 time steps
            # with 3 hr gaps, so 8*3 = 24 hrs)
            for (time_slice, classes) in [time_slices_with_class]:

                # we'll get e.g. ~2000 slices (at 128px), and now we run these
                # through the batch size iterator to get these in 128-size batches
                # for the gpu
                num_examples = time_slice.shape[0]
                for b in range(0, num_examples, batch_size):
                    res = time_slice[b:b+batch_size]
                    cls = classes[b:b+batch_size]
                    cls = cls.astype('int32')
                    yield res, cls
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
    
def plot_clusters(i,x,y,net_cfg, save_dir='.'):
    x = np.squeeze(x)
    hid_L = net_cfg['h_fn'](x)
    ts = TSNE().fit_transform(hid_L)
    plt.clf()
    plt.scatter(ts[:,0], ts[:,1], c=y)
    plt.savefig(save_dir + '/cluster_%i.png'%(i))
    plt.clf()

def plot_recs(i,x,net_cfg, save_dir='.'):
    ind = np.random.randint(0,x.shape[0], size=(1,))
    x=np.squeeze(x)
    #print x.shape
    im = x[ind]
    #print im.shape
    rec = net_cfg['out_fn'](im)
    ch=1
    plt.figure(figsize=(30,30))
    plt.clf()
    for (p_im, p_rec) in zip(im[0],rec[0]):
        p1 = plt.subplot(im.shape[1],2, ch )
        p2 = plt.subplot(im.shape[1],2, ch + 1)
        p1.imshow(p_im)
        p2.imshow(p_rec)
        ch = ch+2
    #pass
    plt.savefig(save_dir +'/recs_%i' %(i))

def plot_filters(network, save_dir='.'):
    plt.figure(figsize=(30,30))
    plt.clf()
    lay_ind = 0
    num_channels_to_plot = 16
    convlayers = [layer for layer in get_all_layers(network) if isinstance(layer, Conv2DLayer)]
    num_layers = len(convlayers)
    spind = 1 
    for layer in convlayers:
        filters = layer.get_params()[0].eval()
        #pick a random filter
        filt = filters[np.random.randint(0,filters.shape[0])]
        for ch_ind in range(num_channels_to_plot):
            p1 = plt.subplot(num_layers,num_channels_to_plot, spind )
            p1.imshow(filt[ch_ind], cmap="gray")
            spind = spind + 1
    
    #pass
    plt.savefig(save_dir +'/filters.png')
            
        
def plot_feature_maps(i, x, network, save_dir='.'):
    plt.figure(figsize=(30,30))
    plt.clf()
    ind = np.random.randint(0,x.shape[0])
    x=np.squeeze(x)

    im = x[ind]
    convlayers = [layer for layer in get_all_layers(network) if not isinstance(layer,DenseLayer)]
    num_layers = len(convlayers)
    spind = 1 
    num_fmaps_to_plot = 16
    for ch in range(num_fmaps_to_plot):
        p1 = plt.subplot(num_layers + 1,num_fmaps_to_plot, spind )
        p1.imshow(im[ch])
        spind = spind + 1
      
    for layer in convlayers:
        # shape is batch_size, num_filters, x,y 
        fmaps = get_output(layer,x ).eval()
        for fmap_ind in range(num_fmaps_to_plot):
            p1 = plt.subplot(num_layers + 1,num_fmaps_to_plot, spind )
            p1.imshow(fmaps[ind][fmap_ind])
            spind = spind + 1
    
    #pass
    plt.savefig(save_dir +'/fmaps.png')

def create_run_dir(custom_rc=False):
    results_dir = os.getcwd() + '/results'
    run_num_file = os.path.join(results_dir, "run_num.txt")
    if not os.path.exists(results_dir):
        print "making results dir"
        os.mkdir(results_dir)

    if not os.path.exists(run_num_file):
        print "making run num file...."
        f = open(run_num_file,'w')
        f.write('0')
        f.close()




    f = open(run_num_file,'r+')

    run_num = int(f.readline()) + 1

    f.seek(0)

    f.write(str(run_num))


    run_dir = os.path.join(results_dir,'run%i'%(run_num))
    os.mkdir(run_dir)
    
    if custom_rc:
        make_custom_config_file(run_dir)
    return run_dir
                    
# little test/example
if __name__ == "__main__":
    for x,y in data_iterator(batch_size=1716, time_chunks_per_example=1):
        
        # if any of the labels are not none
        if y[y>-3].shape[0] > 0:
            labelled_chunks = x[y>-3]
        print x.shape, y.shape

