import netCDF4 as nc
from os import listdir, system
from os.path import isfile, join, isdir
import re
import numpy as np
import os
import sys
from shutil import copyfile
import imp
from lasagne.layers import *
from sklearn.manifold import TSNE
import itertools
import theano
from theano import tensor as T
import lasagne
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.updates import *
from lasagne.utils import *
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
#import common
#sys.path.append("stl10/")
# import stl10
#from psutil import virtual_memory
from pylab import rcParams
rcParams['figure.figsize'] = 15, 20
import pdb
import logging


# -------------------------------------

def get_net(net_cfg, args):
    l_out, ladder = net_cfg(args)
    if args["mode"] == "2d":
        X = T.tensor4('X')
    elif args["mode"] == "3d":
        X = T.tensor5('X')
    #net_out = get_output(l_out, X)
    ladder_output = get_output([l_out] + ladder, X)
    net_out = ladder_output[0]
    # squared error loss is between the first two terms
    loss = squared_error(net_out, X).mean()
    sys.stderr.write("main loss between %s and %s\n" % (str(l_out.output_shape), "X") )
    ladder_output = ladder_output[1::]
    if "ladder" in args:
        sys.stderr.write("using ladder connections for conv\n")        
        for i in range(0, len(ladder_output), 2):
            sys.stderr.write("ladder connection between %s and %s\n" % (str(ladder[i].output_shape), str(ladder[i+1].output_shape)) )
            assert ladder[i].output_shape == ladder[i+1].output_shape
            loss += args["ladder"]*squared_error(ladder_output[i], ladder_output[i+1]).mean()
    
    net_out_det = get_output(l_out, X, deterministic=True)
    # this is deterministic + doesn't have the reg params added to the end
    loss_det = squared_error(net_out_det, X).mean()
    params = get_all_params(l_out, trainable=True)
    lr = theano.shared(floatX(args["learning_rate"]))
    if "optim" not in args:
        updates = nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
    else:
        if args["optim"] == "rmsprop":
            updates = rmsprop(loss, params, learning_rate=lr)
        elif args["optim"] == "adam":
            updates = adam(loss, params, learning_rate=lr)
    
    #updates = adadelta(loss, params, learning_rate=lr)
    #updates = rmsprop(loss, params, learning_rate=lr)
    train_fn = theano.function([X], [loss, loss_det], updates=updates)
    loss_fn = theano.function([X], loss)
    out_fn = theano.function([X], net_out_det)
    return {
        "train_fn": train_fn,
        "loss_fn": loss_fn,
        "out_fn": out_fn,
        "lr": lr,
        "l_out": l_out
    }


def iterate(X_train, bs=32):
    b = 0
    while True:
        if b*bs >= X_train.shape[0]:
            break
        yield X_train[b*bs:(b+1)*bs]
        b += 1
    
        
def get_iterator(name, batch_size, data_dir, start_day, end_day, img_size, time_chunks_per_example, step_size, time_steps):
    # for stl10, 'days' and 'data_dir' does not make
    # any sense
    assert name in ["climate", "stl10"]
    if name == "climate":
        return data_iterator(batch_size, data_dir, start_day=start_day, end_day=end_day, img_size=img_size, time_chunks_per_example=time_chunks_per_example, step_size=step_size, time_steps=time_steps)
    elif name == "stl10":
        return stl10.data_iterator(batch_size)
        
def train(cfg,
        num_epochs,
        out_folder,
        sched={},
        batch_size=128,
        model_folder="/storeSSD/cbeckham/nersc/models/",
        tmp_folder="tmp",
        training_days=[1,20],
        validation_days=[345,365],
        time_chunks_per_example=1,
        step_size=20,
        data_dir="/storeSSD/cbeckham/nersc/big_images/",
        dataset="climate",
        img_size=128,
        resume=None,
        debug=True,
        time_steps=8):
    
    def prep_batch(X_batch):
        if dataset == "climate":
            if time_chunks_per_example == 1:
                # shape is (32, 1, 16, 128, 128), so collapse to a 4d tensor
                X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[2], X_batch.shape[3], X_batch.shape[4])
            else:
                # right now it is: (bs, time, nchannels, height, width)
                # needs to be: (bs, nchannels, time, height, width)
                X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[2], X_batch.shape[1], X_batch.shape[3], X_batch.shape[4])
        else:
            pass # nothing needs to be done for stl-10
        return X_batch

    def plot_image(img_composite):
        if dataset == "climate":
            for j in range(0,32):
                plt.subplot(8,4,j+1)
                if time_chunks_per_example > 1:
                    plt.imshow(img_composite[j][0])
                else:
                    plt.imshow(img_composite[j])
                plt.axis('off')
        elif dataset == "stl10":
            for j in range(0,6):
                plt.subplot(3,2,j+1)
                plt.imshow(img_composite[j])
                plt.axis('off')
        plt.savefig('%s/%i.png' % (out_folder, epoch))
        pyplot.clf()
        
    # extract methods
    train_fn, loss_fn, out_fn, l_out = cfg["train_fn"], cfg["loss_fn"], cfg["out_fn"], cfg["l_out"]
    lr = cfg["lr"]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    #if resume != None:
    #    
    
    def get_logger():
        logger = logging.getLogger('log_train')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler('%s/results.txt'%(out_folder))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        logger.addHandler(fh)
        return logger
    
    logger = get_logger()
    num_train = (training_days[1] - training_days[0] + 1) * time_steps * (((1152 - img_size) / step_size) + 1) * (((768 - img_size) / step_size) + 1)
    logger.info("train size: %i"%(num_train))
    for layer in get_all_layers(l_out):
        logger.info(str(layer) + ' ' +  str(layer.output_shape))
    for epoch in range(0, num_epochs):
        t0 = time()
        # learning rate schedule
        if epoch+1 in sched:
            lr.set_value( floatX(sched[epoch+1]) )
            logger.info("changing learning rate to: %f\n" % sched[epoch+1])
        train_losses = []
        train_losses_det = []
        valid_losses = []
        first_minibatch = True
        # TRAINING LOOP
        for X_train, y_train in get_iterator(dataset, batch_size, data_dir, start_day=training_days[0], end_day=training_days[1],
                                        img_size=img_size,step_size=step_size,time_chunks_per_example=time_chunks_per_example, time_steps=time_steps):
            X_train = prep_batch(X_train)
            if first_minibatch:
                X_train_sample = X_train[0:1]
                first_minibatch = False
            this_loss, this_loss_det = train_fn(X_train)
            if debug:
                logger.info("iteration loss: %f" %this_loss)
            train_losses.append(this_loss)
            train_losses_det.append(this_loss_det)
        # if debug:
        #     mem = virtual_memory()
        #     print mem
        # VALIDATION LOOP
        for X_valid, y_valid in get_iterator(dataset, batch_size, data_dir, start_day=validation_days[0], end_day=validation_days[1],
                                              img_size=img_size, time_chunks_per_example=time_chunks_per_example, step_size=step_size, time_steps=time_steps):
            X_valid = prep_batch(X_valid)
            val_loss = loss_fn(X_valid)
            if debug:
                logger.info("iteration val loss: %f" %val_loss)
            valid_losses.append(val_loss)
        # DEBUG: visualise the reconstructions
        img_orig = X_train_sample
        img_reconstruct = out_fn(img_orig)
        img_composite = np.vstack((img_orig[0],img_reconstruct[0]))
        plot_image(img_composite)
        # STATISTICS
        time_taken = time() - t0

        logger.info("epoch %i of %i \n time: %f \n train loss: %6.3f \n val loss: %6.3f" % (epoch+1, num_epochs, time_taken, np.mean(train_losses), np.mean(valid_losses)))

        # save model at each epoch
        if not os.path.exists("%s/models/" % (out_folder)):
            os.makedirs("%s/models/" % (out_folder))
        with open("%s/models/%i.model" % (out_folder, epoch), "wb") as g:
            pickle.dump( get_all_param_values(cfg["l_out"]), g, pickle.HIGHEST_PROTOCOL )

def make_dense_conv_encoder(args):
    conv_kwargs = dict(filter_size=3, pad=1, nonlinearity=args['nonlinearity'], W=lasagne.init.HeNormal())
    inp = InputLayer(args["input_shape"])
    if "sigma" in args:
        inp = GaussianNoiseLayer(inp, sigma=args['sigma'])
    conc = Conv2DLayer(inp, num_filters=args['k0'], **conv_kwargs)
    conv_kwargs.update({'num_filters': args['k'], 'nonlinearity': None})
    for j in range(args['B']):
        conc = make_dense_block(conc, args, conv_kwargs=conv_kwargs)
        if j < args['B'] - 1:
            conc = make_trans_layer(conc, args)
    bn = BatchNormLayer(conc)
    bn_relu = NonlinearityLayer(bn ,nonlinearity=args['nonlinearity'])
    conc = GlobalPoolLayer(bn_relu)
    return conc



            
def make_dense_conv_classifier(args):  
    conc = make_dense_conv_encoder(args)
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
            bn_relu = NonlinearityLayer(bn, nonlinearity=args['nonlinearity'])
            bn_relu_conv = Conv2DLayer(bn_relu, **conv_kwargs)
            block_layers.append(bn_relu_conv)
            conc = ConcatLayer([conc, bn_relu_conv], axis=1)
        return conc

def make_trans_layer(inp,args):
    conc = inp
    bn = BatchNormLayer(conc)
    bn_relu = NonlinearityLayer(bn ,nonlinearity=args['nonlinearity'])
    conv = Conv2DLayer(bn_relu, num_filters=conc.output_shape[1], filter_size=1)
    conc = Pool2DLayer(conv, pool_size=2,stride=2, mode='average_exc_pad')
    return conc

def make_inverse_trans_layer(inp, layer,args):
    conc = inp
    #because trans layers are 2 layerrs log
    for lay in get_all_layers(layer)[::-1][:2]:
        conc = make_inverse(conc, lay,args)
    
    #return whole network and next layer we are going to invert
    return conc, lay.input_layer

def make_inverse_dense_block(inp, layer, args):
    conc = inp

    #3 layers per comp unit and args['L'] units per block
    for lay in get_all_layers(layer)[::-1][:4*args['L']]:
        if isinstance(lay, ConcatLayer):
            conc = make_inverse(conc,lay,args)
            
#             conc = BatchNormLayer(conc)
#             conc = make_inverse(conc, lay, filters=args['k'])
#             conc = NonlinearityLayer(conc, nonlinearity=args['nonlinearity'])
#             block_layers.append(conc)
#             if args['concat_inver']:
#                 conc = ConcatLayer(block_layers,axis=1)
            
    return conc, lay.input_layer
            
        
    

def make_inverse(l_in, layer,args):
    
    if isinstance(layer, Conv2DLayer):
        if 'dec_batch_norm' in args and args['dec_batch_norm']:
            l_in = batch_norm(l_in)
        if 'tied_weights' in args and args['tied_weights']:
            l_in = InverseLayer(l_in,layer)
        else:
            l_in = Deconv2DLayer(l_in, layer.input_shape[1], layer.filter_size, stride=layer.stride, crop=layer.pad, nonlinearity=layer.nonlinearity)
        return NonlinearityLayer(l_in, nonlinearity=args['nonlinearity'])
        
#         if filters is None:
#             filters = layer.input_shape[1]
#         return 
    elif isinstance(layer, Pool2DLayer) or isinstance(layer, GlobalPoolLayer) or isinstance(layer, DenseLayer):
        return InverseLayer(l_in, layer)
    elif isinstance(layer, ConcatLayer):
        first_input_shape = layer.input_shapes[0][layer.axis]
        return SliceLayer(l_in,indices=slice(0,first_input_shape), axis=layer.axis)
               #SliceLayer(l_in,indices=slice(first_input_shape, -1), axis=layer.axis )
    else:
        return l_in

def make_dense_conv_autoencoder(args):
    conc = make_dense_conv_encoder(args)
    hid_lay = DenseLayer(conc, num_units=args['num_fc_units'])
    conc = hid_lay
    for layer in get_all_layers(conc)[::-1]:
        if isinstance(layer, ConcatLayer):
            break
        else:
            conc = make_inverse(conc,layer,args)
        
    for i in range(args['B']):
        conc, layer = make_inverse_dense_block(conc, layer, args)
        if i < args['B'] - 1:
                conc, layer = make_inverse_trans_layer(conc, layer, args)
    
        

    for lay in get_all_layers(layer)[::-1]:
        if isinstance(lay, InputLayer):
            break
        args['nonlinearity'] = tanh
        conc = make_inverse(conc, lay,args)
        
    for layer_ in get_all_layers(conc):
            print  layer_, layer_.output_shape
    print count_params(layer_)
    
    
    return conc, hid_lay
            
            
    
    
    
            
            
    
    
    


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

