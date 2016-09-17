
import matplotlib; matplotlib.use("agg")


import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
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
import time
if __name__ == "__main__":
    sys.path.insert(0,'..')
    from common import create_run_dir, plot_learn_curve
else:
    from ..common import create_run_dir, plot_learn_curve
from viz import draw_to_file
from sklearn.manifold import TSNE

import itertools



def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    if batchsize > inputs.shape[0]:
        batchsize=inputs.shape[0]
    for start_idx in range(0,len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx: start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def get_net_classif(net_cfg, args):
    l_out = net_cfg(args)
    X = T.tensor4('X')
    Y= T.ivector('Y')
    net_out = get_output(l_out, X)
    loss = categorical_crossentropy(net_out, Y).mean()
    net_out_det = get_output(l_out, X, deterministic=True)
    acc = lasagne.objectives.categorical_accuracy(net_out_det,Y).mean()
    params = get_all_params(l_out, trainable=True)
    lr = theano.shared(floatX(args["learning_rate"]))
    if "rmsprop" in args:
        updates = rmsprop(loss, params, learning_rate=lr)
    else:
        updates = nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
    #updates = adadelta(loss, params, learning_rate=lr)
    #updates = rmsprop(loss, params, learning_rate=lr)
    train_fn = theano.function([X, Y], loss, updates=updates)
    loss_fn = theano.function([X, Y], loss)
    val_fn = theano.function([X,Y], [loss,acc])
    out_fn = theano.function([X], net_out_det)
    return {
        "train_fn": train_fn,
        "loss_fn": loss_fn,
        "out_fn": out_fn,
        "val_fn": val_fn,
        "lr": lr,
        "l_out": l_out
    }



def get_net(net_cfg, args):
    l_out, hid_lay = net_cfg(args)            
    X = T.tensor4('X')
    net_out = get_output(l_out, X)
#     ladder_output = get_output(ladder, X)
    # squared error loss is between the first two terms
    loss = squared_error(net_out, X).mean()
    sys.stderr.write("main loss between %s and %s\n" % (str(l_out.output_shape), "X") )
#     if "ladder" in args:
#         sys.stderr.write("using ladder connections for conv\n")        
#         for i in range(0, len(ladder_output), 2):
#             sys.stderr.write("ladder connection between %s and %s\n" % (str(ladder[i].output_shape), str(ladder[i+1].output_shape)) )
#             assert ladder[i].output_shape == ladder[i+1].output_shape
#             loss += args["ladder"]*squared_error(ladder_output[i], ladder_output[i+1]).mean()
    net_out_det = get_output(l_out, X, deterministic=True)
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
    train_fn = theano.function([X], loss, updates=updates)
    loss_fn = theano.function([X], loss)
    out_fn = theano.function([X], net_out_det)
    hid_fn = theano.function([X], get_output(hid_lay, X))
    return {
        "train_fn": train_fn,
        "loss_fn": loss_fn,
        "out_fn": out_fn,
        "lr": lr,
        "l_out": l_out,
        "h_fn" : hid_fn
    }



def get_net_ae(net_cfg, args):
    l_out, hid_layer = net_cfg(args)
 
    X = T.tensor4('X')
    Y = T.ivector('Y')
    net_out = get_output(l_out, X)
    hid_out = get_output(hid_layer, X)

    
    clsf_loss = get_classifier_loss(hid_layer,X,Y,args)
    rec_loss = squared_error(net_out, X).mean()
    if args['with_classif_loss']:
        loss = args['lrec'] * rec_loss + clsf_loss
        inputs = [X,Y]
    else:
        loss = rec_loss
        inputs=[X]
    params = get_all_params(l_out, trainable=True)
    lr = theano.shared(floatX(args["learning_rate"]))
    updates = nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
    train_fn = theano.function(inputs, loss, updates=updates)
    loss_fn = theano.function(inputs, loss)
    out_fn = theano.function([X], net_out)
    hid_fn = theano.function([X],hid_out)
    return {
        "train_fn": train_fn,
        "loss_fn": loss_fn,
        "out_fn": out_fn,
        "lr": lr,
        "l_out": l_out,
        "h_fn": hid_fn,

    }







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
            bn_relu = NonlinearityLayer(bn ,nonlinearity=args['nonlinearity'])
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
            
            
    
    
    



import argparse
if any(["jupyter" in arg for arg in sys.argv]):
    sys.argv=sys.argv[:1]


parser = argparse.ArgumentParser()

parser.add_argument('-n', '--num_ims', default=5000, type=int,help='number of total images')
parser.add_argument('-L', '--num_layers_per_block', default=2, type=int)
parser.add_argument('-B', '--num_blocks', default=2, type=int)
parser.add_argument('-k', '--num_filters_per_conv', default=3, type=int)
parser.add_argument('-K', '--num_filters_in_first_conv', default=16, type=int)
parser.add_argument('-t', '--tied_weights', dest='tied_weights', action='store_true')
parser.set_defaults(tied_weights=False)
parser.add_argument('-b', '--batch_norm_dec', dest='dec_batch_norm', action='store_true')
parser.set_defaults(dec_batch_norm=False)
parser.add_argument('--fc', default=100, type=int)
parser.add_argument('-s', '--sigma', default=0.5, type=float)
parser.add_argument( '--batchsize', default=128, type=int)
parser.add_argument( '--num_epochs', default=1000, type=int)
parser.add_argument( '--learn_rate', default=0.1, type=float)
pargs = parser.parse_args()



from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')

ims = mnist['data']

ims.shape

ims = ims.reshape(ims.shape[0],1, 28,28).astype('float64')

lbls = mnist['target'].astype('int32')
ims= ims[:pargs.num_ims]
lbls = lbls[:pargs.num_ims]
ims -= np.mean(ims)
ims /= np.var(ims)

num_ims = ims.shape[0]
inds = np.arange(num_ims)
np.random.RandomState(11).shuffle(inds)
ims= ims[inds]
lbls =lbls[inds]

im_tr, lbl_tr, im_val, lbl_val = ims[:int(0.8*num_ims)], lbls[:int(0.8*num_ims)],                                  ims[int(0.8*num_ims):], lbls[int(0.8*num_ims):]



args = {"B":pargs.num_blocks, "L": pargs.num_layers_per_block, 'k':pargs.num_filters_per_conv,
        'k0':pargs.num_filters_in_first_conv, "num_classes":10,'dec_batch_norm':pargs.dec_batch_norm,
        'tied_weights':pargs.tied_weights, "num_fc_units":pargs.fc,
        "input_shape": (None,1,28,28), "learning_rate": pargs.learn_rate,
        "sigma":pargs.sigma, "nonlinearity":elu, "f":128}
#net_cfg = get_net(make_dense_conv_classifier, args)



def plot_clusters(i,x,y,net_cfg, save_dir='.'):
    hid_L = net_cfg['h_fn'](x)
    ts = TSNE().fit_transform(hid_L)
    plt.clf()
    plt.scatter(ts[:,0], ts[:,1], c=y)
    plt.savefig(save_dir + '/cluster_%i.png'%(i))
    plt.clf()

def plot_recs(i,x,net_cfg, save_dir='.'):
    ind = np.random.randint(0,x.shape[0], size=(1,))
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

def plot_filters(network,num_filter_channels_to_plot=1, save_dir='.'):
    plt.figure(figsize=(30,30))
    plt.clf()
    lay_ind = 0
    num_channels_to_plot = num_filter_channels_to_plot
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

    im = x[ind]
    convlayers = [layer for layer in get_all_layers(network) if isinstance(layer,Conv2DLayer) or                   isinstance(layer, TransposedConv2DLayer) or isinstance(layer,Pool2DLayer)]
    num_layers = len(convlayers)
    spind = 1 
    num_fmaps_to_plot = 1
    for ch in range(im.shape[0]):
        p1 = plt.subplot(num_layers + 2,num_fmaps_to_plot, spind )
        p1.imshow(im[ch], cmap="gray")
        spind = spind + 1
    spind = num_fmaps_to_plot +1
    for layer in convlayers:
        # shape is batch_size, num_filters, x,y 
        fmaps = get_output(layer,x ).eval()
        for fmap_ind in range(num_fmaps_to_plot):
            p1 = plt.subplot(num_layers + 2,num_fmaps_to_plot, spind )
            p1.imshow(fmaps[ind][fmap_ind], cmap="gray")
            spind = spind + 1
    
    out = get_output(network,x).eval()
    for ch in range(out.shape[1]):
        p1 = plt.subplot(num_layers + 2,num_fmaps_to_plot, spind )
        p1.imshow(out[ind][ch], cmap="gray")
        spind = spind + 1
    
    #pass
    plt.savefig(save_dir +'/fmaps.png')



import logging
run_dir = create_run_dir()
try:
    print logger
except:
    logger = logging.getLogger('log_train')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('%s/training.log'%(run_dir))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.addHandler(fh)
logger.info("train size = %i, val size = %i"%(im_tr.shape[0], im_val.shape[0])) 

net_cfg = get_net(make_dense_conv_autoencoder,args)

draw_to_file(get_all_layers(net_cfg['l_out']), run_dir + "/network_topo.eps")
tr_losses = []
val_losses = []

for epoch in range(pargs.num_epochs):
    start = time.time() 
    tr_loss = 0
    for iteration, (x, y) in enumerate(iterate_minibatches(im_tr,lbl_tr,batchsize=pargs.batchsize)):  
        loss = net_cfg['train_fn'](x)
        print loss
        tr_loss += loss
    
    train_end = time.time()
    tr_avgloss = tr_loss / (iteration + 1)
    
    
    logger.info("train time : %5.2f seconds" % (train_end - start))
    logger.info(" epoch %i of %i train loss is %f" % (epoch, pargs.num_epochs, tr_avgloss))
    tr_losses.append(tr_avgloss)
    
    val_loss = 0
    val_acc =0 
    for iteration, (xval, yval) in enumerate(iterate_minibatches(im_val,lbl_val,batchsize=pargs.batchsize)):
        loss = net_cfg['loss_fn'](xval)
        #val_acc += acc
        val_loss += loss
    
    #val_avgacc = val_acc / (iteration + 1) 
    val_avgloss = val_loss / (iteration + 1)   
    logger.info("val time : %5.2f seconds" % (time.time() - train_end))

    logger.info(" epoch %i of %i val loss is %f" % (epoch, pargs.num_epochs, val_avgloss))
    #logger.info(" epoch %i of %i val acc is %f percent" % (epoch, num_epochs, 100*val_avgacc))
    val_losses.append(val_avgloss)
    
    plot_learn_curve(tr_losses, val_losses, save_dir=run_dir)
    if epoch % 1 == 0:
        plot_filters(net_cfg['l_out'], save_dir=run_dir)
        for iteration, (x,y) in enumerate(iterate_minibatches(im_tr,lbl_tr,batchsize=pargs.batchsize)):
            plot_recs(iteration, x, net_cfg=net_cfg, save_dir=run_dir)
            plot_clusters(iteration, x, y, net_cfg=net_cfg, save_dir=run_dir)
            plot_feature_maps(iteration,x,net_cfg['l_out'], save_dir=run_dir)
            break;





