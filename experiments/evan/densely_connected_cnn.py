
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



import argparse
if any(["jupyter" in arg for arg in sys.argv]):
    sys.argv=sys.argv[:1]


parser = argparse.ArgumentParser()

parser.add_argument('-n', '--num_ims', default=2000, type=int,
    help='number of total images')
args = parser.parse_args()



from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')

ims = mnist['data']

ims.shape

ims = ims.reshape(ims.shape[0],1, 28,28).astype('float64')

lbls = mnist['target'].astype('int32')
ims= ims[:args.num_ims]
lbls = lbls[:args.num_ims]
ims -= np.mean(ims)
ims /= np.var(ims)

num_ims = ims.shape[0]
inds = np.arange(num_ims)
np.random.RandomState(11).shuffle(inds)
ims= ims[inds]
lbls =lbls[inds]

im_tr, lbl_tr, im_val, lbl_val = ims[:int(0.8*num_ims)], lbls[:int(0.8*num_ims)],                                  ims[int(0.8*num_ims):], lbls[int(0.8*num_ims):]



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



def get_net(net_cfg, args):
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
    bn = BatchNormLayer(conc)
    bn_relu = NonlinearityLayer(bn ,nonlinearity=args['nonlinearity'])
    conc = GlobalPoolLayer(bn_relu) #Pool2DLayer(conc, pool_size=2, stride=2,mode='average_exc_pad')
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
            
            
    
    
    



args = {"B":2, "L": 2, 'k':3, 'k0':16, "num_classes":10, "num_fc_units":100,'concat_inver':True, "input_shape": (None,1,28,28), "learning_rate": 0.01, "sigma":0.1, "nonlinearity":elu, "f":128, "tied":False }
#net_cfg = get_net(make_dense_conv_classifier, args)



cnet = make_dense_conv_classifier(args)



net = make_dense_conv_autoencoder(args)



from nolearn import lasagne as ls



ls.visualize




num_epochs = 5000
batch_size = 128
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
tr_losses = []
val_losses = []

for epoch in range(num_epochs):
    start = time.time() 
    tr_loss = 0
    for iteration, (x, y) in enumerate(iterate_minibatches(im_tr,lbl_tr,batchsize=batch_size)):  
        loss = net_cfg['train_fn'](x,y)
        print loss
        tr_loss += loss
    
    train_end = time.time()
    tr_avgloss = tr_loss / (iteration + 1)
    
    
    logger.info("train time : %5.2f seconds" % (train_end - start))
    logger.info(" epoch %i of %i train loss is %f" % (epoch, num_epochs, tr_avgloss))
    tr_losses.append(tr_avgloss)
    
    val_loss = 0
    val_acc =0 
    for iteration, (xval, yval) in enumerate(iterate_minibatches(im_val,lbl_val,batchsize=batch_size)):
        loss,acc = net_cfg['val_fn'](xval, yval)
        val_acc += acc
        val_loss += loss
    
    val_avgacc = val_acc / (iteration + 1) 
    val_avgloss = val_loss / (iteration + 1)   
    logger.info("val time : %5.2f seconds" % (time.time() - train_end))

    logger.info(" epoch %i of %i val loss is %f" % (epoch, num_epochs, val_avgloss))
    logger.info(" epoch %i of %i val acc is %f percent" % (epoch, num_epochs, 100*val_avgacc))
    val_losses.append(val_avgloss)
    
    plot_learn_curve(tr_losses, val_losses, save_dir=run_dir)
#     if epoch % 5 == 0:
#         plot_filters(net_cfg['l_out'], save_dir=run_dir)
#         for iteration, (x,y) in enumerate(data_iterator(batch_size=batch_size, step_size=128, days=1, month1='01', day1='01')):
#             plot_recs(iteration,x,net_cfg=net_cfg, save_dir=run_dir)
#             plot_clusters(iteration,x,y,net_cfg=net_cfg, save_dir=run_dir)
#             plot_feature_maps(iteration,x,net_cfg['l_out'], save_dir=run_dir)
#             break;



from viz import draw_to_file
draw_to_file(get_all_layers(cnet), "./dense_net_classifier.eps")
draw_to_file(get_all_layers(net), "./dense_net_autoencoder.eps")






matplotlib.use('agg')





