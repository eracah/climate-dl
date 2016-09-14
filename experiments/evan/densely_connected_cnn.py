
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



def dense_conv(args):
    conv_kwargs = dict(filter_size=3, pad=1, nonlinearity=args["nonlinearity"])
    inp = InputLayer(args["input_shape"])
    conc = Conv2DLayer(inp, num_filters=args['k0'], **conv_kwargs)
    conv_kwargs.update({'num_filters': args['k']})
    block_layers = [conc]
    for j in range(args['B']):
        for i in range(args['L']):
            bn = BatchNormLayer(conc)
            bn_relu = NonlinearityLayer(bn ,nonlinearity=args['nonlinearity'])
            bn_relu_conv = Conv2DLayer(bn_relu, **conv_kwargs)
            block_layers.append(bn_relu_conv)
            conc = ConcatLayer(block_layers, axis=1)
        
        if j < args['B']:
            conv = Conv2DLayer(conc, num_filters=conc.output_shape[1], filter_size=1)
            conc = Pool2DLayer(conv,pool_size=2,stride=2, mode='average_exc_pad')
            block_layers=[conc]
    
    conc = Pool2DLayer(conc, pool_size=2, stride=2,mode='average_exc_pad')
    sm = DenseLayer(conc, num_units=args['num_classes'], nonlinearity=softmax)
    for layer in get_all_layers(sm):
        print  layer, layer.output_shape
    print count_params(layer)
    print sm.output_shape
    return sm

    



args = {"B":2, "L": 5, 'k':3, 'k0':16, "num_classes":10, "input_shape": (None,1,28,28), "learning_rate": 0.01, "sigma":0.1, "nonlinearity":elu, "f":128, "tied":False }
net_cfg = get_net(dense_conv, args)




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









