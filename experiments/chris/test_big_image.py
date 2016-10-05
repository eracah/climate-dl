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
from psutil import virtual_memory
from pylab import rcParams
rcParams['figure.figsize'] = 8, 12
import pdb
sys.path.append("..")
import common

def net():
    conv = InputLayer((None, 16, 768, 1152))
    conv = Conv2DLayer(conv, num_filters=128, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=768, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1024, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1280, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1536, filter_size=5, stride=2)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    return conv

def net_bigger():
    conv = InputLayer((None, 16, 768, 1152))
    conv = Conv2DLayer(conv, num_filters=128, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1024, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1280, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1536, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=2048, filter_size=5, stride=2)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    return conv

def net_bigger2():
    conv = InputLayer((None, 16, 768, 1152))
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1024, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1280, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1536, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1792, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=2048, filter_size=5, stride=2)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    return conv

def net_bigger3():
    conv = InputLayer((None, 16, 768, 1152))
    conv = Conv2DLayer(conv, num_filters=384, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1024, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1280, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1536, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1792, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=2048, filter_size=5, stride=2)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    return conv

def net_bigger4():
    conv = InputLayer((None, 16, 768, 1152))
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=768, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1024, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1280, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1536, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1792, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=2048, filter_size=5, stride=2)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    return conv

def net_bigger4_but_cut():
    conv = InputLayer((None, 16, 768, 1152))
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=768, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1024, filter_size=5, stride=2)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    return conv

def net_bigger4_but_cut_less_fms():
    conv = InputLayer((None, 16, 768, 1152))
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=768, filter_size=5, stride=2)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    return conv


def net_bigger4_but_cut2():
    conv = InputLayer((None, 16, 768, 1152))
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=768, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1024, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1280, filter_size=5, stride=2)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    return conv

def net_bigger4_but_cut2_less_fms():
    conv = InputLayer((None, 16, 768, 1152))
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=768, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1024, filter_size=5, stride=2)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    return conv

def net_bigger4_but_cut2_less_fms2():
    conv = InputLayer((None, 16, 768, 1152))
    conv = Conv2DLayer(conv, num_filters=128, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=768, filter_size=5, stride=2)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    return conv

def net_bigger4_but_cut3_less_fms2():
    conv = InputLayer((None, 16, 768, 1152))
    conv = Conv2DLayer(conv, num_filters=128, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=768, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1024, filter_size=5, stride=2)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    return conv

def net_bigger4_but_cut4_less_fms2():
    conv = InputLayer((None, 16, 768, 1152))
    conv = Conv2DLayer(conv, num_filters=128, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=768, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1024, filter_size=5, stride=2)
    conv = Conv2DLayer(conv, num_filters=1280, filter_size=5, stride=2)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    return conv



def net_bigger4_but_cut2_less_fms2_suminv():
    l_in = InputLayer((None, 16, 768, 1152))
    conv = Conv2DLayer(l_in, num_filters=128, filter_size=5, stride=2)
    conv2 = Conv2DLayer(conv, num_filters=256, filter_size=5, stride=2)
    conv3 = Conv2DLayer(conv2, num_filters=512, filter_size=5, stride=2)
    conv4 = Conv2DLayer(conv3, num_filters=768, filter_size=5, stride=2)
    #inv1 = InverseLayer(conv4, conv)
    #inv2 = InverseLayer(conv3, conv)
    #inv3 = InverseLayer(conv2, conv)
    #inv4 = InverseLayer(conv, conv)
    #sum_ = ElemwiseSumLayer([inv1,inv2,inv3,inv4])
    #return sum_
    conv=conv4
    for layer in get_all_layers(conv4)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    return conv


l_out = net_bigger4_but_cut4_less_fms2()
for layer in get_all_layers(l_out):
    print type(layer), layer.output_shape
print l_out.output_shape
print count_params(l_out)

X = T.tensor4('X')
net_out = get_output(l_out, X)
loss = squared_error(net_out, X).mean()
params = get_all_params(l_out, trainable=True)
updates = nesterov_momentum(loss, params, 0.01, 0.9)
train_fn = theano.function([X], loss, updates=updates)

folder_name = "big_image_test_5_but_cut4_less_fms2"

f = open("output/%s/sgd.txt" % folder_name,"wb")
g = open("output/%s/results.txt" % folder_name,"wb")
f.write("per_example_loss\n")
g.write("train_loss\n")
for epoch in range(0,1):
    day = 1
    losses = []
    for x, _ in common.data._day_iterator(data_dir="/storeSSD/cbeckham/nersc/big_images/", shuffle=True):
        print "day:", day
        # x is a tensor of (8, 16, 768, 1152)
        img = x
        # normalise the image on a per feature map basis
        for j in range(0, img.shape[1]):
            range_ = np.max(img[:,j,:,:]) - np.min(img[:,j,:,:])
            midrange = (np.max(img[:,j,:,:]) + np.min(img[:,j,:,:])) / 2.
            img[:,j,:,:] = (img[:,j,:,:] - midrange) / ( range_ / 2.)
        #pdb.set_trace()
        #print img.shape
        for b in range(0, img.shape[0]):
            loss = train_fn(img[b:b+1])
            losses.append(loss)
            f.write("%f\n" % loss)
            f.flush()
        day += 1
    mean_loss = np.mean(losses)
    g.write("%f\n" % mean_loss)
    g.flush()
f.close()
g.close()
