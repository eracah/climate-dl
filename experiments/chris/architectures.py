import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.updates import *
from lasagne.utils import *
try:
    from lasagne.layers import dnn
except:
    pass
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
import common
sys.path.append("stl10/")
import stl10
from psutil import virtual_memory
from pylab import rcParams
rcParams['figure.figsize'] = 15, 20
import pdb


def make_inverse(l_in, layer):
    if isinstance(layer, Conv2DLayer):
        return Deconv2DLayer(l_in, layer.input_shape[1], layer.filter_size, stride=layer.stride, crop=layer.pad,
                             nonlinearity=layer.nonlinearity)
    else:
        return InverseLayer(l_in, layer)

def get_encoder_and_decoder(l_out):
    encoder_layers = [layer for layer in get_all_layers(l_out) if isinstance(layer, Conv2DLayer) ]
    decoder_layers = []
    conv = l_out
    for layer in get_all_layers(l_out)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
        if isinstance(conv.input_layers[-1], Conv2DLayer):
            decoder_layers.append(conv)
    return conv, decoder_layers, encoder_layers


def print_network(l_out):
    for layer in get_all_layers(l_out):
        print layer, layer.output_shape
    print count_params(layer)

def what_where_stl10_arch(args={"nonlinearity": rectify, "tied":False, "tanh_end":False, "input_channels":16, "bottleneck":None, "strides":[2,2,1,1]}):
    #(64)3c-4p-(64)3c-3p-(128)3c-(128)3c-2p-(256)3c-(256)3c-(256)3c-(512)3c-(512)3c-(512)3c-2p-10fc 
    # not sure if i got the strides right? they don't make it
    # clear in their paper...
    # the vggnet paper do 2x2 max-pooling with a stride of 2, so maybe these authors mean stride 2,
    # only that you can't do that for the last two max pools else you get a negative output dimension
    # ---------
    # also, they mention having batch-norm on a per feature map basis, so let's do this too
    conv = InputLayer((None, args["input_channels"], 96, 96))
    conv = Conv2DLayer(conv, num_filters=64, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = MaxPool2DLayer(conv, pool_size=4, stride=args["strides"][0])
    conv = Conv2DLayer(conv, num_filters=64, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = MaxPool2DLayer(conv, pool_size=3, stride=args["strides"][1])
    conv = Conv2DLayer(conv, num_filters=128, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = Conv2DLayer(conv, num_filters=128, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = MaxPool2DLayer(conv, pool_size=2, stride=args["strides"][2])
    conv = Conv2DLayer(conv, num_filters=256, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = Conv2DLayer(conv, num_filters=256, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = Conv2DLayer(conv, num_filters=256, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = Conv2DLayer(conv, num_filters=512, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = Conv2DLayer(conv, num_filters=512, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = Conv2DLayer(conv, num_filters=512, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = MaxPool2DLayer(conv, pool_size=2, stride=args["strides"][3])
    if args["bottleneck"] != None:
        conv = DenseLayer(conv, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    print "encoder:"
    for layer in get_all_layers(conv):
        print layer, layer.output_shape
    print "number of params: ", count_params(conv)
    #conv = DenseLayer(conv, num_units=10)
    l_out = conv
    final_out, decoder_layers, encoder_layers = get_encoder_and_decoder(l_out)
    ladder = []
    for a,b in zip(decoder_layers, encoder_layers[::-1]):
        ladder += [a,b]
    return final_out, ladder

    
def what_where_stl10_arch_beefier(args={"nonlinearity": rectify, "tied":False, "tanh_end":False}):
    conv = InputLayer((None, 16, 96, 96))
    conv = Conv2DLayer(conv, num_filters=128, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = MaxPool2DLayer(conv, pool_size=4, stride=4)
    conv = Conv2DLayer(conv, num_filters=128, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = MaxPool2DLayer(conv, pool_size=3, stride=1)
    conv = Conv2DLayer(conv, num_filters=256, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = Conv2DLayer(conv, num_filters=256, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = MaxPool2DLayer(conv, pool_size=2, stride=1)
    conv = Conv2DLayer(conv, num_filters=384, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = Conv2DLayer(conv, num_filters=384, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = Conv2DLayer(conv, num_filters=384, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = Conv2DLayer(conv, num_filters=512, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = Conv2DLayer(conv, num_filters=512, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = Conv2DLayer(conv, num_filters=512, filter_size=3, nonlinearity=args["nonlinearity"]); conv = BatchNormLayer(conv, axes=[0, 2, 3])
    conv = MaxPool2DLayer(conv, pool_size=2, stride=1)
    #conv = DenseLayer(conv, num_units=10)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        if args["tied"]:
            conv = InverseLayer(conv, layer)
        else:
            conv = make_inverse(conv, layer)
    if args["tanh_end"]:
        # evan pre-procced the imgs to be [-1, 1]
        conv = NonlinearityLayer(conv, nonlinearity=tanh)
    for layer in get_all_layers(conv):
        print layer, layer.output_shape
    print count_params(layer)
    return conv

    

def stl10_test(args={}):
    conv = InputLayer((None,3,96,96))
    conv = Conv2DLayer(conv, num_filters=64, filter_size=5, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=128, filter_size=5, stride=2, nonlinearity=rectify)
    l_out = conv
    print_network(l_out)
    final_out, decoder_layers, encoder_layers = get_encoder_and_decoder(l_out)
    ladder = []
    for a,b in zip(decoder_layers, encoder_layers[::-1]):
        ladder += [a,b]
    return final_out, ladder

def stl10_test_2(args={}):
    conv = InputLayer((None,3,96,96))
    conv = Conv2DLayer(conv, num_filters=64, filter_size=5, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=128, filter_size=5, stride=2, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, nonlinearity=rectify)
    l_out = conv
    print_network(l_out)
    final_out, decoder_layers, encoder_layers = get_encoder_and_decoder(l_out)
    ladder = []
    for a,b in zip(decoder_layers, encoder_layers[::-1]):
        ladder += [a,b]
    return final_out, ladder


def stl10_test_3(args={}):
    conv = InputLayer((None,3,96,96))
    conv = Conv2DLayer(conv, num_filters=64, filter_size=5, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=128, filter_size=5, stride=2, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, nonlinearity=rectify)    
    l_out = conv
    print_network(l_out)
    final_out, decoder_layers, encoder_layers = get_encoder_and_decoder(l_out)
    ladder = []
    for a,b in zip(decoder_layers, encoder_layers[::-1]):
        ladder += [a,b]
    return final_out, ladder


def stl10_test_4(args={"sigma":0.}):
    conv = InputLayer((None,3,96,96))
    conv = GaussianNoiseLayer(conv, sigma=args["sigma"])
    conv = Conv2DLayer(conv, num_filters=128, filter_size=5, stride=2, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, stride=2, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2, nonlinearity=rectify)
    if "bottleneck" in args:
        conv = DenseLayer(conv, num_units=args["bottleneck"], nonlinearity=rectify)
    l_out = conv
    print_network(l_out)
    final_out, decoder_layers, encoder_layers = get_encoder_and_decoder(l_out)
    ladder = []
    for a,b in zip(decoder_layers, encoder_layers[::-1]):
        ladder += [a,b]
    return final_out, ladder

def climate_test_1(args={"sigma":0.}):
    conv = InputLayer((None,16,96,96))
    conv = GaussianNoiseLayer(conv, sigma=args["sigma"])
    conv = Conv2DLayer(conv, num_filters=128, filter_size=5, stride=2, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, stride=2, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2, nonlinearity=rectify)
    if "bottleneck" in args:
        conv = DenseLayer(conv, num_units=args["bottleneck"], nonlinearity=rectify)
    l_out = conv
    print_network(l_out)
    final_out, decoder_layers, encoder_layers = get_encoder_and_decoder(l_out)
    ladder = []
    for a,b in zip(decoder_layers, encoder_layers[::-1]):
        ladder += [a,b]
    return final_out, ladder


def climate_test_1_densifiable(args):
    transition_layers = []
    if not args["densify"]:
        args["k"] = 0
        args["L"] = 0
    conv_kwargs = dict(num_filters=args['k'], filter_size=3, pad=1, nonlinearity=rectify)
    conv = InputLayer((None,16,96,96))
    conv = Conv2DLayer(conv, num_filters=args["nf"][0]-(args["k"]*args["L"]), filter_size=5, stride=2, nonlinearity=rectify, name="c1")
    transition_layers.append(conv)
    if args["densify"]:
        conv = common.helper_fxns.make_dense_block(conv, args, conv_kwargs)
    conv = Conv2DLayer(conv, num_filters=args["nf"][1]-(args["k"]*args["L"]), filter_size=5, stride=2, nonlinearity=rectify, name="c2")
    transition_layers.append(conv)
    if args["densify"]:
        conv = common.helper_fxns.make_dense_block(conv, args, conv_kwargs)
    conv = Conv2DLayer(conv, num_filters=args["nf"][2]-(args["k"]*args["L"]), filter_size=5, stride=2, nonlinearity=rectify, name="c3")
    transition_layers.append(conv)
    if args["densify"]:
        conv = common.helper_fxns.make_dense_block(conv, args, conv_kwargs)
    if "bottleneck" in args:
        conv = DenseLayer(conv, num_units=512, nonlinearity=rectify)
        conv = InverseLayer(conv, conv)
    
    # -----------
    # decoder
    # todo: i could for-loop the bottom, but i needed to do some hacky padding stuff
    # -----------

    c2 = SliceLayer(conv, indices=slice(0,args["nf"][0]-(args["k"]*args["L"])), axis=1)
    c2a = TransposedConv2DLayer(c2, transition_layers[-1].input_shape[1],
         transition_layers[-1].filter_size, stride=transition_layers[-1].stride, crop=transition_layers[-1].pad,
         W=transition_layers[-1].W, flip_filters=not transition_layers[-1].flip_filters)    
    c1 = SliceLayer(c2a, indices=slice(0,args["nf"][1]-(args["k"]*args["L"])), axis=1)
    c1a = TransposedConv2DLayer(c1, transition_layers[-2].input_shape[1],
             transition_layers[-2].filter_size, stride=transition_layers[-2].stride, crop=transition_layers[-2].pad,
             W=transition_layers[-2].W, flip_filters=not transition_layers[-2].flip_filters)
    c1a = PadLayer(c1a, width=[(1,0),(1,0)])
    c0 = SliceLayer(c1a, indices=slice(0,args["nf"][2]-(args["k"]*args["L"])), axis=1)
    c0a = TransposedConv2DLayer(c0, transition_layers[-3].input_shape[1],
             transition_layers[-3].filter_size, stride=transition_layers[-3].stride, crop=transition_layers[-3].pad,
             W=transition_layers[-3].W, flip_filters=not transition_layers[-3].flip_filters)
    c0a = PadLayer(c0a, width=[(1,0),(1,0)])    

    # todo: add ladder??
    return c0a, None


def climate_test_1_3d(args={"sigma":0.}):
    conv = InputLayer((None,16,8,96,96))
    conv = GaussianNoiseLayer(conv, sigma=args["sigma"])
    conv = dnn.Conv3DDNNLayer(conv, num_filters=128, filter_size=(4,5,5), stride=(1,2,2), nonlinearity=rectify)
    conv = dnn.Conv3DDNNLayer(conv, num_filters=256, filter_size=(3,5,5), stride=(1,2,2), nonlinearity=rectify)
    conv = dnn.Conv3DDNNLayer(conv, num_filters=512, filter_size=(3,5,5), stride=(1,2,2), nonlinearity=rectify)
    if "bottleneck" in args:
        conv = DenseLayer(conv, num_units=args["bottleneck"], nonlinearity=rectify)
    l_out = conv
    print_network(l_out)
    final_out, decoder_layers, encoder_layers = get_encoder_and_decoder(l_out)
    ladder = []
    for a,b in zip(decoder_layers, encoder_layers[::-1]):
        ladder += [a,b]
    return final_out, ladder

        
def massive_1_deep(args):
    conv = InputLayer((None,16,128,128))
    conv = GaussianNoiseLayer(conv, args["sigma"])
    for i in range(0,1):
        conv = Conv2DLayer(conv, num_filters=1024, filter_size=9, stride=1, nonlinearity=args["nonlinearity"])
        conv = MaxPool2DLayer(conv, pool_size=5, stride=4)
    #conv = DenseLayer(conv, num_units=1024, nonlinearity=args["nonlinearity"])
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    for layer in get_all_layers(conv):
        print layer, layer.output_shape
    print count_params(layer)
    return conv    

def autoencoder_basic(args={"f":32, "d":4096, "tied":True}):
    conv = InputLayer((None,16,128,128))
    conv = GaussianNoiseLayer(conv, args["sigma"])
    for i in range(0, 4):
        conv = Conv2DLayer(conv, num_filters=(i+1)*args["f"], filter_size=3, nonlinearity=args["nonlinearity"])
        #if i != 3:
        conv = MaxPool2DLayer(conv, pool_size=2)
    # coding layer
    if "d" in args:
        conv = DenseLayer(conv, num_units=args["d"], nonlinearity=args["nonlinearity"])
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        if args["tied"]:
            conv = InverseLayer(conv, layer)
        else:
            conv = make_inverse(conv, layer)
    for layer in get_all_layers(conv):
        print layer, layer.output_shape
    print count_params(layer)
    return conv

def autoencoder_basic_double_up(args={"f":32, "d":4096}):
    conv = InputLayer((None,16,128,128))
    conv = GaussianNoiseLayer(conv, args["sigma"])
    for i in range(0, 4):
        for j in range(2):
            conv = Conv2DLayer(conv, num_filters=(i+1)*args["f"], filter_size=3, nonlinearity=args["nonlinearity"])
        #if i != 3:
        conv = MaxPool2DLayer(conv, pool_size=2)
    # coding layer
    if "d" in args:
        conv = DenseLayer(conv, num_units=args["d"], nonlinearity=args["nonlinearity"])
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    for layer in get_all_layers(conv):
        print layer, layer.output_shape
    print count_params(layer)
    return conv

def autoencoder_basic_double_up_512(args):
    conv = InputLayer((None,16,128,128))
    conv = GaussianNoiseLayer(conv, args["sigma"])
    for i in range(0, 4):
        for j in range(2):
            conv = Conv2DLayer(conv, num_filters=(i+1)*args["f"], filter_size=3, nonlinearity=args["nonlinearity"])
        if i != 3:
            conv = MaxPool2DLayer(conv, pool_size=2)
    for j in range(2):
        conv = Conv2DLayer(conv, num_filters=512, filter_size=3, nonlinearity=args["nonlinearity"])
    # coding layer
    if "d" in args:
        conv = DenseLayer(conv, num_units=args["d"], nonlinearity=args["nonlinearity"])
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    for layer in get_all_layers(conv):
        print layer, layer.output_shape
    print count_params(layer)
    return conv

def autoencoder_basic_double_up_512_stride(args={"f":32, "d":4096}):
    conv = InputLayer((None,16,128,128))
    conv = GaussianNoiseLayer(conv, args["sigma"])
    for i in range(0, 4):
        for j in range(2):
            conv = Conv2DLayer(conv, num_filters=(i+1)*args["f"], filter_size=3, nonlinearity=softplus)
        if i != 3:
            conv = Conv2DLayer(conv, num_filters=(i+1)*args["f"], filter_size=3, stride=2, nonlinearity=softplus)
    for j in range(2):
        conv = Conv2DLayer(conv, num_filters=512, filter_size=3, nonlinearity=softplus)
    # coding layer
    if "d" in args:
        conv = DenseLayer(conv, num_units=args["d"], nonlinearity=softplus)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    for layer in get_all_layers(conv):
        print layer, layer.output_shape
    print count_params(layer)
    return conv


# ------------------------------------------


def full_image_net_1(args):
    conv = InputLayer((None, 16, 768, 1152))
    conv = GaussianNoiseLayer(conv, sigma=args["sigma"])
    encoder_layers = []
    conv = Conv2DLayer(conv, num_filters=128, filter_size=5, stride=2); encoder_layers.append(conv)
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, stride=2); encoder_layers.append(conv)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2); encoder_layers.append(conv)
    conv = Conv2DLayer(conv, num_filters=768, filter_size=5, stride=2); encoder_layers.append(conv)
    conv = Conv2DLayer(conv, num_filters=1024, filter_size=5, stride=2); encoder_layers.append(conv)
    conv = Conv2DLayer(conv, num_filters=1280, filter_size=5, stride=2); encoder_layers.append(conv)
    decoder_layers = []
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
        decoder_layers.append(conv)
    l_out = conv

    #num_events = 1
    #l_out_s = DenseLayer(l_out, num_units=num_events*(768/4)*(1152/4))
    
    # bug: get_encoder_and_decoder doesn't work for this
    #final_out, decoder_layers, encoder_layers = get_encoder_and_decoder(l_out)
    ladder = []
    for a,b in zip(decoder_layers, encoder_layers[::-1]):
        ladder += [a,b]
    return l_out, ladder


def full_image_net_1_3d(args):
    conv = InputLayer((None, 16, 8, 768, 1152))
    conv = GaussianNoiseLayer(conv, sigma=args["sigma"])
    encoder_layers = []
    conv = dnn.Conv3DDNNLayer(conv, num_filters=128, filter_size=(3,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    conv = dnn.Conv3DDNNLayer(conv, num_filters=256, filter_size=(2,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    conv = dnn.Conv3DDNNLayer(conv, num_filters=512, filter_size=(2,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    conv = dnn.Conv3DDNNLayer(conv, num_filters=768, filter_size=(2,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    conv = dnn.Conv3DDNNLayer(conv, num_filters=1024, filter_size=(2,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    conv = dnn.Conv3DDNNLayer(conv, num_filters=1280, filter_size=(2,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    decoder_layers = []
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
        decoder_layers.append(conv)
    l_out = conv    
    # bug: get_encoder_and_decoder doesn't work for this
    #final_out, decoder_layers, encoder_layers = get_encoder_and_decoder(l_out)
    ladder = []
    for a,b in zip(decoder_layers, encoder_layers[::-1]):
        ladder += [a,b]
    print_network(l_out)
    return l_out, ladder


def full_image_net_1_3d_v2(args):
    conv = InputLayer((None, 16, 8, 768, 1152))
    conv = GaussianNoiseLayer(conv, sigma=args["sigma"])
    encoder_layers = []
    conv = dnn.Conv3DDNNLayer(conv, num_filters=128, filter_size=(3,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    conv = dnn.Conv3DDNNLayer(conv, num_filters=256, filter_size=(2,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    conv = dnn.Conv3DDNNLayer(conv, num_filters=512, filter_size=(2,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    conv = dnn.Conv3DDNNLayer(conv, num_filters=768, filter_size=(2,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    conv = dnn.Conv3DDNNLayer(conv, num_filters=1024, filter_size=(2,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    decoder_layers = []
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
        decoder_layers.append(conv)
    l_out = conv    
    # bug: get_encoder_and_decoder doesn't work for this
    #final_out, decoder_layers, encoder_layers = get_encoder_and_decoder(l_out)
    ladder = []
    for a,b in zip(decoder_layers, encoder_layers[::-1]):
        ladder += [a,b]
    print_network(l_out)
    return l_out, ladder

def full_image_net_1_3d_v3(args):
    conv = InputLayer((None, 16, 8, 768, 1152))
    conv = GaussianNoiseLayer(conv, sigma=args["sigma"])
    encoder_layers = []
    conv = dnn.Conv3DDNNLayer(conv, num_filters=196, filter_size=(3,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    #conv = dnn.Conv3DDNNLayer(conv, num_filters=256, filter_size=(3,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    #conv = dnn.Conv3DDNNLayer(conv, num_filters=512, filter_size=(3,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    #conv = dnn.Conv3DDNNLayer(conv, num_filters=1024, filter_size=(2,5,5), stride=(1,2,2)); encoder_layers.append(conv)
    decoder_layers = []
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
        decoder_layers.append(conv)
    l_out = conv    
    # bug: get_encoder_and_decoder doesn't work for this
    #final_out, decoder_layers, encoder_layers = get_encoder_and_decoder(l_out)
    ladder = []
    for a,b in zip(decoder_layers, encoder_layers[::-1]):
        ladder += [a,b]
    print_network(l_out)
    return l_out, ladder





if __name__ == '__main__':
    l_out, _ = full_image_net_1_3d({"sigma":0.})
    for layer in get_all_layers(l_out):
        print type(layer), layer.output_shape
    print count_params(l_out)
