
import matplotlib; matplotlib.use("agg")


from lasagne.layers import *



#!/usr/bin/env python


"""
Creates a DenseNet model in Lasagne, following the paper
"Densely Connected Convolutional Networks"
by Gao Huang, Zhuang Liu, Kilian Q. Weinberger, 2016.
https://arxiv.org/abs/1608.06993

Author: Jan Schlüter
"""

import lasagne
from lasagne.layers import (InputLayer, Conv2DLayer, ConcatLayer, DenseLayer,
                            DropoutLayer, BatchNormLayer, Pool2DLayer,
                            GlobalPoolLayer, NonlinearityLayer)
from lasagne.nonlinearities import rectify, softmax

def build_densenet(input_shape=(None, 3, 32, 32), input_var=None, classes=10,
                   depth=40, first_output=16, growth_rate=12, num_blocks=3,
                   dropout=0):
    """
    Creates a DenseNet model in Lasagne.
    
    Parameters
    ----------
    input_shape : tuple
        The shape of the input layer, as ``(batchsize, channels, rows, cols)``.
        Any entry except ``channels`` can be ``None`` to indicate free size.
    input_var : Theano expression or None
        Symbolic input variable. Will be created automatically if not given.
    classes : int
        The number of classes of the softmax output.
    depth : int
        Depth of the network. Must be ``num_blocks * n + 1`` for some ``n``.
        (Parameterizing by depth rather than n makes it easier to follow the
        paper.)
    first_output : int
        Number of channels of initial convolution before entering the first
        dense block, should be of comparable size to `growth_rate`.
    growth_rate : int
        Number of feature maps added per layer.
    num_blocks : int
        Number of dense blocks (defaults to 3, as in the original paper).
    dropout : float
        The dropout rate. Set to zero (the default) to disable dropout.
    batchsize : int or None
        The batch size to build the model for, or ``None`` (the default) to
        allow any batch size.
    inputsize : int, tuple of int or None        
    
    Returns
    -------
    network : Layer instance
        Lasagne Layer instance for the output layer.
    
    References
    ----------
    .. [1] Gao Huang et al. (2016):
           Densely Connected Convolutional Networks.
           https://arxiv.org/abs/1608.06993
    """
    if (depth - 1) % num_blocks != 0:
        raise ValueError("depth must be num_blocks * n + 1 for some n")
    
    # input and initial convolution
    network = InputLayer(input_shape, input_var, name='input')
    network = Conv2DLayer(network, first_output, 3, pad='same',
                          W=lasagne.init.HeNormal(gain='relu'),
                          b=None, nonlinearity=None, name='pre_conv')
    if dropout:
        network = DropoutLayer(network, dropout)
    # dense blocks with transitions in between
    n = (depth - 1) // num_blocks
    for b in range(num_blocks):
        network = dense_block(network, n - 1, growth_rate, dropout,
                              name_prefix='block%d' % (b + 1))
        if b < num_blocks - 1:
            network = transition(network, dropout,
                                 name_prefix='block%d_trs' % (b + 1))
    # post processing until prediction
    network = BatchNormLayer(network, name='post_bn')
    network = NonlinearityLayer(network, nonlinearity=rectify,
                                name='post_relu')
    network = GlobalPoolLayer(network, name='post_pool')
    network = DenseLayer(network, classes, nonlinearity=softmax,
                         W=lasagne.init.HeNormal(gain=1), name='output')
    return network

def dense_block(network, num_layers, growth_rate, dropout, name_prefix):
    # concatenated 3x3 convolutions
    for n in range(num_layers):
        conv = bn_relu_conv(network, channels=growth_rate,
                            filter_size=3, dropout=dropout,
                            name_prefix=name_prefix + '_l%02d' % (n + 1))
        network = ConcatLayer([network, conv], axis=1,
                              name=name_prefix + '_l%02d_join' % (n + 1))
    return network

def transition(network, dropout, name_prefix):
    # a transition 1x1 convolution followed by avg-pooling
    network = bn_relu_conv(network, channels=network.output_shape[1],
                           filter_size=1, dropout=dropout,
                           name_prefix=name_prefix)
    network = Pool2DLayer(network, 2, mode='average_inc_pad',
                          name=name_prefix + '_pool')
    return network

def bn_relu_conv(network, channels, filter_size, dropout, name_prefix):
    network = BatchNormLayer(network, name=name_prefix + '_bn')
    network = NonlinearityLayer(network, nonlinearity=rectify,
                                name=name_prefix + '_relu')
    network = Conv2DLayer(network, channels, filter_size, pad='same',
                          W=lasagne.init.HeNormal(gain='relu'),
                          b=None, nonlinearity=None,
                          name=name_prefix + '_conv')
    if dropout:
        network = DropoutLayer(network, dropout)
    return network



from viz import draw_to_file



net= build_densenet(input_shape=(None, 1, 28, 28), input_var=None, classes=10,
                   depth=7, first_output=16, growth_rate=3, num_blocks=2,
                   dropout=0)



draw_to_file(get_all_layers(net), "jandense.eps")





