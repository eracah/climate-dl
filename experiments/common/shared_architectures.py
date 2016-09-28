from lasagne.layers import *
import theano
from theano import tensor as T
import lasagne
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.updates import *
from lasagne.utils import *
from common.helper_fxns import *



def climate_test_1(args={"sigma":0.}):
    conv = InputLayer((None,16,96,96))
    conv = GaussianNoiseLayer(conv, sigma=args["sigma"])
    conv = Conv2DLayer(conv, num_filters=128, filter_size=5, stride=2, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, stride=2, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2, nonlinearity=rectify)
    if "bottleneck" in args:
        conv = DenseLayer(conv, num_units=args["bottleneck"], nonlinearity=rectify)
    l_out = conv
    #print_network(l_out)
    final_out, decoder_layers, encoder_layers = get_encoder_and_decoder(l_out, args)
    print_network(final_out)
    ladder = []
    for a,b in zip(decoder_layers, encoder_layers[::-1]):
        ladder += [a,b]
    return final_out, ladder


def climate_test_dense(args):
    conv_kwargs = dict(num_filters=args['k'], filter_size=3, pad=1, nonlinearity=rectify, W=lasagne.init.HeNormal())
    conv = InputLayer((None,16,96,96))
    conv = GaussianNoiseLayer(conv, sigma=args["sigma"])
    conv = Conv2DLayer(conv, num_filters=args["nf"][0], filter_size=5, stride=2, nonlinearity=rectify)
    conv = make_dense_block(conv, args, conv_kwargs)
    conv = Conv2DLayer(conv, num_filters=args["nf"][1], filter_size=5, stride=2, nonlinearity=rectify)
    conv = make_dense_block(conv, args, conv_kwargs)
    conv = Conv2DLayer(conv, num_filters=args["nf"][2], filter_size=5, stride=2, nonlinearity=rectify)
    conv = make_dense_block(conv, args, conv_kwargs)
    if "bottleneck" in args:
        conv = DenseLayer(conv, num_units=args["bottleneck"], nonlinearity=rectify)
    l_out = conv


    final_out, decoder_layers, encoder_layers = get_encoder_and_decoder_for_dense_net(l_out, args)
    print_network(final_out)
    ladder = []
    for a,b in zip(decoder_layers, encoder_layers[::-1]):
        ladder += [a,b]
    return final_out, ladder


            
def get_encoder_and_decoder_for_dense_net(l_out, args):
    encoder_layers = [layer for layer in get_all_layers(l_out) if isinstance(layer, Conv2DLayer)  ]
    decoder_layers = []

    layer = l_out
    #get the dense layer
    conv = InverseLayer(layer, layer)
    layer=layer.input_layer
    for i in range(3):
        conv, layer = make_inverse_dense_block(conv, layer, args)
        #get the in between conv
        conv = InverseLayer(conv, layer)
        
        #add only the in between conv to decoder layers
        decoder_layers.append(conv)
        layer = layer.input_layer

            
    return conv, decoder_layers, encoder_layers
