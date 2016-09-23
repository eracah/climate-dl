import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.layers import dnn
import sys
sys.path.append("..")
import common

l_conv = InputLayer((None,16,8,96,96))
l_conv = dnn.Conv3DDNNLayer(l_conv, num_filters=128, filter_size=(4,5,5), stride=(1,2,2))
l_conv = dnn.Conv3DDNNLayer(l_conv, num_filters=256, filter_size=(3,5,5), stride=(1,2,2))
l_conv = dnn.Conv3DDNNLayer(l_conv, num_filters=512, filter_size=(3,5,5), stride=(1,2,2))
for layer in get_all_layers(l_conv)[::-1]:
    if isinstance(layer, InputLayer):
        break
    l_conv = InverseLayer(l_conv, layer)

for layer in get_all_layers(l_conv):
    print layer, layer.output_shape
print count_params(l_conv)

for x, _ in common.data_iterator(32, "/storeSSD/cbeckham/nersc/big_images/", start_day=1, end_day=1, img_size=96, time_chunks_per_example=8):
    # right now it is: (bs, time, nchannels, height, width)
    # needs to be: (bs, nchannels, time, height, width)
    print x.shape
    x = x.reshape((x.shape[0], x.shape[2], x.shape[1], x.shape[3], x.shape[4]))
    print x.shape
#    break
