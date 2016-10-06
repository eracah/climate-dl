import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
#from lasagne.layers import dnn
import sys
sys.path.append("..")
import common
from collections import Counter
import numpy as np
from time import time
import pdb

import scipy.misc

batch_size=1
for x, _ in common.data._day_iterator(data_dir="/storeSSD/cbeckham/nersc/big_images/"):
    #for x, _ in common.data_iterator(batch_size, "/storeSSD/cbeckham/nersc/big_images/", start_day=1, end_day=365, img_size=2000, time_chunks_per_example=1):
    # right now it is: (bs, time, nchannels, height, width)
    # needs to be: (bs, nchannels, time, height, width)
    for i in range(0, x.shape[0]):
        for j in range(0,16):
            print x[i].shape
            scipy.misc.imsave("tmp/%i_%i.png" % (i,j), x[i][j])
    break


