import sys
sys.path.append("..")
import common
import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import pyplot
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
import architectures


###################

###################
out_folder = "vis/out_test"
#architecture_name = "full_image_net_1_3d"
#architecture_method = getattr(architectures, architecture_name)
args = {"sigma":0.}
fm_idx = 1
#l_out = architecture_method(args)
l_out, _ = architectures.full_image_net_1_3d(args)
model_file = "/storeSSD/cbeckham/nersc/models/output/full_image_1_3d/32.model"
with open(model_file) as f:
    set_all_param_values(l_out, pickle.load(f))
X = T.tensor5('X')
net_out = get_output(l_out, X)
out_fn = theano.function([X], net_out)
###################

tot=0
for x,y in common.data.data_iterator(batch_size=1, time_chunks_per_example=8, img_size=-1, data_dir="/storeSSD/cbeckham/nersc/big_images/", shuffle=False):
    # x is of the shape (1, 8, 16, 768, 1152)
    # reshape it to (1,16,8,768,1152)
    x = np.swapaxes(x, 1, 2)
    # this will also be (1,16,8,768,1152)
    img_out = out_fn(x)
    # get rid of the batch index, and choose a specific feature map
    img_out = img_out[0][fm_idx,:]
    for t in range(0, img_out.shape[0]):
        plt.imshow(img_out[t])
        plt.axis('off')
        filename = "%s/%i/%04d.png" % (out_folder, fm_idx, tot)
        plt.savefig(filename)
        pyplot.clf()
        tot += 1
