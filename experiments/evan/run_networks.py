
import matplotlib; matplotlib.use("agg")


import sys
import time
import numpy as np
if __name__ == "__main__":
    sys.path.insert(0,'..')
    from common import *
else:
    from ..common import *
from lasagne.layers import *
import theano
import lasagne



import argparse
if any(["jupyter" in arg for arg in sys.argv]):
    sys.argv=sys.argv[:1]


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--train_days', default=1, type=int)
parser.add_argument('--bottleneck', default=512, type=int)
parser.add_argument('-s', '--sigma', default=0.5, type=float)
parser.add_argument( '--batch_size', default=128, type=int)
parser.add_argument( '--num_epochs', default=5000, type=int)
parser.add_argument( '--learn_rate', default=0.1, type=float)
parser.add_argument( '--im_size', default=96, type=int)
parser.add_argument( '--data_dir', default="/project/projectdirs/dasrepo/gordon_bell/climate/data/big_images/")
pargs = parser.parse_args()

args = { "learning_rate": 0.01, "sigma":pargs.sigma, "bottleneck":pargs.bottleneck, 'mode':'2d', 'L':1, 'k':3 }
train_kwargs = dict(
                    data_dir=pargs.data_dir,
                    training_days=[1, pargs.train_days], 
                    validation_days=[364,364], 
                    debug=False,
                    num_epochs=pargs.num_epochs,
                    img_size=pargs.im_size,
                    dataset="climate",
                    step_size=512,
                    time_steps=1,
                    batch_size=pargs.batch_size
                    )

total_train_days = train_kwargs['training_days'][1] - train_kwargs['training_days'][0]  + 1
exp_dirname = '_'.join(map(str,args.values())) + '_' +  str(total_train_days)
net_cfg = get_net(climate_test_dense, args)
train(net_cfg, 
      out_folder="output/" +exp_dirname, model_folder="output/" +exp_dirname, **train_kwargs)





