import sys
import time
if __name__ == "__main__":
    sys.path.insert(0,'..')
    from common import *
else:
    from ..common import *

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

def climate_test_1(args={"sigma":0.}):
    conv = InputLayer((None,16,96,96))
    conv = GaussianNoiseLayer(conv, sigma=args["sigma"])
    conv = Conv2DLayer(conv, num_filters=128, filter_size=5, stride=2, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=256, filter_size=5, stride=2, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=512, filter_size=5, stride=2, nonlinearity=rectify)
    if "bottleneck" in args:
        conv = DenseLayer(conv, num_units=args["bottleneck"], nonlinearity=rectify)
    l_out = conv

    final_out, decoder_layers, encoder_layers = get_encoder_and_decoder(l_out)
    print_network(final_out)
    ladder = []
    for a,b in zip(decoder_layers, encoder_layers[::-1]):
        ladder += [a,b]
    return final_out, ladder


import argparse
if any(["jupyter" in arg for arg in sys.argv]):
    sys.argv=sys.argv[:1]


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--train_days', default=5, type=int)
parser.add_argument('--bottleneck', default=512, type=int)
parser.add_argument('-s', '--sigma', default=0.5, type=float)
parser.add_argument( '--batch_size', default=128, type=int)
parser.add_argument( '--num_epochs', default=5000, type=int)
parser.add_argument( '--learn_rate', default=0.1, type=float)
parser.add_argument( '--im_size', default=96, type=int)
parser.add_argument( '--data_dir', default="/project/projectdirs/dasrepo/gordon_bell/climate/data/big_images/")
pargs = parser.parse_args()

pargs.train_days=1
args = { "learning_rate": 0.01, "sigma":pargs.sigma, "bottleneck":pargs.bottleneck, 'mode':'2d' }
train_kwargs = dict(
                    data_dir=pargs.data_dir,
                    training_days=[1, pargs.train_days], 
                    validation_days=[364,364], 
                    debug=False,
                    num_epochs=pargs.num_epochs,
                    img_size=pargs.im_size,
                    dataset="climate",
                    step_size=20,
                    time_steps=8,
                    batch_size=pargs.batch_size
                    )

total_train_days = train_kwargs['training_days'][1] - train_kwargs['training_days'][0]  + 1
exp_dirname = '_'.join(map(str,args.values())) + '_' +  str(total_train_days)
seed = 1
lasagne.random.set_rng( np.random.RandomState(seed) )
net_cfg = get_net(climate_test_1, args)
train(net_cfg, 
      out_folder="output/" +exp_dirname, model_folder="output/" +exp_dirname, **train_kwargs)