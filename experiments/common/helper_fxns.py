from lasagne.layers import *
import theano
from theano import tensor as T
import lasagne
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.updates import *
from lasagne.utils import *
import sys
import logging

def get_logger(out_folder):
        logger=logging.getLogger('trainlog')
        if len(logger.handlers) < 1:
            logger = logging.getLogger('log_train')
            logger.setLevel(logging.INFO)
            fh = logging.FileHandler('%s/results.txt'%(out_folder))
            fh.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            logger.addHandler(ch)
            logger.addHandler(fh)
        return logger

def get_net(net_cfg, args):
    l_out, ladder = net_cfg(args)
    if args["mode"] == "2d":
        X = T.tensor4('X')
    elif args["mode"] == "3d":
        X = T.tensor5('X')

    ladder_output = get_output([l_out] + ladder, X)
    net_out = ladder_output[0]
    # squared error loss is between the first two terms
    loss = squared_error(net_out, X).mean()

    ladder_output = ladder_output[1::]
    
    
    if "ladder" in args:
        sys.stderr.write("using ladder connections for conv\n")        
        for i in range(0, len(ladder_output), 2):
            sys.stderr.write("ladder connection between %s and %s\n" % (str(ladder[i].output_shape), str(ladder[i+1].output_shape)) )
            assert ladder[i].output_shape == ladder[i+1].output_shape
            loss += args["ladder"]*squared_error(ladder_output[i], ladder_output[i+1]).mean()
    
    net_out_det = get_output(l_out, X, deterministic=True)
    # this is deterministic + doesn't have the reg params added to the end
    loss_det = squared_error(net_out_det, X).mean()
    params = get_all_params(l_out, trainable=True)
    lr = theano.shared(floatX(args["learning_rate"]))
    if "optim" not in args:
        updates = nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
    else:
        if args["optim"] == "rmsprop":
            updates = rmsprop(loss, params, learning_rate=lr)
        elif args["optim"] == "adam":
            updates = adam(loss, params, learning_rate=lr)
    
    train_fn = theano.function([X], [loss, loss_det], updates=updates)
    loss_fn = theano.function([X], loss)
    out_fn = theano.function([X], net_out_det)
    return {
        "train_fn": train_fn,
        "loss_fn": loss_fn,
        "out_fn": out_fn,
        "lr": lr,
        "l_out": l_out
    }



def make_dense_block(inp, args, conv_kwargs=dict(filter_size=3, pad=1)):
        conc = inp
        for i in range(args['L']):
            # if "bn_relu" in args:
            #     bn = BatchNormLayer(conc)
            #     bn_relu = NonlinearityLayer(bn, nonlinearity=args['nonlinearity'])
            # else:
            conv = Conv2DLayer(conc, **conv_kwargs)
            conc = ConcatLayer([conc, conv], axis=1)
        return conc

def make_inverse_dense_block(inp, layer, args):
    conc = inp
    inv_layers = get_all_layers(layer)[::-1]
    #3 layers per comp unit and args['L'] units per block
    first_concat_lay = inv_layers[2*args['L']-2]
    print first_concat_lay
    conc = make_concat_inverse(conc,first_concat_lay)
            
    return conc, inv_layers[2*args['L']]


def make_concat_inverse(inp, concat_layer):
    first_input_shape = concat_layer.input_shapes[0][concat_layer.axis]
    return SliceLayer(inp,indices=slice(0,first_input_shape), axis=concat_layer.axis)


def print_network(l_out):
    for layer in get_all_layers(l_out):
        print layer, layer.output_shape
    print count_params(layer)
    



def create_run_dir(custom_rc=False):
    results_dir = os.getcwd() + '/results'
    run_num_file = os.path.join(results_dir, "run_num.txt")
    if not os.path.exists(results_dir):
        print "making results dir"
        os.mkdir(results_dir)

    if not os.path.exists(run_num_file):
        print "making run num file...."
        f = open(run_num_file,'w')
        f.write('0')
        f.close()

    f = open(run_num_file,'r+')

    run_num = int(f.readline()) + 1

    f.seek(0)

    f.write(str(run_num))


    run_dir = os.path.join(results_dir,'run%i'%(run_num))
    os.mkdir(run_dir)
    
    if custom_rc:
        make_custom_config_file(run_dir)
    return run_dir
                    