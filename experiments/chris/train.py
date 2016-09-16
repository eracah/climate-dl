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
sys.path.append("..")
import common
sys.path.append("stl10/")
import stl10
from psutil import virtual_memory
from pylab import rcParams
rcParams['figure.figsize'] = 15, 20
import pdb
from architectures import *

# -------------------------------------

def get_net(net_cfg, args):
    l_out, ladder = net_cfg(args)            
    X = T.tensor4('X')
    net_out = get_output(l_out, X)
    ladder_output = get_output(ladder, X)
    # squared error loss is between the first two terms
    loss = squared_error(net_out, X).mean()
    sys.stderr.write("main loss between %s and %s\n" % (str(l_out.output_shape), "X") )
    if "ladder" in args:
        sys.stderr.write("using ladder connections for conv\n")        
        for i in range(0, len(ladder_output), 2):
            sys.stderr.write("ladder connection between %s and %s\n" % (str(ladder[i].output_shape), str(ladder[i+1].output_shape)) )
            assert ladder[i].output_shape == ladder[i+1].output_shape
            loss += args["ladder"]*squared_error(ladder_output[i], ladder_output[i+1]).mean()
    net_out_det = get_output(l_out, X, deterministic=True)
    params = get_all_params(l_out, trainable=True)
    lr = theano.shared(floatX(args["learning_rate"]))
    if "optim" not in args:
        updates = nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
    else:
        if args["optim"] == "rmsprop":
            updates = rmsprop(loss, params, learning_rate=lr)
        elif args["optim"] == "adam":
            updates = adam(loss, params, learning_rate=lr)
    #updates = adadelta(loss, params, learning_rate=lr)
    #updates = rmsprop(loss, params, learning_rate=lr)
    train_fn = theano.function([X], loss, updates=updates)
    loss_fn = theano.function([X], loss)
    out_fn = theano.function([X], net_out_det)
    return {
        "train_fn": train_fn,
        "loss_fn": loss_fn,
        "out_fn": out_fn,
        "lr": lr,
        "l_out": l_out
    }


def iterate(X_train, bs=32):
    b = 0
    while True:
        if b*bs >= X_train.shape[0]:
            break
        yield X_train[b*bs:(b+1)*bs]
        b += 1
    
        
def get_iterator(name, batch_size, data_dir, days, img_size):
    # for stl10, 'days' and 'data_dir' does not make
    # any sense
    assert name in ["climate", "stl10"]
    if name == "climate":
        return common.data_iterator(batch_size, data_dir, days=days, img_size=img_size)
    elif name == "stl10":
        return stl10.data_iterator(batch_size)
        
def train(cfg,
        num_epochs,
        out_folder,
        sched={},
        batch_size=128,
        model_folder="/storeSSD/cbeckham/nersc/models/",
        tmp_folder="tmp",
        days=1,
        data_dir="/storeSSD/cbeckham/nersc/big_images/",
        dataset="climate",
        img_size=128,
        resume=None,
        debug=True):
    # extract methods
    train_fn, loss_fn, out_fn, l_out = cfg["train_fn"], cfg["loss_fn"], cfg["out_fn"], cfg["l_out"]
    lr = cfg["lr"]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    #if resume != None:
    #    
    with open("%s/results.txt" % out_folder, "wb") as f:
        f.write("epoch,avg_train_loss,avg_valid_loss,time\n")
        print "epoch,avg_train_loss,avg_valid_loss,time"
        for epoch in range(0, num_epochs):
            t0 = time()
            # learning rate schedule
            if epoch+1 in sched:
                lr.set_value( floatX(sched[epoch+1]) )
                sys.stderr.write("changing learning rate to: %f\n" % sched[epoch+1])
            train_losses = []
            first_minibatch = True
            for X_train, y_train in get_iterator(dataset, batch_size, data_dir, days=days, img_size=img_size):
                if dataset == "climate":
                    # shape is (32, 1, 16, 128, 128), so collapse to a 4d tensor
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[3], X_train.shape[4])
                if first_minibatch:
                    X_train_sample = X_train[0:1]
                    first_minibatch = False
                train_losses.append(train_fn(X_train))
                #pdb.set_trace()                
            if debug:
                mem = virtual_memory()
                print mem
            # DEBUG: visualise the reconstructions
            img_orig = X_train_sample
            img_reconstruct = out_fn(img_orig)
            #pdb.set_trace()
            img_composite = np.vstack((img_orig[0],img_reconstruct[0]))
            if dataset == "climate":
                for j in range(0,32):
                    plt.subplot(8,4,j+1)
                    plt.imshow(img_composite[j])
                    plt.axis('off')
            elif dataset == "stl10":
                for j in range(0,6):
                    plt.subplot(3,2,j+1)
                    plt.imshow(img_composite[j])
                    plt.axis('off')
            plt.savefig('%s/%i.png' % (out_folder, epoch))
            pyplot.clf()
             
            valid_losses = []
            # todo
            #
            #
            # time
            time_taken = time() - t0
            # print statistics
            print "%i,%f,%f,%f" % (epoch+1, np.mean(train_losses), np.mean(valid_losses), time_taken)
            f.write("%i,%f,%f,%f\n" % (epoch+1, np.mean(train_losses), np.mean(valid_losses), time_taken))
            f.flush()
            # save model at each epoch
            if not os.path.exists("%s/%s" % (model_folder, out_folder)):
                os.makedirs("%s/%s" % (model_folder, out_folder))
            with open("%s/%s/%i.model" % (model_folder, out_folder, epoch), "wb") as g:
                pickle.dump( get_all_param_values(cfg["l_out"]), g, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    if "BASIC_TEST_1_DAY" in os.environ:
        # - no noising
        # - somewhat beefy architecture
        # - 1 day only (to keep training fast)
        args = { "learning_rate": 0.01, "sigma":0. }
        net_cfg = get_net(autoencoder_basic, args)
        train(net_cfg, num_epochs=300, batch_size=64, out_file="output/basic_test_1_day.txt", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_RMSPROP" in os.environ:
        # - no work
        args = { "learning_rate": 0.01, "sigma":0., "rmsprop":True, "f":32, "d":4096 }
        net_cfg = get_net(autoencoder_basic_128, args)
        train(net_cfg, num_epochs=300, batch_size=64, out_file="output/basic_test_1_day_rmsprop.txt")
    if "BASIC_TEST_1_DAY_2" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "f":64, "d": 4096 }
        net_cfg = get_net(autoencoder_basic_128, args)
        train(net_cfg, num_epochs=300, batch_size=64, out_file="output/basic_test_1_day_beefier.txt", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_3" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "f":64 }
        net_cfg = get_net(autoencoder_basic_128, args)
        train(net_cfg, num_epochs=300, batch_size=64, out_folder="output/basic_test_1_day_beefier_no_dense", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_4" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "f":64, "d":4096 }
        net_cfg = get_net(autoencoder_basic_128_double_up, args)
        train(net_cfg, num_epochs=300, batch_size=32, out_folder="output/basic_test_1_day_beefier_double_up", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_5" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "f":64, "d":4096 }
        net_cfg = get_net(autoencoder_basic_double_up_512, args)
        train(net_cfg, num_epochs=300, batch_size=32, out_folder="output/basic_test_1_day_beefier_double_up_512", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_5_RELU" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "f":64, "d":4096, "nonlinearity":rectify }
        net_cfg = get_net(autoencoder_basic_double_up_512, args)
        train(net_cfg, num_epochs=300, batch_size=32, out_folder="output/basic_test_1_day_beefier_double_up_512_relu", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_5_SIGMA1" in os.environ:
        args = { "learning_rate": 0.1, "sigma":1., "f":64, "d":4096 }
        net_cfg = get_net(autoencoder_basic_double_up_512, args)
        train(net_cfg, num_epochs=300, batch_size=32, out_folder="output/basic_test_1_day_beefier_double_up_512_sigma1", sched={100:0.001,200:0.0001})

    # ----------
    # using relu
    # ----------

    if "BASIC_TEST_1_DAY_RELU_F64" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":rectify, "f":64, "d":4096 }
        net_cfg = get_net(autoencoder_basic, args)
        train(net_cfg, num_epochs=300, batch_size=64, out_folder="output/basic_test_1_day_relu", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_RELU_F64_SIGMA1" in os.environ:
        args = { "learning_rate": 0.01, "sigma":1., "nonlinearity":rectify, "f":64, "d":4096 }
        net_cfg = get_net(autoencoder_basic, args)
        train(net_cfg, num_epochs=300, batch_size=64, out_folder="output/basic_test_1_day_relu_sigma1", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_RELU_F64_UT" in os.environ:
        args = { "learning_rate": 0.01, "sigma":1., "nonlinearity":rectify, "f":64, "d":4096, "tied":False }
        net_cfg = get_net(autoencoder_basic, args)
        train(net_cfg, num_epochs=300, batch_size=64, out_folder="output/basic_test_1_day_relu_ut", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_RELU_F128_ND_SIGMA1" in os.environ:
        args = { "learning_rate": 0.01, "sigma":1., "nonlinearity":rectify, "f":128 }
        net_cfg = get_net(autoencoder_basic, args)
        train(net_cfg, num_epochs=300, batch_size=32, out_folder="output/basic_test_1_day_f128_nd_relu_sigma1", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_RELU_F128_ND_UT" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":rectify, "f":128, "tied":False }
        net_cfg = get_net(autoencoder_basic, args)
        train(net_cfg, num_epochs=300, batch_size=32, out_folder="output/basic_test_1_day_f128_nd_relu_ut", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_RELU_F128_ND_UT_SOFTPLUS" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":softplus, "f":128, "tied":False }
        net_cfg = get_net(autoencoder_basic, args)
        train(net_cfg, num_epochs=300, batch_size=32, out_folder="output/basic_test_1_day_f128_nd_relu_ut_softplus", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_RELU_F128_ND_UT_ELU" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":elu, "f":128, "tied":False }
        net_cfg = get_net(autoencoder_basic, args)
        train(net_cfg, num_epochs=300, batch_size=32, out_folder="output/basic_test_1_day_f128_nd_relu_ut_elu", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_RELU_F128_ND_SIGMA1_UT" in os.environ:
        args = { "learning_rate": 0.01, "sigma":1., "nonlinearity":rectify, "f":128, "tied":False }
        net_cfg = get_net(autoencoder_basic, args)
        train(net_cfg, num_epochs=300, batch_size=32, out_folder="output/basic_test_1_day_f128_nd_relu_sigma1_ut", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_RELU_F128_ND_UT_ELU_SIGMA0P1" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0.1, "nonlinearity":elu, "f":128, "tied":False }
        net_cfg = get_net(autoencoder_basic, args)
        train(net_cfg, num_epochs=300, batch_size=32, out_folder="output/basic_test_1_day_f128_nd_relu_ut_elu_sigma0.1", sched={100:0.001,200:0.0001})
    if "BASIC_TEST_1_DAY_RELU_F128_UT_ELU_SIGMA0P1" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0.1, "nonlinearity":elu, "f":128, "d":4096, "tied":False }
        net_cfg = get_net(autoencoder_basic, args)
        train(net_cfg, num_epochs=300, batch_size=32, out_folder="output/basic_test_1_day_f128_relu_ut_elu_sigma0.1", sched={100:0.001,200:0.0001})

    if "DDD" in os.environ:
        args = { "learning_rate": 0.01, "sigma":1., "nonlinearity":rectify }
        net_cfg = get_net(massive_1_deep, args)
        train(net_cfg, num_epochs=300, batch_size=64, out_folder="output/massive_1_deep", sched={100:0.001,200:0.0001})
        
    if "DEBUG" in os.environ:
        print "untied"
        args = { "learning_rate": 0.01, "sigma":1., "nonlinearity":rectify, "f":128, "tied":False }
        net_cfg = get_net(autoencoder_basic, args)
        print "untied"
        args = { "learning_rate": 0.01, "sigma":1., "nonlinearity":rectify, "f":128, "tied":True }
        net_cfg = get_net(autoencoder_basic, args)

    # --------------------------------------------------------
    # what-where autoencoder inspired architecture experiments
    # --------------------------------------------------------

    if "WW_STL10_EXP1" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":rectify, "tied":False }
        net_cfg = get_net(what_where_stl10_arch, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, out_folder="output/ww_stl10_exp1", sched={100:0.001,200:0.0001})
    if "WW_STL10_EXP1_ADAM" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":rectify, "tied":False, "tanh_end":False }
        net_cfg = get_net(what_where_stl10_arch, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, out_folder="output/ww_stl10_exp1_adam", sched={100:0.001,200:0.0001})
    if "WW_STL10_EXP1_ELU" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":elu, "tied":False }
        net_cfg = get_net(what_where_stl10_arch, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, out_folder="output/ww_stl10_exp1_elu", sched={100:0.001,200:0.0001})    
    if "WW_STL10_EXP1_ELU_TANH_END" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":elu, "tied":False, "tanh_end":True }
        net_cfg = get_net(what_where_stl10_arch, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, out_folder="output/ww_stl10_exp1_elu_tanh_end", sched={100:0.001,200:0.0001})    
    if "WW_STL10_EXP1_ELU_TANH_END_ADAM" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":elu, "tied":False, "tanh_end":True, "optim":"adam" }
        net_cfg = get_net(what_where_stl10_arch, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, out_folder="output/ww_stl10_exp1_elu_tanh_end_adam", sched={100:0.001,200:0.0001})    
    if "WW_STL10_EXP2_ELU_TANH_END_ADAM" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":elu, "tied":False, "tanh_end":True, "optim":"adam" }
        net_cfg = get_net(what_where_stl10_arch_beefier, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, out_folder="output/ww_stl10_exp2_elu_tanh_end_adam", sched={100:0.001,200:0.0001})            
    if "WW_STL10_EXP1_ELU_ADAM_LADDER" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":elu, "tied":False, "ladder":1., "optim":"adam" }
        net_cfg = get_net(what_where_stl10_arch, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, out_folder="output/ww_stl10_exp1_elu_adam_ladder", sched={100:0.001,200:0.0001})                            
    if "WW_STL10_EXP1_ELU_ADAM_LADDER2" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":elu, "tied":False, "ladder":0.001, "optim":"adam" }
        net_cfg = get_net(what_where_stl10_arch, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, out_folder="output/ww_stl10_exp1_elu_adam_ladder2", sched={100:0.001,200:0.0001})                            
    if "WW_STL10_EXP1_ELU_ADAM_LADDER3" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":elu, "tied":False, "ladder":0.00001, "optim":"adam" }
        net_cfg = get_net(what_where_stl10_arch, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, out_folder="output/ww_stl10_exp1_elu_adam_ladder3", sched={100:0.001,200:0.0001})                            
    if "WW_STL10_EXP1_ELU_LADDER3" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":elu, "tied":False, "ladder":0.00001 }
        net_cfg = get_net(what_where_stl10_arch, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, out_folder="output/ww_stl10_exp1_elu_ladder3", sched={100:0.001,200:0.0001})                            

    # ---------

    if "STL10_TEST" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":elu, "tied":False, "input_channels":3 }
        net_cfg = get_net(what_where_stl10_arch, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="stl10", out_folder="output/stl10_test", sched={100:0.001,200:0.0001})                            
    if "STL10_TEST_BS128" in os.environ:
        args = { "learning_rate": 0.1, "sigma":0., "nonlinearity":rectify, "tied":False, "input_channels":3 }
        net_cfg = get_net(what_where_stl10_arch, args)
        train(net_cfg, num_epochs=300, batch_size=128, img_size=96, dataset="stl10", out_folder="output/stl10_test_bs128", sched={100:0.001,200:0.0001})
    if "STL10_TEST_BS128_BN512" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":rectify, "tied":False, "input_channels":3, "bottleneck":512 }
        net_cfg = get_net(what_where_stl10_arch, args)
        train(net_cfg, num_epochs=300, batch_size=128, img_size=96, dataset="stl10", out_folder="output/stl10_test_bs128_bn512")

    # #######################################################################################################
    # I had to run some experiments on STL-10 since I couldn't get anything reasonable on the climate data.
    # Because I had to luck reproducing the autoencoder architecture in what-where, I used this arch instead:
    # https://arxiv.org/pdf/1511.06409.pdf (section 3.2)
    # #######################################################################################################
        
    if "DELETEME4" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., }
        net_cfg = get_net(stl10_test_4, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="stl10", out_folder="output/stl_test_4")
    if "DELETEME4_SIGMA5" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0.5, }
        net_cfg = get_net(stl10_test_4, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="stl10", out_folder="output/stl_test_4_sigma0.5")
    if "DELETEME4_BOTTLENECK" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "bottleneck":512 }
        net_cfg = get_net(stl10_test_4, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="stl10", out_folder="output/stl_test_4_bottleneck512")
    if "DELETEME4_BOTTLENECK_SIGMA5" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0.5, "bottleneck":512 }
        net_cfg = get_net(stl10_test_4, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="stl10", out_folder="output/stl_test_4_bottleneck512_sigma5")

    # ##################################
    # Ok, let's try this s--t on climate
    # ##################################
        
    if "CLIMATE_BOTTLENECK512" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "bottleneck":512 }
        net_cfg = get_net(climate_test_1, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="climate", out_folder="output/climate_bottleneck512")        
    if "CLIMATE_BOTTLENECK512_SIGMA05" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0.5, "bottleneck":512 }
        net_cfg = get_net(climate_test_1, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="climate", out_folder="output/climate_bottleneck512_sigma0.5")
    if "CLIMATE_BOTTLENECK512_SIGMA10" in os.environ:
        args = { "learning_rate": 0.01, "sigma":1., "bottleneck":512 }
        net_cfg = get_net(climate_test_1, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="climate", out_folder="output/climate_bottleneck512_sigma1.0")

    # #######################
    # Try 5 days instead of 1
    # #######################
        
    if "CLIMATE_5DAY_BOTTLENECK512" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "bottleneck":512 }
        net_cfg = get_net(climate_test_1, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="climate", days=5, out_folder="output/climate_5day_bottleneck512")

        
    if "STL10_TEST_BS128_ADAM" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":elu, "tied":False, "input_channels":3, "optim":"adam" }
        net_cfg = get_net(what_where_stl10_arch, args)
        train(net_cfg, num_epochs=300, batch_size=128, img_size=96, dataset="stl10", out_folder="output/stl10_test_bs128_adam", sched={100:0.001,200:0.0001})                            
    
        
        
    """
    for epoch in range(10):
        for X_train, y_train in common.data_iterator(128, "/storeSSD/cbeckham/nersc/big_images/", days=1):
            mem = virtual_memory()
            print mem
            pass
    """
    
