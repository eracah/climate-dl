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
import os
import sys
from time import time

# -------------------------------------

def get_net(net_cfg, args):
    l_out = net_cfg(args)
    X = T.tensor4('X')
    net_out = get_output(l_out, X)
    loss = squared_error(net_out, X).mean()
    params = get_all_params(l_out, trainable=True)
    lr = theano.shared(floatX(args["learning_rate"]))
    updates = nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
    #updates = adadelta(loss, params, learning_rate=lr)
    #updates = rmsprop(loss, params, learning_rate=lr)
    train_fn = theano.function([X], loss, updates=updates)
    loss_fn = theano.function([X], loss)
    out_fn = theano.function([X], net_out)
    return {
        "train_fn": train_fn,
        "loss_fn": loss_fn,
        "out_fn": out_fn,
        "lr": lr,
        "l_out": l_out
    }


def autoencoder_basic_32(args):
    conv = InputLayer((None,16,32,32))
    conv = GaussianNoiseLayer(conv, args["sigma"])
    for i in range(0, 3):
        conv = Conv2DLayer(conv, num_filters=(i+1)*16, filter_size=3)
        conv = MaxPool2DLayer(conv, pool_size=2)
    for layer in get_all_layers(conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        conv = InverseLayer(conv, layer)
    for layer in get_all_layers(conv):
        print layer, layer.output_shape
    print count_params(layer)
    return conv


def iterate(X_train, bs=32):
    b = 0
    while True:
        if b*bs >= X_train.shape[0]:
            break
        yield X_train[b*bs:(b+1)*bs]
        b += 1

def train(cfg, data, num_epochs, out_file, sched={}, batch_size=32, model_folder="models"):
    # extract methods
    train_fn, loss_fn, out_fn = cfg["train_fn"], cfg["loss_fn"], cfg["out_fn"]
    # extract data
    npy_train, npy_valid = data
    lr = cfg["lr"]
    with open(out_file, "wb") as f:
        f.write("epoch,avg_train_loss,avg_valid_loss,time\n")
        print "epoch,avg_train_loss,avg_valid_loss,time"
        for epoch in range(0, num_epochs):
            np.random.shuffle(npy_train)
            t0 = time()
            # learning rate schedule
            if epoch+1 in sched:
                lr.set_value( floatX(sched[epoch+1]) )
                sys.stderr.write("changing learning rate to: %f\n" % sched[epoch+1])
            # train set
            train_losses = []
            for npy in npy_train:
                X_train = np.load(npy)
                idxs = [x for x in range(0, X_train.shape[0])]
                np.random.shuffle(idxs)
                X_train = X_train[idxs]
                for X_batch in iterate(X_train, bs=batch_size):
                    train_losses.append(train_fn(X_train))
            # valid set
            valid_losses = []
            for npy in npy_valid:
                X_valid = np.load(npy)
                for X_batch in iterate(X_valid, bs=batch_size):
                    valid_losses.append(loss_fn(X_valid))
            # time
            time_taken = time() - t0
            # print statistics
            print "%i,%f,%f,%f" % (epoch+1, np.mean(train_losses), np.mean(valid_losses), time_taken)
            f.write("%i,%f,%f,%f\n" % (epoch+1, np.mean(train_losses), np.mean(valid_losses), time_taken))
            # save model at each epoch
            with open("%s/%s.model" % (model_folder, out_file), "wb") as g:
                pickle.dump( get_all_param_values(cfg["l_out"]), g, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    if "EXP1" in os.environ:
        args = { "learning_rate": 0.01, "sigma":1. }
        net_cfg = get_net(autoencoder_basic_32, args)
        train_npys = [
            "/data/lisatmp4/beckhamc/climate-dl/fake_data/01.npy"
        ]
        valid_npys = [
            "/data/lisatmp4/beckhamc/climate-dl/fake_data/01.npy"    
        ]
        data = (train_npys, valid_npys)
        train(net_cfg, data, num_epochs=10, out_file="deleteme.txt")
