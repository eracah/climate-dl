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
from psutil import virtual_memory
from pylab import rcParams
import pdb

import data

def get_net(net_cfg, args):

    # encoder, classification, decoder = net_cfg(...)
    l_out_for_classifier, l_out_for_decimator, l_out_for_decoder = net_cfg(args)
    
    if args["dim"] == "2d":
        X = T.tensor4('X')
        Y = T.tensor4('Y')
    elif args["dim"] == "3d":
        X = T.tensor5('X')
        Y = T.tensor5('Y')

    net_out_for_classifier = get_output(l_out_for_classifier, X) # returns the sigmoid masks
    net_out_for_decimator = get_output(l_out_for_decimator, Y)
    net_out_for_decoder = get_output(l_out_for_decoder, X)

    loss_reconstruction = squared_error(net_out_for_decoder,X).mean()
    loss_segmentation = binary_crossentropy(net_out_for_classifier, net_out_for_decimator).mean()

    if args["mode"] == "segmentation":
        sys.stderr.write("mode = segmentation (loss_combined = loss_segmentation)\n")
        loss_combined = loss_segmentation
    elif args["mode"] == "mixed":
        sys.stderr.write("mode = mixed (loss_combined = loss_segmentation + loss_reconstruction)\n")
        loss_combined = loss_segmentation + loss_reconstruction

    params = get_all_params(l_out_for_decoder, trainable=True) # this needs to be fixed for when we do future prediction
    lr = theano.shared(floatX(args["learning_rate"]))
    
    if "optim" not in args:
        updates = nesterov_momentum(loss_combined, params, learning_rate=lr, momentum=0.9)
    else:
        if args["optim"] == "rmsprop":
            sys.stderr.write("using rmsprop optimisation\n")
            updates = rmsprop(loss_combined, params, learning_rate=lr)
        elif args["optim"] == "adam":
            sys.stderr.write("using adam optimisation\n")
            updates = adam(loss_combined, params, learning_rate=lr)
    
    # todo: fix [loss,loss]
    train_fn = theano.function([X, Y], [loss_reconstruction, loss_segmentation, loss_combined], updates=updates, on_unused_input='warn')
    loss_fn = theano.function([X, Y], [loss_reconstruction, loss_segmentation, loss_combined], on_unused_input='warn')
    classifier_out_fn = theano.function([X], net_out_for_classifier, on_unused_input='warn')
    decoder_out_fn = theano.function([X], net_out_for_decoder, on_unused_input='warn')
    return {
        "train_fn": train_fn,
        "loss_fn": loss_fn,
        "classifier_out_fn": classifier_out_fn,
        "decoder_out_fn": decoder_out_fn,
        "lr": lr,
        "l_out_for_classifier": l_out_for_classifier,
        "l_out_for_decimator": l_out_for_decimator,
        "l_out_for_decoder": l_out_for_decoder
    }

def get_iterator(name, batch_size, data_dir, start_day, end_day, time_chunks_per_example, shuffle):
    # for stl10, 'days' and 'data_dir' does not make
    # any sense
    assert name in ["climate", "segmentation"]
    if name == "climate":
        return data.data_iterator(batch_size, data_dir, start_day=start_day, end_day=end_day, time_chunks_per_example=time_chunks_per_example, shuffle=shuffle)
    elif name =="segmentation":
        return data.segmentation_iterator(start_day=start_day, end_day=end_day, data_dir=data_dir, shuffle=shuffle)
        

def train(cfg,
        num_epochs,
        out_folder,
        sched={},
        batch_size=1,
        model_folder="/storeSSD/cbeckham/nersc/models/",
        tmp_folder="tmp",
        training_days=[1,20],
        validation_days=[345,365],
        time_chunks_per_example=1,
        data_dir="/storeSSD/cbeckham/nersc/big_images/",
        dataset="climate",
        save_images_every=-1,
        resume=None,
        debug=True,
        vis=False):
    
    def plot_reconstruction_image(img_composite, filename):
        for j in range(0,32):
            plt.subplot(8,4,j+1)
            if time_chunks_per_example > 1:
                plt.imshow(img_composite[j][0])
            else:
                plt.imshow(img_composite[j])
            plt.axis('off')
        plt.savefig(filename)
        pyplot.clf()

    def plot_img_mask(img_mask, filename):
        plt.imshow(img_mask[0][1][0], cmap="gray") # index batch size, get hurricane index, get first time step
        plt.axis('off')
        plt.savefig('%s/classifier%i_%i.png' % (out_folder, epoch, nbatches))
        pyplot.clf()
    
    # extract methods
    train_fn, loss_fn, classifier_out_fn, decoder_out_fn, l_out_for_decoder = \
      cfg["train_fn"], cfg["loss_fn"], cfg["classifier_out_fn"], cfg["decoder_out_fn"], cfg["l_out_for_decoder"]
    lr = cfg["lr"]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # results on a per-epoch basis
    f = open("%s/results.txt" % out_folder, "wb")
    # results on a per-minibatch basis
    f_train_raw = open("%s/results_train_raw.txt" % out_folder, "wb")
    f_valid_raw = open("%s/results_valid_raw.txt" % out_folder, "wb")

    headers = ["epoch", "train_loss_reconstruction", "train_loss_segmentation", "train_loss_combined", "valid_loss_reconstruction", "valid_loss_segmentation", "valid_loss_combined", "time"]
    headers_train_raw = ["train_loss_reconstruction", "train_loss_segmentation", "train_loss_combined"]
    headers_valid_raw = ["valid_loss_reconstruction", "valid_loss_segmentation", "valid_loss_combined"]
    
    f.write("%s\n" % ",".join(headers))
    f_train_raw.write("%s\n" % ",".join(headers_train_raw))
    f_valid_raw.write("%s\n" % ",".join(headers_valid_raw))

    print "%s" % (",".join(headers))
    for epoch in range(0, num_epochs):
        t0 = time()
        # learning rate schedule
        if epoch+1 in sched:
            lr.set_value( floatX(sched[epoch+1]) )
            sys.stderr.write("changing learning rate to: %f\n" % sched[epoch+1])
        
            
        train_losses_reconstruction, train_losses_segmentation, train_losses_combined = [], [], []
        valid_losses_reconstruction, valid_losses_segmentation, valid_losses_combined = [], [], []
        first_minibatch = True
        nbatches = 0
        for X_train, y_train in get_iterator(dataset, batch_size, data_dir, start_day=training_days[0], end_day=training_days[1],
                                             time_chunks_per_example=time_chunks_per_example, shuffle=True):
            if first_minibatch:
                X_train_sample = X_train[0:1]
                first_minibatch = False
            
            this_loss_reconstruction, this_loss_segmentation, this_loss_combined = train_fn(X_train, y_train)
            train_losses_reconstruction.append(this_loss_reconstruction)
            train_losses_segmentation.append(this_loss_segmentation)
            train_losses_combined.append(this_loss_combined)
            
            f_train_raw.write("%f,%f,%f\n" % (this_loss_reconstruction, this_loss_segmentation, this_loss_combined))
            f_train_raw.flush()

            nbatches += 1

            if vis and save_images_every != -1 and nbatches % save_images_every == 0:
                # plot the reconstructions
                img_orig = X_train_sample
                img_reconstruct = decoder_out_fn(img_orig)
                img_composite = np.vstack((img_orig[0],img_reconstruct[0]))
                plot_reconstruction_image(img_composite, '%s/reconstruct%i_%i.png' % (out_folder, epoch, nbatches))
                # plot the segment mask
                img_masks = classifier_out_fn(img_orig)
                plot_img_mask(img_masks, '%s/classifier%i_%i.png' % (out_folder, epoch, nbatches))
                #pdb.set_trace()
            
        if debug:
            mem = virtual_memory()
            print mem
            
        # VALIDATION LOOP
        for X_valid, y_valid in get_iterator(dataset, batch_size, data_dir, start_day=validation_days[0], end_day=validation_days[1],
                                              time_chunks_per_example=time_chunks_per_example, shuffle=False):
            
            this_loss_reconstruction, this_loss_segmentation, this_loss_combined = loss_fn(X_valid, y_valid)
            valid_losses_reconstruction.append(this_loss_reconstruction)
            valid_losses_segmentation.append(this_loss_segmentation)
            valid_losses_combined.append(this_loss_combined)
            
            f_valid_raw.write("%f,%f,%f\n" % (this_loss_reconstruction, this_loss_segmentation, this_loss_combined))
            f_valid_raw.flush()

        headers = ["epoch", "train_loss_reconstruction", "train_loss_segmentation", "train_loss_combined", "valid_loss_reconstruction", "valid_loss_segmentation", "valid_loss_combined", "time"]
                    
        # STATISTICS
        time_taken = time() - t0
        stats = (epoch+1,
                np.mean(train_losses_reconstruction),
                np.mean(train_losses_segmentation),
                np.mean(train_losses_combined),
                np.mean(valid_losses_reconstruction),
                np.mean(valid_losses_segmentation),
                np.mean(valid_losses_combined),
                time_taken)
        out_str = "%i,%f,%f,%f,%f,%f,%f,%f" % stats
        print out_str
        f.write("%s\n" % out_str)
        f.flush()
        # save model at each epoch
        if not os.path.exists("%s/%s" % (model_folder, out_folder)):
            os.makedirs("%s/%s" % (model_folder, out_folder))
        with open("%s/%s/%i.model" % (model_folder, out_folder, epoch), "wb") as g:
            pickle.dump( get_all_param_values(cfg["l_out_for_decoder"]), g, pickle.HIGHEST_PROTOCOL )
