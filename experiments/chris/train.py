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
import gan
import architectures

# -------------------------------------

def get_net(net_cfg, args):
    
    l_out, ladder = net_cfg(args)
    if args["dim"] == "2d":
        X = T.tensor4('X')
        Y = T.tensor4('Y')
    elif args["dim"] == "3d":
        X = T.tensor5('X')
        Y = T.tensor5('Y')
    #net_out = get_output(l_out, X)
    ladder_output = get_output([l_out] + ladder, X)
    net_out = ladder_output[0]
    
    # squared error loss is between the first two terms
    if args["mode"] == "autoencoder":
        sys.stderr.write("mode = autoencoder\n")
        loss = squared_error(net_out, X).mean()
    elif args["mode"] == "classification":
        sys.stderr.write("mode = classification\n")
        loss = squared_error(net_out, Y).mean()
    
    sys.stderr.write("main loss between %s and %s\n" % (str(l_out.output_shape), "X") )
    ladder_output = ladder_output[1::]
    if "ladder" in args:
        sys.stderr.write("using ladder connections for conv\n")        
        for i in range(0, len(ladder_output), 2):
            sys.stderr.write("ladder connection between %s and %s\n" % (str(ladder[i].output_shape), str(ladder[i+1].output_shape)) )
            assert ladder[i].output_shape == ladder[i+1].output_shape
            loss += args["ladder"]*squared_error(ladder_output[i], ladder_output[i+1]).mean()
    net_out_det = get_output(l_out, X, deterministic=True)

    # this is deterministic + doesn't have the reg params added to the end
    if args["mode"] == "autoencoder":
        loss_det = squared_error(net_out_det, X).mean()
    elif args["mode"] == "classification":
        loss_det = squared_error(net_out_det, Y).mean()

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

    # this function takes in an X and Y, and for the case of autoencoder,
    # we only really use X, but for the classification part, we actually
    # use X and Y
    train_fn = theano.function([X, Y], [loss, loss_det], updates=updates, on_unused_input='warn')
    loss_fn = theano.function([X, Y], loss, on_unused_input='warn')
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
    

def get_iterator(name, batch_size, data_dir, start_day, end_day, img_size, time_chunks_per_example, shuffle):
    # for stl10, 'days' and 'data_dir' does not make
    # any sense
    assert name in ["climate", "climate_classification", "stl10"]
    if name == "climate":
        return common.data_iterator(batch_size, data_dir, start_day=start_day, end_day=end_day, img_size=img_size, time_chunks_per_example=time_chunks_per_example, shuffle=shuffle)
    elif name =="climate_classification":
        sys.stderr.write("warning: batch_size, img_size, time_chunks_per_example are ignored for climate_classification\n")
        return common.data.consecutive_day_iterator_3d(start_day=start_day, end_day=end_day, data_dir=data_dir, shuffle=shuffle)
    #def consecutive_day_iterator_3d(start_day, end_day, data_dir="/project/projectdirs/dasrepo/gordon_bell/climate/data/big_images/", shuffle=False)
    elif name == "stl10":
        return stl10.data_iterator(batch_size)
        

def train(cfg,
        num_epochs,
        out_folder,
        sched={},
        batch_size=128,
        model_folder="/storeSSD/cbeckham/nersc/models/",
        tmp_folder="tmp",
        training_days=[1,20],
        validation_days=[345,365],
        time_chunks_per_example=1,
        data_dir="/storeSSD/cbeckham/nersc/big_images/",
        dataset="climate",
        img_size=128,
        save_images_every=-1,
        resume=None,
        debug=True):
    
    def prep_batch(X_batch):
        if dataset in ["climate", "climate_classification"]:
            if time_chunks_per_example == 1:
                # shape is (32, 1, 16, 128, 128), so collapse to a 4d tensor
                X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[2], X_batch.shape[3], X_batch.shape[4])
            else:
                # right now it is: (bs, time, nchannels, height, width)
                # needs to be: (bs, nchannels, time, height, width)
                # BUG: this does not do what I think it does
                #X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[2], X_batch.shape[1], X_batch.shape[3], X_batch.shape[4])
                X_batch = np.swapaxes(X_batch, 1, 2)
        else:
            pass # nothing needs to be done for stl-10
        return X_batch

    def plot_image(img_composite, filename):
        if dataset in ["climate", "climate_classification"]:
            for j in range(0,32):
                plt.subplot(8,4,j+1)
                if time_chunks_per_example > 1:
                    pdb.set_trace()
                    plt.imshow(img_composite[j][0])
                else:
                    plt.imshow(img_composite[j])
                plt.axis('off')
        elif dataset == "stl10":
            for j in range(0,6):
                plt.subplot(3,2,j+1)
                plt.imshow(img_composite[j])
                plt.axis('off')
        plt.savefig(filename)
        pyplot.clf()
        
    # extract methods
    train_fn, loss_fn, out_fn, l_out = cfg["train_fn"], cfg["loss_fn"], cfg["out_fn"], cfg["l_out"]
    lr = cfg["lr"]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # results on a per-epoch basis
    f = open("%s/results.txt" % out_folder, "wb")
    # results on a per-minibatch basis
    f_train_raw = open("%s/results_train_raw.txt" % out_folder, "wb")
    f_valid_raw = open("%s/results_valid_raw.txt" % out_folder, "wb")

    headers = ["epoch", "avg_train_loss", "avg_train_loss_det", "avg_valid_loss", "time"]
    headers_train_raw = ["train_loss", "train_loss_det"]
    headers_valid_raw = ["valid_loss"]
    
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
        train_losses = []
        train_losses_det = []
        valid_losses = []
        
        first_minibatch = True
        # TRAINING LOOP
        nbatches = 0
        for X_train, y_train in get_iterator(dataset, batch_size, data_dir, start_day=training_days[0], end_day=training_days[1],
                                             img_size=img_size, time_chunks_per_example=time_chunks_per_example, shuffle=True):
            X_train = prep_batch(X_train)
            
            if first_minibatch:
                X_train_sample = X_train[0:1]
                first_minibatch = False

            # if we're doing the autoencoder, then y can simply be
            # anything, since we only use X in the loss,
            # otherwise if it's classification we use the
            # y_train that is returned by the iterator
            if dataset != "climate_classification":
                y_train = np.zeros(X_train.shape, dtype=X_train.dtype)
            else:
                y_train = prep_batch(y_train)
            
            this_loss, this_loss_det = train_fn(X_train, y_train)
            train_losses.append(this_loss)
            train_losses_det.append(this_loss_det)
            f_train_raw.write("%f,%f\n" % (this_loss, this_loss_det))
            f_train_raw.flush()

            nbatches += 1

            if save_images_every != -1 and nbatches % save_images_every == 0:
                img_orig = X_train_sample
                #fake_y = np.zeros(img_orig.shape, dtype=img_orig.dtype)
                img_reconstruct = out_fn(img_orig)
                img_composite = np.vstack((img_orig[0],img_reconstruct[0]))
                plot_image(img_composite, '%s/%i_%i.png' % (out_folder, epoch, nbatches))
            
        if debug:
            mem = virtual_memory()
            print mem
            
        # VALIDATION LOOP
        for X_valid, y_valid in get_iterator(dataset, batch_size, data_dir, start_day=validation_days[0], end_day=validation_days[1],
                                              img_size=img_size, time_chunks_per_example=time_chunks_per_example, shuffle=False):
            X_valid = prep_batch(X_valid)

            # if we're doing the autoencoder, then y can simply be
            # anything, since we only use X in the loss,
            # otherwise if it's classification we use the
            # y_train that is returned by the iterator
            if dataset != "climate_classification":
                y_valid = np.zeros(X_valid.shape, dtype=X_valid.dtype)
            else:
                y_valid = prep_batch(y_valid)
            
            this_loss = loss_fn(X_valid, y_valid)
            valid_losses.append(this_loss)
            f_valid_raw.write("%f\n" % this_loss)
            f_valid_raw.flush()
            
        # DEBUG: visualise the reconstructions
        img_orig = X_train_sample
        #fake_y = np.zeros(img_orig.shape, dtype=img_orig.dtype) # hacky!!
        img_reconstruct = out_fn(img_orig)
        img_composite = np.vstack((img_orig[0],img_reconstruct[0]))
        plot_image(img_composite, '%s/%i.png' % (out_folder, epoch))
        
        # STATISTICS
        time_taken = time() - t0
        print "%i,%f,%f,%f,%f" % (epoch+1, np.mean(train_losses), np.mean(train_losses_det), np.mean(valid_losses), time_taken)
        f.write("%i,%f,%f,%f,%f\n" % (epoch+1, np.mean(train_losses), np.mean(train_losses_det), np.mean(valid_losses), time_taken))
        f.flush()
        # save model at each epoch
        if not os.path.exists("%s/%s" % (model_folder, out_folder)):
            os.makedirs("%s/%s" % (model_folder, out_folder))
        with open("%s/%s/%i.model" % (model_folder, out_folder, epoch), "wb") as g:
            pickle.dump( get_all_param_values(cfg["l_out"]), g, pickle.HIGHEST_PROTOCOL )







def get_net_gan(network_cfg,args={}):
    cnn_model_dict = network_cfg(args)

    # TODO: right now, just uses sgd without nesterov momentum

    print "args"
    print args
    
    print('compiling theano functions for training')
    print('  encoder/decoder')
    encoder_decoder_update = gan.create_encoder_decoder_func(
        cnn_model_dict, apply_updates=True, learning_rate=args["lr_encoder_decoder"])
    print('  discriminator')
    discriminator_update = gan.create_discriminator_func(
        cnn_model_dict, apply_updates=True, learning_rate=args["lr_discriminator"], coef=args["coef_discriminator"])
    print('  generator')
    generator_update = gan.create_generator_func(
        cnn_model_dict, apply_updates=True, learning_rate=args["lr_generator"], coef=args["coef_generator"])

    print('compiling theano functions for testing')
    print('  encoder/decoder')
    encoder_decoder_test = gan.create_encoder_decoder_func(
        cnn_model_dict, apply_updates=False)
    print('  discriminator')
    discriminator_test = gan.create_discriminator_func(
        cnn_model_dict, apply_updates=False)
    print('  generator')
    generator_test = gan.create_generator_func(
        cnn_model_dict, apply_updates=False)
    print('  reconstruction')
    reconstruction_test = gan.get_reconstruction(
        cnn_model_dict)
    print('  encoder func')
    encoder_test = gan.create_encoder_func(cnn_model_dict)

    return {
        "encoder_decoder_update":encoder_decoder_update,
        "discriminator_update":discriminator_update,
        "generator_update":generator_update,
        "encoder_decoder_test":encoder_decoder_test,
        "discriminator_test":discriminator_test,
        "reconstruction_test":reconstruction_test,
        "encoder_test":encoder_test,
        "l_out":cnn_model_dict["l_decoder_out"],
        "code_size":cnn_model_dict["l_encoder_out"].output_shape[1]
    }
    
def extract_hidden_codes(
        cfg,
        model,
        img_size,
        days,
        out_file,
        data_dir="/storeSSD/cbeckham/nersc/big_images/",
        time_chunks_per_example=1,
        dataset="climate"):
    encoder_test = cfg["encoder_test"]
    l_out = cfg["l_out"]
    with open(model) as g:
        set_all_param_values(l_out, pickle.load(g))
    with open(out_file, "wb") as f:
        for X_batch, y_batch in get_iterator(dataset, 1, data_dir, start_day=days[0], end_day=days[1],
                img_size=img_size, time_chunks_per_example=time_chunks_per_example):
            this_encoding = [ str(elem) for elem in encoder_test(X_batch[0]).flatten().tolist() ]
            f.write(",".join(this_encoding) + "\n")

from collections import Counter
            
def test_classes():
    classes = []
    t0 = time()
    for X_batch, y_batch in get_iterator(name="climate", batch_size=1, data_dir="/storeSSD/cbeckham/nersc/big_images/", start_day=1, end_day=2,
        img_size=96, time_chunks_per_example=1):
        classes.append(y_batch[0])
    ct = Counter(classes)
    print time()-t0
    pdb.set_trace()
                    

def train_gan(
        cfg,
        num_epochs,
        out_folder,
        sched={},
        batch_size=128,
        model_folder="/storeSSD/cbeckham/nersc/models/",
        tmp_folder="tmp",
        training_days=[1,20],
        validation_days=[345,365],
        time_chunks_per_example=1,
        data_dir="/storeSSD/cbeckham/nersc/big_images/",
        dataset="climate",
        img_size=128,
        ignore_gan_part=False,
        debug_terminate=False,
        resume=None,
        debug=True):
    
    def prep_batch(X_batch):
        if dataset == "climate":
            if time_chunks_per_example == 1:
                # shape is (32, 1, 16, 128, 128), so collapse to a 4d tensor
                X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[2], X_batch.shape[3], X_batch.shape[4])
            else:
                # right now it is: (bs, time, nchannels, height, width)
                # needs to be: (bs, nchannels, time, height, width)
                X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[2], X_batch.shape[1], X_batch.shape[3], X_batch.shape[4])
        else:
            pass # nothing needs to be done for stl-10
        return X_batch

    def plot_image(img_composite):
        if dataset == "climate":
            for j in range(0,32):
                plt.subplot(8,4,j+1)
                if time_chunks_per_example > 1:
                    plt.imshow(img_composite[j][0])
                else:
                    plt.imshow(img_composite[j])
                plt.axis('off')
        elif dataset == "stl10":
            for j in range(0,6):
                plt.subplot(3,2,j+1)
                plt.imshow(img_composite[j])
                plt.axis('off')
        plt.savefig('%s/%i.png' % (out_folder, epoch))
        pyplot.clf()

    # load config
    encoder_decoder_update = cfg["encoder_decoder_update"]
    discriminator_update = cfg["discriminator_update"]
    generator_update = cfg["generator_update"]
    encoder_decoder_test = cfg["encoder_decoder_test"]
    discriminator_test = cfg["discriminator_test"]
    reconstruction_test = cfg["reconstruction_test"]
    l_out = cfg["l_out"]

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    if debug:
        print "code size:", cfg["code_size"]

    if resume != None:
        sys.stderr.write("resuming training...\n")
        f = open("%s/results.txt" % out_folder, "ab")
        with open(resume) as g:
            set_all_param_values(l_out, pickle.load(g))
    else:
        f = open("%s/results.txt" % out_folder, "wb")

    #lr = cfg["lr"]
    headers = [
        "epoch",
        "train_reconstruction_loss",
        "train_discriminative_loss",
        "train_generative_loss",
        "valid_reconstruction_loss",
        "valid_discriminative_loss",
        "valid_generative_loss",
        "time"
    ]
    f.write("%s\n" % ",".join(headers))
    print "%s" % (",".join(headers))
    for epoch in range(0, num_epochs):
        t0 = time()
        # learning rate schedule
        if epoch+1 in sched:
            # TODO
            #lr.set_value( floatX(sched[epoch+1]) )
            #sys.stderr.write("changing learning rate to: %f\n" % sched[epoch+1])
            pass

        # TRAINING LOOP
        train_reconstruction_losses = []
        train_discriminative_losses = []
        train_generative_losses = []
        first_minibatch = True
        for X_train_batch, y_train_batch in get_iterator(dataset, batch_size, data_dir, start_day=training_days[0], end_day=training_days[1],
                                             img_size=img_size, time_chunks_per_example=time_chunks_per_example):
            X_train_batch = prep_batch(X_train_batch)
            if first_minibatch:
                X_train_sample = X_train_batch[0:1]
                first_minibatch = False

            # 1.) update the encoder/decoder to min. reconstruction loss
            train_batch_reconstruction_loss = encoder_decoder_update(X_train_batch)
            train_reconstruction_losses.append(train_batch_reconstruction_loss)

            if not ignore_gan_part:

                # sample from p(z)
                pz_train_batch = np.random.uniform(low=-2, high=2, size=(X_train_batch.shape[0], cfg["code_size"])).astype(np.float32)

                # 2.) update discriminator to separate q(z|x) from p(z)
                train_batch_discriminative_loss = discriminator_update(X_train_batch, pz_train_batch)

                # 3.) update generator to output q(z|x) that mimic p(z)
                train_batch_generative_loss = generator_update(X_train_batch)

                train_discriminative_losses.append(train_batch_discriminative_loss)
                train_generative_losses.append(train_batch_generative_loss)

            if debug_terminate:
                break
                
        train_reconstruction_losses_mean = np.mean(train_reconstruction_losses)
        train_discriminative_losses_mean = np.mean(train_discriminative_losses)
        train_generative_losses_mean = np.mean(train_generative_losses)

        # VALIDATION LOOP
        valid_reconstruction_losses = []
        valid_discriminative_losses = []
        valid_generative_losses = []
        for X_valid_batch, y_valid_batch in get_iterator(dataset, batch_size, data_dir, start_day=validation_days[0], end_day=validation_days[1],
                                              img_size=img_size, time_chunks_per_example=time_chunks_per_example):
            X_valid_batch = prep_batch(X_valid_batch)
    
            # 1.) evaluate the encoder/decoder on the valid
            valid_reconstruction_losses.append(encoder_decoder_test(X_valid_batch))

            if not ignore_gan_part:
                
                # sample from p(z)
                pz_valid_batch = np.random.uniform(low=-2, high=2, size=(X_valid_batch.shape[0], cfg["code_size"])).astype(np.float32)

                # 2.) update discriminator to separate q(z|x) from p(z)
                valid_batch_discriminative_loss = discriminator_update(X_valid_batch, pz_valid_batch)

                # 3.) update generator to output q(z|x) that mimic p(z)
                valid_batch_generative_loss = generator_update(X_valid_batch)

                valid_discriminative_losses.append(valid_batch_discriminative_loss)
                valid_generative_losses.append(valid_batch_generative_loss)

            if debug_terminate:
                break

        valid_reconstruction_losses_mean = np.mean(valid_reconstruction_losses)
        valid_discriminative_losses_mean = np.mean(valid_discriminative_losses)
        valid_generative_losses_mean = np.mean(valid_generative_losses)
        
        # DEBUG: visualise the reconstructions
        img_orig = X_train_sample
        img_reconstruct = reconstruction_test(img_orig)
        img_composite = np.vstack((img_orig[0],img_reconstruct[0]))
        plot_image(img_composite)

        # STATISTICS
        time_taken = time() - t0
        result_vector = (epoch+1, np.mean(train_reconstruction_losses), np.mean(train_discriminative_losses), np.mean(train_generative_losses),
             np.mean(valid_reconstruction_losses), np.mean(valid_discriminative_losses), np.mean(valid_generative_losses), time_taken)
        print "%i,%f,%f,%f,%f,%f,%f,%f" % result_vector
        f.write("%i,%f,%f,%f,%f,%f,%f,%f\n" % result_vector)
        f.flush()
        # save model at each epoch
        if not os.path.exists("%s/%s" % (model_folder, out_folder)):
            os.makedirs("%s/%s" % (model_folder, out_folder))
        with open("%s/%s/%i.model" % (model_folder, out_folder, epoch), "wb") as g:
            pickle.dump( get_all_param_values(cfg["l_out"]), g, pickle.HIGHEST_PROTOCOL )















                

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
    if "CLIMATE_5DAY_BOTTLENECK512_LADDER0001" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "bottleneck":512, "ladder":0.001 }
        net_cfg = get_net(climate_test_1, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="climate", days=5, out_folder="output/climate_5day_bottleneck512_ladder0001")

    # ##########
    # Dense nets
    # ##########

    if "DENSE_NET_1" in os.environ:
        #k = 3
        #L = 3
        for k in [4,5,6]:
            for L in [3,4,5]:
                seed = 1
                lasagne.random.set_rng( np.random.RandomState(seed) )
                args = { "learning_rate": 0.01, "sigma":0., "bottleneck":512, "nf":[128-(k*L), 256-(k*L), 512-(k*L)], "k":k, "L":L, "mode":"2d" }
                net_cfg = get_net(common.shared_architectures.climate_test_dense, args)
                train(net_cfg, num_epochs=30, batch_size=128, img_size=96, dataset="climate", training_days=[1,2], validation_days=[320,321], out_folder="output/dense_net_1_128-256-512_k%i_l%i.%i" % (k, L, seed))
    if "BASELINE_NO_DENSE_NET" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        args = { "learning_rate": 0.01, "sigma":0., "bottleneck":512, "mode":"2d" }
        net_cfg = get_net(climate_test_1, args)
        train(net_cfg, num_epochs=30, batch_size=128, img_size=96, dataset="climate", training_days=[1,2], validation_days=[320,321], out_folder="output/no_dense_net_1_128-256-512.%i" % (seed))
        
        

    # ###############################
    # Generative adverserial networks
    # ###############################

    # accidentally used relu for q(z|x) in this
    if "GAN_TEST" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "bottleneck":512 }
        net_cfg = get_net_gan(gan.build_model_cnn, args)
        train_gan(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="climate", training_days=[1,2], validation_days=[3,4], out_folder="output/gan_test")

    # repeat of above but relu fix
    if "GAN_TEST_2" in os.environ:
        args = {"learning_rate": 0.01, "sigma":0., "bottleneck":512, "lr_encoder_decoder":0.01, "lr_discriminator":0.001,"lr_generator":0.001}
        net_cfg = get_net_gan(gan.build_model_cnn, args)
        train_gan(
            net_cfg,
            num_epochs=300,
            batch_size=32,
            img_size=96,
            dataset="climate",
            training_days=[1,2],
            validation_days=[3,4],
            out_folder="output/gan_test_fix_relu",
            resume="/storeSSD/cbeckham/nersc/models/output/gan_test_fix_relu/4.model.bak",
        )

    # repeat of above but relu fix
    if "GAN_TEST_2_DAMPEN1" in os.environ:
        args = {"learning_rate": 0.01, "sigma":0., "bottleneck":512, "lr_encoder_decoder":0.01, "lr_discriminator":0.001,"lr_generator":0.001, "coef_discriminator":0.1, "coef_generator":0.1}
        net_cfg = get_net_gan(gan.build_model_cnn, args)
        train_gan(
            net_cfg,
            num_epochs=300,
            batch_size=32,
            img_size=96,
            dataset="climate",
            training_days=[1,2],
            validation_days=[3,4],
            out_folder="output/gan_test_fix_relu_dampen1",
        )

    # repeat of above but relu fix
    if "GAN_TEST_2_DAMPEN2" in os.environ:
        args = {"learning_rate": 0.01, "sigma":0., "bottleneck":512, "lr_encoder_decoder":0.01, "lr_discriminator":0.001,"lr_generator":0.001, "coef_discriminator":0.5, "coef_generator":0.5}
        net_cfg = get_net_gan(gan.build_model_cnn, args)
        train_gan(
            net_cfg,
            num_epochs=300,
            batch_size=32,
            img_size=96,
            dataset="climate",
            training_days=[1,2],
            validation_days=[3,4],
            out_folder="output/gan_test_fix_relu_dampen2",
        )

    if "GAN_TEST_2_DAMPEN1_BS64" in os.environ:
        args = {"learning_rate": 0.01, "sigma":0., "bottleneck":512, "lr_encoder_decoder":0.01, "lr_discriminator":0.001,"lr_generator":0.001, "coef_discriminator":0.5, "coef_generator":0.1}
        net_cfg = get_net_gan(gan.build_model_cnn, args)
        train_gan(
            net_cfg,
            num_epochs=300,
            batch_size=64,
            img_size=96,
            dataset="climate",
            training_days=[1,2],
            validation_days=[3,4],
            out_folder="output/gan_test_fix_relu_dampen1_bs64",
        )
        

    # --------------------
    # GANs on 20 day data
    # --------------------

    if "GAN_20DAY_DAMPEN1_BS64" in os.environ:
        args = {"learning_rate": 0.01, "sigma":0., "bottleneck":512, "lr_encoder_decoder":0.01, "lr_discriminator":0.001,"lr_generator":0.001, "coef_discriminator":0.5, "coef_generator":0.1}
        net_cfg = get_net_gan(gan.build_model_cnn, args)
        train_gan(
            net_cfg,
            num_epochs=300,
            batch_size=128,
            img_size=96,
            dataset="climate",
            training_days=[1,20],
            validation_days=[325,345],
            out_folder="output/gan_20day_dampen1_bs128",
        )
    if "GAN_20DAY_NO_GAN_BS64" in os.environ:
        args = {"learning_rate": 0.01, "sigma":0., "bottleneck":512, "lr_encoder_decoder":0.01, "lr_discriminator":0.001,"lr_generator":0.001, "coef_discriminator":0.5, "coef_generator":0.1}
        net_cfg = get_net_gan(gan.build_model_cnn, args)
        train_gan(
            net_cfg,
            num_epochs=300,
            batch_size=128,
            img_size=96,
            dataset="climate",
            training_days=[1,20],
            validation_days=[325,345],
            out_folder="output/gan_no_gan_20day_dampen1_bs128",
            ignore_gan_part=True
        )

    if "GAN_VIS" in os.environ:
        args = {"learning_rate": 0.01, "sigma":0., "bottleneck":512, "lr_encoder_decoder":0.01, "lr_discriminator":0.001,"lr_generator":0.001, "coef_discriminator":0.5, "coef_generator":0.1}
        net_cfg = get_net_gan(gan.build_model_cnn, args)
        extract_hidden_codes(cfg=net_cfg, model="/storeSSD/cbeckham/nersc/models/output/gan_test_fix_relu_dampen1/92.model", out_file="codes/code.txt", img_size=96, days=[1,2])

               
        #def extract_hidden_codes(cfg, model, data_dir, img_size, days, time_chunks_per_example=1, dataset="climate"):

        
        

    if "CLIMATE_20DAY_BOTTLENECK512" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "bottleneck":512 }
        net_cfg = get_net(climate_test_1, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="climate", days=20, out_folder="output/climate_20day_bottleneck512")
    if "CLIMATE_20DAY_BOTTLENECK512_DET" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        args = { "learning_rate": 0.01, "sigma":0., "bottleneck":512 }
        net_cfg = get_net(climate_test_1, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="climate", training_days=[1,20], validation_days=[325,345], out_folder="output/climate_20day_bottleneck512_det")

    if "TEST" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        args = { "learning_rate": 0.01, "sigma":0., "bottleneck":512, "mode":"2d" }
        net_cfg = get_net(climate_test_1, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="climate", training_days=[1,1], validation_days=[2,2], out_folder="output/deleteme")

        
    # ---------------------
    # let's try out 3d conv
    # ---------------------

    if "CLIMATE_1DAY_3D_TEST" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        args = { "learning_rate": 0.01, "sigma":0., "bottleneck":1024, "mode":"3d" }
        net_cfg = get_net(climate_test_1_3d, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="climate",
              training_days=[1,2], validation_days=[10,11], out_folder="output/climate_1day_3d_test", time_chunks_per_example=8)


    if "CLIMATE_1DAY_3D_TEST_2" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        args = { "learning_rate": 0.1, "sigma":0., "bottleneck":1024, "mode":"3d" }
        net_cfg = get_net(climate_test_1_3d, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="climate",
              training_days=[1,2], validation_days=[10,11], out_folder="output/climate_1day_3d_test_2", time_chunks_per_example=8)


        
        
    if "CLIMATE_50DAY_BOTTLENECK512" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "bottleneck":512 }
        net_cfg = get_net(climate_test_1, args)
        train(net_cfg, num_epochs=300, batch_size=32, img_size=96, dataset="climate", days=50, out_folder="output/climate_50day_bottleneck512")
        
        
    if "STL10_TEST_BS128_ADAM" in os.environ:
        args = { "learning_rate": 0.01, "sigma":0., "nonlinearity":elu, "tied":False, "input_channels":3, "optim":"adam" }
        net_cfg = get_net(what_where_stl10_arch, args)
        train(net_cfg, num_epochs=300, batch_size=128, img_size=96, dataset="stl10", out_folder="output/stl10_test_bs128_adam", sched={100:0.001,200:0.0001})

    # ###############################
    # Experiments on full-sized image
    # ###############################

    if "FULL_IMAGE_1" in os.environ:
        args = {"learning_rate": 0.01, "mode":"2d" }
        net_cfg = get_net(architectures.full_image_net_1, args)
        train(
            net_cfg,
            num_epochs=100,
            batch_size=1,
            img_size=-1,
            dataset="climate",
            training_days=[1,319],
            validation_days=[320,345],
            out_folder="output/full_image_1",
            save_images_every=20
        )
    if "FULL_IMAGE_1_SIGMA5" in os.environ:
        args = {"learning_rate": 0.01, "mode":"2d", "sigma":0.5 }
        net_cfg = get_net(architectures.full_image_net_1, args)
        train(
            net_cfg,
            num_epochs=100,
            batch_size=1,
            img_size=-1,
            dataset="climate",
            training_days=[1,319],
            validation_days=[320,345],
            out_folder="output/full_image_1_sigma0.5",
            save_images_every=100
        )
    if "FULL_IMAGE_1_SIGMA10" in os.environ:
        args = {"learning_rate": 0.01, "mode":"2d", "sigma":1.0 }
        net_cfg = get_net(architectures.full_image_net_1, args)
        train(
            net_cfg,
            num_epochs=100,
            batch_size=1,
            img_size=-1,
            dataset="climate",
            training_days=[1,319],
            validation_days=[320,345],
            out_folder="output/full_image_1_sigma1.0",
            save_images_every=500
        )
    if "FULL_IMAGE_1_3D" in os.environ:
        args = {"learning_rate": 0.01, "dim":"3d", "mode":"autoencoder", "sigma":0. }
        net_cfg = get_net(architectures.full_image_net_1_3d, args)
        train(
            net_cfg,
            num_epochs=100,
            batch_size=1,
            img_size=-1,
            dataset="climate",
            training_days=[1,319],
            validation_days=[320,345],
            out_folder="output/full_image_1_3d_deleteme",
            time_chunks_per_example=8,
            save_images_every=100000
        )
    if "FULL_IMAGE_1_3D_V2" in os.environ:
        args = {"learning_rate": 0.01, "mode":"3d", "sigma":0. }
        net_cfg = get_net(architectures.full_image_net_1_3d_v2, args)
        train(
            net_cfg,
            num_epochs=100,
            batch_size=1,
            img_size=-1,
            dataset="climate",
            training_days=[1,319],
            validation_days=[320,345],
            out_folder="output/full_image_1_3d_v2_test",
            time_chunks_per_example=8,
            save_images_every=1
        )
    if "FULL_IMAGE_1_3D_V2_RMSPROP" in os.environ:
        args = {"learning_rate": 0.01, "mode":"3d", "sigma":0., "optim":"rmsprop" }
        net_cfg = get_net(architectures.full_image_net_1_3d_v2, args)
        train(
            net_cfg,
            num_epochs=100,
            batch_size=1,
            img_size=-1,
            dataset="climate",
            training_days=[1,319],
            validation_days=[320,345],
            out_folder="output/full_image_1_3d_v2_rmsprop",
            time_chunks_per_example=8,
            save_images_every=500
        )

    if "FULL_IMAGE_1_3D_V3B" in os.environ:
        args = {"learning_rate": 0.01, "mode":"3d", "sigma":0. }
        net_cfg = get_net(architectures.full_image_net_1_3d_v3b, args)
        train(
            net_cfg,
            num_epochs=100,
            batch_size=1,
            img_size=-1,
            dataset="climate",
            training_days=[1,319],
            validation_days=[320,345],
            out_folder="output/full_image_1_3d_v3b",
            time_chunks_per_example=8,
            save_images_every=500
        )

    # ----------------- classification -----------------

    if "FULL_IMAGE_1_3D_CLASSIFICATION" in os.environ:
        args = {"learning_rate": 0.01, "dim":"3d", "mode":"classification", "sigma":0. }
        net_cfg = get_net(architectures.full_image_net_1_3d, args)
        train(
            net_cfg,
            num_epochs=100,
            batch_size=1,
            img_size=-1,
            dataset="climate_classification",
            training_days=[1,319],
            validation_days=[320,345],
            out_folder="output/full_image_1_3d_classification_deleteme",
            time_chunks_per_example=8,
            save_images_every=1
        )

        
        
    """
    for epoch in range(10):
        for X_train, y_train in common.data_iterator(128, "/storeSSD/cbeckham/nersc/big_images/", days=1):
            mem = virtual_memory()
            print mem
            pass
    """
    
