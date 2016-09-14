
import matplotlib; matplotlib.use("agg")


import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.updates import *
from lasagne.utils import *
from lasagne.init import *
import numpy as np
#import cPickle as pickle
#import gzip
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import sys
from time import time
if __name__ == "__main__":
    sys.path.insert(0,'..')
    from common import *
else:
    from ..common import *
import time
import logging
from sklearn.manifold import TSNE



def get_net(net_cfg, args):
    l_out, hid_layer = net_cfg(args)
 
    X = T.tensor4('X')
    Y = T.ivector('Y')
    net_out = get_output(l_out, X)
    hid_out = get_output(hid_layer, X)

    
    clsf_loss = get_classifier_loss(hid_layer,X,Y,args)
    rec_loss = squared_error(net_out, X).mean()
    if args['with_classif_loss']:
        loss = args['lrec'] * rec_loss + clsf_loss
        inputs = [X,Y]
    else:
        loss = rec_loss
        inputs=[X]
    params = get_all_params(l_out, trainable=True)
    lr = theano.shared(floatX(args["learning_rate"]))
    updates = nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
    train_fn = theano.function(inputs, loss, updates=updates)
    loss_fn = theano.function(inputs, loss)
    out_fn = theano.function([X], net_out)
    hid_fn = theano.function([X],hid_out)
    return {
        "train_fn": train_fn,
        "loss_fn": loss_fn,
        "out_fn": out_fn,
        "lr": lr,
        "l_out": l_out,
        "h_fn": hid_fn,

    }

def get_classifier_loss(hid_layer,x,y, args):
    
    clsf = DenseLayer(hid_layer, num_units=args['num_classes'], nonlinearity=softmax)
    label_inds = y > -3
    
    #get x's with labels
    x_lbl = x[label_inds.nonzero()]
    y_lbl = y[label_inds.nonzero()]
    y_lbl = y_lbl + 2
    
    clsf_out = get_output(clsf, x_lbl)
    clsf_loss = categorical_crossentropy(clsf_out, y_lbl).mean()
    return clsf_loss

def autoencoder_basic_32(args):
    conv_kwargs = {'nonlinearity': rectify, 'W': HeNormal()}
    net = InputLayer(args['shape'])
    net = GaussianNoiseLayer(net, args["sigma"])
    for i in range(5):
        net = Conv2DLayer(net, num_filters=args["nfilters"], filter_size=2,stride=2, **conv_kwargs)
        #net = MaxPool2DLayer(net, pool_size=2)
    last_conv_shape = tuple([k if k is not None else [i] for i,k in enumerate(get_output_shape(net,args['shape']))] )
    
    hid_layer = DenseLayer(net, num_units=args['code_size'], **conv_kwargs)
    net = DenseLayer(hid_layer, num_units=np.prod(last_conv_shape[1:]))
    net = ReshapeLayer(net, shape=last_conv_shape)
    
    for layer in get_all_layers(net)[::-1]:
        if isinstance(layer, MaxPool2DLayer):
            net = InverseLayer(net, layer)
            
        if isinstance(layer, Conv2DLayer):
            conv_dict = {key:getattr(layer, key) for key in ["stride", "pad", "num_filters", "filter_size"]}
            conv_dict['crop'] = conv_dict['pad']
            del conv_dict['pad']
            
            if not isinstance(layer.input_layer,Conv2DLayer):
                conv_dict['num_filters'] = args["shape"][1]
                conv_dict['nonlinearity'] = linear
            net = Deconv2DLayer(net, **conv_dict)

    for layer in get_all_layers(net):
        logger.info(str(layer) + str(layer.output_shape))
    print count_params(layer)
    return net, hid_layer


    



# def plot_learn_curve(tr_losses, val_losses, save_dir='.'):
#     plt.clf()
#     plt.plot(tr_losses)
#     plt.plot(val_losses)
#     plt.savefig(save_dir + '/learn_curve.png')
#     plt.clf()
    
# def plot_clusters(i,x,y, save_dir='.'):
#     x = np.squeeze(x)
#     hid_L = net_cfg['h_fn'](x)
#     ts = TSNE().fit_transform(hid_L)
#     plt.clf()
#     plt.scatter(ts[:,0], ts[:,1], c=y)
#     plt.savefig(save_dir + '/cluster_%i.png'%(i))
#     plt.clf()

# def plot_recs(i,x,net_cfg, save_dir='.'):
#     ind = np.random.randint(0,x.shape[0], size=(1,))
#     x=np.squeeze(x)
#     #print x.shape
#     im = x[ind]
#     #print im.shape
#     rec = net_cfg['out_fn'](im)
#     ch=1
#     plt.figure(figsize=(30,30))
#     plt.clf()
#     for (p_im, p_rec) in zip(im[0],rec[0]):
#         p1 = plt.subplot(im.shape[1],2, ch )
#         p2 = plt.subplot(im.shape[1],2, ch + 1)
#         p1.imshow(p_im)
#         p2.imshow(p_rec)
#         ch = ch+2
#     #pass
#     plt.savefig(save_dir +'/recs_%i' %(i))

# def plot_filters(network, save_dir='.'):
#     plt.figure(figsize=(30,30))
#     plt.clf()
#     lay_ind = 0
#     num_channels_to_plot = 16
#     convlayers = [layer for layer in get_all_layers(network) if isinstance(layer, Conv2DLayer)]
#     num_layers = len(convlayers)
#     spind = 1 
#     for layer in convlayers:
#         filters = layer.get_params()[0].eval()
#         #pick a random filter
#         filt = filters[np.random.randint(0,filters.shape[0])]
#         for ch_ind in range(num_channels_to_plot):
#             p1 = plt.subplot(num_layers,num_channels_to_plot, spind )
#             p1.imshow(filt[ch_ind], cmap="gray")
#             spind = spind + 1
    
#     #pass
#     plt.savefig(save_dir +'/filters.png')
            
        
# def plot_feature_maps(i, x, network, save_dir='.'):
#     plt.figure(figsize=(30,30))
#     plt.clf()
#     ind = np.random.randint(0,x.shape[0])
#     x=np.squeeze(x)

#     im = x[ind]
#     convlayers = [layer for layer in get_all_layers(network) if not isinstance(layer,DenseLayer)]
#     num_layers = len(convlayers)
#     spind = 1 
#     num_fmaps_to_plot = 16
#     for ch in range(num_fmaps_to_plot):
#         p1 = plt.subplot(num_layers + 1,num_fmaps_to_plot, spind )
#         p1.imshow(im[ch])
#         spind = spind + 1
      
#     for layer in convlayers:
#         # shape is batch_size, num_filters, x,y 
#         fmaps = get_output(layer,x ).eval()
#         for fmap_ind in range(num_fmaps_to_plot):
#             p1 = plt.subplot(num_layers + 1,num_fmaps_to_plot, spind )
#             p1.imshow(fmaps[ind][fmap_ind])
#             spind = spind + 1
    
#     #pass
#     plt.savefig(save_dir +'/fmaps.png')



num_epochs = 5000
batch_size = 128

run_dir = create_run_dir()
try:
    print logger
except:
    logger = logging.getLogger('log_train')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('%s/training.log'%(run_dir))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    
args = { "learning_rate": 0.01, "sigma":0.1, "shape": (None,16,128,128),
        'code_size': 16384 , 'nfilters': 128, 'lrec': 1, 'num_classes': 3, "with_classif_loss": False }
net_cfg = get_net(autoencoder_basic_32, args)



tr_losses = []
val_losses = []

for epoch in range(num_epochs):
    tr_iterator = data_iterator(batch_size=batch_size, step_size=128, days=1, month1='01', day1='01')
    val_iterator = data_iterator(batch_size=batch_size, step_size=128, days=1, month1='10', day1='28')
    start = time.time() 
    tr_loss = 0
    for iteration, (x, y) in enumerate(tr_iterator):
        #print iteration
        x = np.squeeze(x)
        print x.shape
        
        loss = net_cfg['train_fn'](x)
        tr_loss += loss
    
    train_end = time.time()
    tr_avgloss = tr_loss / (iteration + 1)
    
    
    logger.info("train time : %5.2f seconds" % (train_end - start))
    logger.info(" epoch %i of %i train loss is %f" % (epoch, num_epochs, tr_avgloss))
    tr_losses.append(tr_avgloss)
    
    val_loss = 0

    for iteration, (xval, yval) in enumerate(val_iterator):
        xval = np.squeeze(xval)
        loss = net_cfg['loss_fn'](xval)
        val_loss += loss
    
    val_avgloss = val_loss / (iteration + 1)   
    logger.info("val time : %5.2f seconds" % (time.time() - train_end))

    logger.info(" epoch %i of %i val loss is %f" % (epoch, num_epochs, val_avgloss))
    val_losses.append(val_avgloss)
    
    plot_learn_curve(tr_losses, val_losses, save_dir=run_dir)
    if epoch % 5 == 0:
        plot_filters(net_cfg['l_out'], save_dir=run_dir)
        for iteration, (x,y) in enumerate(data_iterator(batch_size=batch_size, step_size=128, days=1, month1='01', day1='01')):
            plot_recs(iteration,x,net_cfg=net_cfg, save_dir=run_dir)
            plot_clusters(iteration,x,y,net_cfg=net_cfg, save_dir=run_dir)
            plot_feature_maps(iteration,x,net_cfg['l_out'], save_dir=run_dir)
            break;
        
                     






