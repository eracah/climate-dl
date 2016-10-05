import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.init import *
from lasagne.objectives import *
import numpy as np
from lasagne.updates import *


def build_model_cnn(args={}):

    num_code = args["bottleneck"]
    num_hidden = 64
    img_size = 96
    num_channels = 16

    l_encoder_in = InputLayer((None, num_channels, img_size, img_size), name='l_encoder_in')
    
    l_conv1 = Conv2DLayer(l_encoder_in, filter_size=5, num_filters=128, name="conv1")
    l_conv1.params[l_conv1.W].add("generator")
    l_conv1.params[l_conv1.b].add("generator")
                   
    #l_mp1 = Pool2DLayer(l_conv1, pool_size=2, name="mp1", mode="average_inc_pad")
                   
    l_conv2 = Conv2DLayer(l_conv1, filter_size=5, num_filters=256, name="conv2", stride=2)
    l_conv2.params[l_conv2.W].add("generator")
    l_conv2.params[l_conv2.b].add("generator")
                   
    #l_mp2 = Pool2DLayer(l_conv2, pool_size=2, name="mp2", mode="average_inc_pad")
    
    # output of the encoder/generator: q(z|x)
    l_encoder_out = DenseLayer(l_conv2, num_units=num_code, nonlinearity=linear, name="l_encoder_out")
    l_encoder_out.params[l_encoder_out.W].add('generator')
    l_encoder_out.params[l_encoder_out.b].add('generator')
        
    l_decoder_out = l_encoder_out
    k = 0
    for layer in get_all_layers(l_encoder_out)[::-1]:
        if isinstance(layer, InputLayer):
            break
        l_decoder_out = InverseLayer(l_decoder_out, layer, name="inv%i" % k)
        k += 1
    # output of the decoder: p(x|z)
    l_decoder_out.name = "l_decoder_out"
    l_decoder_out.nonlinearity = sigmoid

    # print output of encoder
    print "encoder/decoder:"
    for layer in get_all_layers(l_decoder_out):
        print layer, layer.name, layer.output_shape

    # --------------------------------------------------------------------------

    # input layer providing samples from p(z)
    l_prior = InputLayer((None, num_code), name='l_prior_in')

    # concatenate samples from q(z|x) to samples from p(z)
    l_concat = ConcatLayer(
        [l_encoder_out, l_prior], axis=0, name='l_prior_encoder_concat',
    )

    # first layer of the discriminator
    l_dense6 = DenseLayer(
        l_concat, num_units=num_hidden, nonlinearity=rectify,
        name='l_discriminator_dense1',
    )
    l_dense6.params[l_dense6.W].add('discriminator')
    l_dense6.params[l_dense6.b].add('discriminator')

    # second layer of the discriminator
    l_dense7 = DenseLayer(
        l_dense6, num_units=num_hidden, nonlinearity=rectify,
        name='l_discriminator_dense2',
    )
    l_dense7.params[l_dense7.W].add('discriminator')
    l_dense7.params[l_dense7.b].add('discriminator')

    # output layer of the discriminator
    l_discriminator_out = DenseLayer(
        l_dense7, num_units=1, nonlinearity=sigmoid,
        name='l_discriminator_out',
    )
    l_discriminator_out.params[l_discriminator_out.W].add('discriminator')
    l_discriminator_out.params[l_discriminator_out.b].add('discriminator')

    print "discriminator:"
    for layer in get_all_layers(l_discriminator_out):
        print layer, layer.name, layer.output_shape
        
    model_layers = get_all_layers([l_decoder_out, l_discriminator_out])

    # put all layers in a dictionary for convenience
    return {layer.name: layer for layer in model_layers}
    



# forward pass for the encoder, q(z|x)
def create_encoder_func(layers):
    X = T.tensor4('X')
    X_batch = T.tensor4('X_batch')
    Z = get_output(layers['l_encoder_out'], X, deterministic=True)
    encoder_func = theano.function(
        inputs=[theano.In(X_batch)],
        outputs=Z,
        givens={
            X: X_batch,
        },
    )
    return encoder_func

# forward pass for the decoder, p(x|z)
# note: this won't work if you use
# inverse layer, since it requires the grad
# wrt layers before the start of the decoder
# (i.e. the encoder part)
def create_decoder_func(layers):
    Z = T.fmatrix('Z')
    Z_batch = T.fmatrix('Z_batch')
    X = get_output(
        layers['l_decoder_out'],
        inputs={
            layers['l_encoder_out']: Z
        },
        deterministic=True
    )
    decoder_func = theano.function(
        inputs=[theano.In(Z_batch)],
        outputs=X,
        givens={
            Z: Z_batch,
        },
    )
    return decoder_func


# forward/backward (optional) pass for the encoder/decoder pair
def create_encoder_decoder_func(layers, apply_updates=False, learning_rate=0.01):
    X = T.tensor4('X')
    X_batch = T.tensor4('X_batch')
    X_hat = get_output(layers['l_decoder_out'], X, deterministic=False)
    # reconstruction loss
    encoder_decoder_loss = T.mean(
        T.mean(T.sqr(X - X_hat), axis=1)
    )
    if apply_updates:
        # all layers that participate in the forward pass should be updated
        encoder_decoder_params = get_all_params(
            layers['l_decoder_out'], trainable=True)

        encoder_decoder_updates = nesterov_momentum(
            encoder_decoder_loss, encoder_decoder_params, learning_rate, 0.9)
    else:
        encoder_decoder_updates = None
    encoder_decoder_func = theano.function(
        inputs=[theano.In(X_batch)],
        outputs=encoder_decoder_loss,
        updates=encoder_decoder_updates,
        givens={
            X: X_batch,
        },
    )
    return encoder_decoder_func

# forward/backward (optional) pass for discriminator
def create_discriminator_func(layers, apply_updates=False, learning_rate=0.001, coef=1.):
    X = T.tensor4('X')
    pz = T.fmatrix('pz')
    X_batch = T.tensor4('X_batch')
    pz_batch = T.fmatrix('pz_batch')
    # the discriminator receives samples from q(z|x) and p(z)
    # and should predict to which distribution each sample belongs
    discriminator_outputs = get_output(
        layers['l_discriminator_out'],
        inputs={
            layers['l_prior_in']: pz,
            layers['l_encoder_in']: X,
        },
        deterministic=False,
    )
    # label samples from q(z|x) as 1 and samples from p(z) as 0
    discriminator_targets = T.vertical_stack(
        T.ones((X_batch.shape[0], 1)),
        T.zeros((pz_batch.shape[0], 1))
    )
    discriminator_loss = coef*T.mean(
        T.nnet.binary_crossentropy(
            discriminator_outputs,
            discriminator_targets,
        )
    )
    if apply_updates:
        # only layers that are part of the discriminator should be updated
        discriminator_params = get_all_params(
            layers['l_discriminator_out'], trainable=True, discriminator=True)

        discriminator_updates = nesterov_momentum(
            discriminator_loss, discriminator_params, learning_rate, 0.0)
    else:
        discriminator_updates = None
    discriminator_func = theano.function(
        inputs=[
            theano.In(X_batch),
            theano.In(pz_batch),
        ],
        outputs=discriminator_loss,
        updates=discriminator_updates,
        givens={
            X: X_batch,
            pz: pz_batch,
        },
    )
    return discriminator_func

# forward/backward (optional) pass for the generator
# note that the generator is the same network as the encoder,
# but updated separately
def create_generator_func(layers, apply_updates=False, learning_rate=0.001, coef=1.):
    X = T.tensor4('X')
    X_batch = T.tensor4('X_batch')
    # no need to pass an input to l_prior_in here
    generator_outputs = get_output(
        layers['l_encoder_out'], X, deterministic=False)
    # so pass the output of the generator as the output of the concat layer
    discriminator_outputs = get_output(
        layers['l_discriminator_out'],
        inputs={
            layers['l_prior_encoder_concat']: generator_outputs,
        },
        deterministic=False
    )
    # the discriminator learns to predict 1 for q(z|x),
    # so the generator should fool it into predicting 0
    generator_targets = T.zeros_like(X_batch.shape[0])
    # so the generator needs to push the discriminator's output to 0
    generator_loss = coef*T.mean(
        T.nnet.binary_crossentropy(
            discriminator_outputs,
            generator_targets,
        )
    )
    if apply_updates:
        # only layers that are part of the generator (i.e., encoder)
        # should be updated
        generator_params = get_all_params(
            layers['l_discriminator_out'], trainable=True, generator=True)

        generator_updates = nesterov_momentum(
            generator_loss, generator_params, learning_rate, 0.0)
    else:
        generator_updates = None
    generator_func = theano.function(
        inputs=[
            theano.In(X_batch),
        ],
        outputs=generator_loss,
        updates=generator_updates,
        givens={
            X: X_batch,
        },
    )
    return generator_func

# forward/backward (optional) pass for the encoder/decoder pair
def get_reconstruction(layers):
    X = T.tensor4('X')
    X_batch = T.tensor4('X_batch')
    X_hat = get_output(layers['l_decoder_out'], X, deterministic=False)
    reconstruction_func = theano.function(
        inputs=[theano.In(X_batch)],
        outputs=X_hat,
        givens={
            X: X_batch,
        },
    )
    return reconstruction_func
