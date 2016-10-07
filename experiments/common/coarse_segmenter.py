from lasagne.layers import *  
from lasagne.objectives import *
from lasagne.nonlinearities import *
import numpy as np
import theano
import sys
sys.path.append('../common/')
from data import segmentation_iterator

#FCN used softmax w/ xent, but they also tried sigmoid w/ xent and 
# did pretty well
def downsample_mask(mask_t, num_decimation_layers):
    '''gt_mask (Theano Tensor4) -> 768 by 1162 by num_classes
    num_decimation_layer (int) -> number of stride 2 or max pool layers'''
    net = InputLayer(input_var=mask_t, shape=(None,1,768,1152))
    for i in range(num_decimation_layers):
        net = MaxPool2DLayer(net,pool_size=5,stride=2)
    return get_output(net)
            

def get_classwise_scores(last_enc_layer, num_classes):
    '''last_enc_layer  (Theano Tensor4) -> last encoder layer
    num_classes (int) -> should be equal to depth of gt_mask'''
    
    #get hyperplanes for each class
    net = Conv2DLayer(last_enc_layer,
                      num_filters=num_classes,
                      filter_size=1,
                      nonlinearity=sigmoid)
    return get_output(net)
    
    
def calc_seg_loss(gt_coarse_mask, class_scores):
    '''gt_coarse_mask (Theano Tensor4) -> 5 by 11 by num_classes or something'''
    '''class_scores (Theano Tensor4) -> same as above'''
    loss = binary_crossentropy(class_scores, gt_coarse_mask).mean()
    return loss

if __name__ == "__main__":
    
    net = InputLayer(input_var=x, shape=(None,10, 18, 30))
    net = Conv2DLayer(net, num_filters=1200, stride=64, filter_size=3, pad=1)
    x = theano.tensor.tensor4('x')
    mask_t = theano.tensor.tensor4('m')
    class_scores = get_classwise_scores(net, 1)

    new_mask = downsample_mask(mask_t, 6)

    loss = calc_seg_loss(new_mask, class_scores)

    loss.eval({mask_t: mask, x:np.random.random(((10, 10, 18, 30))).astype('float32')})