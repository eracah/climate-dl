import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
#from lasagne.layers import dnn
import sys
sys.path.append("..")
import common
from collections import Counter
import numpy as np
from time import time
import pdb

batch_size=128

# first determine the size of the tensor for each day
g = open("/storeSSD/cbeckham/nersc/big_images/1979_memmap/shapes.txt", "wb")
counter=0
for x, y in common.data_iterator(batch_size, "/storeSSD/cbeckham/nersc/big_images/", start_day=1, end_day=1, img_size=96, time_chunks_per_example=1):
    counter += x.shape[0]
g.write("%i,%i,%i,%i,%i\n" % (counter,16,1,96,96))
g.close()

batch_size=128
training_days=range(1,50+1)
valid_days=range(325,345+1)
for day in training_days+valid_days:
    print "day:",day
    fp = np.memmap("/storeSSD/cbeckham/nersc/big_images/1979_memmap/%i_float32.npy" % day, dtype='float32', mode='w+', shape=(counter,16,1,96,96))
    print counter
    counter = 0
    for x, y in common.data_iterator(batch_size, "/storeSSD/cbeckham/nersc/big_images/", start_day=day, end_day=day, img_size=96, time_chunks_per_example=1):
        # right now it is: (bs, time, nchannels, height, width)
        # needs to be: (bs, nchannels, time, height, width)
        x = x.reshape((x.shape[0], x.shape[2], x.shape[1], x.shape[3], x.shape[4]))
        print x.shape, "mapped to", counter, ":", counter+x.shape[0]
        fp[counter:counter+x.shape[0]] = x
        counter += x.shape[0]
        print counter


"""
# vanilla
t0 = time()
for x, y in common.data_iterator(batch_size, "/storeSSD/cbeckham/nersc/big_images/", start_day=1, end_day=6, img_size=96, time_chunks_per_example=1):
    tmp = x
print time()-t0

# new
t0 = time()
for i in range(1,6+1):
    newfp = np.memmap("/storeSSD/cbeckham/nersc/big_images/1979_memmap/%i.npy" % i, dtype='float32', mode='r', shape=(14312,16,1,96,96))
    for b in range(0, newfp.shape[0], batch_size):
        tmp = newfp[b*batch_size:(b+1)*batch_size]
print time()-t0
"""
