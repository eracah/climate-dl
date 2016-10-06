#!/bin/bash
FULL_IMAGE_1_3D_CLASSIFICATION=1 \
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,lib.cnmem=0.95,allow_gc=True,floatX=float32,nvcc.fastmath=True,profile=False \
  python -u train.py
