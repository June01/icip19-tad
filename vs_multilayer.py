from __future__ import division

import numpy as np
import tensorflow as tf

# components
from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import pooling_layer as pool
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from util.ua_cnn import relu_layer as relu_ua
from util.ua_cnn import fc_layer as fc_ua

def vs_multilayer(input_batch, name, middle_layer_dim=1000, class_num=20, dropout=False, reuse=False):
    """This function is inherited from CBR project(https://github.com/jiyanggao/CBR)
    """
    print('--I am using vs_multilayer--')

    with tf.variable_scope(name):
        if reuse==True:
            print(name+" reuse variables")
            tf.get_variable_scope().reuse_variables()
        else:
            print(name+" doesn't reuse variables")

        layer1 = fc_relu('layer1', input_batch, output_dim=middle_layer_dim)
        if dropout:
            layer1 = drop(layer1, 0.5)
        sim_score = fc('layer2', layer1, output_dim=(class_num+1)*3)
    return sim_score
