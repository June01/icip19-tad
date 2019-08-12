from __future__ import division 

import tensorflow as tf

def fc_layer(name, x_mean, x_var, output_dim, bias_term=True, weights_initializer=None,
             biases_initializer=None):
    # flatten bottom input
    # input has shape [batch, in_height, in_width, in_channels]
    shape = x_mean.get_shape().as_list()

    input_dim = shape[1]

    # weights and biases variables
    with tf.variable_scope(name):
        if weights_initializer is None and biases_initializer is None:
            # initialize the variables
            if weights_initializer is None:
                weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            if bias_term and biases_initializer is None:
                biases_initializer = tf.constant_initializer(0.0)

            # weights has shape [input_dim, output_dim]
            weights = tf.get_variable("weights", [input_dim, output_dim],
                initializer=weights_initializer)
            if bias_term:
                biases = tf.get_variable("biases", output_dim,
                    initializer=biases_initializer)

            print(weights.name+" initialized as random or retrieved from graph")
            if bias_term:
                print(biases.name+" initialized as random or retrieved from graph")
        else:
            weights = tf.get_variable("weights", shape=None,
                initializer=weights_initializer)
            if bias_term:
                biases = tf.get_variable("biases", shape=None,
                    initializer=biases_initializer)

            print(weights.name+" initialized from pre-trained parameters or retrieved from graph")
            if bias_term:
                print(biases.name+" initialized from pre-trained parameters or retrieved from graph")

    if bias_term:
        y_mean = tf.nn.xw_plus_b(x_mean, weights, biases)
    else:
        y_mean = tf.matmul(x_mean, weights)

    square_w = tf.square(weights)
    y_var = tf.matmul(x_var, square_w)

    return y_mean, y_var

def relu_layer(x_mean, x_var):

    p_mask = x_mean > 0

    y_mean = tf.multiply(x_mean, tf.to_float(p_mask)) 
    y_var = tf.multiply(x_var, tf.to_float(p_mask))
    y_var = tf.maximum(y_var, 1e-6)
    return y_mean, y_var
