import numpy as np
import tensorflow as tf

np.random.seed(100)

def get_mean_variance_concat(vec):
    '''
        Input: It should be has two dimension(n, d)
        Output: (1, d)
    '''
    mean = np.mean(vec, axis=1)
    # mean_x_square = np.mean(vec**2, axis=0)
    # var = np.mean(mean_x_square)-mean**2
    var = np.var(vec, axis=1)

    return mean, var

def min_max_norm(x_mean, x_var):
	
	x_max = tf.reduce_max(x_mean, axis=0)
	x_min = tf.reduce_min(x_mean, axis=0)

	denom = tf.maximum(x_max-x_min, 1e-6)
	# denom = x_max-x_min
	alpha = 1.0/denom
	beta = (-1.0)*x_min/denom

	x_mean_norm = tf.add(tf.multiply(alpha, x_mean), beta)
	x_var_norm = tf.multiply(tf.square(alpha), x_var)

	return x_mean_norm, x_var_norm

inp_mean = tf.placeholder(tf.float32, shape=(40,500))
inp_var = tf.placeholder(tf.float32, shape=(40,500))

var_0 = tf.fill(tf.shape(inp_var), 1e-6)
            
cls_score_var = tf.maximum(inp_var, var_0)
mask_p = tf.subtract(tf.multiply(tf.to_float(inp_mean>0),2.0), 1)

distance = 0.25*tf.log(0.25*(tf.div(var_0, inp_var)+tf.div(inp_var, var_0)+2))+0.25*(tf.div(tf.square(inp_mean), tf.add(var_0, inp_var)))

cls_score_mean = distance*mask_p



inp = np.random.rand(40,4,500)
mean, var = get_mean_variance_concat(inp)

var_0 = np.zeros(40,500)
var_0.fill(1e-6)
var = np.maximum(var, var_0)
d = 0.25*np.log(0.25*(var/var_0+var_0/var+2))+0.25*(np.square(action_score-mean_0)/(var+var_0))
action_score = d*mask




print('-----------------------------')
print(np.linalg.norm(b1-a1))
print(np.linalg.norm(b2-a2))