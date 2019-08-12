import tensorflow as tf
import numpy as np

x_mean = tf.placeholder(tf.float32, shape=(20,500))
x_var = tf.placeholder(tf.float32, shape=(20,500))

mean_norm = tf.math.l2_normalize(x_mean, axis=1)

col_norm = tf.reshape(tf.sqrt(tf.maximum(tf.reduce_sum(x_mean**2, axis=1), 1e-12)), shape=(20,1))
col_norm_e = tf.tile(col_norm, [1,500])

var_norm_acc = tf.multiply(x_var, tf.square(tf.div(mean_norm,x_mean)))
var_norm2_man = tf.div(x_var, tf.square(col_norm_e))

mean_norm_man = tf.div(x_mean,col_norm_e)

sess=tf.Session()

bo, co, b1, c1 = sess.run([mean_norm, mean_norm_man, var_norm_acc, var_norm2_man], feed_dict={x_mean: np.random.rand(20,500), x_var: np.random.rand(20,500)})

# print(b1)
# print('------------------------')
# print(c1)

print(np.linalg.norm(b1-c1))
print(np.linalg.norm(bo-co))

