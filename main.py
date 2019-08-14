""" This is the main file of the project

Here you can start to train the model, or evaluate the model

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

import logging
tf.get_logger().setLevel(logging.ERROR)

import cbr_model

from config import Config, parser

def run_evaluating(config):
	""" Evaluation function
	"""
	res_path_name = config.test_model_path

	model = cbr_model.CBR_Model(config)

	model.build()

	res_path = os.path.join('model', res_path_name, config.test_iter+'.ckpt')
	model.restore_session(res_path)

	model.do_eval_slidingclips(res_path_name+'_cas'+str(config.cas_step)+'_'+str(config.prop_method)+'_'+str(config.test_iter))

def run_training(config):
	"""Training function
	"""
	model = cbr_model.CBR_Model(config)
	model.build()
	model.train()


def main(_):

	config = Config()
	if config.mode == 'train':
		run_training(config)
	else:
		run_evaluating(config)


if __name__ == '__main__':
	tf.app.run()

