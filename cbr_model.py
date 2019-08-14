import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf


import logging
tf.get_logger().setLevel(logging.ERROR)

from six.moves import xrange
import pickle
import time
import os

import math

import vs_multilayer

from dataset import TestingDataSet
from dataset import TrainingDataSet
import dataset

import tools

class CBR_Model(object):
    """ This is the body of the network we are using

    Here you will get access to the network structure, function of training, evaluation

    """

    def __init__(self, config):
        """Initialization
        """
        self.config = config

        self.sess = None
        self.saver = None

        self.train_clip_path = self.config.train_clip_path
        self.background_path = self.config.background_path
        self.test_clip_path = self.config.test_clip_path
        self.train_flow_feature_dir = self.config.train_flow_feature_dir
        self.train_appr_feature_dir = self.config.train_appr_feature_dir
        self.test_flow_feature_dir = self.config.test_flow_feature_dir
        self.test_appr_feature_dir = self.config.test_appr_feature_dir
        self.test_len_dict = self.config.test_len_dict

        self.batch_size = self.config.batch_size

        self.test_batch_size = 1
        self.middle_layer_size = 1000

        self.lambda_reg = float(self.config.lambda_reg)
        self.action_class_num = self.config.action_class_num
        self.feat_type = self.config.feat_type
        self.visual_feature_dim = self.config.visual_feature_dim

        # Initialize the training data and testing data
        self.train_set = TrainingDataSet(self.config, self.train_flow_feature_dir, self.train_appr_feature_dir, self.train_clip_path, self.background_path)
        self.test_set = TestingDataSet(self.config,self.test_flow_feature_dir, self.test_appr_feature_dir, self.test_clip_path, self.test_batch_size, self.test_len_dict)

        # Path to save the summary of the models
        self.summary_dir = os.path.join('./summary', self.config.save_name)

        if not os.path.exists(self.summary_dir):
            os.mkdir(self.summary_dir)

        if self.config.issave == 'Yes':
            self.model_dir = os.path.join('./model', self.config.save_name)

            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)


    def init_session(self):
        """Create a session in tensorflow
        """
        print('Initializing of session')

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3) # 30% memory of TITAN is enough
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)
        if self.config.ispretrain == 'Yes':
            self.restore_session('model/xx.ckpt')


    def get_feed_dict(self, lr_by_step):
        """Prepare training samples in each batch size to the network
        """
        image_batch, label_batch, offset_batch, one_hot_label_batch = self.train_set.next_batch()

        input_feed = {
            self.visual_featmap_ph_train: image_batch,
            self.label_ph: label_batch,
            self.offset_ph: offset_batch,
            self.one_hot_label_ph: one_hot_label_batch,
            self.vs_lr: lr_by_step
        }

        return input_feed


    def add_loss_op(self, visual_feature, offsets, labels, one_hot_labels, name='CBR'):
        """This function is to compute the loss in tensorflow graph

        Args:
            visual_feature: Tensor, feature, (batch_size, visual_feature_dim)
            offsets: Tensor, boundary offset(both to the start and end in frame-level), (batch_size, 2)
            labels: Tensor, label, (batch_size)
            one_hot_labels: Tensor, one hot label, (batch_size, action_class_num+1)

        Returns:
            loss: loss_cls + lambda_reg * loss_reg
            loss_reg: L1 loss between ground truth offsets and prediction offsets
            loss_cls: cross entropy loss

        """
        print('Add the standard loss')

        cls_reg_vec = vs_multilayer.vs_multilayer(visual_feature, name, middle_layer_dim=self.middle_layer_size, class_num=self.action_class_num, dropout=self.config.dropout)

        cls_reg_vec = tf.reshape(cls_reg_vec, [self.batch_size, (self.action_class_num+1)*3])
        cls_score_vec = cls_reg_vec[:, :self.action_class_num+1]
        start_offset_pred = cls_reg_vec[:, self.action_class_num+1:(self.action_class_num+1)*2]
        end_offset_pred = cls_reg_vec[:, (self.action_class_num+1)*2:]

        # l1 loss
        loss_l1 = tf.reduce_mean(tf.abs(cls_score_vec))

        # classification loss
        loss_cls_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score_vec, labels=labels)
        loss_cls = tf.reduce_mean(loss_cls_vec)

        # regression loss
        pick_start_offset_pred = []
        pick_end_offset_pred = []
        for k in range(self.batch_size):

            pick_start_offset_pred.append(start_offset_pred[k, labels[k]])
            pick_end_offset_pred.append(end_offset_pred[k, labels[k]])

        pick_start_offset_pred = tf.reshape(tf.stack(pick_start_offset_pred),[self.batch_size, 1])
        pick_end_offset_pred = tf.reshape(tf.stack(pick_end_offset_pred), [self.batch_size, 1])
        labels_1 = tf.to_float(tf.not_equal(labels,0))
        label_tmp = tf.to_float(tf.reshape(labels_1, [self.batch_size, 1]))
        label_for_reg = tf.concat([label_tmp, label_tmp], 1)
        offset_pred = tf.concat((pick_start_offset_pred, pick_end_offset_pred), 1)

        loss_reg = tf.reduce_mean(tf.multiply(tf.abs(tf.subtract(offset_pred, offsets)), label_for_reg))

        loss = tf.add(tf.multiply(self.lambda_reg, loss_reg), loss_cls)

        if self.config.l1_loss:
            loss = tf.add(loss, loss_l1)
        else:
            loss = loss

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("loss_reg", loss_reg)
        tf.summary.scalar("loss_cls", loss_cls)

        return loss, loss_reg, loss_cls


    def add_placeholders(self):
        """Add placeholders
        """
        print('Add placeholders')

        self.visual_featmap_ph_train = tf.placeholder(tf.float32, name='train_featmap', shape=(self.batch_size, self.visual_feature_dim))
        self.visual_featmap_ph_test = tf.placeholder(tf.float32, name='test_featmap', shape=(self.test_batch_size, self.visual_feature_dim))

        self.label_ph = tf.placeholder(tf.int32, name='label', shape=(self.batch_size))
        self.offset_ph = tf.placeholder(tf.float32, name='offset', shape=(self.batch_size, 2))
        self.one_hot_label_ph = tf.placeholder(tf.float32, name='one_hot_label', shape=(self.batch_size, self.action_class_num+1))
        self.vs_lr = tf.placeholder(tf.float32, name='lr')


    def add_summary(self):
        """Add summary
        """
        print('Add summay')
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)


    def save_session(self, step):
        """Save the session if needed
        """
        if self.config.issave == 'Yes':

            print('Save session')

            model_name = os.path.join(self.model_dir, str(step)+'.ckpt')
            self.saver.save(self.sess, model_name)


    def restore_session(self, dir_model):
        """Restore session
        """
        print('Restore the Session')

        self.saver.restore(self.sess, dir_model)


    def close_session(self):
        """ Close session once finished
        """
        print('Close session')

        self.sess.close()


    def predict(self, visual_feature_test):
        """Inference during testing

        Args:
            visual_feature_test: Tensor, feature,  (test_batch_size, visual_feature_dim)

        Returns:
            sim_score: Tensor, (action_class_num+1)*3 (Note: [0:action_class_num+1]: classification scores;
                [action_class_num+1:(action_class_num+1)*2: start offsets; [(action_class_num+1)*2:(action_class_num+1)*3]: end offsets)

        """
        print('To predict the label')

        sim_score = vs_multilayer.vs_multilayer(visual_feature_test, "CBR", middle_layer_dim=self.middle_layer_size, class_num=self.action_class_num, dropout=False, reuse=True)
        sim_score = tf.reshape(sim_score, [(self.action_class_num+1)*3])

        return sim_score


    def get_variables_by_name(self, name_list):
        """Get variables by name
        """
        v_list = tf.trainable_variables()

        v_dict = {}
        for name in name_list:
            v_dict[name] = []
        for v in v_list:
            for name in name_list:
                if name in v.name: v_dict[name].append(v)

        for name in name_list:
            print("Variables of <"+name+">")
            for v in v_dict[name]:
                print( "    "+v.name)
        return v_dict


    def add_train_op(self, loss):
        """Add train operation
        """
        print('Add train operation')

        v_dict = self.get_variables_by_name(["CBR"])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if self.config.opm_type == 'adam_wd':
            vs_optimizer = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
            optimizer = vs_optimizer(weight_decay=1e-4, learning_rate=self.vs_lr, name='vs_adam')
            with tf.control_dependencies(update_ops):
                vs_train_op = optimizer.minimize(loss, var_list=v_dict["CBR"])
        elif self.config.opm_type == 'adam':
            vs_optimizer = tf.train.AdamOptimizer(self.vs_lr, name='vs_adam')
            with tf.control_dependencies(update_ops):
                vs_train_op = vs_optimizer.minimize(loss, var_list=v_dict["CBR"])

        return vs_train_op


    def build(self):
        """Build the model
        """
        print('Construct the network')

        self.add_placeholders()

        if self.config.norm == 'l2':
            visual_featmap_ph_train_norm = tf.nn.l2_normalize(self.visual_featmap_ph_train, dim=1)
            visual_featmap_ph_test_norm = tf.nn.l2_normalize(self.visual_featmap_ph_test, dim=1)
        elif self.config.norm == 'No':
            visual_featmap_ph_train_norm = self.visual_featmap_ph_train
            visual_featmap_ph_test_norm = self.visual_featmap_ph_test

        self.loss, self.loss_reg, self.loss_cls = self.add_loss_op(visual_featmap_ph_train_norm, self.offset_ph, self.label_ph, self.one_hot_label_ph)
        self.vs_train_op = self.add_train_op(self.loss)
        self.vs_eval_op = self.predict(visual_featmap_ph_test_norm)

        self.init_session()


    def train(self):
        """Training
        """
        self.add_summary()

        for step in xrange(self.config.max_steps):

            # if step <= 3000:
            lr = self.config.lr
            # else:
            #    lr = self.config.lr/10

            start_time = time.time()

            feed_dict = self.get_feed_dict(lr)
            duration1 = time.time()-start_time

            [_, loss_value, loss_reg_value, loss_cls_value, summary] = self.sess.run([self.vs_train_op, self.loss, self.loss_reg, self.loss_cls, self.merged], feed_dict=feed_dict)

            duration2 = time.time()-start_time

            print('Step %d: loss=%.2f, loss_reg=%.2f, loss_cls=%.2f, (%.3f sec),(%.3f sec)' % (step, loss_value, loss_reg_value, loss_cls_value, duration1, duration2))

            self.file_writer.add_summary(summary, step)
            if (step+1)==4000 or (step+1) % self.config.test_steps==0:
                self.save_session(step+1)


    def do_eval_slidingclips(self, save_name):
        """Do evaluation based on proposals and save the coresponding score and offset to a pickle file in './eval/test_results' folder
        """
        test_len_dict = tools.load_length_dict(type='test')
        reg_result_dict = {}

        for k,test_sample in enumerate(self.test_set.test_samples): 

            reg_result_dict[k] = []

            if k%1000==0:
                print(str(k)+"/"+str(len(self.test_set.test_samples)))
            movie_name = test_sample[0]

            init_clip_start = test_sample[1]
            init_clip_end = test_sample[2]

            clip_start = init_clip_start
            clip_end = init_clip_end
            final_action_prob = np.zeros([(self.config.action_class_num+1)*3*self.config.cas_step])

            if clip_start >= clip_end:
                reg_result_dict[k].append(final_action_prob)
                continue

            for i in range(self.config.cas_step):
                if clip_start >= clip_end:
                    break

                if self.config.feat_type == 'Pool':

                    featmap = dataset.get_pooling_feature(self.test_set.flow_feat_dir, self.test_set.appr_feat_dir, movie_name,clip_start, clip_end, self.config.pool_level, self.config.unit_size, self.config.unit_feature_size, self.config.fusion_type)
                    left_feat = dataset.get_left_context_feature(self.test_set.flow_feat_dir, self.test_set.appr_feat_dir, movie_name, clip_start, clip_end, self.config.ctx_num, self.config.unit_size, self.config.unit_feature_size, self.config.fusion_type)
                    right_feat = dataset.get_right_context_feature(self.test_set.flow_feat_dir, self.test_set.appr_feat_dir, movie_name, clip_start, clip_end, self.config.ctx_num, self.config.unit_size, self.config.unit_feature_size, self.config.fusion_type)

                    mean_ = np.hstack((left_feat, featmap, right_feat))

                    feat = mean_

                feat = np.reshape(feat, [1, self.config.visual_feature_dim])

                feed_dict = {
                    self.visual_featmap_ph_test: feat
                }

                outputs = self.sess.run(self.vs_eval_op, feed_dict=feed_dict)

                action_score = outputs[1:self.config.action_class_num+1]
                action_prob = tools.softmax(action_score)

                final_action_prob[(i)*(self.config.action_class_num+1)*3: (i+1)*(self.config.action_class_num+1)*3]= outputs

                action_cat = np.argmax(action_prob)+1
                round_reg_end = clip_end+round(outputs[(self.config.action_class_num+1)*2+action_cat])*self.config.unit_size
                round_reg_start = clip_start+round(outputs[self.config.action_class_num+1+action_cat])*self.config.unit_size
                if round_reg_start < 0 or round_reg_end > test_len_dict[movie_name]-15 or round_reg_start >= round_reg_end:
                    round_reg_end = clip_end
                    round_reg_start = clip_start
                reg_end = clip_end+outputs[(self.config.action_class_num+1)*2+action_cat]*self.config.unit_size
                reg_start = clip_start+outputs[self.config.action_class_num+1+action_cat]*self.config.unit_size
                clip_start = round_reg_start
                clip_end = round_reg_end

            reg_result_dict[k].append(final_action_prob)

        pickle.dump(reg_result_dict, open("./eval/test_results/"+save_name+"_outputs.pkl","wb"))
