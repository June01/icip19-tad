import numpy as np
from math import sqrt, ceil
import os
import random
import pickle

from scipy.interpolate import interp1d

cat_index_dict={
    "Background":0,
    "BaseballPitch":1,
    "BasketballDunk":2,
    "Billiards":3,
    "CleanAndJerk":4,
    "CliffDiving":5,
    "CricketBowling":6,
    "CricketShot":7,
    "Diving":8,
    "FrisbeeCatch":9,
    "GolfSwing":10,
    "HammerThrow":11,
    "HighJump":12,
    "JavelinThrow":13,
    "LongJump":14,
    "PoleVault":15,
    "Shotput":16,
    "SoccerPenalty":17,
    "TennisSwing":18,
    "ThrowDiscus":19,
    "VolleyballSpiking":20
}

def get_mean_variance_concat(vec):
    '''
        Input: It should be has two dimension(n, d)
        Output: (1, d)
    '''
    mean = np.mean(vec, axis=0)
    # mean_x_square = np.mean(vec**2, axis=0)
    # var = np.mean(mean_x_square)-mean**2
    var = np.var(vec, axis=0)

    return mean, var

def calculate_regoffset(clip_start, clip_end, round_gt_start, round_gt_end, unit_size):
    start_offset = (round_gt_start-clip_start)/unit_size
    end_offset = (round_gt_end-clip_end)/unit_size
    return start_offset, end_offset

def temporal_iou(span_A, span_B):
    """
    Calculates the intersection over union of two temporal "bounding boxes"

    span_A: (start, end)
    span_B: (start, end)
    """
    union = min(span_A[0], span_B[0]), max(span_A[1], span_B[1])
    inter = max(span_A[0], span_B[0]), min(span_A[1], span_B[1])

    if inter[0] >= inter[1]:
        return 0
    else:
        return float(inter[1] - inter[0]) / float(union[1] - union[0])

# Devide equally by pool_level
def get_pooling_feature(flow_feat_dir, appr_feat_dir, movie_name, start, end, pool_level, unit_size, unit_feature_size, which_feat):
    swin_step = unit_size
    all_feat = np.zeros([0, unit_feature_size], dtype=np.float32)
    current_pos = start
    
    xlen_unit = int(end-start)/unit_size
    
    while current_pos<end:
        swin_start = current_pos
        swin_end = swin_start+swin_step
        if which_feat == 'flow':
            flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            feat = flow_feat
        elif which_feat == 'rgb':
            appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            feat = appr_feat
        else:
            flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            feat = np.hstack((flow_feat, appr_feat))
        all_feat = np.vstack((all_feat, feat))
        current_pos+=swin_step
    all_feat = np.reshape(all_feat, (-1, unit_feature_size))
    
    pool_feat=[]
    pool_var = []
    if pool_level == 1:
        pool_feat, pool_var = get_mean_variance_concat(all_feat)
    else:
        index = [int((xlen_unit/float(pool_level))*i) for i in range(pool_level)]
        index.append(int(xlen_unit-1))
        # print(index)
        for i in range(pool_level):
            if index[i] == index[i+1]:
                f_vec = all_feat[index[i]:index[i]+1]
            else:
                f_vec = all_feat[index[i]:index[i+1]]

            mean, var = get_mean_variance_concat(f_vec)
            mean = np.reshape(mean, (-1, unit_feature_size))
            var = np.reshape(var, (-1, unit_feature_size))
            pool_feat.append(mean)
            pool_var.append(var)
           
       
        pool_feat = np.array(pool_feat)
        pool_var = np.array(pool_var)

    pool_feat = np.reshape(pool_feat, (-1))
    pool_var = np.reshape(pool_var, (-1))

    return pool_feat, pool_var

def get_left_context_feature(flow_feat_dir, appr_feat_dir, movie_name, start, end, ctx_num, unit_size, unit_feature_size, which_feat):
    swin_step = unit_size
    all_feat = np.zeros([0, unit_feature_size], dtype=np.float32)
    count = 0
    current_pos = start
    context_ext = False
    while  count<ctx_num:
        swin_start = current_pos-swin_step
        swin_end = current_pos
        if os.path.exists(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
            if which_feat == 'flow':
                flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
                feat = flow_feat
            elif which_feat == 'rgb':
                appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
                feat = appr_feat
            else:
                flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
                appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
                feat = np.hstack((flow_feat, appr_feat))

            all_feat = np.vstack((all_feat,feat))
            context_ext = True
        current_pos-=swin_step
        count+=1

    if context_ext:
        feat_vec = all_feat
    else:
        feat_vec = np.zeros([1, unit_feature_size], dtype=np.float32)

    pool_feat, pool_var = get_mean_variance_concat(feat_vec)
    pool_feat = np.reshape(pool_feat, -1)
    pool_var = np.reshape(pool_var, -1)

    return pool_feat, pool_var

def get_right_context_feature(flow_feat_dir, appr_feat_dir, movie_name, start, end, ctx_num, unit_size, unit_feature_size, which_feat):
    swin_step = unit_size
    all_feat = np.zeros([0, unit_feature_size], dtype=np.float32)
    count = 0
    current_pos = end
    context_ext = False
    while  count<ctx_num:
        swin_start = current_pos
        swin_end = current_pos+swin_step
        if os.path.exists(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
            if which_feat == 'flow':
                flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
                feat = flow_feat
            elif which_feat == 'rgb':
                appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
                feat = appr_feat
            else:
                flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
                appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
                feat = np.hstack((flow_feat, appr_feat))
            all_feat = np.vstack((all_feat,feat))
            context_ext = True
        current_pos+=swin_step
        count+=1
    
    if context_ext:
        feat_vec = all_feat
    else:
        feat_vec = np.zeros([1, unit_feature_size], dtype=np.float32)

    pool_feat, pool_var = get_mean_variance_concat(feat_vec)
    pool_feat = np.reshape(pool_feat, -1)
    pool_var = np.reshape(pool_var, -1)

    return pool_feat, pool_var

def load_feature(flow_feat_dir, appr_feat_dir, movie_name, start, end, unit_size, unit_feature_size):
    swin_step = unit_size
    all_feat = np.zeros([0, unit_feature_size], dtype=np.float32)
    current_pos = start
    zero_feature = np.zeros(unit_feature_size, dtype=np.float32)
    while current_pos<=end:
        swin_start = current_pos
        swin_end = swin_start+swin_step
        if os.path.exists(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
            flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            # flow_feat = self.reduce_resolution(flow_feat)
            appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            # appr_feat = self.reduce_resolution(appr_feat)
            feat = np.hstack((flow_feat, appr_feat))
        else:
            feat = zero_feature
        all_feat = np.vstack((all_feat, feat))
        current_pos+=swin_step
    #pca_feat = PCA(n_components = 1024).fit_transform(all_feat)        
    return all_feat

def get_BSP_feature(flow_feat_dir, appr_feat_dir, movie_name, start_f, end_f, unit_size, bsp_level):

    num_sample_start = bsp_level/4
    num_sample_action = bsp_level/2
    num_sample_end = bsp_level/4
    num_sample_interpld = 1
    
    end = int(end_f/unit_size)
    start = int(start_f/unit_size)

    xlen = (end-start) + 1

    # map the function from segment num to features
    start_ext = start - int(xlen/2)
    end_ext = end + int(xlen/2)
    tmp_x = [i for i in range(start_ext, end_ext+1)]

    feature = load_feature(flow_feat_dir, appr_feat_dir, movie_name, float(start_ext*16+1), float(end_ext*16+1), unit_size, unit_feature_size)
    # print(feature.shape)
    # print(len(tmp_x), len(feature))
    f_action=interp1d(tmp_x,feature,axis=0)
    
    #insert(len(feature) == len(tmp_x))

    xmin_0=start-xlen/5
    xmin_1=start+xlen/5
    xmax_0=end-xlen/5
    xmax_1=end+xlen/5
   
    #start
    plen_start= (xmin_1-xmin_0)/(num_sample_start-1)
    plen_sample = plen_start / num_sample_interpld
    tmp_x_new = [ xmin_0 - plen_start/2 + plen_sample * ii for ii in range(num_sample_start*num_sample_interpld +1 )] 
    tmp_y_new_start_action=f_action(tmp_x_new)
    tmp_y_new_start = np.array([np.mean(tmp_y_new_start_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1], axis=0) for ii in range(num_sample_start) ])
    #end
    plen_end= (xmax_1-xmax_0)/(num_sample_end-1)
    plen_sample = plen_end / num_sample_interpld
    tmp_y_new = [ xmax_0 - plen_end/2 + plen_sample * ii for ii in range(num_sample_end*num_sample_interpld +1 )] 

    tmp_y_new_end_action=f_action(tmp_y_new)
    tmp_y_new_end = np.array([np.mean(tmp_y_new_end_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1], axis=0) for ii in range(num_sample_end) ])
    #action
    if num_sample_action == 1:
        final_feat = np.mean(feature, axis=0)
    else:
        plen_action= (end-start)/(num_sample_action-1)
        plen_sample = plen_action / num_sample_interpld
        tmp_x_new = [ start - plen_action/2 + plen_sample * ii for ii in range(num_sample_action*num_sample_interpld +1 )] 
        tmp_y_new_action=f_action(tmp_x_new)
        tmp_y_new_action = np.array([np.mean(tmp_y_new_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1], axis=0) for ii in range(num_sample_action) ])
        
        tmp_feature = np.concatenate([tmp_y_new_start,tmp_y_new_action,tmp_y_new_end])
        final_feat = np.reshape(tmp_feature, (-1))
        # final_feat = np.reshape(tmp_y_new_action, (-1))
    # print(final_feat.shape)
    return final_feat

def get_SSN_feature(flow_feat_dir, appr_feat_dir, movie_name, start, end, unit_size, unit_feature_size, which_feat):
    swin_step = unit_size
    all_feat = np.zeros([0, unit_feature_size], dtype=np.float32)
    current_pos = start
    
    xlen_unit = int(end-start)/unit_size
    start_unit = int(start/unit_size)
    end_unit = int(end/unit_size)
    
    left_feat = get_left_context_feature(flow_feat_dir, appr_feat_dir, movie_name, start+swin_step, end, which_feat)
    left_feat = np.reshape(left_feat, (-1))
    right_feat = get_right_context_feature(flow_feat_dir, appr_feat_dir, movie_name, start, end-swin_step, which_feat)
    right_feat = np.reshape(right_feat, (-1))

    while current_pos<end:
        swin_start = current_pos
        swin_end = swin_start+swin_step
        if which_feat == 'flow':
            flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            feat = flow_feat
        elif which_feat == 'rgb':
            appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            feat = appr_feat
        else:
            flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            feat = np.hstack((flow_feat, appr_feat))
        all_feat = np.vstack((all_feat, feat))
        current_pos+=swin_step
    all_feat = np.reshape(all_feat, (-1, unit_feature_size))
    
    pool_feat=[]

    middle_index = int(xlen_unit/2)
    left_middle = int(middle_index/2)
    right_middle = middle_index+left_middle
    
    start_feat = np.mean(all_feat[0:middle_index+1], axis=0)
    start_feat = np.reshape(start_feat, (-1))
    end_feat = np.mean(all_feat[middle_index:], axis=0)
    end_feat = np.reshape(end_feat, (-1))
    middle_feat = np.mean(all_feat, axis=0)   
    middle_feat = np.reshape(middle_feat, (-1))

    s1_feat = np.mean(all_feat[0:left_middle+1], axis=0)
    s1_feat = np.reshape(s1_feat, (-1))
    s2_feat = np.mean(all_feat[left_middle:middle_index+1], axis=0)
    s2_feat = np.reshape(s2_feat, (-1))
    s3_feat = np.mean(all_feat[middle_index:right_middle+1], axis=0)
    s3_feat = np.reshape(s3_feat, (-1))
    s4_feat = np.mean(all_feat[right_middle:], axis=0)
    s4_feat = np.reshape(s4_feat, (-1))

    final_feat = np.hstack((left_feat, s1_feat, s2_feat, middle_feat, s3_feat, s4_feat, right_feat))
    # final_feat = np.hstack((left_feat, s1_feat, start_feat, s2_feat, middle_feat, s3_feat, end_feat, s4_feat, right_feat))

    return final_feat

class TrainingDataSet(object):
    def __init__(self, config, flow_feat_dir, appr_feat_dir, clip_gt_path, background_path, model = 'CBR'):
        #it_path: image_token_file path
        self.config = config
        self.batch_size = self.config.batch_size
        print("Reading training data list from "+clip_gt_path+" and "+background_path)
        self.ctx_num = self.config.ctx_num
        self.feat_type = self.config.feat_type
        self.fusion_type = self.config.fusion_type
        self.bsp_level = self.config.bsp_level
        self.pool_level = self.config.pool_level

        self.unit_feature_size = self.config.unit_feature_size
        self.flow_feat_dir = flow_feat_dir
        self.appr_feat_dir = appr_feat_dir
        self.training_samples = []
        self.unit_size = self.config.unit_size
        self.action_class_num = self.config.action_class_num
        
        with open(clip_gt_path) as f:
            for l in f:
                movie_name = l.rstrip().split(" ")[0]
                clip_start = float(l.rstrip().split(" ")[1])
                clip_end = float(l.rstrip().split(" ")[2])
                gt_start = float(l.rstrip().split(" ")[3])
                gt_end = float(l.rstrip().split(" ")[4])
                round_gt_start = np.round(gt_start/self.unit_size)*self.unit_size+1
                round_gt_end = np.round(gt_end/self.unit_size)*self.unit_size+1
                if model == 'CBR':
                    category = l.rstrip().split(" ")[5]
                    cat_index = cat_index_dict[category]
                    iou_one_hot = 1.0
                else:
                    cat_index = int(l.rstrip().split(" ")[5])
                    iou_one_hot = float(l.rstrip().split(" ")[6])

                one_hot_label = np.zeros([self.action_class_num+1],dtype=np.float32)
                one_hot_label[cat_index] = 1.0

                self.training_samples.append((movie_name, clip_start, clip_end, gt_start, gt_end, round_gt_start, round_gt_end, cat_index, one_hot_label, iou_one_hot))
            
        print(str(len(self.training_samples))+" training samples are read")
        positive_num = len(self.training_samples)*1.0
        with open(background_path) as f:
            for l in f:
                # control the number of background samples
                if random.random()>1.0*positive_num/self.action_class_num/279584: continue
                movie_name = l.rstrip().split(" ")[0]
                clip_start = float(l.rstrip().split(" ")[1])
                clip_end = float(l.rstrip().split(" ")[2])
                one_hot_label = np.zeros([self.action_class_num+1], dtype=np.float32)

                self.training_samples.append((movie_name, clip_start, clip_end, 0, 0, 0, 0, 0, one_hot_label, 0))
        self.num_samples = len(self.training_samples)
        print(str(len(self.training_samples))+" training samples are read")

    def next_batch(self, model='CBR'):

        random_batch_index = random.sample(range(self.num_samples), self.batch_size)
        image_batch = np.zeros([self.batch_size, self.config.visual_feature_dim])
        length_batch = np.zeros([self.batch_size], dtype=np.int32)

        label_batch = np.zeros([self.batch_size], dtype=np.int32)

        iou_batch = np.zeros([self.batch_size], dtype=np.float32)
        offset_batch = np.zeros([self.batch_size,2], dtype=np.float32)
        one_hot_label_batch = np.zeros([self.batch_size, self.action_class_num+1], dtype=np.float32)

        index = 0
        while index < self.batch_size:
            k = random_batch_index[index]
            movie_name = self.training_samples[k][0]
            if self.training_samples[k][7]!=0:
                clip_start = self.training_samples[k][1]
                clip_end = self.training_samples[k][2]
                round_gt_start = self.training_samples[k][5]
                round_gt_end = self.training_samples[k][6]
                iou = self.training_samples[k][9]
                start_offset, end_offset = calculate_regoffset(clip_start, clip_end, round_gt_start, round_gt_end, self.unit_size) 
                # print(movie_name, clip_start, clip_end)
                if self.feat_type == 'Pool':
                    
                    featmap, middle_var = get_pooling_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name,clip_start, clip_end, self.pool_level, self.unit_size, self.unit_feature_size, self.fusion_type)
                    left_feat, left_var = get_left_context_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name, clip_start, clip_end, self.ctx_num, self.unit_size, self.unit_feature_size, self.fusion_type)
                    right_feat, right_var = get_right_context_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name, clip_start, clip_end, self.ctx_num, self.unit_size, self.unit_feature_size, self.fusion_type)
                    
                    mean = np.hstack((left_feat, featmap, right_feat))
                    var = np.hstack((left_var, middle_var, right_var))
                    
                    image_batch[index,:] = mean

                elif self.feat_type == 'SSN':
                    image_batch[index,:] = get_SSN_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name, clip_start, clip_end, self.unit_size, self.unit_feature_size, self.fusion_type)
                else:
                    image_batch[index,:] = get_BSP_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name, clip_start, clip_end, self.unit_size, self.bsp_level)

                label_batch[index] = self.training_samples[k][7]
                one_hot_label_batch[index,:] = self.training_samples[k][8]
                offset_batch[index,0] = start_offset
                offset_batch[index,1] = end_offset
                # iou_batch[index, :] = iou
                iou_batch[index] = iou
                length_batch[index] = clip_end-clip_start
                index+=1
            else:
                clip_start = self.training_samples[k][1]
                clip_end = self.training_samples[k][2]
                if self.feat_type == 'Pool':
                    
                    featmap, middle_var = get_pooling_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name,clip_start, clip_end, self.pool_level, self.unit_size, self.unit_feature_size, self.fusion_type)
                    left_feat, left_var = get_left_context_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name, clip_start, clip_end, self.ctx_num, self.unit_size, self.unit_feature_size, self.fusion_type)
                    right_feat, right_var = get_right_context_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name, clip_start, clip_end, self.ctx_num, self.unit_size, self.unit_feature_size, self.fusion_type)
                    
                    mean = np.hstack((left_feat, featmap, right_feat))
                    var = np.hstack((left_var, middle_var, right_var))
  
                    image_batch[index,:] = mean
                elif self.feat_type == 'SSN':
                    image_batch[index,:] = get_SSN_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name, clip_start, clip_end, self.unit_size, self.unit_feature_size, self.fusion_type)
                else:
                    image_batch[index,:] = get_BSP_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name, clip_start, clip_end, self.unit_size, self.bsp_level)
                
                label_batch[index] = 0
                length_batch[index] = 0
                one_hot_label_batch[index,:] = self.training_samples[k][8]
                offset_batch[index,0] = 0
                offset_batch[index,1] = 0
                # iou_batch[index,:] = self.training_samples[k][9]
                iou_batch[index] = self.training_samples[k][9]
                index+=1  

        return image_batch, label_batch, offset_batch, one_hot_label_batch
            
def load_video_length(file_path):
    video_len_dict = {}
    file = open(file_path)
    lines = file.readlines()
    for line in lines:
        content = line.strip('\n').split(' ')
        name = content[0]
        length = int(content[2])
        video_len_dict[name] = length

    return video_len_dict

class TestingDataSet(object):
    def __init__(self, config, flow_feat_dir, appr_feat_dir, test_clip_path, batch_size, test_len_dict):
        self.config = config
        self.batch_size = batch_size
        self.flow_feat_dir = flow_feat_dir
        self.appr_feat_dir = appr_feat_dir
        print("Reading testing data list from "+test_clip_path)
        # video_length_dict = load_video_length('/home/june/code/CBR/thumos14_video_length_test.txt')
        self.test_samples = []
        self.unit_size = self.config.unit_size
        with open(test_clip_path) as f:
            for l in f:
                movie_name = l.rstrip().split(" ")[0]
                clip_start = float(l.rstrip().split(" ")[1])
                clip_end = float(l.rstrip().split(" ")[2])
                clip_start = max(0.0, clip_start)
                clip_end = min(float(test_len_dict[movie_name]), clip_end)
                round_start = np.round(clip_start/self.unit_size)*self.unit_size+1
                round_end = np.round(clip_end/self.unit_size)*self.unit_size+1
                if round_end > test_len_dict[movie_name]-15:
                    round_end = round_end-self.config.unit_size
                self.test_samples.append((movie_name, round_start, round_end))
        self.num_samples = len(self.test_samples)
        print("test clips number: "+str(len(self.test_samples)))
       
 


