import numpy as np

import pickle
import sys
import time

action_class_num = 20
unit_size = 16.0

pkl_appr = sys.argv[1]
pkl_flow = sys.argv[2]
t_v = sys.argv[3]

def softmax(x):
    return np.exp(x/float(t_v))/np.sum(np.exp(x/float(t_v)), axis=0)

feq_refine_early=True
act_refine_freq = np.load('val_training_samples_16-512_window_freq.npy')
binary_flag = True

if pkl_appr == pkl_flow:
    name = pkl_appr.split('.pkl')[0]
else:
    name = 'joint'

appr_dict=pickle.load(open("./test_results/"+pkl_appr, 'rb'), encoding='latin1')
flow_dict=pickle.load(open("./test_results/"+pkl_flow, 'rb'), encoding='latin1')

result_dict = {}

pickle_file=open("./test_results/"+name+'_'+str(t_v)+".pkl","wb")

test_clip_path = '../props/test_proposals_from_TURN.txt'

f = open(test_clip_path)

len_appr = len(appr_dict)

count = -1
cnt_v=0
for line in f:
    count+=1
    # print(count, appr_dict[count])
    if appr_dict[count] == [] or flow_dict[count] == []:
        continue
    [outputs_appr] = appr_dict[count]

    outputs_appr1 = outputs_appr[:63]
    outputs_appr2 = outputs_appr[63:126]
    outputs_appr3 = outputs_appr[126:189]

    outputs_a = (outputs_appr1+outputs_appr2+outputs_appr3)

    [outputs_flow] = flow_dict[count]
    outputs_flow1 = outputs_flow[:63]
    outputs_flow2 = outputs_flow[63:126]
    outputs_flow3 = outputs_flow[126:189]

    outputs_f = (outputs_flow1+outputs_flow2+outputs_flow3)

    outputs = (outputs_a+outputs_f)/2.0

    movie_name = line.rstrip().split(" ")[0]
    clip_start = float(line.rstrip().split(" ")[1])
    clip_end = float(line.rstrip().split(" ")[2])
    # print(movie_name)

    if not movie_name in result_dict:
        result_dict[movie_name] = []
        result_dict[movie_name].append([]) # start
        result_dict[movie_name].append([]) # end
        result_dict[movie_name].append([]) # cls score
        result_dict[movie_name].append([]) # label category


    action_score = np.array(outputs[:action_class_num+1])

    action_prob = softmax(action_score)
    # print(action_prob)
    best_score = np.max(action_prob)
    action_cat = np.argmax(action_prob)

    # print(best_score, action_cat)
    if action_cat == 0:
        continue
    else:
        cnt_v+=1

    reg_start = clip_start+outputs[action_class_num+1:(action_class_num+1)*2]*unit_size
    reg_end = clip_end+outputs[(action_class_num+1)*2:]*unit_size

    reg_start = reg_start[1:]
    reg_end = reg_end[1:]
    # print('reg start and end are: ')
    # # print(reg_start, reg_end)
    # print(outputs[action_class_num+1:(action_class_num+1)*2], outputs[(action_class_num+1)*2:])

    cls_score = action_prob[1:]
    # print(act_refine_freq.shape)
    if feq_refine_early:
        clip_length_index=[[16,32,64,128,256,512].index(min([16,32,64,128,256,512],key=lambda x:abs(x-int(reg_end[i]-reg_start[i])))) for i in range(len(reg_start))]
        cls_score = [act_refine_freq[i+1, clip_length_index[i]]*cls_score[i] for i in range(len(cls_score))]

    class_idx = np.argmax(cls_score)
    st = reg_start[class_idx]
    ed = reg_end[class_idx]

    if st < ed and ed - st > 15:
        result_dict[movie_name][0].append(st)
        result_dict[movie_name][1].append(ed)

        result_dict[movie_name][2].append(cls_score[class_idx])
        result_dict[movie_name][3].append(class_idx+1)
        # print((clip_start, clip_end), (st, ed))

pickle.dump(result_dict, pickle_file)
print (cnt_v)