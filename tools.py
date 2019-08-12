import numpy as np

def load_length_dict(type='test'):
    
    len_dict = {}

    if type=='test':
        file_path = 'thumos14_video_length_test.txt'
    else:
        file_path = 'thumos14_video_length_val.txt'

    with open(file_path) as f:
        contents = f.readlines()

    for line in contents:
        line_s = line.strip().split()
        movie_name = line_s[0]
        movie_length = int(line_s[2])
        len_dict[movie_name] = movie_length

    return len_dict

def load_video_length(file):
    dic = {}
    with open(file) as f:
        content = f.readlines()

    for line in content:
        a,b,c = line.strip().split()
        video_name = a
        video_length = int(c)
        dic[video_name] = video_length
    return dic

def load_video_length_second(file):
    dic = {}
    with open(file) as f:
        content = f.readlines()

    for line in content:
        a,b,c = line.strip().split()
        video_name = a
        video_length = round(float(b),1)
        dic[video_name] = video_length
    return dic

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)
