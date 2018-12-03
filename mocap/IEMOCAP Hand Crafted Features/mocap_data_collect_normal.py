import numpy as np
import os
import json
import codecs
from features import *
from helper import *


code_path = os.path.dirname(os.path.realpath(os.getcwd()))
emotions_used = np.array(['ang', 'exc', 'neu', 'sad'])
data_path = code_path + "/../data/sessions/"
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
framerate = 16000

def generate_mocap_rot_dict(path_to_mocap_rot, filename):
    dict1 = {}
    dict2 = {}
    f = open(path_to_mocap_rot + filename, 'r').read()
    f = np.array(f.split('\n'))
    mocap_rot = []
    mocap_rot_avg = []
    f_head_1 = f[0]
    f_head_2 = f[1]
    data2_1 = f_head_1.split(' ')[2:]
    data2_2 = f_head_2.split(' ')

    i = 0
    for data3 in data2_1:
        dict1[i]=data3
        i += 1
    
    i = 0
    for data3 in data2_2:
        dict2[i]=data3   
        i += 1
    return dict1, dict2


def read_iemocap_mocap():
    data = []
    ids = {}
    for session in sessions:
        path_to_wav = data_path + session + '/dialog/wav/'
        path_to_emotions = data_path + session + '/dialog/EmoEvaluation/'
        
        path_to_mocap_rot = data_path + session + '/dialog/MOCAP_rotated/'

        files2 = os.listdir(path_to_wav)

        files = []
        for f in files2:
            if f.endswith(".wav"):
                if f[0] == '.':
                    files.append(f[2:-4])
                else:
                    files.append(f[:-4])
                    

        f = files[0]     
        print(f)
        mocap_f = f
        dict1, dict2 = generate_mocap_rot_dict(path_to_mocap_rot, mocap_f + '.txt')
        with codecs.open('name.json','a', 'utf-8') as outf:
            json.dump(dic, outf, ensure_ascii=False)
            outfile.write('\n')
read_iemocap_mocap()

