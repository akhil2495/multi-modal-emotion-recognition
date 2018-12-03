import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

from features import *
from helper import *


batch_size = 64
nb_feat = 34
nb_class = 4
nb_epoch = 80

optimizer = 'Adadelta'


code_path = os.path.dirname(os.path.realpath(os.getcwd()))
emotions_used = np.array(['ang', 'exc', 'neu', 'sad'])
data_path = code_path + "/../data/sessions/"
# sessions = ['Session1', 'Session2']
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
framerate = 16000

def get_mocap_rot(path_to_mocap_rot, filename, start,end):
    f = open(path_to_mocap_rot + filename, 'r').read()
    f = np.array(f.split('\n'))
    mocap_rot = []
    mocap_rot_avg = []
    f = f[2:]
    counter = 0
    for data in f:
        counter+=1
        data2 = data.split(' ')
        if(len(data2)<2):
            continue
        if(float(data2[1])>start and float(data2[1])<end):
            mocap_rot_avg.append(np.array(data2[2:]).astype(np.float))
        if(float(data2[1])>end):
            break      
    mocap_rot_avg = np.array_split(np.array(mocap_rot_avg), 200)
    for spl in mocap_rot_avg:
        mocap_rot.append(np.mean(spl, axis=0))
    return np.array(mocap_rot)


def get_mocap_rot_raw(path_to_mocap_rot, filename):
    f = open(path_to_mocap_rot + filename, 'r').read()
    f = np.array(f.split('\n'))
    mocap_rot = []
    f = f[2:]
    counter = 0
    for data in f:
        data2 = data.split(' ')
        if(len(data2)<2):
            continue
        counter+=1
        data2 = np.array(data2[2:]).astype(np.float)
        data2[np.isnan(data2)]=0
        mocap_rot.append(data2)
    return mocap_rot, counter


def replace_ori_mocap_rot(path_to_mocap_rot, filename, path_to_mocap_rot_pca, data_mat):
    f = open(path_to_mocap_rot + filename, 'r').read()
    f = np.array(f.split('\n'))
    mocap_rot = []
    f_head = f[0:2]
    f = f[2:]
    i = 0
    for data in f:
        data2 = data.split(' ')
        if(len(data2)<2):
            continue
        data2_head = data2[0:2]
        data_mat_list = np.ndarray.tolist(data_mat[i])
        data2 = data2_head + data_mat_list
        mocap_rot.append(data2)
        i += 1
    total = f_head.tolist() + mocap_rot

    with open(path_to_mocap_rot_pca + filename, 'w') as f:
        for item in total:
            f.write("%s\n" % item)


def read_iemocap_mocap():
    total_raw_data = []
    total_raw_data_index = [0]
    data = []
    ids = {}

    for session in sessions:
        path_to_wav = data_path + session + '/dialog/wav/'

        path_to_mocap_rot = data_path + session + '/dialog/MOCAP_rotated/'

        files2 = os.listdir(path_to_wav)

        files = []
        for f in files2:
            if f.endswith(".wav"):
                if f[0] == '.':
                    files.append(f[2:-4])
                else:
                    files.append(f[:-4])
        

        for f in files:       
            print(f)
            mocap_f = f
            raw_data, counter = get_mocap_rot_raw(path_to_mocap_rot, mocap_f + '.txt')
            total_raw_data_index.append(len(total_raw_data) + counter)
            total_raw_data = total_raw_data + raw_data
    
    total_raw_data = np.array(total_raw_data)
    # total_num = total_raw_data.shape[0]
    
    # total_raw_data = np.asarray(total_raw_data).reshape(1, total_num, 165)[0]

    ipca = PCA(n_components=0.99)
    total_raw_data = ipca.fit_transform(total_raw_data)

    print("\nStart Replacing:")
    i = 0
    for session in sessions:
        path_to_wav = data_path + session + '/dialog/wav/'
        path_to_mocap_rot = data_path + session + '/dialog/MOCAP_rotated/'
        path_to_mocap_rot_pca = data_path + session + '/dialog/pca/MOCAP_rotated/'
        isCreated = os.path.exists(path_to_mocap_rot_pca)
        if not isCreated:
            os.makedirs(path_to_mocap_rot_pca)

        files2 = os.listdir(path_to_wav)

        files = []
        for f in files2:
            if f.endswith(".wav"):
                if f[0] == '.':
                    files.append(f[2:-4])
                else:
                    files.append(f[:-4])
        
        
        print()
        for f in files:       
            print(f)
            mocap_f = f
            start = total_raw_data_index[i]
            end = total_raw_data_index[i+1]
            data_mat = total_raw_data[start:end]
            replace_ori_mocap_rot(path_to_mocap_rot, mocap_f + '.txt', path_to_mocap_rot_pca, data_mat)
            i += 1
            
    del total_raw_data


read_iemocap_mocap()


