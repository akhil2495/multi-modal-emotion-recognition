from python_speech_features import mfcc
import scipy.io.wavfile as wav
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import time
import librosa

class speech_feat_extract():
    
    def __init__(self, sessions, datapath, categories = ['ang', 'exc', 'neu', 'sad']):
        self.sessions = sessions
        self.datapath = datapath
        
        self.category_list = categories
        self.mfcc = {}
        self.zcr = {}
        self.pitch = {}
        self.mfcc_stat = {}
        self.zcr_stat = {}
        self.pitch_stat = {}
        
        self.id2label = {}
        self.label2ids = {}

        def parse_groundtruth(fpath):
            with open(fpath) as fil:
	        for line in fil:
                    if line.startswith('['):
                        words = line.rstrip().split('\t')
                        self.id2label[words[1]] = words[2]
                        if self.label2ids.has_key(words[2]):
                            self.label2ids[words[2]].append(words[1])
                        else:
                            self.label2ids[words[2]] = [words[1]]
        
        for session in sessions:
            for subdir, dirs, files in os.walk(datapath + session + '/dialog/EmoEvaluation/all'):
                for f in files:
                    fpath = subdir + os.sep + f
                    if fpath.endswith('.txt'):
                        parse_groundtruth(fpath)
                        
        self.cat2catid = {}
        for i in range(len(categories)):
            self.cat2catid[categories[i]] = i+1
            
        self.wavpaths = self.filepath_helper('/sentences/wav', '.wav')
        print len(self.wavpaths)
            
    def filepath_helper(self, relpath, filetype):
        filepaths = []
        for session in self.sessions:
            for subdir, dirs, files in os.walk(self.datapath + session + relpath):
                for f in files:
                    if f.endswith(filetype) and self.id2label[f[:-4]] in self.category_list:
                        filepaths.append([subdir + os.sep + f, f[:-4]])
        return filepaths
                        
        
    def compute_mfcc(self):
        wavpath = '/sentences/wav/'
        filepaths = self.wavpaths
        print 'Computing MFCC for ' + str(len(filepaths)) + ' utterances'
        for f in filepaths:
            (rate,sig) = wav.read(f[0])
            self.mfcc[f[1]] = mfcc(sig,rate)
        
    def compute_zcr(self):
        filepaths = self.wavpaths
        for f in filepaths:
            sig, sr = librosa.load(f[0])
            self.zcr[f[1]] = librosa.feature.zero_crossing_rate(sig)[0]
        
    def compute_pitch(self):
        filepaths = self.wavpaths
        for f in filepaths:
            subprocess.call(['/usr/bin/praat', '--run', 'extract_pitch.praat', f[0]])
            pitch = {}
            with open('temp.pitch') as fil:
                for i in fil:
                    val = i.split()
                    if val[1] != '--undefined--' and val[0] != 'Pitch':
                        pitch[float(val[0])] = float(val[1])
            self.pitch[f[1]] = pitch
        
    def compute_stat(self, feat):
        if feat == 'mfcc':
            if self.mfcc:
                X = self.mfcc
            else:
                self.compute_mfcc()
                X = self.mfcc
        elif feat == 'zcr':
            if self.zcr:
                X = self.zcr
            else:
                self.compute_zcr()
                X = self.zcr
        
        X_mean = np.empty((0,13), float)
        X_var = np.empty((0,13), float)
        for k,v in X.items():
            A = np.mean(v, axis=0)
            X_mean = np.concatenate((X_mean, np.reshape(A, (-1, A.shape[0]))), axis=0)
            A = np.var(v, axis=0)
            X_var = np.concatenate((X_var, np.reshape(A, (-1, A.shape[0]))), axis=0)
                
        if feat == 'mfcc':
            self.mfcc_stat = list(np.concatenate((X_mean, X_var), axis = 1))
        elif feat == 'zcr':
            self.zcr_stat = list(np.concatenate((X_mean, X_var), axis=1))
