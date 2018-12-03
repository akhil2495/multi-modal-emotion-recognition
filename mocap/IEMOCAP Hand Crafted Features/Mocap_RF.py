import codecs
import json
import math
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

code_path = os.path.dirname(os.path.realpath(os.getcwd()))
emotions_used = np.array(['ang', 'exc', 'neu', 'sad'])
data_path = code_path + "/data/sessions/"
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
framerate = 16000

with open(data_path + '/../'+'data_collected.pickle', 'rb') as handle:
    data2 = pickle.load(handle)


Y = []
for ses_mod in data2:
    Y.append(ses_mod['emotion'])

Y = label_binarize(Y, emotions_used)


mocap_rot = []

for ses_mod in data2:
    x_rot = ses_mod['mocap_rot']
    if(x_rot.shape != (200, 165)):
        x_rot = np.zeros((200, 165))
    x_rot[np.isnan(x_rot)] = 0
    x_rot = np.amax(x_rot, axis=0)
    mocap_rot.append(x_rot)

mocap_rot = np.array(mocap_rot)


data = []
with codecs.open("name.json", "r", "utf-8") as f:
    for line in f:
        dic = json.loads(line)
        data.append(dic)

names = data[3]

rf = RandomForestClassifier(
    n_jobs=-1, n_estimators=566, random_state=666, oob_score=True)
rf.fit(mocap_rot, Y)

print(rf.oob_score_)


fea_imp = sorted(
    zip(map(lambda x: round(x, 10), rf.feature_importances_), names), reverse=True)
with codecs.open('feature_importances.json', 'w', 'utf-8') as outf:
    for data in fea_imp:
        json.dump(data, outf, ensure_ascii=False)
        outf.write('\n')


# print("Features sorted by their score:")
# print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))
