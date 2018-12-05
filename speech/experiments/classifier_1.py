from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from features import speech_feat_extract
import numpy as np
import pickle

sess = ['Session'+str(i+1) for i in range(5)]

data = speech_feat_extract(sess, '../../../../HBA/IEMOCAP_full_release/')
data.compute_stat('mfcc')

with open('mfcc_stat.pkl', 'wb') as f:
    pickle.dump(data.mfcc_stat, f)

with open('labels.pkl', 'wb') as f:
    pickle.dump(data.label, f)

#with open('mfcc_stat.pkl') as f:
#    data = pickle.load(f)

#y = []
X = []
for v in data.mfcc_stat:
    X.append(v)
#    y.append(k)

X = np.array(X)
y = np.array(data.label)

kf = KFold(n_splits=5)
acc = []
#X = np.transpose(X)
for train_index, test_index in kf.split(X):
    train_index = np.array([i for i in range(3501)])
    test_index = np.array([i for i in range(3501, 4936, 1)])
    tr_x, te_x = X[train_index], X[test_index]
    tr_y, te_y = y[train_index], y[test_index]
    clf = svm.SVC()
    clf.fit(tr_x, tr_y)
    pred = clf.predict(te_x)
    acc.append(accuracy_score(te_y, pred))

print sum(acc)/float(len(acc))
