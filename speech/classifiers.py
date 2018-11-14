from sklearn.mixture import GaussianMixture
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from features import speech_feat_extract
import numpy as np
import pickle

sess = ['Session'+str(i+1) for i in range(5)]

data = speech_feat_extract(sess, '../../../../HBA/IEMOCAP_full_release/')
data.compute_mfcc()#stat('mfcc')
#print type(data.mfcc)
#X = np.array(data.mfcc_stat)
y = np.array([data.cat2catid[data.id2label[i]] for i in data.mfcc.keys()])

X = []
for dat in data.mfcc.values():
    model = GaussianMixture(n_components = 8, init_params = 'kmeans', max_iter = 50, random_state = 1234)
    model.fit(dat)
    X.append(model.means_.ravel())

with open('supervectors.pkl', 'wb') as f:
    pickle.dump(X, f)

with open('labels.pkl', 'wb') as f:
    pickle.dump(y, f)

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

kf = KFold(n_splits=5)
acc = []
X = np.transpose(X)
for train_index, test_index in kf.split(X):
    tr_x, te_x = X[train_index], X[test_index]
    tr_y, te_y = y[train_index], y[test_index]
    clf = svm.SVC()
    clf.fit(tr_x, tr_y)
    pred = clf.predict(te_x)
    acc.append(accuracy_score(te_y, pred))

print sum(acc)/float(len(acc))


