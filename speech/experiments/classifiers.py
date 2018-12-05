from sklearn.mixture import GaussianMixture
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from features import speech_feat_extract
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

sess = ['Session'+str(i+1) for i in range(5)]

data = speech_feat_extract(sess, '../../../../HBA/IEMOCAP_full_release/')
data.compute_mfcc()

y = []
val = []
for k,v in data.mfcc.items():
    y.append(data.cat2catid[data.id2label[k]])
    val.append(v)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(val, y, test_size=0.3, random_state=1234)
X = X_train
y = y_train
print len(X)

#y = np.array([data.cat2catid[data.id2label[i]] for i in data.zcr.keys()])

X = []
for dat in X_train:
    for l in list(dat):
	X.append(l)
X = np.array(X)
y = np.array(y)

model = GaussianMixture(n_components = 2, init_params = 'kmeans', max_iter=1)
model.fit(X)
model_mean = model.means_
model_weight = model.weights_
model_precisions = model.precisions_

X = []
for dat in X_train:
    model = GaussianMixture(n_components = 2, weights_init = model_weight, means_init = model_mean, precisions_init = model_precisions, random_state = 1234)
    model.fit(dat, (-1,1))
    X.append(model.means_.ravel())
X = np.array(X)

with open('supervectors_mfcc.pkl', 'wb') as f:
    pickle.dump(X, f)

with open('labels.pkl', 'wb') as f:
    pickle.dump(y, f)

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

kf = KFold(n_splits=5)
acc = []
#X = np.transpose(X)
for train_index, test_index in kf.split(X):
    tr_x, te_x = X[train_index], X[test_index]
    tr_y, te_y = y[train_index], y[test_index]
    clf = svm.SVC()
    clf.fit(tr_x, tr_y)
    pred = clf.predict(te_x)
    acc.append(accuracy_score(te_y, pred))

print sum(acc)/float(len(acc))


