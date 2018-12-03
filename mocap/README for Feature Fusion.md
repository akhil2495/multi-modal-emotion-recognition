code:

**data process:**   data.py & normal_feature.py

divide dataset into train-test according to split_as_in_original_paper.pickle

get CNN/LSTM/ 3 types of text// 2 types of speech//normal  features (See in data)2

**SVM model**:   svm.py

using svm model to classify emotions

**RandomForest feature selection:**   rf.py

using randomforest  to train the model, and choose the best 256 features according to the weight of different features.

**autoencoder feature selection:**  autoencoder.py

the form of  data in the hiden layer is 256. 

make the form of  output layer the same as input layer.

get the hiden layer features 



