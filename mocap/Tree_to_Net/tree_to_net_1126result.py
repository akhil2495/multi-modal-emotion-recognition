from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn import tree
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
import os

def get_netpara_tree(tree,feature_names,class_num):
    threshold = tree.tree_.threshold
    feature = tree.tree_.feature
    value = tree.tree_.value
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    n=len(threshold)
    m=len(feature_names)
    k=int((n+1)/2)
    a = np.zeros((k-1,m))
    b = np.zeros((k,k-1))
    c = np.zeros((k,class_num))
    j=0
    node_ixs=np.zeros(n)
    node_ixs=node_ixs-1
    for i in range(n):
        if threshold[i] != -2:
            a[j,feature[i]]=threshold[i]
            node_ixs[i]=j
            j=j+1
    def find(x,right,left):
        for i in range(len(right)):
            if right[i]==x:
                return i,-1
        for i in range(len(left)):
            if left[i]==x:
                return i,1
    j=0
    for i in range(n):
        if right[i]== -1:
            x,y= find(i,right,left)
            b[j,int(node_ixs[x])]=y
            while(x!=0):
                x,y= find(x,right,left)
                b[j,int(node_ixs[x])]=y
            c[j]=value[i]
            j=j+1
    return a,b,c

def change_parameters1(fcn,a_c):
    a,b=fcn.named_parameters()
    n = len(a)
    for i in range(len(a[1])):
        for j in range(len(a[1][0])):
            if a_c[i][j]==0:
                (a[1].data)[i][j] = 0
            else:
                (a[1].data)[i][j] = 1
                (b[1].data)[i] = -a_c[i][j]
#     print (a)
#     print (b)
    
def change_parameters2(fcn,c_c):
    a,b=fcn.named_parameters()
    n = len(a)
    for i in range(len(a[1])):
        for j in range(len(a[1][0])):
            (a[1].data)[i,j] = c_c[j][i]
    for i in range(len(b[1])):
        (b[1].data)[i] = 0
    
class Tree_net(nn.Module):
    def __init__(self,a,b,c,class_num):
        super(Tree_net, self).__init__()
        self.fcn1 = nn.Linear(len(a[0]), len(a))
        self.fcn2 = nn.Linear(len(b), class_num)
        self.b=b 
        self.a=a
        self.c=c
        change_parameters1(self.fcn1,self.a)
        change_parameters2(self.fcn2,self.c)
    def forward(self, input):
        a,b=self.fcn1.named_parameters()
        batch_size = len(input)
        x = self.fcn1(input)
        m = len(self.b)
        x = x*10
        x = torch.sigmoid(x)
        x_c = Variable(torch.ones(batch_size,m))
        if torch.cuda.is_available():
            x_c = x_c.cuda()
        for i in range(m):
            for j in range(len(self.b[i])):
                if self.b[i][j] == 1:
                    if torch.cuda.is_available():
                        # x_c[:,i] = x_c[:,i].clone().cuda() *(1-x[:,j])
                        x_c[:,i] = x_c[:,i].clone() *(1-x[:,j]) 
                    else:
                        x_c[:,i] = x_c[:,i].clone() *(1-x[:,j]) 
                if self.b[i][j] == -1:
                    if torch.cuda.is_available():
                        # x_c[:,i] = x_c[:,i].clone().cuda() *x[:,j]
                        x_c[:,i] = x_c[:,i].clone() *x[:,j] 
                    else:
                        x_c[:,i] = x_c[:,i].clone() *x[:,j]
        output=self.fcn2(x_c)
        return output


class random_forest_net(nn.Module):
    def __init__(self,random_forest,feature_names,class_num):
        super(random_forest_net, self).__init__()
        self.n_estimators=len(random_forest.estimators_)
        self.forest_net=nn.ModuleList()
        x = Variable(torch.zeros(class_num))
        for i in range(self.n_estimators):
            tree = random_forest.estimators_[i]
            a,b,c=get_netpara_tree(tree,feature_names,class_num)
            tree_net=Tree_net(a,b,c,class_num)
            if torch.cuda.is_available():
                tree_net = tree_net.cuda()
            (self.forest_net).append(tree_net)
            
    def forward(self, input):
        x = Variable(torch.zeros(class_num))
        if torch.cuda.is_available():
            x=x.cuda()
        for i in range(self.n_estimators):
            # print(i)
            x=x+(self.forest_net[i])(input)/self.n_estimators
        return x

def get_accuracy(y_predict,y_true):
    m=0.
    for i in range(len(y_predict)):
        if y_predict[i]== y_true[i]:
            m=m+1
    return m/len(y_predict)

def test_tree_net(rfc_net,X_test,y_test):
    n=len(X_test)
    m=int(n/100)
    accuracy_all = 0
    for i in range(m):
        x=torch.FloatTensor(X_test[0+100*i:100+100*i])
        if torch.cuda.is_available():
            x=x.cuda()
        out = rfc_net(x)
        if torch.cuda.is_available():
            y_predict=(torch.max(F.softmax(out), 1)[1]).cpu().numpy()
        else:
            y_predict=(torch.max(F.softmax(out), 1)[1]).numpy()
        y_true=y_test[0+100*i:100+100*i]
        accuracy_all = accuracy_all+get_accuracy(y_predict,y_true)
    accuracy_all=accuracy_all/m
    # print ("net_result:")
    # print (accuracy_all)
    return accuracy_all

def train(rfc_net,X_train,y_train,X_test,ytest,epch,f):
    for i in range(epch):
        loss_all = 0
        X_train,y_train = shuffle(X_train,y_train)
        for j in range(int(len(X_train)/100)):
            learning_rate = 0.00001 # If you set this too high, it might explode. If too low, it might not learn 
            criterion=nn.CrossEntropyLoss()
            optimizer = optim.SGD(rfc_net.parameters(), lr=learning_rate, momentum=0.9)
            inputs=torch.FloatTensor(X_train[0+j*100:100+j*100])
            lable=torch.LongTensor(y_train[0+j*100:100+j*100])
            if torch.cuda.is_available():
                inputs=inputs.cuda()
                lable=lable.cuda()
            output= rfc_net(inputs)
        #     print (output)
            loss = criterion(output, lable)
            loss.backward()
            optimizer.step()
            loss_all = loss+loss_all
        loss_all = loss_all/10
        print (loss_all)
        if i%5==0:
            accuracy_all=test_tree_net(rfc_net,X_train,y_train)
            print ("train_net_result:"+str(accuracy_all))
            f.write(str(i)+":train_net_result:"+str(accuracy_all)+'\n')
            accuracy_all=test_tree_net(rfc_net,X_test,ytest)
            print ("test_net_result:"+str(accuracy_all))
            f.write(str(i)+":test_net_result:"+str(accuracy_all)+'\n')
    return rfc_net

def get_test(Y_ori):
    y_new= np.ones(len(Y_ori))
    for i in range(len(y_new)):
        for j in range(4):
            if Y_ori[i,j] == 1:
                y_new[i] = j
    return y_new

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
    # X_train = np.loadtxt("data/lstm_train.txt")
    # X_test = np.loadtxt("data/lstm_test.txt")
    # Y_train = np.loadtxt("data/y_train.txt")
    # Y_test = np.loadtxt("data/y_test.txt")
    X_train = np.loadtxt("data/Adam_output_train.txt")
    X_test = np.loadtxt("data/Adam_output_test.txt")
    Y_train = np.loadtxt("data/ytrain.txt")
    Y_test = np.loadtxt("data/ytest.txt")
    # X_train = np.loadtxt("data/SGD_output_train.txt")
    # X_test = np.loadtxt("data/SGD_output_test.txt")
    # Y_train = np.loadtxt("data/ySGDtrain.txt")
    # Y_test = np.loadtxt("data/ySGDtest.txt")
    ytrain = get_test(Y_train)
    ytest = get_test(Y_test) 
    feature_names = range(256)
    class_num = 4
    rfc = RandomForestClassifier(n_estimators=20,max_depth=5)
    rfc.fit(X_train, ytrain)
    y_predict=rfc.predict(X_test)
    arc = get_accuracy(y_predict,ytest)
    f=open("result/result.txt","w")
    print("rfc_test:" + str(arc))
    f.write("rfc_test:" + str(arc)+'\n')
    rfc_net = random_forest_net(rfc,feature_names,class_num)
    # x=torch.FloatTensor(X_test[0:10])
    # if torch.cuda.is_available():
    #     x=x.cuda()
    # out = rfc_net(x)
    arc = test_tree_net(rfc_net,X_test,ytest)
    print("rfc_net_test:" + str(arc))
    f.write("rfc_net_test:" + str(arc)+'\n')
    rfc_net = train(rfc_net,X_train,ytrain,X_test,ytest,4000,f)
    f.close()