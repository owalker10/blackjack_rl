import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde as kde


def fit(x,y,model=LogisticRegression()):
    model.fit(x,y)
    return model


data = pd.read_csv('discard_data_1000000.csv')

labels = data['result']
features = data.iloc[:,2:]


# train a model that is only given a fraction each discard pile (i.e. if n = 0.5 each discard pile is cut in half and only one half is used to train/test
def train_partial(n):
    global labels,features

    # emulates the removal of 1-n fraction of cards in the discard pile, takes ~30 minutes for n = 0.5 and rows = 1,000,000
    def partial(data,n):
        for i in range(data.shape[0]): # traverse through the features row-wise
            #if i % 1000 == 0: print(i)
            row = data.iloc[i,:]
            start = row.sum() # starting number of cards in deck-features
            cnt = start
            cards = [] # cards by value
            for card in data.columns:
                cards.extend([card]*row[card]) # make a list of cards from the one hots
            while cnt > int(start*n)+1: # while we still have more than fraction n of cards left, keep taking them off
                card = np.random.choice(cards)
                cards.remove(card)
                row.loc[card] = row.loc[card]-1
                cnt-=1
        return data

    features2 = partial(features.copy(),n)

    x_train,x_test,y_train,y_test = train_test_split(features2,labels,train_size=0.8,test_size=0.2,random_state=333) # split data

    # to ensure no bias towards losses in training the model, we drop rows in the data so there are an equal number of 1's and 0's in the 'results' column
    x_train.loc[:,'result'] = y_train
    index = min(x_train[x_train['result']==0].shape[0],x_train[x_train['result']==1].shape[0])
    pos = x_train[x_train['result']==1].iloc[:index,:]
    neg = x_train[x_train['result']==0].iloc[:index,:]
    pos = pos.append(neg)
    pos=pos.sample(frac=1.0)
    y_train = pos['result']
    x_train = pos.drop('result',axis=1)

    m = LogisticRegression(max_iter=1000)
    
    # train model
    trained_model = fit(x_train,y_train,model=m)
    print(trained_model.get_params())

    # predict labels
    train_predict = trained_model.predict(x_train)
    test_predict = trained_model.predict(x_test)

    
    print(m)
    print(n)
    print('train accuracy: ',accuracy_score(y_train,train_predict))
    print('test accuracy:  ',accuracy_score(y_test,test_predict))
    print('confusion matrix:\n',confusion_matrix(y_test,test_predict),'\n\n')
    for card,coef in zip(x_test.columns,trained_model.coef_.T):
        print(card,coef)
    

    #joblib.dump(trained_model,'cvmodel_partial'+str(int(n*100))+'.pkl') # UNCOMMENT THIS LINE TO SAVE THE MODEL
    
# train a regular CV model with full data, trained on 800,000 rows and tested on 200,000
def train_reg(m=LogisticRegression(random_state=200)):
    global labels,features,data
    x_train,x_test,y_train,y_test = train_test_split(features,labels,train_size=0.8,test_size=0.2,random_state=420) # split data
    
    # to ensure no bias towards losses in training the model, we drop rows in the data so there are an equal number of 1's and 0's in the 'results' column
    x_train.loc[:,'result'] = y_train
    n = min(x_train[x_train['result']==0].shape[0],x_train[x_train['result']==1].shape[0])
    pos = x_train[x_train['result']==1].iloc[:n,:]
    neg = x_train[x_train['result']==0].iloc[:n,:]
    pos = pos.append(neg)
    pos=pos.sample(frac=1.0)
    y_train = pos['result']
    x_train = pos.drop('result',axis=1)


    #train mdoel
    trained_model = fit(x_train,y_train,model=m)
    #print(trained_model.get_params())

    # predict labels
    train_predict = trained_model.predict(x_train)
    test_predict = trained_model.predict(x_test)
    probs = trained_model.predict_proba(x_test)
    #print(probs)

    
    print(m)
    print('train accuracy: ',accuracy_score(y_train,train_predict))
    print('test accuracy:  ',accuracy_score(y_test,test_predict))
    print('confusion matrix:\n',confusion_matrix(y_test,test_predict),'\n\ncoefficients:')
    for card,coef in zip(x_test.columns,trained_model.coef_.T): # this is for Logistic Regression ONLY
        print(card,coef)
    
    #joblib.dump(trained_model,'cvmodel_new.pkl') # UNCOMMENT THIS LINE TO SAVE THE MODEL

#for n in [0.1,0.3,0.5,0.8,1.0]:
#    train_partial(n)

#train_reg()

#for m in LogisticRegression(),MLPClassifier(),RandomForestClassifier(),SVC():
#    train_reg(m=m)


