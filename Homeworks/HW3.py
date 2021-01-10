import  pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #train ve test olarak ayırmak için kullanacağız
from sklearn import metrics # accuracy  calculation
from sklearn.metrics import precision_score , recall_score ,accuracy_score


X ,y = make_blobs(n_samples=2000,n_features=3,random_state=0,cluster_std=5.5)#Generate dataset
print(X.shape)#show shape
print(y.shape)#show shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
sns.set_style("whitegrid")
sns.scatterplot(X_train.T[0], X_train.T[1],color='g')
plt.show()
sns.scatterplot(X_test.T[0], X_test.T[1],color='g')
plt.show()
sns.distplot(X_train.T[0])
plt.show()
sns.distplot(X_train.T[1])
plt.show()


#model building treedecision:
clf=DecisionTreeClassifier()
clf =clf.fit(X_train,y_train)
#prediction
y_pred = clf.predict(X_test)
#accuracy ,model score
print("Accuracy : ", metrics.accuracy_score(y_test,y_pred))

# XGBoost Algorithm
dmatrix_train = xgb.DMatrix(data=X_train, label=y_train)
dmatrix_test = xgb.DMatrix(data=X_test, label=y_test)
param = {'max_depth':3,
         'eta':1,
         'objective':'multi:softprob',
         'num_class':3}

num_round = 5
model = xgb.train(param, dmatrix_train, num_round)
preds = model.predict(dmatrix_test)
print(preds[:10])
best_preds = np.asarray([np.argmax(line) for line in preds])
print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy  = {}".format(accuracy_score(y_test, best_preds)))


#Accuracy :  0.5783333333333334 DecisonTree
#Precision = 0.6377422691681836 xgboost
#Recall = 0.6255106209150326  xgboost
#Accuracy = 0.6266666666666667 xgboost

#comment :This model is Underfitting because accuray score is  0.578333333333333 when we use Decision Tree model Al.
#the error for the model on the training data goes down and so does the error on the test dataset. If we train for too
#long, the performance on the training dataset may continue to decrease because the model is overfitting and learning the
# irrelevant detail and noise in the training dataset.
