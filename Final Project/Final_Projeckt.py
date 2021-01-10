import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score ,r2_score

data = pd.read_csv("winequality.csv") #import data
#DATA DESCRİBE
"""
print(data.head())# test objesct has the rgiht type
print(data.info())#prints information about a DataFrame including the index dtype
print(data.describe())#view some basic statistical details l
print(data.isna().sum())#returns the number of missing values in each column.
"""
#Exploratory Data Analysis
"""
plt.rcParams['figure.figsize'] = (20.0, 10.0)
sns.countplot(x="quality", alpha=0.7, data=data)
plt.show()

data.groupby(by="quality").count()
sns.distplot(data.citricacid,color="g")
plt.show()
"""
#Data Preprocessing
X,y=data.iloc[:,:-1],data.iloc[:,-1]
#print(X)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

#model building Decision Tree Al.
print("MODEL : DECİS,ON TREE AL.")
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train,y_train)
print("Accuracy of train:",clf.score(X_train,y_train))
print("Accuracy of test:",clf.score(X_test,y_test))
pred = clf.predict(X_test)
print(classification_report(y_test,pred))
pred = clf.predict(X_test)
print(classification_report(y_test,pred))
print("************************************************************************")
#modelb building LinearRegression
print("Model :  LinearRegression  ")
modelb = LinearRegression(normalize=False)
modelb.fit(X_train,y_train)
def adj_r2 (X,y,model):
    r_squared = model.score(X,y)
    return(1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1))
adj_r2(X_test, y_test, modelb)

modelb = LinearRegression(normalize=False)
modelb.fit(X_train,y_train)

print("Score of the train set",modelb.score(X_train,y_train))
print("Score of the test set",modelb.score(X_test,y_test))
print("Adj. R2 of the train set",adj_r2(X_train,y_train,modelb))
print("Adj. R2 of the test set",adj_r2(X_test,y_test,modelb))
#Score of the train set 0.36453704922715846
#Score of the test set 0.3310001542880008
#Adj. R2 of the train set 0.35822260256184557
#Adj. R2 of the test set 0.3152757989400692

print("************************************************************************")
#model building Logistic Regression
print("MODEL : Logistic Regression ")
modelc = LogisticRegression(random_state = 42)
modelc.fit(X, y)
print("Score of the train set",modelc.score(X_train,y_train))
print("Score of the test set",modelc.score(X_test,y_test))
print("Adj. R2 of the train set",adj_r2(X_train,y_train,modelc))
print("Adj. R2 of the test set",adj_r2(X_test,y_test,modelc))

#Score of the train set 0.5880250223413762
#Score of the test set 0.5604166666666667
#Adj. R2 of the train set 0.5839313233763854
#Adj. R2 of the test set 0.5500845797720797
#I selected Decision Tree model why I got it best score of Accuray , R2 ,F1-score.This model did not memorize. How we know ? If this model memorise , All of Score is closed to 1.
#I can get better result if i add more data and make clear analysis.