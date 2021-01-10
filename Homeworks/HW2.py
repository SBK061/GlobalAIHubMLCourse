import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression ,Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X,y = load_boston(return_X_y=True)
dataset = preprocessing.scale(X)
dataset =pd.DataFrame(X,columns=load_boston().feature_names)

print(dataset) #show us data and get info
print(dataset.describe())#recognize data
print(dataset.isna().sum())

#model LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
modelb = LinearRegression(normalize=False)
modelb.fit(X_train,y_train)

print("Score of the train set",modelb.score(X_train,y_train))
print("Score of the test set",modelb.score(X_test,y_test))

#model Ridge
X_r= load_boston().data
y_r = load_boston().target
scaler = StandardScaler()
X_std = scaler.fit_transform(X_r)
# Create ridge regression with an alpha value
five_value = [0.5 , 0.4 , 0.8 ,0.9 ,0.1]
print("Ridge model")
for a in five_value:
    regr = Ridge(alpha=a)
    # Fit the linear regression
    model = regr.fit(X_std, y_r)
    print(f'Alpha value : {a}  and Ridge model coef: {regr.coef_}')
    # Ridge
    print("Ridge Train: ", regr.score(X_train, y_train))
    print("Ridge Test: ", regr.score(X_test, y_test))
    print("------------------------------------------------------")

five_val = [0.001 , 0.002 , 0.003 , 0.004 ,0.005]
#model Lasso
print("Lasso model")
for b in five_val:
    lasso_model = Lasso(alpha = 0.001)
    lasso_model.fit(X_train, y_train)
    print(f'Alpha VAl : {b} Lasso model coef: {lasso_model.coef_}')
    #Lasso
    print("Lasso Train: ", lasso_model.score(X_train, y_train))
    print("Lasso Test: ", lasso_model.score(X_test, y_test))
    print("------------------------------------------------------")
#simple linear model
print('Simple linear model')
print("Simple Train: ", modelb.score(X_train, y_train))
print("Simple Test: ", modelb.score(X_test, y_test))
print('*************************')


#Lasso model is overfitting because accuracy score is what we want for every alpha value.And we didnt loss data for test and train. thats so i chose the Lasso model.



