#importing the required librariesimport re
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
%matplotlib inline
# Question 1
#Importing datasets
from sklearn.datasets import load_digits
# Question 2
digit=load_digits()
x=digit.data
y=digit.target
x.shape,y.shape #64 columns
plt.imshow(x[4].reshape(8,8),cmap=plt.cm.gray) #imshow will show 4th digit nd will reshape the figure into 8*8 matrix nd will print in gray
# Question 3
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
# Question 4
reg = LogisticRegression()
reg.fit(x_train,y_train)
pred = reg.predict(x_test)
result = pd.DataFrame({"Predicted":pred,"Actual":y_test})
result
reg.score(x_test,y_test)
#Question 5 
# A
kfold=model_selection.KFold(n_splits=10,random_state=7)
results=model_selection.cross_val_score(reg,x,y,cv=kfold,scoring="accuracy")
results
results.sum()/10
# B
results=model_selection.cross_val_score(reg,x,y,cv=kfold,scoring="neg_log_loss")
results
results.sum()/10
# C
results1=model_selection.cross_val_score(reg,x,y,cv=kfold,scoring="r2")
results1
# D
mean_absolute_error(y_test,y_pred)
# E 
mean_squared_error(y_test,y_pred)
# F
confusion_matrix(y_test,pred)
# G
print(classification_report(y_test,pred))
regression=LinearRegression()
regression.fit(x_train,y_train)
regression.intercept_
regression.coef_
y_pred = regression.predict(x_test)
y_pred
a=pd.DataFrame({'actual':y_test,'prediction':y_pred})
a
