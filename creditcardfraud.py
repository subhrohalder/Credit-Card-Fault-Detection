"""
Created on Tue Apr 14 17:03:27 2020

@author: subhrohalder

It is a imbalanced Data set from Kaggle
Link: https://www.kaggle.com/mlg-ulb/creditcardfraud
"""
#imports
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from pylab import rcParams
from  sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


rcParams['figure.figsize']=10,6
warnings.filterwarnings('ignore')
sns.set(style='darkgrid')

#To Generate Report
def generate_model_report(y_actual,y_predicted):
    print('accuracy_score:',accuracy_score(y_actual,y_predicted))
    print('precision_score:',precision_score(y_actual,y_predicted))
    print('recall_score:',recall_score(y_actual,y_predicted))
    print('f1_score:',f1_score(y_actual,y_predicted))
    
#Reading the CSV
df=pd.read_csv('creditcard.csv')
df.head()

#Data Selection
target='Class'
X=df.loc[:,df.columns!=target]
y=df.loc[:,df.columns==target]

#Split Train and Test set
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25)


#plot the data
ax=sns.countplot(x=target,data=df)
print(df[target].value_counts())

#Percentages of Zeros and Ones
percentage_of_zeros=100*(492/float(df.shape[0]))
percentage_of_ones=100*(284315/float(df.shape[0]))

#Model Fitting with complete imbalanced data
model=LogisticRegression()
model.fit(X_train,y_train)

#Prediction
y_pred= model.predict(X_test)

#accuracy report
generate_model_report(y_test,y_pred)


"""
Using logistic regression using imbalanced dataset
accuracy_score: 0.9989326142524086
precision_score: 0.7307692307692307
recall_score: 0.6129032258064516
f1_score: 0.6666666666666667
"""

#Class reweighting
from sklearn.model_selection import GridSearchCV
weights=np.linspace(0.05,0.95,20) 
gsc=GridSearchCV(estimator=LogisticRegression(),
                 param_grid={'class_weight':[{0:x,1:1.0-x} for x in weights]},
                 scoring='f1',
                 cv=5
                 )
grid_result=gsc.fit(X_train,y_train)
print("Best Parameters:",grid_result.best_params_)

#Model Fitting with complete imbalanced data
model=LogisticRegression(**grid_result.best_params_).fit(X_train,y_train)
model.fit(X_train,y_train)

#Prediction
y_pred= model.predict(X_test)

#accuracy report
generate_model_report(y_test,y_pred)
"""
Class reweighting
accuracy_score: 0.9994382180275835
precision_score: 0.8782608695652174
recall_score: 0.7952755905511811
f1_score: 0.8347107438016529"""

#Doing Under Sampling

#Minority Class Data
minority_class_len=len(df[df[target]==1])
minority_class_index=(df[df[target]==1]).index
print(minority_class_index)
print(minority_class_len)

#Majority Class Data indices
majority_class_index=(df[df[target]==0]).index
print(majority_class_index)

#Selecting Random Data indices from All the indices
random_majority_index=np.random.choice(majority_class_index,minority_class_len,replace=False)
print(random_majority_index)

#concatenate both minority and majority indices
under_sample_indices=np.concatenate([minority_class_index,random_majority_index])

#Data selection for undersampling
under_sample=df.loc[under_sample_indices]

#plot the data
ax=sns.countplot(x=target,data=under_sample)

#Data Selection
X=under_sample.loc[:,df.columns!=target]
y=under_sample.loc[:,df.columns==target]

#Split Train and Test set
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=42)

#Model Fitting after under sampling
model=LogisticRegression()
model.fit(X_train,y_train)

#Prediction
y_pred= model.predict(X_test)

#accuracy after under sampling
generate_model_report(y_test,y_pred)

"""
Accuracy using logistic Regression after under sampling
Using Small X_test :
accuracy_score: 0.943089430894309
precision_score: 0.9912280701754386
recall_score: 0.8968253968253969
f1_score: 0.9416666666666667
Using Big X_test :
accuracy_score: 0.999522485323446
precision_score: 0.9387755102040817
recall_score: 0.7666666666666667
f1_score: 0.8440366972477065
"""

""" Random Forest Algorithm """

#Reading the CSV
df=pd.read_csv('creditcard.csv')
df.head()

#Data Selection
target='Class'
X=df.loc[:,df.columns!=target]
y=df.loc[:,df.columns==target]

#Split Train and Test set
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25)

#Model building using Random Forest
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=400,max_depth=10)

#Model Fitting
model.fit(X_train,y_train)

#model Accuracy
model.score(X_test,y_test)

#Prediction
y_pred= model.predict(X_test)

#accuracy after under sampling
generate_model_report(y_test,y_pred)


"""
Accuracy using Random Forest algorithm
generate_model_report(y_test,y_pred)
accuracy_score: 0.9994663071262043
precision_score: 1.0
recall_score: 0.6607142857142857
f1_score: 0.7956989247311829
"""


"""xgboost """

#Reading the CSV
df=pd.read_csv('creditcard.csv')
df.head()

#Data Selection
target='Class'
X=df.loc[:,df.columns!=target]
y=df.loc[:,df.columns==target]

#Split Train and Test set
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25)

#Model building Xgboost
from xgboost import XGBClassifier
model = XGBClassifier()

#Model Fitting
model.fit(X_train,y_train)

#model Accuracy
model.score(X_test,y_test)

#Prediction
y_pred= model.predict(X_test)

#accuracy after under sampling
generate_model_report(y_test,y_pred)


"""
Accuracy using xgboost
accuracy_score: 0.999522485323446
precision_score: 0.9387755102040817
recall_score: 0.7666666666666667
f1_score: 0.8440366972477065
"""



    






