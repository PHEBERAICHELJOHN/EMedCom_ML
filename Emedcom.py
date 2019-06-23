#!/usr/bin/env python
# coding: utf-8

# In[42]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("dataset.csv")

#preprocessing of data

from sklearn.preprocessing import LabelEncoder
processed_data = pd.DataFrame()
#Label Encoder converts into numerical value
lb_make = LabelEncoder()
processed_data["GENERIC NAME"] = lb_make.fit_transform(data["GENERIC NAME"])
processed_data["SOURCE"] = lb_make.fit_transform(data["SOURCE"])
processed_data["COUNT"] = lb_make.fit_transform(data["COUNT"])
processed_data["MONTH"] = lb_make.fit_transform(data["MONTH"])
processed_data["DESTINATION"] = lb_make.fit_transform(data["DESTINATION"])

# In[47]:
# to extract the features
#printing the columns

features = processed_data.columns

# In[48]:
#printing the last column which is used to predict 

predict_class = data.columns[-1]

# In[49]:
#print the value to be predicted

predictions = data[predict_class]

# In[50]:

#printing the numerical values of DESTINATION to be predicted

prediction_class = processed_data["DESTINATION"]

# In[51]:

features_data = processed_data[["GENERIC NAME","SOURCE","COUNT","MONTH"]]

# In[52]:
#training

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features_data,prediction_class,train_size = 0.80,test_size=0.20,random_state = 1)
np.random.seed(0) #to make random numbers predictable

# In[55]:
#fitting into KNN

from sklearn import neighbors
from sklearn.model_selection import cross_val_score
knn = neighbors.KNeighborsClassifier(n_neighbors = 3)
cv_scores = cross_val_score(knn, features_data, prediction_class, cv = 10)

# In[58]:
#printing confusion matrix

knn.fit(X_train,y_train)
predictions_test = knn.predict(X_test)
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  

# In[76]:
#giving input 
import sys
#15, 1, 112, 3
a = sys.argv[1]
b = sys.argv[2]
c = sys.argv[3]
d = sys.argv[4]
new=pd.DataFrame([[a,b,c,d]])

# In[65]:
test=knn.predict(new)
arr=["Aleppey","Ernakulam","Idukki","Kannur","Kasarkode","Kollam","Kottayam","Kozhikode","Malappuram","Palakkad","Pathanamthitta","Thiruvananthapuram","Thrissur","Wayanad"];
pre_out = arr[int(test)]

# In[66]:
print(pre_out)
