
# coding: utf-8

# In[1]:
#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("new.csv")
type(data)


# In[4]:


data


# In[5]:


#preprocessing of data

from sklearn.preprocessing import LabelEncoder
processed_data = pd.DataFrame()

#Label Encoder converts into numerical value

lb_make = LabelEncoder()
processed_data["GENERIC NAME"] = lb_make.fit_transform(data["GENERIC NAME"])
#processed_data["GENERIC"] = data["GENERIC NAME"]
processed_data["CONDITION"] = lb_make.fit_transform(data["CONDITION"])
#processed_data["CONDITIO"] = data["CONDITION"]
processed_data["SOURCE"] = lb_make.fit_transform(data["SOURCE"])
#processed_data["src"] = data["SOURCE"]
processed_data["COUNT"] = lb_make.fit_transform(data["COUNT"])
#processed_data["cnt"] = data["COUNT"]
processed_data["MONTH"] = lb_make.fit_transform(data["MONTH"])
#processed_data["Mon"] = data["MONTH"]
processed_data["DESTINATION"] = lb_make.fit_transform(data["DESTINATION"])
#processed_data["DESTINATIO"] = data["DESTINATION"]


# In[6]:


processed_data


# In[7]:


# to extract the features

#printing the columns

features = processed_data.columns
print(features)


# In[8]:


#printing the last column which is used to predict 

predict_class = data.columns[-1]
predict_class


# In[9]:


#print the value to be predicted

predictions = data[predict_class]
predictions


# In[10]:


#printing the numerical values of DESTINATION to be predicted

prediction_class = processed_data["DESTINATION"]
prediction_class


# In[11]:


features_data = processed_data[["GENERIC NAME","CONDITION","SOURCE","COUNT","MONTH"]]

#    features_data = processed_data[["GENERIC NAME","CONDITION","SOURCE","MONTH"]]

#features_data = processed_data[["GENERIC NAME","CONDITION","SOURCE"]]
features_data


# In[12]:


#training

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features_data,prediction_class,train_size = 0.80,test_size=0.20,random_state = 1)

np.random.seed(1234)

# Show the results of the split
#print "Training set has {} samples.".format(X_train.shape[0])
#print "Testing set has {} samples.".format(X_test.shape[0])

#print(X_train.shape, y_train.shape) 
#print(X_test.shape, y_test.shape) 

print("Training samples")
print(X_train.shape[0])
print("Testing samples")
print(X_test.shape[0])


# In[13]:


X_train


# In[14]:


y_train


# In[15]:


#fitting into KNN

from sklearn import neighbors
from sklearn.model_selection import cross_val_score

knn = neighbors.KNeighborsClassifier(n_neighbors = 3)
cv_scores = cross_val_score(knn, features_data, prediction_class, cv = 10)

cv_scores.mean()


# In[16]:


#printing confusion matrix

knn.fit(X_train,y_train)
predictions_test = knn.predict(X_test)
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  

# Model Accuracy: how often is the classifier correct?
print(confusion_matrix(y_test,predictions_test))


# In[17]:


#finding accuracy
print('Accuracy is',metrics.accuracy_score(y_test,predictions_test))


# In[18]:



print(classification_report(y_test,predictions_test))


# In[25]:


#giving input 
new=pd.DataFrame([[15,1,1,112,3]])
new


# In[28]:


pre_out = knn.predict(new)


# In[29]:


print(pre_out)

