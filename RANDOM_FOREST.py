#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets


# In[2]:


iris = datasets.load_iris()


# In[3]:


iris


# In[4]:


print(iris.target_names)
print(iris.feature_names)


# In[11]:


import pandas as pd
data = pd.DataFrame({"sepal length":iris.data[:,0],"sepal width": iris.data[:,1],
              "petal length":iris.data[:,2], "petal width": iris.data[:,3],"species": iris.target})


# In[40]:


data   # Show the first five records


# In[39]:


# Now Splitting data into dependent and independent columns
X = data[['sepal length','sepal width' ,'petal length','petal width']]
Y = data['species']


# In[17]:


X


# In[19]:


Y


# # splitting the data 

# In[20]:


from sklearn.model_selection import train_test_split


# In[41]:


X_train, X_test, y_train, y_test= train_test_split(X,Y,test_size=0.3) #70% for training and 30 for test 


# random forest

# In[23]:


from sklearn.ensemble import RandomForestClassifier


# In[62]:


algo=RandomForestClassifier(n_estimators=100 , criterion='gini')
# n_estimators tell us how many number of trees we want to generate 


# In[63]:


# Train the model
algo.fit(X_train,y_train)


# In[59]:


algo.predict(X_test)


# In[60]:


X_test.head()


# In[61]:


# Calculating Score of the model
algo.score(X_test,y_test)


# In[64]:


import numpy as np 


# In[68]:


np.load('testAccelerometer.npy',
    mmap_mode=None,
    allow_pickle=False,
    fix_imports=True,
    encoding='ASCII',
)


# In[ ]:




