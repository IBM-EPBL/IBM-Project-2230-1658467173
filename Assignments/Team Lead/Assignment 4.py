#!/usr/bin/env python
# coding: utf-8

# SPAM Classifier

# Import Required Libraries

# In[33]:


import pandas as pd
import numpy as np
import nltk
import re


# In[34]:


nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


# Reading Dataset

# In[35]:


df = pd.read_csv('C:\\Users\\sarav\\Downloads\\spam.csv',encoding ='ISO-8859-1')
df.shape


# Analysing Dataset

# In[36]:


df


# In[37]:


df.info


# In[38]:


df.describe


# In[39]:


print(f'Checking is there any columns having null values \n{df.isnull().any()}\n')
print(f'Checking is there any columns having only null values \n{df.isnull().all()}\n')
print(f'Checking total number of null values in all colunms \n{df.isnull().sum()}\n')
print(df.shape)


# Pre-Processing Data to create model

# In[40]:


df1 = df.copy()


# In[41]:


df1 = df1.iloc[:,0:2]
df1.shape


# In[42]:


df1.isnull().sum()


# In[43]:


train_set_x = df1.iloc[:,1:2]
train_set_y = df1.iloc[:,0:1]
print(train_set_x)
print(train_set_y)


# Creating an Object for doing Pre-Processing

# In[64]:


class SMSProcessor():
    
    def __init__(self,x,y):
        try:
            if len(x) == len(y):
                self.x = x
                self.y = y
                self.data = []
                self.ps = PorterStemmer()
                self.cv = CountVectorizer()
                self.re = re
                self.limit = self.x.shape[0]
        except:
            raise 'The given independent column - x  and dependent column - y   sizes are not matching'
    def sentence_process(self,string):
        v2 = str(string)
        v2 = self.re.sub('[^a-zA-Z]',' ',v2)
        v2 = v2.lower()
        v2 = v2.split()
        v2 = [self.ps.stem(word) for word in v2 if word not in set(stopwords.words('english'))]
        v2 = ' '.join(v2)
        return v2
  
    def sentence_updater(self):
        for i in range(0,self.limit):
            data = self.sentence_process(self.x.values[i])
            self.data.append(data)
    def train_process(self):
        self.x = self.cv.fit_transform(self.data).toarray()
        self.y = pd.get_dummies(self.y).drop('v1_spam', axis=1)
  
    def x_y_formater(self):
        self.sentence_updater()
        self.train_process()
        return self.x, self.y
  
    def test_process(self,string):
        string = self.sentence_process(string)
        string = self.cv.transform([string]).toarray()
        return string


# Preprocessing Dataset

# In[63]:


processor = SMSProcessor(train_set_x, train_set_y)

x_train,y_train = processor.x_y_formater()
print(x_train)
print(y_train)


# Model training
# 
# Importing required libraries for model training

# In[65]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Creating Model Skeleton

# In[66]:


model = Sequential()
model.add(Dense(1000, activation='relu'))
model.add(Dense(1500, activation='relu'))
model.add(Dense(3000, activation='relu'))
model.add(Dense(5000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[67]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[68]:


model.fit(x_train,y_train,epochs=15)


# In[69]:


model.save('sms.h5')


# Testing Model

# In[70]:


sample_input = input('Enter the sms here : \n')
sms = processor.test_process(sample_input)
pred = model.predict(sms)
print(f'\n\nThe prodicted binary output is : {pred[0][0]}')
print(f"The SMS is {'HAM' if pred>0.5 else 'SPAM'}")


# In[ ]:




