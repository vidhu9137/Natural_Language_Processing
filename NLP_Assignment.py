#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


import os


# # Importing Dataset

# In[3]:


df = pd.read_csv('Text_Similarity_Dataset.csv')


# In[4]:


df


# # Importing librabries for text preprocessing

# In[5]:


import re
import nltk


# In[6]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[7]:


from nltk.stem.porter import PorterStemmer


# # Text Preprocessing

# In[120]:


#Preprocessing for the first row of dataset


# In[8]:


X = re.sub('[^a-zA-Z]', ' ', df['text1'][0])   #only takes things made up from letters, a-z and A-Z  #No ;%&,No numbers
X = X.lower()
X = word_tokenize(X)  #split all words


# In[9]:


Y = re.sub('[^a-zA-Z]', ' ', df['text2'][0])   
Y = Y.lower()
Y = word_tokenize(Y)  


# In[10]:


# sw contains the list of stopwords
sw = stopwords.words('english')
#ps.stem() used to stem the word to their original Eg: playing to play
ps = PorterStemmer()
l1 =[];l2 =[]


# In[11]:


# remove stop words from the string
X_set = {ps.stem(w) for w in X if not w in sw} 
Y_set = {ps.stem(w) for w in Y if not w in sw}


# In[12]:


X_set


# In[13]:


Y_set


# In[14]:


# form a set containing keywords of both strings 
rvector = X_set.union(Y_set) 
for w in rvector:
    if w in X_set: l1.append(1) # create a vector
    else: l1.append(0)
    if w in Y_set: l2.append(1)
    else: l2.append(0)


# In[15]:


len(l1)


# In[16]:


len(l2)


# In[17]:


c = 0
  
# cosine formula for finding similarity score b/w l1 and l2
for i in range(len(rvector)):
    c+= l1[i]*l2[i]
cosine = c / float((sum(l1)*sum(l2))**0.5)
print("similarity: ", cosine)


# In[ ]:


#Applying the above operations on the whole dataframe


# In[18]:


def count_frequency(word_list):   #function for creating dictionary of frequencies for all words
      
    D = {}
      
    for new_word in word_list:
          
        if new_word in D:
            D[new_word] = D[new_word] + 1
              
        else:
            D[new_word] = 1
              
    return D


# In[19]:


import math    

#All functions used for finding cosine value i.e. used for similarity score

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def cosine_angle(v1, v2):
    return dotproduct(v1, v2) / (length(v1) * length(v2))


# In[19]:


#Creating a for loop for preprocessing and similarity score calculation of whole dataframe


# In[20]:


sw = stopwords.words('english') 
ps = PorterStemmer()

for i in range(0, 3000):
    X = re.sub('[^a-zA-Z]', ' ', df['text1'][i])  
    X = X.lower()
    X = word_tokenize(X) 
    
    Y = re.sub('[^a-zA-Z]', ' ', df['text2'][i]) 
    Y = Y.lower()
    Y = word_tokenize(Y)  
    
    l1 =[];l2 =[]
    
    X_set = {ps.stem(w) for w in X if not w in sw} 
    Y_set = {ps.stem(w) for w in Y if not w in sw}
    
    D1 = count_frequency(X_set)
    D2 = count_frequency(Y_set)
    
    rvector = X_set.union(Y_set) 
    for w in rvector:
        if w in X_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in Y_set: l2.append(1)
        else: l2.append(0)
    
    print('Row index: ', i)
    print('Similarity: ', cosine_angle(l1, l2))


# In[ ]:




