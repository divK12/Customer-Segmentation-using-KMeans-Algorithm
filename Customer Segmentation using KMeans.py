#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.simplefilter(action="ignore",category=FutureWarning)
warnings.simplefilter(action="ignore",category=UserWarning)


# In[2]:


df=pd.read_csv("customers.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df.dtypes


# In[8]:


sns.countplot(df,x="Gender",orient="v")


# Dataset comprises of a higher female proportion

# In[9]:


sns.histplot(df,x="Age",kde=True,binwidth=2)


# In[10]:


sns.histplot(df,x="Annual Income (k$)",kde=True,binwidth=5)


# In[11]:


sns.histplot(df,x="Spending Score (1-100)",kde=True,binwidth=5)


# In[12]:


sns.kdeplot(df,x="Spending Score (1-100)",hue="Gender",multiple="layer")


# In[13]:


sns.kdeplot(df,x="Annual Income (k$)",hue="Gender",multiple="layer")


# In[ ]:


sns.pairplot(df)


# In[ ]:


df.groupby(['Gender'])['Age','Annual Income (k$)','Spending Score (1-100)'].mean()


# In[ ]:


df1=df.copy()
df1=df1.drop("CustomerID",axis=1)
corr=df1.corr()
sns.heatmap(corr,annot=True)


# In[ ]:


df1=df1.drop(["Age","Gender"],axis=1)


# In[ ]:


sns.relplot(df1,x="Annual Income (k$)",y="Spending Score (1-100)",kind="scatter")


# WCSS- Within Clusters Sum of Squares to choose optimum no. of clusters
# 

# In[ ]:


dff=pd.get_dummies(df,drop_first=True)


# In[ ]:


dff


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler=StandardScaler()
x=scaler.fit_transform(dff)
dff=pd.DataFrame(x,columns=['CustomerID','Age','Annual Income (k$)','Spending Score (1-100)','Label','Gender_Male'])
dff.head()


# In[ ]:


wcss1=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=56)
    kmeans.fit(dff.values)
    wcss1.append(kmeans.inertia_)


# In[ ]:


sns.set(style="whitegrid")
plt.plot(range(1,11),wcss1)
plt.xlabel("K-values")
plt.ylabel("WCSS")
plt.title("ELBOW CURVE")


# In[ ]:


kmeans=KMeans(n_clusters=5,random_state=82)

y=kmeans.fit_predict(dff)
print(y)
dff['Label']=y


# In[ ]:


plt.figure(figsize=(10,8))
sns.scatterplot(x="Annual Income (k$)",y="Spending Score (1-100)",data=df,hue="Label",palette=['red','green','blue','black','purple'])


# In[ ]:


df.groupby(['Label'])['Age','Annual Income (k$)','Spending Score (1-100)'].mean()


# In[ ]:


pd.crosstab(df['Label'],df['Gender'])

The green-1 cluster signifies people with less salary but high spending 
The black-3 signifies people with high salary and less spending.
The magenta-4 are the ones with less salary and less spending 
The red-0 are the ones with mild salary and mild spending
the blue-2 are the ones with higher income and spending ##TARGET CUSTOMER