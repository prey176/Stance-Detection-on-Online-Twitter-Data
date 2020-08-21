#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
import string
import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics



# In[2]:


def READ (d,D) :    
    df = pd.read_csv("train.csv",sep = ',', names = ["Tweet","Target","Stance","Opinion","Sentiment"],engine = 'python')
    data = np.array(df)
    data = data[1:,:]

    cnt = 0
    for a in data :
        if a[1] not in d :
            d[a[1]] = cnt
            cnt += 1

    for i in range (cnt) :
        D.append ([])

    for a in data :
        D [d[a[1]]].append ([a[0],a[2]])

def READ2 (d,D) :    
    df = pd.read_csv("test.csv",sep = ',', names = ["Tweet","Target","Stance","Opinion","Sentiment"],engine = 'python')
    data = np.array(df)
    data = data[1:,:]

    cnt = 0
    for a in data :
        if a[1] not in d :
            d[a[1]] = cnt
            cnt += 1

    for i in range (cnt) :
        D.append ([])

    for a in data :
        D [d[a[1]]].append ([a[0],a[2]])
# In[3]:


def clean(text) :
    tokens = nltk.word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    ptokens = [w.translate(table) for w in tokens]
    non_blank_tokens = [s.lower() for s in ptokens if s]
    return nltk.pos_tag(non_blank_tokens)


# In[ ]:





# In[4]:


def CHANGE(d,D) :
    for i in range (len(D)) :
        for j in range (len(D[i])) :
            D[i][j][0] = clean (D[i][j][0])    


# In[ ]:





# In[ ]:





# In[6]:


D = []
d = {}
READ(d,D)
CHANGE(d,D)

D2=[]
d2={}
READ2(d2,D2)
CHANGE(d2,D2)





# In[7]:


print (d)
print(d2)


# In[30]:


tagset = {'NN','NNS','NNP','VB','VBD','VBJ','VBN','VBP','VBZ','JJ','JJR','JJS'}


# In[36]:


FEATURES = []
Positions = []

for i in range (len(d)) :
    Features = []
    for a in D[i] :
        for b in a[0] :
            if b[1] in tagset:
                Features.append(b[0])
        
    Features = list(set(Features))

    
    
    Pos = {}
    for a in range (len(Features)) :
        Pos[Features[a]] = a
    Positions.append(Pos)
    FEATURES.append(Features)


# In[39]:


OBS = []
for i in range (len(d)) :
    observations = [] 
    for a in D[i] :
        obs = [0] * len(FEATURES[i])
        for b in a[0] :
            if b[0] in Positions[i] :
                obs [Positions[i][b[0]]]= 1
        observations.append (obs)
    OBS.append (observations)
    

rt={0:3,1:4,2:0,3:1,4:2}
xx={0:"Hillary",1:"Abortion",2:"Atheism",3:"Climate",4:"feminism"}
s=0
n2ov=[]
ypredov=[]
for ll in range(5):

    n1=[]
    n2=[]
    for a in D[ll]:
        if (a[1]=="FAVOR"):
            n1.append(0)
        elif (a[1]=="AGAINST"):
            n1.append(1)
        elif (a[1]=="NONE"):
            n1.append(2)

    for a in D2[rt[ll]]:
        if (a[1]=="FAVOR"):
            n2.append(0)
        elif (a[1]=="AGAINST"):
            n2.append(1)
        elif (a[1]=="NONE"):
            n2.append(2)
    print (len(n1))
    observations2 = [] 
    for a in D2[rt[ll]] :
        obs = [0] * len(FEATURES[ll])
        for b in a[0] :
            if b[0] in Positions[ll] :
                obs [Positions[ll][b[0]]]= 1
        observations2.append (obs)

    a = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1)
    a.fit(OBS[ll], n1)

    # a=SVC(kernel="linear")
    # a.fit(OBS[ll],n1)
    ypred=a.predict(observations2)
    tr1=[]
    pr1=[]
    tr2=[]
    pr2=[]
    for jj in range(len(n2)):
        if (n2[jj]==0):
            tr1.append(n2[jj])
            pr1.append(ypred[jj])
        elif (n2[jj]==1):
            tr2.append(n2[jj])
            pr2.append(ypred[jj])
    #f11=sklearn.metrics.f1_score(tr1,pr1,average="weighted")
    #f22=sklearn.metrics.f1_score(tr2,pr2,average="weighted")
    #accuracy=a.score(observations2,n2)
    f1=sklearn.metrics.f1_score(n2,ypred,average="micro")
    n2ov.extend(n2)
    ypredov.extend(ypred)
    s+=f1
    #print (xx[ll],f1)
    print(metrics.classification_report(n2, ypred, labels=[0, 1, 2]))

print(metrics.classification_report(n2ov, ypredov, labels=[0, 1, 2]))


print ("avg f1 score GradientBoosting: ",s/5)

'''
n1=[]
n2=[]
for a in D[4]:
    if (a[1]=="FAVOR"):
        n1.append(0)
    elif (a[1]=="AGAINST"):
        n1.append(1)
    elif (a[1]=="NONE"):
        n1.append(2)

for a in D2[2]:
    if (a[1]=="FAVOR"):
        n2.append(0)
    elif (a[1]=="AGAINST"):
        n2.append(1)
    elif (a[1]=="NONE"):
        n2.append(2)
print (len(n1))
observations2 = [] 
for a in D2[2] :
    obs = [0] * len(FEATURES[4])
    for b in a[0] :
        if b[0] in Positions[4] :
            obs [Positions[4][b[0]]]= 1
    observations2.append (obs)
a=SVC(kernel="linear")

a.fit(OBS[4],n1)
accuracy=a.score(observations2,n2)
print (accuracy)
# In[ ]:





# In[ ]:

'''
# Import the model we are using

# Instantiate model with 1000 decision trees
