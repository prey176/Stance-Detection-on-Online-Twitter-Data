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

n2ov = []
ypredov = []

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
    rf = RandomForestClassifier(n_estimators = 100)
# Train the model on training data
    rf.fit(OBS[ll], n1);
    ypred=rf.predict(observations2)
    c=0
    t=len(n2)
    for j in range(len(n2)):
        if (ypred[j]==n2[j]):
            c+=1

    n2ov.extend(n2)
    ypredov.extend(ypred)
    
    accuracy=rf.score(observations2,n2)
    s+=accuracy
    print (xx[ll],accuracy)


print(metrics.classification_report(n2ov, ypredov, labels=[0, 1, 2]))

print ("avg f1 score rand_forest_classifier:",s/5)