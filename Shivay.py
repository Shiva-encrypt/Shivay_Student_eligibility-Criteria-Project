#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy.stats import norm, skew
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score as cv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# In[99]:


data= pd.read_csv('DS_DATESET.csv') 
data.head()


# In[100]:


data.info()


# In[101]:


data.describe()


# In[102]:


data.drop(['Emergency Contact Number','Zip Code','Certifications/Achievement/ Research papers','Link to updated Resume (Google/ One Drive link preferred)', 'link to Linkedin profile','State','DOB [DD/MM/YYYY]', 'Email Address','Contact Number','How Did You Hear About This Internship?'], axis=1, inplace=True)


# In[103]:


data.drop(['First Name','Last Name'],axis=1,inplace=True)
data.isnull().sum()


# In[104]:


past = data.select_dtypes(include=np.number)


# In[105]:


past.head()


# In[106]:


present = data.select_dtypes(exclude=np.number)


# In[107]:


present.head()


# In[108]:


for col in present:
    present[col] = LabelEncoder().fit_transform(present[col])
     
present.head()


# In[109]:


Future = pd.concat([past , present],axis=1,ignore_index=True)


# In[110]:


Future.columns = ['Age','CGPA/percentage','Expected Graduation-year','Rate your written communication skills [1-10]','Rate your verbal communication skills [1-10]','City','Gender','College name','University Name','Degree','Major/Area of Study','Course Type','Which-year are you studying in?','Areas of interest','Current Employment Status','Have you worked core Java','Programming Language Known other than Java (one major)','Have you worked on MySQL or Oracle database','Have you studied OOP Concepts','label']
Future.head()


# 
# # SPLITTING THE DATASET INTO INDEPENDANT AND DEPENDANT VARIABLE

# In[111]:


X = Future.iloc[:,:-1] 
###INDEPENDANT VERIABLE
y = Future.iloc[:,-1]
####DEPENDANT VERIABLE

####Using STandard SCALER
X = StandardScaler().fit_transform(X)


# In[112]:


variables = list(Future)
X = pd.DataFrame(X,columns=variables[:-1])
X.head()


# # TRAIN TEST SPLIT

# In[113]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=00)


# In[127]:


pipeline_lr = Pipeline([("scaler" , StandardScaler()),
                        ('pca1',PCA(n_components = 5)),
                        ('LR_classifier',LogisticRegression(random_state = 100, max_iter=1000))])
pipeline_svm = Pipeline([("scaler" , StandardScaler()),
                        ('pca1',PCA(n_components = 5)),
                        ('SVC',SVC(random_state = 100,degree=3 ,gamma='auto'))])
pipeline_tree = Pipeline([("scaler" , StandardScaler()),
                        ('pca1',PCA(n_components = 5)),
                        ('Tree',DecisionTreeClassifier(random_state = 100))])
pipeline_Forest = Pipeline([("scaler" , StandardScaler()),
                        ('pca1',PCA(n_components = 5)),
                        ('RF',RandomForestClassifier(random_state = 100,n_estimators=5 ,max_depth=20))])
pipeline_Knn = Pipeline([("scaler" , StandardScaler()),
                        ('pca1',PCA(n_components = 5)),
                        ('Neighbours',KNeighborsClassifier())])

pipelines = [pipeline_lr,pipeline_svm,pipeline_tree,pipeline_Forest,pipeline_Knn]

best_accuracy = 0.0
best_classifier=0
best_pipeline=''


# In[128]:


pipe_dict = {0 : "LogisticRegression" , 1 :'SVC',2:'DecisionTree',3:'RandomForest',4:'KNN'}


for pipe in pipelines:
    pipe.fit(X,y)


# In[129]:


for i , model in enumerate(pipelines):     #####Just for referancing to choose the 
    print('{} has TestAccuracy Score of : {}'.format(pipe_dict[i],model.score(X_test , y_test)))


# # USING SVM 
# 
# 
# After Trying ALL THE Algorithms SVM HAS THE BEST ACCURACY AND FSCORE

# In[123]:


svc = SVC(random_state=142 ,kernel='poly',degree= 3, gamma="auto",C=20)   ####Polynomial kernal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30) 
svc.fit(X_train, y_train) 
y_pred = svc.predict(X_test)


# In[124]:


###RESULTS OF THE MODEL
print("Final ACCURACY OF SVM MODEL: Poly KERNAL" )
print(accuracy_score(y_test,y_pred))


# In[125]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




