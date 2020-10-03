#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import sys


# In[26]:


data = pd.read_csv("DS_DATESET.csv")


# In[5]:


data.drop(['Certifications/Achievement/ Research papers','Link to updated Resume (Google/ One Drive link preferred)', 'link to Linkedin profile','State','DOB [DD/MM/YYYY]', 'Email Address','Contact Number', 'Emergency Contact Number','Zip Code','Gender'],axis=1,inplace=True)


# In[6]:


pdf = PdfPages('visualizations-output.pdf')
intrest= data['Areas of interest'].value_counts().plot(kind='barh',figsize=(18,8),color='blue')
plt.title("Students applied to diiferent Technologies")
plt.xlabel('Number of Students', fontweight ='bold')
plt.ylabel('Different Technologies', fontweight ='bold')
pdf.savefig()


# In[7]:


DS = data[['Areas of interest','Programming Language Known other than Java (one major)']]
DSP = DS.groupby(['Areas of interest'])
program = DSP.get_group('Data Science ')
program1 = list(program['Programming Language Known other than Java (one major)'])
x=program1
Grab=[]
for i in range(0,len(program1)):
    if (x[i]=="Python"):
        Grab.append("YES")
    else:
        Grab.append("NO")
program2 = pd.DataFrame({"Python":program1})
ax0 = program2['Python'].value_counts().plot(kind='barh',figsize=(8,8),color='red') 
plt.title("Having Knowledge of Python") 
plt.xlabel('Student Count', fontweight ='bold')
plt.ylabel('Students With python Knowledge', fontweight ='bold')
pdf.savefig()


# In[8]:


df2= data['How Did You Hear About This Internship?'].value_counts().plot(kind='barh',figsize=(18,8),color='yellow')
plt.title("Where Did You Found About Internship",fontweight ='bold')
plt.xlabel('Student Count',fontweight ='bold')
plt.ylabel('Various Platforms',fontweight ='bold')
pdf.savefig()


# In[9]:


CGPA= data[['Which-year are you studying in?','CGPA/ percentage']]
grade = CGPA.groupby(['Which-year are you studying in?'])
level = grade.get_group('Fourth-year')
level1 = list(level['CGPA/ percentage'])
x=level1
Y=[]
for i in range(0,len(level1)):
    if x[i]>8.0:
        Y.append("YES")
    elif x[i]<8.0:
        Y.append("NO")
level2= pd.DataFrame({"Fourth Year":Y})
ax2 = level2['Fourth Year'].value_counts().plot(kind="barh",figsize=(8,8),color='cyan')
plt.title("Students With Greater CGPA",fontweight ='bold')
plt.xlabel('Number of Students',fontweight ='bold')
plt.ylabel('Studnets with CGPA greater than 8',fontweight ='bold')
pdf.savefig()


# In[10]:


df2= data[['Areas of interest','Rate your written communication skills [1-10]','Rate your verbal communication skills [1-10]']]
Frame2 = df2.groupby(['Areas of interest'])
Df1 = Frame2.get_group('Digital Marketing ')
prop1 = list(Df1['Rate your written communication skills [1-10]'])
prop2= list(Df1['Rate your verbal communication skills [1-10]'])
x=prop1
y=prop2
Z=[]
for i in range(0,624):
        if x[i]>7.5:
            if y[i]>7.5:
                Z.append("YES")
            else:
                Z.append("NO")
market= pd.DataFrame({"Digital Marketing":Z})
df3 = market['Digital Marketing'].value_counts().plot(kind="barh",figsize=(8,8),color='black')
plt.title("Greater Score in Digital Marketing",fontweight ='bold')
plt.xlabel('Number of Students', fontweight ='bold')
plt.ylabel('Verbal & Wrinting Skills',fontweight ='bold')
pdf.savefig()


# In[11]:


df4 = data['Major/Area of Study'].value_counts().plot(kind='barh',color='red',figsize=(25,8))
plt.title("Area of Study wise Classification of students",fontweight ='bold')
plt.xlabel('Number of Students',fontweight ='bold')
plt.ylabel('Different Areas of Study',fontweight ='bold')
pdf.savefig()


# In[12]:


df5 = data['College name'].value_counts().plot(kind='barh',figsize=(20,5),color='blue')
plt.title("College wise Classification of Students",fontweight ='bold')
plt.xlabel('Number of Students',fontweight ='bold')
plt.ylabel('Different Colleges',fontweight ='bold')
pdf.savefig()


# In[13]:


df6= data['Which-year are you studying in?'].value_counts().plot(kind='barh',figsize=(8,8),color='green')
plt.title("Year-Wise Classification of Students",fontweight ='bold')
plt.xlabel('Number of Students',fontweight ='bold')
plt.ylabel('Different Years of Degree',fontweight ='bold')
pdf.savefig()


# In[14]:


df7 = data['City'].value_counts().plot(kind='barh',color='yellow',figsize=(8,8))
plt.title("City wise Classification of Students",fontweight ='bold')
plt.xlabel('Number of Students',fontweight ='bold')
plt.ylabel('Different Cities',fontweight ='bold')
pdf.savefig()


# In[15]:


for bug in data:
    data[bug] = LabelEncoder().fit_transform(data[bug])
     
data.head()


# In[16]:


plt.figure(figsize=(8, 8))
plt.scatter(data['CGPA/ percentage'],data['Label'])
plt.title("CGPA And Label",fontweight ='bold')
plt.xlabel('Label',fontweight ='bold')
plt.ylabel('CGPA/Percentage',fontweight ='bold')
pdf.savefig()


# In[17]:


plt.figure(figsize=(8, 8))
plt.scatter(data['Areas of interest'],data['Label'])
plt.title("Label and Intrest",fontweight ='bold')
plt.xlabel('Label',fontweight ='bold')
plt.ylabel('Areas of Interest',fontweight ='bold')
pdf.savefig()


# In[18]:


plt.figure(figsize=(8, 8))
plt.scatter(data['Which-year are you studying in?'],data['Label'])
plt.title("Current Year And Label",fontweight ='bold')
plt.xlabel('Label',fontweight ='bold')
plt.ylabel('Current Year',fontweight ='bold')
pdf.savefig()


# In[19]:


plt.figure(figsize=(8, 8))
plt.scatter(data['Major/Area of Study'],data['Label'])
plt.title("Major/Area of Study and Label",fontsize = 18, fontweight ='bold')
plt.xlabel('Label',fontweight ='bold')
plt.ylabel('Major\Area of Study',fontweight ='bold')
pdf.savefig()


# In[20]:


pdf.close()
print("PDF file created")


# In[ ]:





# In[ ]:




