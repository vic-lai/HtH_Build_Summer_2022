#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd

''' When going through the assignment, keep in mind that to actually see these variables you need to use python's print() statement to see their values printed out in replit's console 
'''

# Step 1: Use the appropriate pandas method to read the titanic data into your python file 
titanic_data = pd.read_csv('titanic.csv')
titanic_data


# In[8]:


# Step 2(a): Use the pandas method that reads the first 25 lines of the dataset
first_25_passengers = titanic_data.head(25)
first_25_passengers


# In[9]:


# Step 2(b): Use the pandas method thats reads the last 25 lines of the dataset
last_25_passengers = titanic_data.tail(25)
last_25_passengers


# In[10]:


# Step 3: Use the pandas method that only tells us the number of rows and columns in our data
titanic_shape = titanic_data.shape
titanic_shape


# In[11]:


# Step 4: Describe the titanic data
titanic_description = titanic_data.describe()
titanic_description


# In[12]:


# Step 5(a): How many passengers were between the ages of 0 to 16? 
children = titanic_data[(titanic_data['Age']>=0) & (titanic_data['Age']<=16) ]
children


# In[57]:


# Step 6: How many values are missing from the "age" column
missing_ages = titanic_data[titanic_data['Age'].isnull()]
count = 0
for age in missing_ages:
    if(age):
        count+=1
count


# In[59]:


# Step 7: List out all the available passengers' ages
age_list = titanic_data['Age'].unique()
age_list


# In[31]:


# Step 8: Filter the DataFrame to find all passengers who boarded the Titanic at Port Cherbourg
cherbourg_passengers = titanic_data[titanic_data['Embarked']=='C']
cherbourg_passengers


# In[62]:


# Step 9(a): Find the average age of all female passengers
avg_fem_age = titanic_data[titanic_data['Sex']=='female']['Age'].mean()
avg_fem_age


# In[65]:


# Step 9(b): Find the average age of all male passengers
avg_male_age = titanic_data[titanic_data['Sex']=='male']['Age'].mean()
avg_male_age


# In[52]:


avg_fem_age = titanic_data.groupby("Sex").mean().Age
avg_fem_age


# In[67]:


# Step 10(a): Find the survival percentage of passengers in group "C"
group_c_passengers = titanic_data[ titanic_data['Embarked'] == 'S']
cherbourg_survival = group_c_passengers['Survived'].sum()
cherbourg_survival/len(group_c_passengers) * 100


# In[ ]:




