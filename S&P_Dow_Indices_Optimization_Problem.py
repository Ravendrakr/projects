#!/usr/bin/env python
# coding: utf-8

# Created on Monday Jan 03 10:00:00 2022
# 
# @author: RAVENDRA KUMAR
# 
# python_version~ 3.8.5

# # Process
# * STEP1: Installing the required packages 
# * STEP 2: Importing the required packages 
# * STEP 3: Importing the data and processing it and Check the data and understanding the individal variables (Though in our case we have data which is already processed )
# * STEP 4: Defining the Problem matrix: P and q in 1/2 X*PX - q X s.t. l<= Ax <= u
# * STEP 5: Defining the contrained matrix A,l, and u
# * STEP 6: Upload these matrices in the the OSQP library 
# * STEP 7: Store the result

# # Packages installation

# In[17]:


# !pip install numpy 
# !pip install osqp
# !pip install pandas


# # Main Code

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import osqp
from scipy import sparse 


# ## Loading Raw Data

# In[19]:


df = pd.read_excel("C:/Users/JMD/Dropbox/private_job/S&P_Global/test_input_file.xlsx", sheet_name=3)
df.columns = df.columns.str.replace(' ','_')
df.columns = map(str.lower, df.columns)
df = df.rename(columns={"stock_key": "stockkey" })
df


# In[20]:


df.isnull().sum()


# In[21]:


df.describe()


# ## Subsetting Eligible Universe

# In[22]:


dataframe= df.copy() # Storing df as raw data 
dataframe.dtypes
dataframe["eligible"].unique()
dataframe["eligible"]= dataframe["eligible"].astype(str) # True and False are boolen data type hence needs to change their format
dataframe= dataframe[dataframe.eligible =="True"] # Keeping only eligible universe
dataframe=dataframe.reset_index() 
dataframe=dataframe.drop(dataframe.columns[0], axis=1)
dataframe=dataframe.iloc[:,0:7]
dataframe


# ## Defining the Problem matrix
# writing the problem in following form 
# 
# $\min \dfrac{1}{2}X^TPX  + qX$
# 
# s.t. $ l \le AX \le u$

# ### Defining Objective (P and q)

# In[23]:


a = np.zeros((1008, 1008), int) # Create matrix with only 0
np.fill_diagonal(a, 2) # fill diagonal with 2

P= pd.DataFrame(a) #Creating Dataframe using P matrix

dataframe_pwm= dataframe[['underlying_index_weight']]
dataframe_pwm.reset_index(inplace = True)
dataframe_pwm= dataframe_pwm[["underlying_index_weight"]]
dataframe_pwm= pd.merge(dataframe_pwm, P, left_index=True, right_index=True) # Merging using unique index

P_cols= P.columns
for i in P_cols:
    dataframe_pwm[i]= dataframe_pwm[i]/dataframe_pwm['underlying_index_weight']
    
P = dataframe_pwm.iloc[:,1:] # dropping the underlying_index_weight column


# renaming the columns as our desired variables

columns_name= []
for i in range(1,1009,1):
    x= "x_"+ str(i)
    #print(x)
    columns_name.append(x)    
P.columns=columns_name 


# Defining the linear part involved in the problem matrix 

q_linear_element= np.array([2 for x in range(1008)])

print('P:')
display(P)
print('\n\nq:')
display(q_linear_element)


# ### Defining contraints (A, l, u)

# In[24]:


# Creating contraints for industry group 

indgrp= dataframe["industry_group"].unique()
indgrp_n=[]
for i in indgrp:
    i= str(i)
    indgrp_n.append(i)

for i in indgrp_n:
    dataframe[i]=i
    #print(len(dataframe.columns))
    
dataframe["industry_group"]= dataframe["industry_group"].astype(str)


# Assinging column value 1 if rows matches with the column otherwise filling with zero.

for i in indgrp_n:
    dataframe[i]= np.where(i==dataframe["industry_group"], 1,0)
    j="indgrp_"+i
    dataframe=dataframe.rename(columns={i:j})
   # print(i)
    

# Creating contraints for country

country= dataframe["country"].unique()
for i in country:
    dataframe[i]=i
    #print(len(dataframe.columns))
    
    
# Assinging column value 1 if rows matches with the column otherwise filling with zero.

for i in country:
    dataframe[i]= np.where(i==dataframe["country"], 1,0)
    j="country_"+i
    dataframe=dataframe.rename(columns={i:j})

    
# Creating contraints for stockkeys

dataframe["stockkey"]= dataframe["stockkey"].astype(str)   
stockkey= dataframe["stockkey"].unique()

for i in stockkey:
    dataframe[i]= i
    #print(len(dataframe.columns))

# Assinging column value 1 if rows matches with the column otherwise filling with zero

for i in stockkey:
    dataframe[i]= np.where(i==dataframe["stockkey"], 1,0)
    j="stockkey_"+i
    dataframe=dataframe.rename(columns={i:j})
    
    
dfa=dataframe # Storing dataframe for cross verification and understanding in case of any doubt in calculation 


# Lower Bound constraints formation 

list_lower_bound_max=[]         # Generating Lower bound list to store upper bound for each category
for i in dfa.columns[7:]:
    j=i.split("_")[0]
    if j== "indgrp" or j=="country":             
        dfa[i]= dfa[i].astype(int)  
        lbmp=sum(dfa["underlying_index_weight"]*dfa[i])
        temp= lbmp
        lbmp=max(lbmp-0.05, lbmp*.75) #defined in the methodology sheet
        tuplet= [i, lbmp]
        list_lower_bound_max.append(tuplet)
        #print(i,lbmp)
    else:
        dfa[i]= dfa[i].astype(int)  
        lbmp=sum(dfa["underlying_index_weight"]*dfa[i])
        lbmp=max(0,lbmp-0.02)
        tuplet= [i, lbmp]
        list_lower_bound_max.append(tuplet)
        #print(i,lbmp)

        
# Upper Bound constraints formation 

list_upper_bound_min=[]        # Generating Upper bound list to store upper bound for each category
for i in dfa.columns[7:]:
    j=i.split("_")[0]
    if j== "indgrp" or j=="country":          
        dfa[i]= dfa[i].astype(int)  
        lbmp=sum(dfa["underlying_index_weight"]*dfa[i])
        temp= lbmp
        lbmp=min(lbmp+0.05, lbmp*1.25) #defined in methodology sheet
        tuplet= [i, lbmp]
        list_upper_bound_min.append(tuplet)
        #print(i,lbmp)
    else:
        dfa[i]= dfa[i].astype(int)  
        lbmp=sum(dfa["underlying_index_weight"]*dfa[i])
        lbmp=min(0.08,lbmp*10,lbmp+.02)
        tuplet= [i, lbmp]
        list_upper_bound_min.append(tuplet)
        #print(i,lbmp)   

lower_bound_max=pd.DataFrame(list_lower_bound_max)  #Creating dataframe for lower bound for each category
lower_bound_max=lower_bound_max.transpose() 
lower_bound_max.columns=lower_bound_max.iloc[0]
lower_bound_max=lower_bound_max[1:]
upper_bound_min=pd.DataFrame(list_upper_bound_min)
upper_bound_min=upper_bound_min.transpose()
upper_bound_min.columns=upper_bound_min.iloc[0]
upper_bound_min=upper_bound_min[1:]

# Defining first constraint 

dfa["WAFS_2"] = dfa['factor_score_2']*dfa['underlying_index_weight']
lwr_upr_wafs_2=[0, sum(dfa["WAFS_2"])*.50]
WAFS_2_constraint = dfa['WAFS_2'].tolist()
WAFS_2_constraint= WAFS_2_constraint+ lwr_upr_wafs_2
WASF_2_constraint= pd.DataFrame(WAFS_2_constraint, columns=["wafs_2_target"])
dfa=dfa.drop(dfa.columns[-1], axis=1)


#dfa.underlying_index_weight.dtypes

dfa= pd.concat([dfa,lower_bound_max])  # Generating Lower bound list to store upper bound for each category
dfa= pd.concat([dfa,upper_bound_min])  # Generating Upper bound list to store upper bound for each category

dfa = dfa.reset_index()
dfa= dfa.drop(dfa.columns[0], axis=1)
dfa=pd.merge(dfa, WASF_2_constraint, left_index=True, right_index=True)

A= dfa.iloc[:,7:] # Dropping out first 7 columns as we do not need them anymore. 

A["hundred_percent_weight"]= 1  # Creating a column which says that sum of resulting weights must be 1. 
A= A.transpose()

l = A.iloc[:,-2] 
u = A.iloc[:,-1]
A = A.iloc[:,:-2]


print('A:')
display(A)
print('\n\n bounds: [lower, upper]')
display(np.c_[l, u])

#Checking for wrong calculation: lower bound must be lower than upper bound for each constraint
(l>u).pipe(lambda s: s[s]) # This should be empty


# ## Solving Problem

# In[25]:


# Convert problem/ Constraint dataframe into array form . 

P = P.values
A = A.values

c,b=np.linalg.eig(P) #checking positive semidefinite of problem matrix. If c>= 0 then P is positive-semidefinite matrix

q=np.array(q_linear_element)
l=np.array(l)
u=np.array(u)

P = sparse.csc_matrix(P)
A = sparse.csc_matrix(A.astype(float))


# Create an OSQP object

prob = osqp.OSQP()


# Setup workspace and change alpha parameter

prob.setup(P, q, A, l, u, alpha=1.6)


# In[26]:


res = prob.solve()


# ## Storing Solution

# In[27]:


final_weights= res.x
final_weights=pd.DataFrame(final_weights, columns= ["final_weight"])
dataframe=dataframe.iloc[:,0:7] 
final_result= pd.merge(dataframe,final_weights, left_index=True, right_index=True)
final_result


# In[28]:


final_result['underlying_index_weight'].sum()


# In[29]:


final_result['final_weight'].sum()


# In[31]:


final_result.to_csv("C:/Users/JMD/Dropbox/private_job/S&P_Global/final_result.csv", index= False)

