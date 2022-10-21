#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
import pandas as pd

df = pd.read_csv('cleaned_data.csv')


# In[96]:


import math 
import matplotlib.pyplot as plt 

df_a = df[df['expiration'] == '2020-11-13']
df_b = df_a[df_a['strike'] == 165.0]

# Isolating a specific option so that the underlying bid is unique to each time stamp. 
# We use this to calculate the vol of the underlying

df_b['log_rtn'] = np.log(df_b['underlying_bid']).diff()

window = 1  # trading days in rolling window
dpy = 252  # trading days per year
ann_factor = dpy / window

df_b['real_var'] = df_b['log_rtn'].var() * ann_factor # removed rolling because I don't know how it works
df_b['real_vol'] = np.sqrt(df_b['real_var'])
df_b.head()
#You can check here that volatility shows up in the column on the right. 


# In[93]:


# This is exploratory data analysis on what a "rolling window" does. 
# We have rolling volatility set to a 21-day window here. Unsure if useful

rollist=df_b['log_rtn'].rolling(21) 
rolvol=rollist.std(ddof=0) 
plt.plot(rolvol) 
plt.title('Volatility as measured by standard deviation') 
plt.show() 

#df['underlying_bid'].pct_change().rolling(1500).std()*(252**0.5)


# In[97]:


#defining black scholes function, implied volatility, and the delta. 
# code lifted from UChic, considers each option individually
def d1(self,S,K,T,r,sigma):
    return(np.log(S/K)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T))

def d2(self,S,K,T,r,sigma):
    return self.d1(S,K,T,r,sigma)-sigma*np.sqrt(T)

def bs_call(self,S,K,T,r,sigma):
    return S*norm.cdf(self.d1(S,K,T,r,sigma))-K*np.exp(-r*T)*norm.cdf(self.d2(S,K,T,r,sigma))

def iv_call(self,S,K,T,r,C):
    return max(0, fsolve((lambda sigma: np.abs(self.bs_call(S,K,T,r,sigma) - C)), [1])[0])

def delta_call(self,S,K,T,C):
    sigma = self.iv_call(S,K,T,0,C)
    return 100 * norm.cdf(self.d1(S,K,T,0,sigma))


# In[ ]:


# Use theo and price to get upside, use delta and upside to get weights

# calculate upside
def U(self, bs_call, S):
    U = (bs_call - S)/S
    return U

#total modified upside takes in a vector of upsides. 
#qn:this won't work if we are doing it on an option by option basis? How do we get all the upsides into a vector/equivalent?
def totalU(U_vect, delta_call):
    S = 0
    for i in U_vect:
        S+= U_vect[i]*delta[i]
    return S

# calculate weights 
def W(U, delta_call, totalU):
    W = (U*delta_call)/totalU 
    return W


# In[ ]:


#kelly size
#this code accepts vectors: a vector of weights, a vector of upsides

def aggregateU(W,U):
A = 0
for i in range(len(W)):
    A+=W[i]*U[i]
    return A

def aggregateP(W,P):
B = 0
for i in range(len(W)):
    B+=W[i]*P[i]
    return B
    
def kelly(A,B):
K = B - (1-B)/A
    return K


# In[ ]:


# buy and sell! And implement a delta hedge 

