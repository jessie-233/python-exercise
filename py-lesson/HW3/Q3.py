#!/usr/bin/env python
# coding: utf-8

# In[9]:


def fact(x):
    return x * fact(x-1) if x > 1 else 1


def combi(n, k):
    if(k > n):
        print('input error!')
    else:
        print(int(fact(n) / fact(k) / fact(n - k)))
        
    


# In[10]:


combi(3,2)


# In[12]:


combi(6,3)


# In[ ]:




