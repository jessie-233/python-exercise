#!/usr/bin/env python
# coding: utf-8

# In[3]:


import random  
import matplotlib.pyplot as plt

def MonPi(number):
    N2 = number
    N1 = 0
    inner_points = []
    outer_points = []
    for i in range(N2):
        x = random.random()  #随机生成0~1实数
        y = random.random()
        if x*x+y*y<=1:
            N1+=1
            inner_points.append((x,y))
        else:
            outer_points.append((x,y))
    
    print("PI=", 4*N1/N2)


# In[4]:


MonPi(100000)


# In[5]:


MonPi(10000000)


# In[ ]:




