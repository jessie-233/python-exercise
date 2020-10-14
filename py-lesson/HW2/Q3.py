#!/usr/bin/env python
# coding: utf-8

# In[7]:


for i in range(1,10):
    j = 1
    while j <= i:
        if j != i:
            print('%dx%d=%-4d'%(j,i,j*i), end = '')
        else:
            print('%dx%d=%d'%(j,i,j*i))
        j += 1


# In[ ]:




