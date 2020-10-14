#!/usr/bin/env python
# coding: utf-8

# In[1]:


num = 100
sum = 0
while num <= 999:
    a = num % 10
    b = int((num % 100) / 10)
    c = int(num / 100)
    sum = a**3 + b**3 +c**3
    if sum == num:
        print(num)
    num += 1
        


# In[ ]:




