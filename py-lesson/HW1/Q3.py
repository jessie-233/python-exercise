#!/usr/bin/env python
# coding: utf-8

# In[12]:


def calcul(x):
    result = 1
    if x ==0:
        return result
    elif x>0:
        while x>=1:
            result *= x
            x -=1
        return result
    


# In[22]:


a = int(input('请输入第一个整数：'))
b = int(input('请输入第二个整数：'))
if a<0 or b<0:
    print('error!')
else:
    c = calcul(a) + calcul(b)
    print('两数阶乘之和为：%d'%c)


# In[ ]:




