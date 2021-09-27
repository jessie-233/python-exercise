#!/usr/bin/env python
# coding: utf-8

# In[5]:


def calculate(time):
    all = float(principal) * (1 + float(interest)/100 )
    all *= (1 + float(interest)/100) ** int(time)
    return all


# In[9]:


principal = input('请输入本金金额（元）：')
interest = input('请输入利率（%）：')
time = int(input('请输入自动转存次数：'))
result = calculate(time)
print('自动转存%d次后，期满金额为：%.2f 元' %(time, result))


# In[ ]:




