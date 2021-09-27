#!/usr/bin/env python
# coding: utf-8

# In[2]:


principal = input('请输入本金金额（元）：')
interest = input('请输入利率（%）：')
all = float(principal) * (1 + float(interest)/100 )
print('一年后本息为：%.2f 元'%all)


# In[ ]:




