#!/usr/bin/env python
# coding: utf-8

# In[ ]:


n = eval(input('请输入本金（元）：'))
for i in range(30):
    n *= 1 + (i + 1) / 100 
print('最终资产为：%.2f元'%n)


# In[ ]:




