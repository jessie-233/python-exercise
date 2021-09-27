#!/usr/bin/env python
# coding: utf-8

# In[ ]:


count = -1
i = 1
aver = 0
sum = 0
while i != 0:
    count += 1
    i = eval(input('enter a number: '))
    sum += i
if count == 0:
    print('输入错误')
else:
    aver = sum / count
    print('平均值为：%.2f'%aver)    


# In[ ]:




