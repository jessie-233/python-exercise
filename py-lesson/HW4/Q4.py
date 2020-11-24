#!/usr/bin/env python
# coding: utf-8

# In[3]:


#写不出来了。。
days_dict = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31,8:31, 9:30, 10:31, 11:30, 12:31}
start_week = {1:3, 2:6, 3:0, 4:3, 5:5, 6:1, 7:3, 8:6, 9:2, 10:4, 11:0, 12:2}
with open('Q4-output.txt', 'wb') as f:
    count = 1
    for i in range(1,13):
        f.seek((i-1)*50, 0)          
        f.write(bytes('\t\t\t%d月\n'%i,encoding = 'utf-8'))
        f.seek((i-1)*50, 1)   
        f.write(bytes('日\t一\t二\t三\t四\t五\t六\n', encoding = 'utf-8'))
        f.seek((i-1)*50, 1)
        for k in range(start_week[i]):
            f.write(bytes('\t',encoding = 'utf-8'))
        for j in range(1, days_dict[i]+1):
            f.write(bytes('%d\t'%j, encoding = 'utf-8'))
            if (start_week[i]+j) % 7 == 0:
                f.write(bytes('\n',encoding = 'utf-8'))
                f.seek((i-1)*50, 1)
        if i % 3 == 0:
            count +=1
    


# In[ ]:




