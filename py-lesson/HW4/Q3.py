#!/usr/bin/env python
# coding: utf-8

# In[1]:


days_dict = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31,8:31, 9:30, 10:31, 11:30, 12:31}
start_week = {1:3, 2:6, 3:0, 4:3, 5:5, 6:1, 7:3, 8:6, 9:2, 10:4, 11:0, 12:2}
with open('Q3-output.txt', 'w') as f:
    for i in range(1,13):
        print('\n\n\t\t\t%d月'%i, file = f)
        print('日\t一\t二\t三\t四\t五\t六', file = f)
        for k in range(start_week[i]):
            print('\t', end = '', file = f)
        for j in range(1, days_dict[i]+1):
            print(j, '\t', end = '', file = f)
            if (start_week[i]+j) % 7 == 0:
                print('\n', file = f)
    


# In[ ]:




