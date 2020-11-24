#!/usr/bin/env python
# coding: utf-8

# In[4]:


def sum_digit(n):
    nums = []
    j = 10
    sum = 0
    while not(0 <= j <= 9):
        i = n % 10
        nums.append(i)
        j = int(n / 10)
        n = j
    nums.append(j)
    for num in nums:
        sum += num
    if not(1 <= sum <= 9):
        return(sum_digit(sum))
    else:
        return(sum)


# In[28]:


print(sum_digit(1))


# In[31]:


print(sum_digit(1283))


# In[32]:


sums = []
for x in range(1,100000):
    sums.append(sum_digit(x))
for i in range(1,10):
    time = sums.count(i)
    comp = time / len(sums) *100
    print('%d出现的比例为：%.2f%%'%(i,comp))
   


# In[ ]:




