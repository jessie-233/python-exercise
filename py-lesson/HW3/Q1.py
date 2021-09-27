#!/usr/bin/env python
# coding: utf-8

# In[15]:


def sum_digit(n):
    nums = []
    j = 10
    sum = 0
    while not(1 <= j <= 9):
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


# In[17]:


print(sum_digit(1478))

