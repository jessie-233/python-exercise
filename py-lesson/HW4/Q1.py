#!/usr/bin/env python
# coding: utf-8

# In[81]:


def is_word_palindrome(file: str):
    with open(file, 'r') as f:
        k = 0  #行数
        for line in f.readlines():
            k += 1
            s = list(line.lower()) #字符串转列表，小写
            txt = []
            flag = True
            for item in s:  #提取出纯字母
                if item.isalpha():
                    txt.append(item)
            for i in range(int(len(txt)/2)):  #判断是否回文
                if txt[i] != txt[len(txt)-1-i]:
                    flag = False
                    break
            if flag:
                print('第%d行是回文'%k)
            else:
                print('第%d行不是回文'%k)


# In[82]:


is_word_palindrome('Q1.txt')

