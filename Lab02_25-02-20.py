# In[28]:


def myf1(list1=[]):
    s = 0
    n = len(list1)
    maxsum = 0
    for i in range(n):
        for j in range(1, n):
            t = 0
            for k in range(i, j):
                t = t + list1[k]
                s = s + 1
            if t > maxsum:
                maxsum = t
    return maxsum, s


# In[29]:


def myf2(list1=[]):
    s = 0
    n = len(list1)
    maxsum = 0
    for i in range(n):
        t = 0
        for j in range(i, n):
            t = t + list1[j]
            s = s + 1
            if t > maxsum:
                maxsum = t
    return maxsum, s


# In[30]:


list1 = [4, -3, 5, -2, -1, 2, 6, -2]
list2 = [4, -3, 5, -2, -1, 2, 6, -2, 4, -3, 5, -2, -1, 2, 6, -2]


# In[31]:


print(myf1(list1))
print(myf2(list1))
print(myf1(list2))
print(myf2(list2))


# In[32]:


def myf3(list1=[]):
    n = len(list1)
    
    if n==1:
        return list1[0]
    else:
        left_list = list1[0:(n//2)]
        right_list= list1[(n//2):n]
        
        left_sum = myf3(left_list)
        right_sum= myf3(right_list)
        
        center_sum = 0
        temp_sum_left = 0
        t = 0
        for i in range(n//2-1, -1, -1):
            t = t + list1[i]
            if t > temp_sum_left:
                temp_sum_left = t
        
        temp_sum_right = 0
        t = 0
        for i in range(n//2, n):
            t = t + list1[i]
            if t > temp_sum_right:
                temp_sum_right = t
        
        center_sum = temp_sum_left + temp_sum_right
        
        return max_of_three(left_sum, right_sum, center_sum)


# In[33]:


def max_of_two(a, b):
    if a > b:
        return a
    else:
        return b


# In[34]:


def max_of_three(a, b, c):
    return (max_of_two(a, max_of_two(b, c)))


# In[36]:


print(myf3(list1))


# In[37]:


print(myf3(list2))


# In[38]:


def power_loop(a, b):
    t = 1
    for i in range(b):
        t = t * a
    return t


# In[39]:


def power_recursive(a, b):
    if b==1:
        return a
    return power_recursive(a, b-1)*a


# In[42]:


print(power_loop(2, 8))
print(power_recursive(2,8))


# In[ ]:




