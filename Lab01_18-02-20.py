# In[87]:


def fib_1(n):
    if(n < 2):
        return n
    else:
        return fib_1(n-1) + fib_1(n-2)


# In[88]:


def fib_2(n):
    if(n < 2):
        return n
    else:
        a = 0
        b = 1
        c = a + b
        s = 2
        while(s < n):
            a = b
            b = c
            c = a + b
            s += 1
        return c


# In[92]:


t = 6


# In[93]:


print(t, fib_1(t))


# In[94]:


print(t, fib_2(t))


# In[98]:


def function_1(a, b):
    s = 1
    for i in range(b):
        s = s * a
    return s


# In[102]:


def function_2(a, b):
    if b==0:
        return 1
    elif b==1:
        return a
    else:
        return function_2(a*b, b/2)


# In[103]:


print(function_1(2, 3))


# In[107]:


print(function_2(2, 4))


# In[ ]:




