# In[11]:


def bubbleSort(arr):
    n = len(arr)
    
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]


# In[12]:


arr = [64, 34, 25, 12, 22, 11, 90]


# In[13]:


print("Array: ")
for i in range(len(arr)):
    print("%d" %arr[i])


# In[14]:


bubbleSort(arr)


# In[15]:


print("Sorted array is: ")
for i in range(len(arr)):
    print("%d" %arr[i])


# In[16]:


def insertionSort(arr):
    loop_counter = 0
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
            loop_counter += 1
        arr[j+1] = key


# In[17]:


arr = [12, 11, 13, 5, 6]


# In[18]:


insertionSort(arr)


# In[19]:


for i in range(len(arr)):
    print("%d" %arr[i])


# In[ ]:




