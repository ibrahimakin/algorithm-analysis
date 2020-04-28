# In[1]:


def min_heapify(array, i):
    left = 2 * i + 1
    right = 2 * i + 2
    length = len(array) - 1
    smallest = i
    if left <= length and array[i] > array[left]:
        smallest = left
    if right <= length and array[smallest] > array[right]:
        smallest = right
    if smallest != i:
        array[i], array[smallest] = array[smallest], array[i]
        min_heapify(array, smallest)


# In[2]:


def build_min_heap(array):
    for i in reversed(range(len(array)//2)):
        min_heapify(array, i)


# In[3]:


my_array_1 = [8, 9, 10, 3, 4, 7, 14, 1, 2, 16]


# In[4]:


build_min_heap(my_array_1)


# In[5]:


my_array_1


# In[ ]:




