# In[5]:


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


# In[6]:


def build_min_heap(array):
    for i in reversed(range(len(array)//2)):
        min_heapify(array, i)


# In[7]:


my_array_1 = [8, 9, 10, 3, 4, 7, 14, 1, 2, 16]


# In[8]:


build_min_heap(my_array_1)


# In[9]:


my_array_1


# In[10]:


def insert_heap(array_with_heap_property, item):
    array_with_heap_propertyy_with_heap_property.append(array_with_heap_property, item)
    for i in remove()


# In[13]:


n = 10
while n>0:
    print(n)
    n = n//2


# In[14]:


def heapsort(array):
    array = array.copy()
    build_min_heap(array)
    sorted_array = []
    for _ in range(len(array)):
        array[0], array[-1] = array[-1], array[0]
        sorted_array.append(array.pop())
        min_heapify(array, 0)
    return sorted_array


# In[41]:


my_array_2 = heapsort(my_array_1)
my_array_1, my_array_2


# In[18]:


def insertItemToHeap(my_heap, item):
    my_heap.append(item)
    n = len(my_heap)
    parent_indis = n // 2
    while parent_indis >= 1:
        min_heapify(my_heap, parent_indis)
        parent_indis = parent_indis // 2


# In[19]:


insertItemToHeap(my_array_2, 5)
my_array_2


# In[20]:


n = len(my_array_2)
n


# In[22]:


s = n + 1
while s >= 1:
    print(s)
    s = s // 2


# In[27]:


1//2


# In[28]:


2//2


# In[29]:


for n in range(0, 10):
    print(n, end = " ")
    if(n//2 == n/2):
        print("çift")
    if(n//2 != n/2):
        print("tek")


# In[33]:


for n in range(10):
    if(n//2 == n/2):
        parent = (n//2)-1
    else:
        parent = n//2
    print(n, parent)


# In[76]:


def insertItemToHeap_2(my_heap, item):
    my_heap.append(item)
    n = len(my_heap) - 1
    
    if(n//2 == n/2):
        parent_indis = (n//2)-1
    else:
        parent_indis = n//2
    
    while parent_indis >= 0:
        min_heapify(my_heap, parent_indis)
        # if exchange not exists, then break
        if(parent_indis//2 == parent_indis/2):
            parent_indis = (parent_indis//2)-1
        else:
            parent_indis = parent_indis//2


# In[77]:


def removeItemFromHeap(my_heap):
    my_heap[0], my_heap[-1] = my_heap[-1], my_heap[0]
    s = my_heap.pop()
    min_heapify(my_heap, 0)
    return my_heap


# In[78]:


print(my_array_1)
my_array_2 = heapsort(my_array_1)
print(my_array_2)


# In[79]:


print(my_array_2)
insertItemToHeap_2(my_array_2, 5)
print(my_array_2)


# In[80]:


print(my_array_2)
removeItemFromHeap(my_array_2)
print(my_array_2)


# In[ ]:




