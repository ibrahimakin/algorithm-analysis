# In[1]:


my_list=[]


# In[3]:


for i in range(100):
    my_list.append(i)   # O(1), O(n)
                        # size = 10   size * 2, 2000 + size*1.2
                        # 1 1 1 1 1 1 1 1  n  O(1)
#my_c = DynamicArry(s)


# In[ ]:


# ADT-s
# list array, pointer
# tree BST pointer array


# In[4]:


# ADT-s
#    internal
#        array     amortized  insert 1 1 1 1 1 1 ... n = 1
#        pointer


# In[5]:


#list search O(n), O(logn)


# In[6]:


import ctypes 


# In[39]:


class DynamicArray:
    """A dynamic array class akin to a simplified Python list."""
    def getsize(self):
        import sys
        try:
            return sys.getsizeof(self._A)  
          
        except:
            return 0
        
    def ToString(self):
        try:
            for i in self._A:
                print(i," ")
        except:
            pass
        
    def getLength(self):
        return len(self._A)
    def __init__(self):
        """Create an empty array."""
        self._n = 0                                    # count actual elements
        self._capacity = 500                             # default array capacity
        self._A = self._make_array(self._capacity)     # low-level array
    
    def _make_array(self, c):                        # nonpublic utitity
        """Return new array with capacity c."""  
        return (c * ctypes.py_object)()     #malloc  # see ctypes documentation
    
    def append(self, obj):
        """Add object to end of the array."""
        if self._n == self._capacity:                 # not enough room
            self._resize(2 * self._capacity)             # so double capacity
            print("cost islemi")
        self._A[self._n] = obj
        self._n += 1
    def _resize(self, c):                            # nonpublic utitity
        """Resize internal array to capacity c."""
        B = self._make_array(c)                        # new (bigger) array
        for k in range(self._n):                       # for each existing value
            B[k] = self._A[k]
            print("tasima islemi")
        self._A = B                                    # use the bigger array
        self._capacity = c
    def len_n(self):
        """Return number of elements stored in the array."""
        return self._n
    


# In[35]:

c=DynamicArray()


# In[36]:


c.getLength(), c.getsize()


# In[43]:


c=DynamicArray()
for i in range(500):
    c.append(-100)
c.getLength(), c.getsize(), c.len_n()


# In[21]:


import sys
from pympler import asizeof

s_1=sys.getsizeof(c)
s_2=asizeof.asizeof(c)
print("s_1 : {0}, s_2 : {1}".format(s_1,s_2))


# In[15]:


n = 100000
for i in range(n):
    c.append(12)
    c.append("sdfsdfsdf")
    
s_1=sys.getsizeof(c)
s_2=asizeof.asizeof(c)
print("s_1 : {0}, s_2 : {1}".format(s_1,s_2))


# In[ ]:




