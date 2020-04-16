# In[4]:


import random
def get_n_random_numbers(n = 10, min_ = -5, max_ = 5):
    numbers = []
    for i in range(n):
        numbers.append(random.randint(min_, max_))
    return numbers
get_n_random_numbers()


# In[5]:


def my_linear_search(my_list, item_search):
    found = (-1,-1)    # default, eğer listede yoksa
    n = len(my_list)
    for indis in range(n):
        if my_list[indis] == item_search:
            found = (my_list[indis], indis) # listede bulundu, return bulunn sayı, indisi
            break #, uncomment for last found
    return found


# In[83]:


def my_experimental_study_linear(iterNum = 50):
    cost=[]
    x_low = -100
    x_high = 100
    array_size = 40
    print("array size : ", array_size)
    for i in range(iterNum):
        my_list = get_n_random_numbers(array_size, x_low, x_high)
        my_search_item = get_n_random_numbers(1, x_low, x_high)[0]
    
        result = my_linear_search(my_list, my_search_item)
        if(result[1] == -1):
            cost.append(array_size)
        else:
            cost.append(result[1])
        print(result)
    
    return cost


# In[7]:


c_s = my_experimental_study()


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


plt.plot(c_s)


# In[77]:


def my_binary_search(my_list, item_search):
    found=(-1,-1)
    low = 0
    high = len(my_list) - 1
    s=0
    while low <= high:
        mid = (low + high) // 2
        print(low,high,mid)
        s=s+1
        if my_list[mid] == item_search:
            return my_list[mid], mid, s
        elif my_list[mid] > item_search:
            high = mid - 1
        else:
            low = mid + 1
    print(s)
    return found[0], found[1], s # None


# In[11]:


def my_bubble_sort(my_list):
    n = len(my_list)
    #print(my_list)
    for i in range(n - 1, -1, -1):
        for j in range(0, i):
            if not(my_list[j] < my_list[j + 1]):
                # print("swap işlemi")
                temp = my_list[j]
                my_list[j] = my_list[j + 1]
                my_list[j + 1] = temp
    return my_list


# In[48]:


x_low = -10
x_high = 10
array_size = 4
my_list = get_n_random_numbers(array_size, x_low, x_high)
my_list


# In[49]:


my_list=my_bubble_sort(my_list)
my_list


# In[58]:


my_search_item = get_n_random_numbers(1, x_low, x_high)[0]
my_search_item


# In[59]:


my_binary_search(my_list, my_search_item)


# In[78]:


def my_experimental_study_binary(iterNum = 50):
    cost=[]
    x_low = -100
    x_high = 100
    array_size = 40
    print("array size : ", array_size)
    for i in range(iterNum):
        my_list = get_n_random_numbers(array_size, x_low, x_high)
        my_list = my_bubble_sort(my_list)
        
        my_search_item = get_n_random_numbers(1, x_low, x_high)[0]
    
        result = my_binary_search(my_list, my_search_item)
        cost.append(result[2])
       #if(result[1] == -1):
       #    cost.append(array_size)
       #else:
       #    cost.append(result[1])
       #print(result)
    
    return cost


# In[85]:


c_s_b = my_experimental_study_binary()


# In[86]:


c_s_l = my_experimental_study_linear()


# In[88]:


plt.subplot(1,2,1), plt.plot(c_s_b)
plt.subplot(1,2,2), plt.plot(c_s_l)
plt.show()


# In[ ]:
