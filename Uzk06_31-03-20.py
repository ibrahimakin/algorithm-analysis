# In[1]:


import random


# In[2]:


s = random.randint(1,100)
print(s)


# In[5]:


def get_n_random_numbers(n = 10, min_ = -5, max_ = 5):
    numbers = []
    for i in range(n):
        numbers.append(random.randint(min_, max_))
    return numbers


# In[6]:


print(get_n_random_numbers())


# In[7]:


my_list = get_n_random_numbers(15, -4, 4)
print(my_list)


# In[9]:


# array of tuples format 
histgram_1=[(-4,1),(-3,1),
            (-1,1),(0,2),
            (1,1), (3,1),
            (6,1), (8,1)]


# In[10]:


print(sorted(my_list))


# In[11]:


def my_frequency_with_dict(list):
    frequency_dict = {}
    for item in list:
        if (item in frequency_dict):
            frequency_dict[item] += 1
        else:
            frequency_dict[item] = 1
    return frequency_dict


# In[12]:


print(my_frequency_with_dict(my_list))


# In[13]:


def my_frequency_with_list_of_tuples(list_1):
    frequency_list = []
    for i in range(len(list_1)):
        s = False
        for j in range(len(frequency_list)):
            if (list_1[i] == frequency_list[j][0]):
                frequency_list[j][1] = frequency_list[j][1] + 1
                s = True
        if(s == False):
            frequency_list.append([list_1[i], 1])
    return frequency_list


# In[15]:


my_list  = [2, 3, 2, 5, 8, 2, 4, 3, 3, 2, 8, 5, 2, 4, 4, 4, 4, 4]
result_1 = my_frequency_with_dict(my_list)
result_2 = my_frequency_with_list_of_tuples(my_list)
print(result_1)
print(result_2)


# In[35]:


my_list_1 = get_n_random_numbers(5, -2, 2)
my_hist_d = my_frequency_with_dict(my_list_1)
print(my_hist_d)


# In[36]:


my_hist_l = my_frequency_with_list_of_tuples(my_list_1)
print(my_hist_l)


# In[37]:


frequency_max = -1
mode = -1
for key in my_hist_d.keys():
    print(key, my_hist_d[key])
    if my_hist_d[key] > frequency_max:
        frequency_max = my_hist_d[key]
        mode = key
print("Output ", mode, frequency_max)


# In[38]:


def my_mode_with_dict(my_hist_d):
    frequency_max = -1 
    mode = -1
    for key in my_hist_d.keys():
        # print(key,my_hist_d[key])
        if my_hist_d[key] > frequency_max:
            frequency_max = my_hist_d[key]
            mode = key
    return mode, frequency_max


# In[39]:


my_mode_with_dict(my_hist_d)


# In[40]:


my_list_100 = get_n_random_numbers(100, -40, 40)
my_hist_1 = my_frequency_with_dict(my_list_100)
my_mode_with_dict(my_hist_1)


# In[41]:


print(my_hist_1)


# In[42]:


print(sorted(my_list_100))


# In[43]:


my_list_1 = get_n_random_numbers(10)
my_hist_list = my_frequency_with_list_of_tuples(my_list_1)
print(my_hist_list)


# In[44]:


frequency_max = -1
mode = -1
for item,frequency in my_hist_list:
    print(item,frequency)
    if frequency>frequency_max:
        frequency_max=frequency
        mode=item
print("Output", mode, frequency_max)


# In[45]:


def my_mode_with_list(my_hist_list):
    frequency_max = -1
    mode = -1
    for item,frequency in my_hist_list:
        print(item, frequency)
        if frequency > frequency_max:
            frequency_max = frequency
            mode = item
    return mode, frequency_max


# In[46]:


print(my_mode_with_list(my_hist_list))


# In[47]:


my_list_100 = get_n_random_numbers(20, -4, 4)
my_hist_1 = my_frequency_with_list_of_tuples(my_list_100)
print(my_mode_with_list(my_hist_1))


# In[48]:


print(my_list_100)


# In[50]:


def my_linear_search(my_list, item_search):
    found = (-1,-1)    # default, eğer listede yoksa
    n = len(my_list)
    for indis in range(n):
        if my_list[indis] == item_search:
            found = (my_list[indis], indis)
            # break, uncomment for last found
    return found


# In[51]:


my_list = get_n_random_numbers(10, -5, 5)
print(my_list)


# In[52]:


print(my_linear_search(my_list, 10))


# In[53]:


my_list = get_n_random_numbers(10, -50, 50)
print(my_list)


# In[55]:


s, t = 0, 0
for item in my_list:
    s = s + 1
    t = t + item
mean_ = t / s
print(mean_)


# In[57]:


def my_mean(my_list):
    s, t = 0, 0
    for item in my_list:
        s = s + 1
        t = t + item
    mean_ = t / s
    return mean_


# In[58]:


my_list = get_n_random_numbers(4, -5, 5)
print(my_list)
print(my_mean(my_list))


# In[59]:


print(my_list)


# In[60]:


n = len(my_list)
print(my_list)
for i in range(n - 1, -1, -1):
    for j in range(0, i):
        if not(my_list[j] < my_list[j + 1]):
            # print("swap işlemi")
            temp = my_list[j]
            my_list[j] = my_list[j + 1]
            my_list[j + 1] = temp
print(my_list)


# In[61]:


# with function
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


# In[62]:


my_list = get_n_random_numbers(4, -5, 5)
print(my_list)
print(my_bubble_sort(my_list))


# In[63]:


def my_binary_search(my_list, item_search):
    found=(-1,-1)
    low = 0
    high = len(my_list) - 1
    while low <= high:
        mid = (low + high) // 2
        if my_list[mid] == item_search:
            return my_list[mid],mid
        elif my_list[mid] > item_search:
            high = mid - 1
        else:
            low = mid + 1
    return found # None


# In[65]:


my_list_1 = get_n_random_numbers(10)
print("liste ", my_list_1)
my_list_2 = my_bubble_sort(my_list_1)
print("sırali liste ", my_list_2)
print(my_binary_search(my_list_2, 3)) # 1


# In[73]:


size = input("dizi boyutunu giriniz ")
size = int(size) # convert str to int
my_list_1 = get_n_random_numbers(size)
print("liste ", my_list_1)


# In[74]:


my_list_2 = my_bubble_sort(my_list_1)


# In[75]:


print(my_list_2)
n = len(my_list_2)
if n % 2 == 1:
    middle = int(n / 2) + 1
    median = my_list_2[middle]
    print(median)
else:
    middle_1 = my_list_2[int(n / 2)]
    middle_2 = my_list_2[int(n / 2) + 1]
    median = (middle_1 + middle_2) / 2
    print (median)


# In[79]:


def my_median(my_list):
    my_list_2 = my_bubble_sort(my_list)
    #print(my_list_2)
    n = len(my_list_2)
    if n % 2 == 1:
        middle = int(n / 2) + 1
        median = my_list_2[middle]
        # print(median)
    else:
        middle_1 = my_list_2[int(n/2)]
        middle_2 = my_list_2[int(n/2)+1]
        median = (middle_1+middle_2)/2
        # print(median)
    return median


# In[80]:


my_list_2 = get_n_random_numbers(6, -10, 10)
print(my_median(my_list_2))


# In[ ]:




