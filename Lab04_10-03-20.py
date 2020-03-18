# In[28]:


def my_search(list1, x):
    s=0
    for i in list1:
        s+=1
        if x == i:
            return True,s
    return False,s


# In[4]:


list1=[0,1,2,5]


# In[5]:


list1[1]=-10


# In[6]:


for i in range(10):
    list1.append(i*10)
    


# In[11]:


def check_prime(n):
    s=0
    if n!=1:
        for factor in range(2,n):
            s+=1
            if n%factor == 0:
                return False,s
    else:
        return False,s
    return True,s


# In[12]:


check_prime(10),check_prime(13),check_prime(23),check_prime(310)


# In[21]:


numbers=[10,13,23,310,69,49]


# In[22]:


for num_ in numbers:
    print(num_, check_prime(num_))


# In[30]:


x=1
my_search(numbers,x)


# In[73]:


import random


# In[83]:


def getMyList(s):
    list1=[]
    for i in range(s):
        t=int(random.uniform(0,1000))
        list1.append(t)
    return list1


# In[84]:


def getMyNumber():
    return int(random.uniform(0,1000))


# In[51]:


myList = getMyList(10)
myNumber = getMyNumber()
myNumber, myList


# In[52]:


my_search(myList,myNumber)


# In[58]:



print("Liste boyutu " ,len(myList))
mySearchNumbers=[2,45,78,-34,55]
t=0
for x in mySearchNumbers:
    t1=my_search(myList,x)[1]
    t=t+t1
#print("Ortalama değer ",b)
t, t/len(mySearchNumbers)


# In[96]:


def my_search_complexity(numOfItem=10, numOfTrials=5):
    my_list = getMyList(numOfItem)
    
    mySearchNumbers=getMyList(numOfTrials)
    
    print("Liste boyutu " ,len(my_list))
    t=0
    for x in mySearchNumbers:
        t1=my_search(my_list,x)[1]
        t=t+t1
    #print("Toplam değer, Ortalama değer")
    print(t, t/len(mySearchNumbers))


# In[98]:


my_search_complexity(10,5)
my_search_complexity(50,25)
my_search_complexity(100,50)
my_search_complexity(1000,500)
my_search_complexity(1000,800)
my_search_complexity(1000,900)


# In[ ]:


random.uniform(1,6)


# In[ ]:




