# In[30]:


#  İbrahim AKIN

import random

def getModeMedian(list_1):             
    
    
    n = len(list_1)
    
    # Bubble Sort                      
    for i in range(n):                 
        for j in range(0, n-i-1):      
            if list_1[j] > list_1[j+1]:                          
                list_1[j], list_1[j+1] = list_1[j+1], list_1[j]  
   
    # Medyan                             
    if n % 2 == 0:                       
        median1 = list_1[n//2]           
        median2 = list_1[n//2 - 1]       
        median  = (median1 + median2)/2  
    else: 
        median = list_1[n//2]            
        
    # Mod                                
    count_array = {}
    mode = []
    s = 0
    t = 1
    for i in range(n):                     
        if list_1[i] in count_array.keys():
            count_array[list_1[i]] += 1
            t = count_array[list_1[i]]
        else:
            count_array[list_1[i]] = 1
        if(t > s):
            s = t

    
    for i,j in count_array.items():        
        if(s == j):                        
            mode.append(i)
    return mode, median


# In[34]:


print(getModeMedian([1, 2, 3, 3, 3, 2, 2, 4, 5, 5, 6, 7, 5, 8])) 
                                                                 


# In[47]:


s = 10
list_1=[]
for i in range(s):
    t=int(random.uniform(0,10))
    list_1.append(t)


# In[48]:


print(list_1)
print(getModeMedian(list_1)) 


# In[ ]:




