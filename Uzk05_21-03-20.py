# In[17]:


# Python3 code to demonstrate working of
# Median of list
# Using loop + "~" operator
# initializing list
test_list = [4, 5, 8, 9, 10, 17]
# printing list
print("The original list : " + str(test_list))
# Median of list
# Using loop + "~" operator
test_list.sort()
mid = len(test_list) // 2
res = (test_list[mid] + test_list[~mid]) / 2
# Printing result
print("Median of list is : " + str(res))


# In[18]:


# Python3 code to demonstrate working of
# Median of list
# Using statistics.median()
import statistics
# initializing list
test_list = [4, 5, 8, 9, 10, 17]
# printing list
print("The original list : " + str(test_list))
# Median of list
# Using statistics.median()
res = statistics.median(test_list)
# Printing result
print("Median of list is : " + str(res))


# In[19]:


# Python Program for recursive binary search.
# Returns index of x in arr if present, else -1
def binarySearch (arr, l, r, x):
 # Check base case
    if r >= l:
         mid = l + (r - l)/2
 # If element is present at the middle itself
         if arr[int(mid)] == x:
             return int(mid)

 # If element is smaller than mid, then it can only
 # be present in left subarray
         elif arr[int(mid)] > x:
             return binarySearch(arr, l, mid-1, x)
 # Else the element can only be present in right subarray
         else:
             return binarySearch(arr, mid+1, r, x)
    else:
 # Element is not present in the array
        return -1
# Test array
arr = [ 2, 3, 4, 10, 40 ]
x = 10
# Function call
result = binarySearch(arr, 0, len(arr)-1, x)
if result != -1:
    print("Element is present at index %d" % result)
else:
    print("Element is not present in array")


# In[20]:


# Iterative Binary Search Function
# It returns location of x in given array arr if present,
# else returns -1
def binarySearch(arr, l, r, x):
    while l <= r:
        mid = l + (r - l)/2;
# Check if x is present at mid
        if arr[int(mid)] == x:
            return mid
 # If x is greater, ignore left half
        elif arr[int(mid)] < x:
            l = mid + 1
 # If x is smaller, ignore right half
        else:
            r = mid - 1

 # If we reach here, then the element was not present
    return -1
# Test array
arr = [ 2, 3, 4, 10, 40 ]
x = 10
# Function call
result = binarySearch(arr, 0, len(arr)-1, x)
if result != -1:
    print("Element is present at index %d" % result)
else:
    print("Element is not present in array")


# In[ ]:




