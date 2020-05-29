# In[44]:


class Node: 
    def __init__(self, key): 
        self.left = None
        self.right = None
        self.val = key 
        self.depth = 0
        
    def __str__(self):
        return "val : " + str(self.val) + " depth : " + str(self.depth)


# In[36]:


def insert(root,node): 
    if root is None: 
        root = node 
    else: 
        node.depth += 1
        if root.val < node.val: 
            if root.right is None: 
                root.right = node 
            else: 
                insert(root.right, node) 
        else: 
            if root.left is None: 
                root.left = node 
            else: 
                insert(root.left, node) 


# In[80]:


SumOfDepth = 0
NumOfNodes = 0
def inorder(root): 
    if root: 
        inorder(root.left) 
        print(root.val , " (" , root.depth , ")" , end = " - ")
        global SumOfDepth
        global NumOfNodes
        SumOfDepth += root.depth
        NumOfNodes += 1
        inorder(root.right)


# In[81]:


r = Node(50) 
insert(r,Node(30)) 
insert(r,Node(20)) 
insert(r,Node(40)) 
insert(r,Node(70)) 
insert(r,Node(60)) 
insert(r,Node(80)) 


# In[84]:


import random


# In[85]:


for i in range(100):
    insert(r,Node(random.randint(-1000,1000)))


# In[86]:


import math
inorder(r)
print("\n",SumOfDepth, " - ", NumOfNodes, " - ", SumOfDepth/NumOfNodes, " - ", math.log(NumOfNodes))
SumOfDepth = 0
NumOfNodes = 0


# In[40]:


def search(root,key): 
      
    # Base Cases: root is null or key is present at root 
    if root is None or root.val == key: 
        return root 
  
    # Key is greater than root's key 
    if root.val < key: 
        return search(root.right,key) 
    
    # Key is smaller than root's key 
    return search(root.left,key) 


# In[41]:


result = search(r, 20)
result.val


# In[42]:


def minValueNode(node): 
    current = node 
  
    # loop down to find the leftmost leaf 
    while(current.left is not None): 
        current = current.left  
  
    return current  
  
# Given a binary search tree and a key, this function 
# delete the key and returns the new root 
def deleteNode(root, key): 
  
    # Base Case 
    if root is None: 
        return root  
  
    # If the key to be deleted is smaller than the root's 
    # key then it lies in  left subtree 
    if key < root.val: 
        root.left = deleteNode(root.left, key) 
  
    # If the kye to be delete is greater than the root's key 
    # then it lies in right subtree 
    elif(key > root.val): 
        root.right = deleteNode(root.right, key) 
  
    # If key is same as root's key, then this is the node 
    # to be deleted 
    else: 
          
        # Node with only one child or no child 
        if root.left is None : 
            temp = root.right  
            root = None 
            return temp  
              
        elif root.right is None : 
            temp = root.left  
            root = None
            return temp 
  
        # Node with two children: Get the inorder successor 
        # (smallest in the right subtree) 
        temp = minValueNode(root.right) 
  
        # Copy the inorder successor's content to this node 
        root.key = temp.val 
  
        # Delete the inorder successor 
        root.right = deleteNode(root.right , temp.val) 
  
  
    return root  


# In[43]:


print ("Inorder traversal of the given tree")
inorder(r) 
  
print ("\nDelete 20")
r = deleteNode(r, 60) 
print ("Inorder traversal of the modified tree")
inorder(r) 


# In[ ]: