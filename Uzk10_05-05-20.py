# In[43]:


class Item(object):
    def __init__(self, n, v, w):
        self.name = n
        self.value = float(v)
        self.weight = float(w)
        
    def getName(self):
        return self.name
    
    def getValue(self):
        return self.value
    
    def getWeight(self):
        return self.weight
    
    def __str__(self):
        result = '<' + self.name + ', ' + str(self.value)                + ', ' + str(self.weight) + '>'
        return result
    
def value(item):
    return item.getValue()

def weightInverse(item):
    return 1.0/item.getWeight()

def density(item):
    return item.getValue()/item.getWeight()

def buildItems():
    names = ['clock', 'painting', 'radio', 'vase', 'book', 'computer']
    values = [175,90,20,50,10,200]
    weights = [10,9,4,2,1,20]
    Items = []
    for i in range(len(values)):
        Items.append(Item(names[i], values[i], weights[i]))
    return Items


# In[6]:


def greedy(items, maxWeight, keyFunction):
    """Assumes Items a list, maxWeight >= 0,
        keyFunction maps elements of Items to floats"""
    itemsCopy = sorted(items, key=keyFunction, reverse = True)
    result = []
    totalValue = 0.0
    totalWeight = 0.0
    for i in range(len(itemsCopy)):
        if (totalWeight + itemsCopy[i].getWeight()) <= maxWeight:
            result.append(itemsCopy[i])
            totalWeight += itemsCopy[i].getWeight()
            totalValue += itemsCopy[i].getValue()
    return (result, totalValue)

def testGreedy(items, constraint, keyFunction):
    taken, val = greedy(items, constraint, keyFunction)
    print ('Total value of items taken = ', val)
    for item in taken:
        print (' ', item)

def testGreedys(maxWeight = 20):
    items = buildItems()
    print ('Use greedy by value to fill knapsack of size', maxWeight)
    testGreedy(items, maxWeight, value)
    print ("\nUse greedy by weight to fill knapsack of size", maxWeight)
    testGreedy(items, maxWeight, weightInverse)
    print ("\nUse greedy by density to fill knapsack of size", maxWeight)
    testGreedy(items, maxWeight, density)


# In[7]:


testGreedys()


# In[11]:


from pprint import pprint as pp
from itertools import chain, combinations
 
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


# In[12]:


def chooseBest(pset, maxWeight, getVal, getWeight):
    bestVal = 0.0
    bestSet = None
    for items in pset:
        itemsVal = 0.0
        itemsWeight = 0.0
        for item in items:
            itemsVal += getVal(item)
            itemsWeight += getWeight(item)
        if itemsWeight <= maxWeight and itemsVal > bestVal:
            bestVal = itemsVal
            bestSet = items
    return (bestSet, bestVal)

def testBest(maxWeight = 20):
    items = buildItems()
    pset = set(powerset(items))
    taken, val = chooseBest(pset, maxWeight, Item.getValue, Item.getWeight)
    
    print ('Total value of items taken =', val)
    for item in taken:
        print(item)


# In[13]:


testBest()


# In[15]:


def fib(n):
    """Assumes n is an int >= 0
    Returns Fibonacci of n"""
    print("call -> ", n)
    if n == 0 or n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)


# In[16]:


fib(5)


# In[31]:


def fastFib(n, memo = {}):
    """Assumes n is an int >= 0, memo used only by recursive calls
    Returns Fibonacci of n"""
    print("call -> ", n)
    if n == 0 or n == 1:
        return 1
    try:
        return memo[n]
    except KeyError:
        result = fastFib(n-1, memo) + fastFib(n-2, memo)
        memo[n] = result
        return result


# In[32]:


memo = None
fastFib(5)


# In[47]:


def maxVal(toConsider, avail):
    """Assumes toConsider a list of items, avail a weight
    Returns a tuple of the total weight of a solution to the
    0/1 knapsack problem and the items of that solution"""
    if toConsider == [] or avail == 0:
        result = (0, ())
    elif toConsider[0].getWeight() > avail:
        #Explore right branch only
        result = maxVal(toConsider[1:], avail)
    else:
        nextItem = toConsider[0]
        #Explore left branch
        withVal, withToTake = maxVal(toConsider[1:], avail - nextItem.getWeight())
        withVal += nextItem.getValue()
        #Explore right branch
        withoutVal, withoutToTake = maxVal(toConsider[1:], avail)
        #Choose better branch
        if withVal > withoutVal:
            result = (withVal, withToTake + (nextItem,))
        else:
            result = (withoutVal, withoutToTake)
    return result

def smallTest():
    names = ['a', 'b', 'c', 'd']
    vals = [6, 7, 8, 9]
    weights = [3, 3, 2, 5]
    Items = []
    for i in range(len(vals)):
        Items.append(Item(names[i], vals[i], weights[i]))
    val, taken = maxVal(Items, 5)
    for item in taken:
        print (item)
    print ('Total value of items taken =', val)


# In[65]:


import random
def buildManyItems(numItems, maxVal, maxWeight):
    items = []
    for i in range(numItems):
        items.append(Item(str(i),
                          random.randint(1, maxVal),
                          random.randint(1, maxWeight)))
    return items

def bigTest(numItems):
    items = buildManyItems(numItems, 10, 10)
    val, taken = maxVal(items, 40)
    print ('Items Taken')
    for item in taken:
        print (item)
    print ('Total value of items taken =', val)


# In[49]:


smallTest()


# In[76]:


def fastMaxVal(toConsider, avail, memo = {}):
    """Assumes toConsider a list of items, avail a weight
 memo used only by recursive calls
 Returns a tuple of the total weight of a solution to the
 0/1 knapsack problem and the items of that solution"""
    print("call -> ", len(toConsider))
    if (len(toConsider), avail) in memo:
        result = memo[(len(toConsider), avail)]
    elif toConsider == [] or avail == 0:
        result = (0, ())
    elif toConsider[0].getWeight() > avail:
        #Explore right branch only
        result = fastMaxVal(toConsider[1:], avail, memo)
    else:
        nextItem = toConsider[0]
        #Explore left branch
        withVal, withToTake =        fastMaxVal(toConsider[1:],
                   avail - nextItem.getWeight(), memo)
        withVal += nextItem.getValue()
        #Explore right branch
        withoutVal, withoutToTake = fastMaxVal(toConsider[1:],
                                               avail, memo)
        #Choose better branch
        if withVal > withoutVal:
            result = (withVal, withToTake + (nextItem,))
        else:
            result = (withoutVal, withoutToTake)
    memo[(len(toConsider), avail)] = result
    return result


# In[77]:


numItems = 5
items = buildManyItems(numItems, 10, 10)
fastMaxVal(items, 40)


# In[75]:


memo=None


# In[ ]: