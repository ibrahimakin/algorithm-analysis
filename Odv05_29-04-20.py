# In[ ]:


#    Kaynaklar
# https://en.wikipedia.org/wiki/Perceptron
# https://link.springer.com/chapter/10.1007/978-3-540-39624-6_18
# https://devhunteryz.wordpress.com/2018/07/05/yapay-sinir-agi-egitimi-cok-katmanli-perceptronmulti-layer-perceptron/
# https://medium.com/@thomascountz/19-line-line-by-line-python-perceptron-b6f113b161f3
# https://medium.com/@stanleydukor/neural-representation-of-and-or-not-xor-and-xnor-logic-gates-perceptron-algorithm-b0275375fea1 (Şekil-1)
# https://www.quora.com/How-can-I-implement-AND-logic-using-single-layer-perceptron (Şekil-2)
# https://www.researchgate.net/figure/A-hypothetical-example-of-Multilayer-Perceptron-Network_fig4_303875065 (Şekil-5)
# https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/
# https://www.researchgate.net/figure/An-illustration-of-the-signal-processing-in-a-sigmoid-function_fig2_239269767 (Şekil-6)
# http://ml.informatik.uni-freiburg.de/former/_media/documents/teaching/ss09/ml/perceptrons.pdf (Lemma (worst case running time))
# https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d
# https://www.researchgate.net/post/What_is_the_time_complexity_of_Multilayer_Perceptron_MLP_and_other_neural_networks


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time


# In[127]:


class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=500, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)                       # +1 bias için
        # print(type(self.weights), self.weights)
        
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0            
        return activation

    def train(self, training_inputs, labels):
        sayac_1 = 0
        
        basla = time.perf_counter()
        
        for i in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)    # bias
                sayac_1 += 1
            #if(i == 4):
                # print(self.weights)
                # print("---")
        
        bitir = time.perf_counter()
        print(f"Süre : {bitir - basla:0.4f} saniye")                

        # print(self.weights)

        #print("Dongu sayisi", sayac_1)
        #plt.plot(self.weights[1:])


# In[128]:


training_inputs= []
labels = []

training_inputs.append(np.array([1,1]))
# training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([0,0]))

labels = np.array([1,0,0,0]) # or -> 1,1,1,0   and -> 1,0,0,0
# xor yapılamaz  0,1,1,0
perceptron = Perceptron(2)
perceptron.train(training_inputs, labels)


# In[129]:


inputs = np.array([1, 1])
perceptron.predict(inputs)


# In[130]:


inputs = np.array([1, 0])
perceptron.predict(inputs)


# In[131]:


inputs = np.array([0, 1])
perceptron.predict(inputs)


# In[132]:


inputs = np.array([0, 0])
perceptron.predict(inputs)


# In[133]:


inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])


# In[134]:


# activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return x*(1-x)


# In[135]:


class MLP(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.l = len(inputs)       # number of inputs
        self.li = len(inputs[0])   # length of each input
        
        #self.wi = np.random.random((self.li, self.l)) # first  layer weight
        #self.wh = np.random.random((self.l, 1))       # hidden layer weight
        
        self.wi = np.array([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
        self.wh = np.array([[0.1], [0.1], [0.1], [0.1]])
        self.wi = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        self.wh = np.array([[1.0], [1.0], [1.0], [1.0]])
        #print("l", type(self.l), "li", self.li)
        #print("wh", self.wh, "wi", self.wi) 
    
    def predict(self, inp):
        s1=sigmoid(np.dot(inp, self.wi))
        s2=sigmoid(np.dot(s1, self.wh))
        k1 = sigmoid(np.dot([1, 0], self.wi))
        k2 = sigmoid(np.dot(k1, self.wh))
        #plt.plot(s1)
        #print("s1", s1)
        #plt.plot(s2)
        #print("s2", s2)
        #plt.show()
        #print(s2)
        if s2[0] == k2[0]:
            return 1
        else:
            return 0
        
        return s2

    def train(self, inputs,outputs, it = 500):
        basla = time.perf_counter()

        l1=0
        l2=0
        for _ in range(it):
            l0=inputs                        # input  layer
            l1=sigmoid(np.dot(l0, self.wi))  # first  layer
            l2=sigmoid(np.dot(l1, self.wh))  # hidden layer

            l2_err=outputs - l2
            l2_delta = np.multiply(l2_err, sigmoid_der(l2))

            l1_err=np.dot(l2_delta, self.wh.T)
            l1_delta = np.multiply(l1_err, sigmoid_der(l1))

            self.wh+=np.dot(l1.T, l2_delta)
            self.wi+=np.dot(l0.T, l1_delta)
            
        bitir = time.perf_counter()
        print(f"Süre : {bitir - basla:0.4f} saniye")
       #plt.plot(l1)
       #plt.plot(l2)
       #plt.plot([0.0, 0.0])
       #plt.plot([0.0, 1.0])
       #plt.plot([1.0, 0.0])
       #plt.plot([1.0, 1.0])
       #plt.show()


# In[136]:


inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
outputs = np.array([[0],   [1],   [1],   [0]])


# In[137]:


n=MLP(inputs)
# firs_predict = n.predict(inputs)
# print(firs_predict)
n.train(inputs, outputs)


# In[138]:


print(n.predict([0,0]))
print(n.predict([0,1]))
print(n.predict([1,0]))
print(n.predict([1,1]))


# In[1]:


import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report, confusion_matrix
import time


# In[2]:


data_path = "data/"
mnist = fetch_mldata('MNIST original', data_home=data_path)
X, y = mnist["data"], mnist["target"]


# In[3]:


X = X / 255
y_new = np.zeros(y.shape)
y_new[np.where(y == 0.0)[0]] = 1
y = y_new


# In[4]:


m = 60000
m_test = X.shape[0] - m
print(X.shape, y.shape)
X_train, X_test = X[:m].T, X[m:].T
y_train, y_test = y[:m].reshape(1,m), y[m:].reshape(1,m_test)

# X_train.shape, X_test.shape ,y_train.shape, y_test.shape


# In[5]:


np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]
# X_train.shape, X_test.shape ,y_train.shape, y_test.shape


# In[6]:


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def compute_loss(Y, Y_hat):
    m = Y.shape[1]
    L = -(1./m) * ( np.sum( np.multiply(np.log(Y_hat),Y) ) + np.sum( np.multiply(np.log(1-Y_hat),(1-Y)) ) )
    return L


# In[145]:


basla = time.perf_counter()

learning_rate = 1
X = X_train
Y = y_train

n_x = X.shape[0]
m = X.shape[1]

W = np.random.randn(n_x, 1) * 0.01
b = np.zeros((1, 1))

for i in range(500): # 20
    Z = np.matmul(W.T, X) + b
    A = sigmoid(Z)

    cost = compute_loss(Y, A)

    dW = (1/m) * np.matmul(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y, axis=1, keepdims=True)

    W = W - learning_rate * dW
    b = b - learning_rate * db

    if (i % 100 == 0):
        print("Epoch", i, "cost: ", cost)

print("Final cost:", cost)

bitir = time.perf_counter()
print(f"Süre : {bitir - basla:0.4f} saniye")


# In[146]:


Z = np.matmul(W.T, X_test) + b
A = sigmoid(Z)

predictions = (A>.5)[0,:]
labels = (y_test == 1)[0,:]
print(predictions)
print(labels)
print(confusion_matrix(predictions, labels))


# In[147]:


# 3. One Hidden Layer
basla = time.perf_counter()

X = X_train
Y = y_train

n_x = X.shape[0]
n_h = 64
learning_rate = 1

W1 = np.random.randn(n_h, n_x)
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(1, n_h)
b2 = np.zeros((1, 1))

for i in range(500):

    Z1 = np.matmul(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = sigmoid(Z2)

    cost = compute_loss(Y, A2)

    dZ2 = A2-Y
    dW2 = (1./m) * np.matmul(dZ2, A1.T)
    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(W2.T, dZ2)
    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = (1./m) * np.matmul(dZ1, X.T)
    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    if i % 100 == 0:
        print("Epoch", i, "cost: ", cost)

print("Final cost:", cost)

bitir = time.perf_counter()
print(f"Süre : {bitir - basla:0.4f} saniye")


# In[7]:


Z1 = np.matmul(W1, X_test) + b1
A1 = sigmoid(Z1)
Z2 = np.matmul(W2, A1) + b2
A2 = sigmoid(Z2)

predictions = (A2>.5)[0,:]
labels = (y_test == 1)[0,:]

# print(confusion_matrix(predictions, labels))
# print(classification_report(predictions, labels))


# In[13]:


# 4. Upgrading to Multiclass

mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]

X = X / 255
digits = 10
examples = y.shape[0]

y = y.reshape(1, examples)

Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)


# In[14]:


m = 60000
m_test = X.shape[0] - m

X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:,:m], Y_new[:,m:]

shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]


# In[15]:


#i = 12
#plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
#plt.axis("off")
#plt.show()
#Y_train[:,i]


# In[16]:


# 4.3 Cost Function
def compute_multiclass_loss(Y, Y_hat):

    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1/m) * L_sum

    return L


# In[17]:


# 4.5 Build & Train
basla = time.perf_counter()

n_x = X_train.shape[0]
n_h = 64
learning_rate = 1

W1 = np.random.randn(n_h, n_x)
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(digits, n_h)
b2 = np.zeros((digits, 1))

X = X_train
Y = Y_train

for i in range(500):

    Z1 = np.matmul(W1,X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2,A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

    cost = compute_multiclass_loss(Y, A2)

    dZ2 = A2-Y
    dW2 = (1./m) * np.matmul(dZ2, A1.T)
    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(W2.T, dZ2)
    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = (1./m) * np.matmul(dZ1, X.T)
    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    if (i % 100 == 0):
        print("Epoch", i, "cost: ", cost)

print("Final cost:", cost)

bitir = time.perf_counter()
print(f"Süre : {bitir - basla:0.4f} saniye")


# In[ ]: