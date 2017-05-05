#HW0005

import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

def D_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.linspace(-10,10,100)
y = sigmoid(x)
plt.plot(x,y)
plt.show()

x = np.linspace(-10,10,100)
y = D_sigmoid(x)
plt.plot(x,y)
plt.show()

def tanh(x):
    return (np.exp(x/2)-np.exp(-x/2))/(np.exp(x/2)+np.exp(-x/2))

def D_tanh(x):
    return 1-tanh(x)**2

x = np.linspace(-10,10,100)
y = []
for dig in x:
    y.append(tanh(dig))
plt.plot(x,np.reshape(y,(100,)))
plt.show()

def ReLU(x):
    return max(0,x)

def D_ReLU(x):
    return 1 if x >= 0 else 0

x = np.linspace(-10,10,100)
y = []
for dig in x:
    y.append(ReLU(dig))
plt.plot(x,np.reshape(y,(100,)))
plt.show()



