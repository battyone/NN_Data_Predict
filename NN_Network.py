#import libraries
import numpy as np
import time
import matplotlib.pyplot as plt

#hidden layer is set here
n_hidden = 3 # set your own value
#input value for x
n_in = 5 # set your own value
#output value for y
n_out = 5 #set your own value
#number of samples to test
n_sample = 30

#learning rate can be set to anything
learning_rate = 0.0003
momentum = 0.003000

#Sets the range of values for random
np.random.seed(0)

#transfer functions
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
	out = sigmoid(x)
	return out * (1.0 - out)

def tanh_prime(x):
	return 1.0 - np.tanh(x)**2

#input data, transpose, layer 1, layer 2, biases
def train(x, t, V, W, bv, bw):
	
	#forward -- matrix multiply + biases
	A = np.dot(x, V) + bv
	#Z = np.tanh(A)
	Z = sigmoid(A)
	B = np.dot(Z, W) + bw
	Y = sigmoid(B)
	
	#backpropagation
	Ew = Y - t
	#Ev = tanh_prime(A) * np.dot(W, Ew)
	Ev = sigmoid_prime(A) * np.dot(W, Ew)
	
	#predict our loss
	dW = np.outer(Z, Ew)
	dV = np.outer(x, Ev)
	
	#cross entroppy
	#loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1 - Y))
	loss = 0.5*sum((Y-T)**2)
	
	return loss, (dV, dW, Ev, Ew)
	
def predict(x, V, W, bv, bw):
	A = np.dot(x, V) + bv
	B = np.dot(sigmoid(A), W) + bw
	return (sigmoid(B))
	
#create layers
V = np.random.randn(n_in, n_hidden)
W = np.random.randn(n_hidden, n_out)
bv = np.zeros(n_hidden)
bw = np.zeros(n_out)
params = [V, W, bv, bw]

#generate our data
filename = 'input.csv'
data = np.loadtxt(filename, delimiter=',')
X = data[:63] / 100
T = data[64:129] / 100

#training time
#any number can be set for iteration
for epoch in range(6699):#test with your own iterations
	err = []
	err_mean = []
	upd = [0] * len(params)
	#test the complexity speed
	t0 = time.clock()
	#for each data point, update our weights
	for i in range(X.shape[0]):
		loss, grad = train(X[i], T[i], * params)
		#update loss
		for j in range(len(params)):
			params[j] -= upd[j]
			
		for j in range(len(params)):
			upd[j] = learning_rate * grad[j] + momentum * upd[j]
			
		err.append(loss)
		err_mean.append(np.mean(loss))
	#prints the error, iteration and time it takes for iter	
	print('Epoch: %d, Loss: %.8f, Time: %.4fs'%(epoch, np.mean(err), time.clock()-t0))
	
# try to predict something
x = data[167:168] / 100
print('Lotto prediction')
print(x)
print(predict(x, * params) * 100)
#plt.plot(err)
#plt.show()
