import numpy as np
import time
import matplotlib.pyplot as plt

print("Welcome to the neural network program!!")
print("")

print("Throughout the initiation Ill ask for parameters")
print("")

#variables
#print("Please type number of hidden layers")
#n = input()
n_hidden = 3
#print("Please type input size")
#n = input()
n_in = 5
#print("Please type output size")
#n = input()
n_out = 5
#print("number of samples to play with?")
#n = input()
n_sample = 30

#hyperparameters
#print("Type learning rate value ?")
#n = input()
learning_rate = 0.000003
#print("Type mometum value")
#n = input()
momentum = 0.00003

#non deterministic seeding
np.random.seed(0)

#transfer functions

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
	out = sigmoid(x)
	return out * (1.0 - out)

def tanh_prime(x):
	return 1 - np.tanh(x)**2

#input data, transpose, layer 1, layer 2, biases
def train(x, t, V, W, bv, bw):
	
	#forward -- matrix multiply + biases
	A = np.dot(x, V) + bv
	#Z = np.tanh(A)
	Z = sigmoid(A)
	
	B = np.dot(Z, W) + bw
	Y = sigmoid(B)
	
	#backward
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
V = np.random.randn(n_in, n_hidden) #normal(scale=0.3, size=(n_in, n_hidden))
W = np.random.randn(n_hidden, n_out) #normal(scale=0.3, size=(n_hidden, n_out))

bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V, W, bv, bw]

#generate our data
filename = 'input.csv'
data = np.loadtxt(filename, delimiter=',')
X = data[:63] / 100
#X = np.random.binomial(1, 0.5, (n_sample, n_in))
#T = X ^ 1
T = data[64:129] / 100


#training time
for epoch in range(6699):
	err = []
	err_mean = []
	upd = [0] * len(params)
	
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
		
		
	print('Epoch: %d, Loss: %.8f, Time: %.4fs'%(epoch, np.mean(err), time.clock()-t0))
	
# try to predict something
#x = np.random.binomial(1, 0.5, n_in)
x = data[167:168] / 100
print('Lotto prediction')
print(x)
print(predict(x, * params) * 100)
#plt.plot(err)
#plt.show()