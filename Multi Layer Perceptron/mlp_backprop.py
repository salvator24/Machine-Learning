import numpy as np 
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

X = load_iris().data
Y1 = load_iris().target

Y = LabelBinarizer().fit_transform(Y1)

in_dim = 4
hid_dim_1 = 3
hid_dim_2 = 3
out_dim = 3

alpha = 0.01
num_itr = 100000

w0 = np.random.normal(0., 0.25, (in_dim, hid_dim_1))
w1 = np.random.normal(0., 0.25, (hid_dim_1, hid_dim_2))
w2 = np.random.normal(0., 0.25, (hid_dim_2, out_dim))

b0 = np.random.normal(0., 0.25, (1, hid_dim_1))
b1 = np.random.normal(0., 0.25, (1, hid_dim_2))
b2 = np.random.normal(0., 0.25, (1, out_dim))

def sigmoid(x, deriv = False):
	if deriv:
		return sigmoid(x) * (1 - sigmoid(x))
	return 1 / (1 + np.exp(-x))



for i in range(num_itr):
	a1 = sigmoid(np.dot(X, w0) + b0)
	a2 = sigmoid(np.dot(a1, w1) + b1)
	a3 = sigmoid(np.dot(a2, w2) + b2)
	
	loss = -np.sum((Y * np.log(a3)) + ((1 - Y) * np.log(1 - a3))) / Y.shape[0]
	
	dw2 = a3 - Y
	dw1 = np.dot(dw2, w2.T) * sigmoid(np.dot(a1, w1), deriv=True)
	dw0 = np.dot(dw1, w1.T) * sigmoid(np.dot(X, w0), deriv=True)

	db2 = a3 - Y
	db1 = np.dot(dw2, w2.T) * sigmoid(np.dot(a1, w1), deriv=True)
	db0 = np.dot(dw1, w1.T) * sigmoid(np.dot(X, w0), deriv=True)

	w2 -= (alpha * np.dot(a2.T, dw2)) / Y.shape[0]
	w1 -= (alpha * np.dot(a1.T, dw1)) / Y.shape[0]
	w0 -= (alpha * np.dot(X.T, dw0)) / Y.shape[0]
	
	b2 -= (alpha * np.dot(np.ones((X.shape[0], 1)).T, db2))/ Y.shape[0]
	b1 -= (alpha * np.dot(np.ones((X.shape[0], 1)).T, db1))/ Y.shape[0]
	b0 -= (alpha * np.dot(np.ones((X.shape[0], 1)).T, db0))/ Y.shape[0]

	if i%10000 == 0:
		print loss

print accuracy_score(Y1, np.argmax(a3, axis = 1))


'''
X = X.T 
Y = Y.reshape(1,-1)

for i in range(num_itr):
	a1 = sigmoid(np.dot(w0.T, X))
	a2 = sigmoid(np.dot(w1.T, a1))
	loss = -np.sum((Y * np.log(a2)) + ((1 - Y) * np.log(1 - a2))) / Y.shape[1]
	dw1 = a2 - Y
	dw0 = np.dot(w1, dw1) * sigmoid(np.dot(w0.T, X), deriv=True)
	w1 -= (alpha * np.dot(a1, dw1.T)) / Y.shape[1]
	w0 -= (alpha * np.dot(X, dw0.T)) / Y.shape[1]
	if i%100 == 0:
		print loss

print accuracy_score(Y, np.round(a2))
'''