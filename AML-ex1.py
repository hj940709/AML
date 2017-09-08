import numpy as np
from matplotlib import pyplot as plt
cd "E:\document\AML\dataset"

X = np.loadtxt(open("./ex_1_data.csv","rb"),delimiter=",",skiprows=0).T

np.shape(X)

eigvalue, eigvector = np.linalg.eig(np.cov(X))
eigvalue[0]*eigvector[0:,]==np.cov(X).dot(eigvector[0,:].T)
np.cov(X).dot(eigvector[:,0]) - eigvalue[0] * eigvector[:,0]
#Since eigen values happens to be listed in descending order,
#I did not need to sort them any more.
#eigvalue[np.argsort(-eigvalue)]
#eigvector[:,np.argsort(-eigvalue)]
W = np.mat(eigvector[:,0:2])
Z = W.T.dot(X)

plt.title("Projected Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(Z.T[:,0],Z.T[:,1],"bo")


def reconstruction_error(X,L):
    W = np.mat(eigvector[:,0:L])
    Z = W.T.dot(X)
    X_hat = W.dot(Z)
    temp = np.subtract(X,X_hat)
    temp = np.multiply(temp,temp)
    return temp.sum()
    
x_axis = np.array([2,3,4,5])
y_axis = np.array([]) 
for i in range(0,x_axis.size):
    y_axis = np.append(y_axis,reconstruction_error(X,x_axis[i]))
plt.title("Reconstruction Error vs Dimension")
plt.xlabel("L")
plt.ylabel("Squared Error")
plt.plot(x_axis,y_axis)

