import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

def compute_cost(theta,X,y, C):
    N = 0.5
    preds = np.dot(X, theta)
    J = (np.mean((y - preds)**2))
    return J


def get_grad(theta, X, y, alpha, C):
    N = X.shape[0]
    K = len(theta)
    gradVec = np.zeros(K)
    gradVec = gradVec + np.mean((y - np.dot(X, theta)).reshape(N,1) *  X, 0)
    return 2*gradVec



def split_kfold(X, y, K, i, swap):
    # Given 
    N = len(X)
    S = np.ceil(1.0 * N / K)
    arange = np.arange(N)
    if swap:
        train = np.logical_and(i * S <= arange, arange < (i+1) * S)
        valid = np.logical_not(train)
    else:
        valid = np.logical_and(i * S <= arange, arange < (i+1) * S)
        train = np.logical_not(valid)
    return [X[train],y[train],X[valid],y[valid]]


def grad_theta(theta, y, X, alpha, C):
    N = 0.5
    dgrad = get_grad(theta, X, y, alpha, C)
    theta = theta + alpha * (dgrad - ((1./N)*C * theta))
    return theta

def stochasticgrad_theta(theta, y, X, alpha, C, sample_size):
    N = 0.5
    sample = np.random.choice(len(X),sample_size, replace=False)
    dgrad = get_grad(theta, X[sample], y[sample], alpha, C)
    theta = theta + alpha * (dgrad - ((1./N)*C * theta))
    return theta

def adagrad_theta(theta, y, X, alpha, C, sample_size, scale):
    N = 0.5
    sample = np.random.choice(len(X),sample_size, replace=False)
    dgrad = get_grad(theta, X[sample], y[sample], alpha, C)
    scale = scale + (dgrad)**2
    ada_alpha = alpha / (1 + np.sqrt(scale))
    theta = theta + ada_alpha * dgrad
    return [theta,scale]

def newton_theta(theta, y, X, alpha, C):
    N = len(X)
    hessian = 2 * np.dot(X.T, X) / N
    dgrad = get_grad(theta, X, y, alpha, C)
    hess_inv = np.linalg.inv(hessian)
    newgrad = np.dot(hess_inv, dgrad)
    theta = theta + newgrad
    return theta

def main(theta,X,Y,C, alpha, sample_size, i): 
    k = 0
    old_theta = theta
    allcosts = []
    old_cost = compute_cost(theta,X,Y, C)
    theta_vec = []
    noChange = True
    scale = np.zeros(len(theta))
    while noChange:
        if i == 0:
            theta = grad_theta(old_theta, Y, X, alpha, C)
        elif i == 1:
            theta, scale = adagrad_theta(old_theta, Y, X, 1., C, sample_size, scale)
        elif i == 2:
            theta = stochasticgrad_theta(old_theta, Y, X, alpha, C, sample_size)
        else:
            theta = newton_theta(old_theta, Y, X, alpha, C)
        cost = compute_cost(theta,X,Y, C)
        allcosts.append(cost)
        if i == 2:
            if k > 140:
                break
        if i == 3:
            if k > 10:
                break
        if i == 0:
            if (np.linalg.norm(theta - old_theta) < 1e-4):
                break
        if i == 1:
            if k > 1000:
                break
        old_theta = theta
        theta_vec.append(theta)
        old_cost = cost
        k = k + 1
    return [theta, allcosts, theta_vec, k]

def get_Z(I, J, X, Y, C):
    Z = np.zeros((len(I),len(J)))
    for i in range(len(I)):
        for j in range(len(J)):
            theta = [I[i], J[j]]
            Z[i,j] = compute_cost(theta, X, Y, C)
    return Z
    

if __name__ == '__main__':
    data = np.genfromtxt('exercise_2_3.csv', delimiter=",")
    N = data.shape[0]
    D = data.shape[1] -1
    X, Y = data[:,:-1], data[:,-1]
    theta0 =np.array([-0.3, 1.5])
    alpha = 0.025
    C = 0.
    sample_size = 1
    algo_names = {0:"GD",2:"SGD",1:"SGD with Adagrad",3:"Newton method"}
    # Following plots shows the behaviour of all the algorithms and their
    # convergence on the contour plot 
    fig,axes=plt.subplots(figsize=(8,8), nrows=2, ncols=2,squeeze=False)
    np.random.seed(13)
    for i in range(2):
        for j in range(2):
            res = main(theta0,X,Y,C, alpha, sample_size, (2*i + j))
            theta = res[2]
            theta_new = np.row_stack((theta0,theta))
            K = theta_new.shape[0]
            theta1 = [theta_new[k][0] for k in range(K)]
            theta2 = [theta_new[k][1] for k in range(K)]
            I = np.arange(-2.0, 2, 0.05)
            J = np.arange(-2.0, 2, 0.05)
            Z = get_Z(I, J, X, Y, C)
            levels=np.arange(0.1,1.2,0.3)
            I1, J1 = np.meshgrid(I, J)
            axes[i][j].contour(I1, J1, Z.T, levels)
            axes[i][j].set_title(algo_names[2*i+j])
            axes[i][j].plot(theta1, theta2, "ro-")
            print len(theta1)
    with PdfPages('q3.pdf') as pdf:
        pdf.savefig(fig)
    sample_size = [1, 50]
    algo_names = {0:"SGD with Adagrad using MB=1",1:"SGD with Adagrad using MB=50"}
    # This plot shows how the choice of mini-batch size effects the algorithm like
    # Adagrad with SGD
    fig1,axes=plt.subplots(figsize=(12,6), nrows=1, ncols=2,squeeze=False)
    for k1 in range(len(sample_size)):
        res = main(theta0,X,Y,C, alpha, sample_size[k1], 1)
        theta = res[2]
        theta_new = np.row_stack((theta0,theta))
        K = theta_new.shape[0]
        print i,j
        theta1 = [theta_new[k][0] for k in range(K)]
        theta2 = [theta_new[k][1] for k in range(K)]
        I = np.arange(-2.0, 2, 0.05)
        J = np.arange(-2.0, 2, 0.05)
        Z = get_Z(I, J, X, Y, C)
        levels=np.arange(0.1,1.2,0.3)
        I1, J1 = np.meshgrid(I, J)
        axes[0][k1].contour(I1, J1, Z.T, levels)
        axes[0][k1].set_title(algo_names[k1])
        axes[0][k1].plot(theta1, theta2, "ro-")
        print len(theta1)
    with PdfPages('q4.pdf') as pdf:
        pdf.savefig(fig1)