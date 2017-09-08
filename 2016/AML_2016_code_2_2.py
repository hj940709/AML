import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

def compute_cost(theta,X,y, C):
    preds = np.dot(X, theta)
    J = float(np.mean((np.dot(X, theta) - y) ** 2)) + (C * float(np.dot(theta, theta)))
    return J


def get_grad(theta, X, y, alpha, C):
    N = X.shape[0]
    K = len(theta)
    return ( (2./ len(y))*np.dot(X.T, (np.dot(X, theta) - y))) + (2 * C * theta)


# This function lets you split data matrix X into training and validation sets.
# If K = 2, we split the data into two parts(50% split) and if K = 10, we have data divided
# into 10 parts, argument i selects a specific part as training set and swap=True
# lets you choose 10/90 split over 90/10 split for K = 10. 
# In this exercise we use i = 0, since we are not doing cross-validation.
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

# grad_theta, calls get_grad and updates the theta vector with a step-size alpha
def grad_theta(theta, y, X, alpha, C):
    dgrad = get_grad(theta, X, y, alpha, C)
    theta = theta - (alpha * dgrad)
    return [theta,dgrad]

# Empirical risk minimization
def sol2a(theta, trainY, trainX, alpha, tol, C):
    print "sol 2a: "
    old_theta = np.zeros(len(theta))
    k = 0
    allcosts = []
    old_cost = compute_cost(theta,trainX,trainY, C)
    noChange = True
    logcosts = []
    while noChange:
        theta,grad = grad_theta(theta, trainY, trainX, alpha, C)
        cost = compute_cost(theta,trainX,trainY, C)
        allcosts.append(cost)
        k = k + 1
        new_val = np.linalg.norm(grad)
        logcosts.append(np.log(abs(cost - old_cost)))
        if new_val < 1e-3:
            break
        old_theta = theta
        old_cost = cost
    return [theta, allcosts, k, logcosts]

#Early-stopping
def sol2b(theta, trainY, trainX, alpha, tol, C, validX, validY, stop_b):
    oldVcost = compute_cost(theta,validX,validY, 0.)
    old_cost = compute_cost(theta,trainX,trainY, C)
    old_theta = np.zeros(len(theta))
    k = 0
    allcosts = []
    allvalid = []
    logcosts = []
    G = np.zeros(len(theta))
    noChange = True
    while noChange:
        theta, grad = grad_theta(theta, trainY, trainX, alpha, C)
        cost = compute_cost(theta,trainX,trainY, C)
        validcost = compute_cost(theta,validX,validY, 0.)
        allcosts.append(cost)
        allvalid.append(validcost)
        k = k + 1
        if stop_b:
            if abs(validcost - oldVcost)  < 1e-5:
                print abs(oldVcost - validcost), cost
                break
        else:
            new_val = np.linalg.norm(grad)
            if new_val < 1e-3:
                print abs(old_cost - cost)
                break
        old_theta = theta
        oldVcost = validcost
        old_cost = cost
    print k
    return [theta, allcosts,allvalid, k, logcosts]
# Regularized risk minimization
def sol2c(N, D, K, tol, alpha, train, algo, swap, allC):
    train_error_arr = []
    valid_error_arr = []
    theta_c = []
    for C in allC:
        train_error = 0
        valid_error = 0
        for i in range(1):
            split_data = split_kfold(train[0], train[1], K, 0, swap)
            trainX, trainY, validX, validY = split_data
            newtheta = np.zeros(D)
            res3= sol2b(newtheta, trainY, trainX, alpha, tol, C, validX, validY, False)
            theta_c.append(res3[0])
            train_error = train_error + compute_cost(res3[0],trainX,trainY, 0.)
            valid_error = valid_error + compute_cost(res3[0],validX,validY, 0.)
        train_error_arr.append(train_error / 1)
        valid_error_arr.append(valid_error / 1)
    return [train_error_arr, valid_error_arr, allC, theta_c]


def solution(algo, train,N, D,  alpha, K, tol, swap, allC):
    train = [train_data[:,:-1], train_data[:,-1]]
    split_data = split_kfold(train[0], train[1], K, 0, swap)
    trainX, trainY, validX, validY = split_data
    newtheta = np.zeros(D)
    # a) Empirical risk minimization
    res1 = sol2a(newtheta, trainY, trainX, alpha, tol, 0.)
    theta_erm = res1[0]
    print res1[2]
    # b) Early-stopping
    res2 = sol2b(newtheta, trainY, trainX, alpha, tol, 0., validX, validY, True)
    theta_early = res2[0]
    print res2[3]
    # c) Regularized risk minimization
    res3 = sol2c(N, D, K, tol, alpha, train, algo, swap, allC)
    choice = np.argmin(res3[1])
    print choice, res3[2][choice]
    theta_rrm = res3[3][choice]
    fig3,axes=plt.subplots(figsize=(12,6), nrows=1, ncols=2,squeeze=False)
    axes[0][0].plot(np.arange(res1[2]),res1[1], "b-")
    axes[0][0].set_xlabel("Number of iterations")
    axes[0][0].set_ylabel("Squared error")
    axes[0][0].set_title("Squared error vs Iterations")
    axes[0][1].plot(np.arange(res1[2]),res1[3], "b-")
    axes[0][1].set_xlabel("Number of Iterations")
    axes[0][1].set_ylabel("Log (SquaredLoss(current) - SquaredLoss(optimal)")
    axes[0][1].set_title("Difference in squared errors vs Iterations")
    with PdfPages('convergence.pdf') as pdf:
        pdf.savefig(fig3)
    return [theta_erm, theta_early, theta_rrm, res3]

if __name__ == '__main__':
    algo = "1"
    fig2,axes=plt.subplots(figsize=(16,6), nrows=1, ncols=3,squeeze=False)
    allC = [0.001, 0.003, 0.01, 0.03, 0.1]
    train_data = np.genfromtxt('exercise_2_2_train.csv', delimiter=",")
    test_data = np.genfromtxt('exercise_2_2_test.csv', delimiter=",")
    N = train_data.shape[0]
    D = train_data.shape[1] -1
    train = [train_data[:,:-1], train_data[:,-1]]
    test_x, test_y = test_data[:,:-1], test_data[:,-1]
    swap = True# if swap is true, valid == N / K or valid == N- (N/K)
    res_theta = solution(algo,train, N, D, 0.5, 10, 0.01, swap, allC)
    costs = [compute_cost(res_theta[i],test_x,test_y, 0) for i in range(len(res_theta)-1)]
    print costs
    res3 = res_theta[-1]
    axes[0][0].semilogx(res3[2], res3[0], "g-")
    axes[0][0].semilogx(res3[2], res3[1], "r-")
    axes[0][0].set_ylabel("Error (Residual Sum of Squares)")
    axes[0][0].legend(['Training', 'Validation'])
    axes[0][0].set_ylabel("Error (Residual Sum of Squares)")
    axes[0][0].set_title("10/90 split")
    swap = True
    res_theta = solution(algo,train, N, D, 0.5, 2, 0.01, swap, allC)
    costs = [compute_cost(res_theta[i],test_x,test_y, 0) for i in range(len(res_theta)-1)]
    print costs
    res3 = res_theta[-1]
    axes[0][1].semilogx(res3[2], res3[0], "g-")
    axes[0][1].semilogx(res3[2], res3[1], "r-")
    axes[0][1].legend(['Training', 'Validation'])
    axes[0][1].set_xlabel("Regularization constant lambda")
    axes[0][1].set_ylabel("Error (Residual Sum of Squares)")
    axes[0][1].set_title("50/50 split")
    swap = False
    res_theta = solution(algo,train, N, D, 0.5, 10, 0.01, swap, allC)
    costs = [compute_cost(res_theta[i],test_x,test_y, 0) for i in range(len(res_theta)-1)]
    print costs
    res3 = res_theta[-1]
    axes[0][2].semilogx(res3[2], res3[0], "g-")
    axes[0][2].semilogx(res3[2], res3[1], "r-")
    axes[0][2].legend(['Training', 'Validation'])
    axes[0][2].set_title("90/10 split")
    with PdfPages('RRM.pdf') as pdf:
        pdf.savefig(fig2)
    