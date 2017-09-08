import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.misc import logsumexp
from matplotlib.backends.backend_pdf import PdfPages
import mnist_load_show as mnist


def gt_stump_v2(data, one_y):
    n = data.shape[0]
    N = data.shape[1]
    ri = np.random.randint(N)
    thresh = 1e-10
    tot_counts = sum(one_y)
    tot_prob = (tot_counts / float(n)) + thresh
    E_tot= -1*sum(tot_prob*np.log(tot_prob))
    res =  []
    temp_vec = np.unique(data[:,ri])
    temp_vec.sort()
    for i in temp_vec:
        lookup = data[:,ri] <= i
        not_lookup = np.logical_not(lookup)
        cl_counts = sum(one_y[lookup,:])
        cr_counts = sum(one_y[not_lookup,:])
        if type(cr_counts).__module__ != 'numpy':
            cr_counts = [0]
        if type(cl_counts).__module__ != 'numpy':
            cl_counts = [0]
        if sum(cl_counts) == 0:
            tot_l = 0
            probs_left = np.ones(10)*0.1
            E_left= -1*sum(probs_left*np.log(probs_left))
        else:
            tot_l = sum(cl_counts)
            probs_left = (cl_counts / float(tot_l)) + thresh
            E_left= -1*sum(probs_left*np.log(probs_left))
        if sum(cr_counts) == 0:
            tot_r = 0
            probs_right = np.ones(10)*0.1
            E_right= -1*sum(probs_right*np.log(probs_right))
        else:
            tot_r = sum(cr_counts)
            probs_right = (cr_counts / float(tot_r)) + thresh
            E_right= -1*sum(probs_right*np.log(probs_right))
        IG = E_tot - (tot_l/(1.*n))*E_left - (tot_r/(1.*n))*E_right
        res.append([IG, ri, i, probs_left, probs_right])
    return res


def get_score(M, N, stumps, X, y):
    yhat = np.zeros(N)
    for i in range(N):
        pY = np.zeros(10)
        for l in range(M):
            stump1 = stumps[l]
            if X[i,stump1[1]] < stump1[0]:
                pY = pY + stump1[3]
            else:
                pY = pY + stump1[4]
        pY = pY / float(M)
        y_pred = np.argmax(pY)
        yhat[i] = y_pred
    return sum(yhat == y)/float(N), yhat
    

if __name__ == '__main__':
    Nv = eval(sys.argv[2])
    M = eval(sys.argv[1])
    X1 , y1 = mnist.read_mnist_training_data(10000)
    X = X1[0:5000,:]
    y = y1[0:5000]
    Xv = X1[5000:,:]
    yv = y1[5000:]
    N = X.shape[0]
    D = X.shape[1]
    #init one-hot encoding
    one_y = np.zeros((N,10))
    sample_size = 100
    one_y[np.arange(N), y] = 1
    stumps_M = []
    for k in range(M):
        sample = np.random.choice(len(X),sample_size, replace=False)
        r1 = gt_stump_v2(X[sample,:], one_y[sample,:])
        tempvec = [(r1[i][0], i) for i in range(len(r1))]
        tempvec.sort()
        s1 = tempvec[-1]
        stump = r1[s1[1]]
        stumps_M.append(stump)
    # validation
    yhat = np.zeros(Nv) 
    M_vec = np.arange(10, M + 10, 10)
    M_vec = np.array([1, 10,100,300,600,900,1000])
    valid_errors = M_vec*0.
    train_errors = M_vec*0.
    for m1 in range(len(M_vec)):
        M = M_vec[m1]
        valid_errors[m1] = get_score(M, Nv, stumps_M, Xv[:Nv,:], yv[:Nv])
        train_errors[m1] = get_score(M, N, stumps_M, X, y)
        print m1+1, valid_errors[m1], train_errors[m1]
    plt.plot(M_vec, 1-valid_errors,"b-",label = 'Validation loss')
    plt.plot(M_vec, 1-train_errors,"r-",label = 'Training loss' )
    plt.xlabel("M")
    plt.ylabel("Classification error")
    plt.legend(loc=1, fontsize = 'x-large')
    plt.savefig("p1.pdf")