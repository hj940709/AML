import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys



def get_Kinverse(p, gamma, X, y, lam):
    N = X.shape[0]
    K = (np.dot(X,X.T) + gamma)**p
    return np.linalg.inv(K + lam*np.eye(N))

def get_feature(x, X, gamma,p):
    return (np.dot(X, x) + gamma)**p

def squared_error(p, gamma, lam, Xt, yt, Xv, yv):
    sse = 0.
    Kt = get_Kinverse(p, gamma, Xt, yt, lam)
    for i in range(len(yv)):
        kv = get_feature(Xv[i,:],Xt, gamma, p)
        temp_y = np.dot(np.dot(yt, Kt), kv)
        sse = sse + (temp_y - yv[i])**2
    return sse / len(yv)

def get_results(p, gammas, lams, Xt, yt, Xv, yv):
    n1 = len(lams)
    results = np.zeros((n1,n1))
    for i in range(len(lams)):
        for j in range(len(gammas)):
            results[i,j] = squared_error(p, gammas[j], lams[i], Xt, yt, Xv, yv)
    return results
    

if __name__ == '__main__':
    data = np.genfromtxt('data_bonus.csv', delimiter=",")
    y = data[:,-1]
    X = data[:,0:2]
    Xt = X[0:200,:]
    Xv = X[200:400,:]
    yt = y[0:200]
    yv = y[200:400]
    #gammas = np.arange(0.01,10,2)
    #lams = np.arange(1e-3,10,2)
    gammas = np.logspace(-3,2,num=6)
    lams = np.logspace(-3,2,num=6)
    # Plots
    p = 1
    results = get_results(p, gammas, lams, Xt, yt, Xv, yv)
    fig1,axes1=plt.subplots(figsize=(6,6), nrows=1, ncols=1,squeeze=False)
    for k in range(len(results)):
        axes1[0][0].semilogx(gammas, results[k,:], "o-")
    axes1[0][0].set_ylabel("Mean square error")
    axes1[0][0].set_xlabel("gamma")
    axes1[0][0].set_title("p = 1")
    with PdfPages('p1.pdf') as pdf:
            pdf.savefig(fig1)
    p = 2
    results = get_results(p, gammas, lams, Xt, yt, Xv, yv)
    fig2,axes2=plt.subplots(figsize=(6,6), nrows=1, ncols=1,squeeze=False)
    for k in range(len(results)):
        axes2[0][0].semilogx(gammas, results[k,:], "o-")
    axes2[0][0].set_ylabel("Mean square error")
    axes2[0][0].set_xlabel("gamma")
    axes2[0][0].set_title("p = 2")
    with PdfPages('p2.pdf') as pdf:
            pdf.savefig(fig2)
    p = 3
    results = get_results(p, gammas, lams, Xt, yt, Xv, yv)
    fig3,axes3=plt.subplots(figsize=(6,6), nrows=1, ncols=1,squeeze=False)
    for k in range(len(results)):
        axes3[0][0].semilogx(gammas, results[k,:], "o-", label='lambda = %s' % lams[k])
    axes3[0][0].set_ylabel("Error (Residual Sum of Squares)")
    axes3[0][0].set_xlabel("gamma")
    axes3[0][0].set_title("p = 3")
    axes3[0][0].legend()
    with PdfPages('p3.pdf') as pdf:
            pdf.savefig(fig3)
    
