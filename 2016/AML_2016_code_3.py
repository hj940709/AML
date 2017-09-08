## Command line usage: python spectral.py e A M

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
from sklearn.cluster import KMeans

# The function get_W returns the new representation matrix (N x M) dimension.
# It also returns the eigenvalues and its eigenvectore followed by the order vector
# ascending order.
def get_W(X, flag, e, A, M):
    N = X.shape[0]
    dist_vec = [np.linalg.norm(X[i,:]-X[j,:]) for i in range(len(X)) for j in range(len(X))]
    dist_mat = np.reshape(dist_vec,(N,N))
    if (flag==1):
        W = 1*(dist_mat <= e)
    else:
        W = np.zeros((N,N))
        for i in range(N):
            tempvec1 = [(dist_mat[i,jj],jj) for jj in range(N)]
            tempvec1.sort()
            dsorder1 = [tempvec1[ii][1] for ii in range(N)]
            pick = dsorder1[1:(1+A)]
            W[i,pick] = 1
            W[pick,i] = 1
    D = np.eye(N) *  np.sum(W,1)
    L = D - W
    evals, evecs = np.linalg.eig(L)
    tempvec = [(evals[i],i) for i in range(N)]
    tempvec.sort()
    dsorder = [tempvec[i][1] for i in range(len(evecs))]
    temp_Y = np.zeros((N,M))
    for jj in range(M):
        temp_Y[:,jj] = evecs[:,dsorder[jj]]
    return [temp_Y, evals, evecs, dsorder]

def new_kmeans(data, num_clus = 2):
    kmes = KMeans(n_clusters=num_clus)
    kmes.fit(data)
    cols = ["blue", "green"]
    labels = kmes.labels_
    cc_cols = [cols[i] for i in labels]
    return cc_cols

def y_plots(Y, flag):
    fname = ['scatter_new_e.pdf', 'scatter_new_knn.pdf']
    add_tag = ['(e = 0.5)', '(K = 8)']
    fig6, axes=plt.subplots(figsize=(12,6), nrows=1, ncols=2,squeeze=False)
    axes[0][0].set_title("plot with evec 1 and evec 2 " + add_tag[flag])
    axes[0][0].scatter(Ye[:,0], Ye[:,1])
    axes[0][1].set_title("plot with evec 2 and evec 3 " + add_tag[flag])
    axes[0][1].scatter(Ye[:,1], Ye[:,2])
    with PdfPages(fname[flag]) as pdf:
        pdf.savefig(fig6)
    plt.close()

if __name__ == '__main__':
    e = eval(sys.argv[1])
    A = eval(sys.argv[2])
    M = eval(sys.argv[3])
    X = np.genfromtxt('exercise_3_3.csv', delimiter=",")
    N = X.shape[0]
    [Ye, evals, evecs, dsorder] = get_W(X, 1, e, M)
    print evecs
    fig1,axes=plt.subplots(figsize=(12,6), nrows=1, ncols=2,squeeze=False)
    axes[0][1].plot(evals[dsorder], "o-", alpha = 0.5)
    axes[0][1].set_title("Eigenvalues(e = 0.5)")
    axes[0][0].set_title("first 4 eigenvectors(e = 0.5)")
    for jj in range(M):
        axes[0][0].plot(evecs[:,dsorder[jj]], "-")
    axes[0][0].legend(['evec 1', 'evec 2', 'evec 3', 'evec 4'])
    with PdfPages('evecs_d.pdf') as pdf:
        pdf.savefig(fig1)
    plt.close()
    ccols = new_kmeans(Ye, 2)
    fig2,axes=plt.subplots(figsize=(6,6), nrows=1, ncols=1,squeeze=False)
    axes[0][0].scatter(X[:,0], X[:,1], color = ccols)
    axes[0][0].set_title("e = 0.5")
    with PdfPages('scatter_d.pdf') as pdf:
        pdf.savefig(fig2)
    plt.close()
    # K-NN
    [Yknn, evals, evecs, dsorder] = get_W(X, 2, A, M)
    fig3,axes=plt.subplots(figsize=(12,6), nrows=1, ncols=2,squeeze=False)
    axes[0][1].plot(evals[dsorder], "o-", alpha = 0.5)
    axes[0][1].set_title("Eigenvalues(K = 8)")
    axes[0][0].set_title("first 4 eigenvectors(K = 8)")
    for jj in range(M):
        axes[0][0].plot(evecs[:,dsorder[jj]], "-")
    axes[0][0].legend(['evec 1', 'evec 2', 'evec 3', 'evec 4'])
    with PdfPages('evecs_knn.pdf') as pdf:
        pdf.savefig(fig3)
    plt.close()
    ccols = new_kmeans(Yknn, 2)
    fig4,axes=plt.subplots(figsize=(6,6), nrows=1, ncols=1,squeeze=False)
    axes[0][0].scatter(X[:,0], X[:,1], color = ccols)
    axes[0][0].set_title("K = 8")
    with PdfPages('scatter_knn.pdf') as pdf:
        pdf.savefig(fig4)
    plt.close()
    ori_kmes = KMeans(n_clusters=2)
    ori_kmes.fit(X)
    cols = ["blue", "green"]
    labels = ori_kmes.labels_
    ori_cols = [cols[i] for i in labels]
    fig5,axes=plt.subplots(figsize=(6,6), nrows=1, ncols=1,squeeze=False)
    axes[0][0].scatter(X[:,0], X[:,1], color = ori_cols)
    axes[0][0].set_title("Kmeans (k = 2)")
    with PdfPages('scatter_kmeans.pdf') as pdf:
        pdf.savefig(fig5)
    plt.close()
    # uncomment for further scatter plots
    #y_plots(Ye,0)
    #y_plots(Yknn,1)