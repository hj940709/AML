import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import sys
from scipy.misc import logsumexp
from matplotlib.backends.backend_pdf import PdfPages
import mnist_load_show as mnist



def get_prob_mat(data, sigma):
    N = data.shape[0]
    sum_data = np.sum(np.square(data), 1)
    q1 = 1. * np.add(np.add(-2 * np.dot(data, data.T), sum_data).T, sum_data)
    q1 = q1 / (2.*sigma**2)
    q1[range(N), range(N)] = np.inf
    q1_mins = np.min(q1,1)
    new_q = ((-1.*q1).T + q1_mins).T
    q11 = np.exp(new_q)
    denom_q = (np.sum(q11, 1) - q11.T).T
    q2 = q11 / denom_q
    q2 = np.maximum(q2, 1e-10)
    return q2



def get_grad(z, p, q):
    N = z.shape[0]
    D = z.shape[1]
    pq = p-q
    grad_z = np.zeros((N, D))
    for i in range(N):
        grad_z[i,:] =  np.sum(((z[i,:]- z).T*(pq[i,:] + pq[:,i])).T, 0)
    return 2*grad_z



def grad_descent(q, p, z, alpha, sigma, y, iplot):
    old_cost = np.sum(p * np.log(p / q))
    print old_cost
    N =  z.shape[0]
    count = 0
    if iplot:
        fig,axes=plt.subplots(figsize=(6,6), nrows=1, ncols=1,squeeze=False)
        plt.ion()
        g1 = axes[0][0].scatter(z[:,0],z[:,1], c = y)
        plt.show()
    while count < 1000:
        z = z - alpha * get_grad(z, p, q)
        # to add jitter every 10th iteration uncomment the if block.
        #if count % 10 == 0:
        #    z = z + np.random.normal(0,0.003, (N,2))
        q = get_prob_mat(z, sigma)
        cost = np.sum(p * np.log(p / q))
        print cost, count
        count += 1
        if iplot:
            g1.set_offsets(z[:,:2])
            plt.pause(0.0001)
    plt.close()
    return z





if __name__ == '__main__':
    # PCA of 2 dimensions
    X , y = mnist.read_mnist_training_data(1000)
    # low dimensions
    N = X.shape[0]
    D = X.shape[1]
    pca = PCA(n_components=2)
    pca.fit(X)
    sigmaq = 1.
    sigmap = 10000.
    alpha = 0.001
    Znew = pca.transform(X)
    Znew = Znew / np.std(Znew, 0)
    q_ij = get_prob_mat(Znew, sigmaq)
    p_ij = get_prob_mat(X, sigmap)
    final_z = grad_descent(q_ij, p_ij, Znew, alpha, sigmaq, y, True)
    fig1,axes=plt.subplots(figsize=(12,6), nrows=1, ncols=2,squeeze=False)
    axes[0][0].scatter(final_z[:,0],final_z[:,1], c = y)
    axes[0][0].set_title("SNE")
    axes[0][1].scatter(Znew[:,0],Znew[:,1], c = y)
    axes[0][1].set_title("PCA")
    with PdfPages('sne_final1.pdf') as pdf:
        pdf.savefig(fig1)