import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA

def visualize(image, l):
    if image.ndim == 1:
        image = np.array([image])
    cols = int(np.ceil(np.sqrt(image.shape[0])))
    img_number = 0
    for row in xrange(0, cols):
        for col in xrange(0, cols):
            if img_number > image.shape[0] - 1:
                break
            else:
                ax = plt.subplot2grid((cols, cols), (row, col))
                ax.axes.axes.get_xaxis().set_visible(False)
                ax.axes.axes.get_yaxis().set_visible(False)
                imgplot = ax.imshow(image[img_number].reshape(l, l), cmap='Greys_r')
                imgplot.set_interpolation('nearest')
                ax.xaxis.set_ticks_position('top')
                ax.yaxis.set_ticks_position('left')
                img_number += 1
    plt.show()


def nmf(X, K, N1, tol, verbose=True):
    N = X.shape[0]
    D = X.shape[1]
    pick_train = np.arange(0,N1)
    pick_valid = np.arange(N1,N)
    W = np.random.rand(N1,K)
    W_valid = np.random.rand(N-N1, K)
    H = np.random.rand(K,D)
    Xtrain = X[pick_train, :]
    Xvalid = X[pick_valid, :]
    old_cost = (np.linalg.norm(Xtrain - np.dot(W,H))**2)/ (1.0*N1)
    notChanged = True
    count = 0
    costs_all = []
    valid_all = []
    while notChanged:
        # H update
        num1 = np.dot(W.T, Xtrain)
        denom1 = np.dot(W.T, np.dot(W,H))
        alphaH = num1 / denom1
        H = H*alphaH
        # W update
        num2 = np.dot(Xtrain,H.T)
        num2_valid = np.dot(Xvalid, H.T)
        denom2 = np.dot(W,np.dot(H,H.T))
        denom2_valid = np.dot(W_valid,np.dot(H,H.T))
        alphaW = num2 / denom2
        alphaW_valid = num2_valid / denom2_valid
        W = W * alphaW
        W_valid = W_valid*alphaW_valid
        cost = (np.linalg.norm(Xtrain - np.dot(W,H))**2)/ (1.0*N1)
        cost_valid = (np.linalg.norm(Xvalid - np.dot(W_valid,H))**2) / (1.0*(N-N1))
        if verbose:
            print str(cost) + "  " + str(cost_valid) + "  " + str(abs(old_cost -cost)) + " "+str(count)
        costs_all.append(cost)
        valid_all.append(cost_valid)
        if  abs(old_cost - cost) < 1e-2:
            notChanged = False
        old_cost = cost
        count += 1
    return W,H, costs_all, valid_all

if __name__ == '__main__':
    X = np.genfromtxt('exercise_4.csv', delimiter=",")
    N = X.shape[0]
    N1 = 3500
    K =  8
    res8 = nmf(X, K, N1, 0.1)
    pca = PCA(n_components=K)
    pca.fit(X)
    X1 = X[np.arange(0,N1),:]
    X2 = X[np.arange(N1,N),:]
    Z1 = pca.transform(X1)
    pca_training_loss = np.linalg.norm(X1 - pca.inverse_transform(Z1))**2
    Z2 = pca.transform(X2)
    pca_validation_loss = np.linalg.norm(X2 - pca.inverse_transform(Z2))**2
    pca_8 = [pca_training_loss, pca_validation_loss]
    
    K =  16
    res16 = nmf(X, K, N1, 0.1)
    pca = PCA(n_components=K)
    pca.fit(X)
    X1 = X[np.arange(0,N1),:]
    X2 = X[np.arange(N1,N),:]
    Z1 = pca.transform(X1)
    pca_training_loss = np.linalg.norm(X1 - pca.inverse_transform(Z1))**2
    Z2 = pca.transform(X2)
    pca_validation_loss = np.linalg.norm(X2 - pca.inverse_transform(Z2))**2
    pca_16 = [pca_training_loss, pca_validation_loss]
    
    K =  64
    res64 = nmf(X, K, N1, 0.1)
    pca = PCA(n_components=K)
    pca.fit(X)
    X1 = X[np.arange(0,N1),:]
    X2 = X[np.arange(N1,N),:]
    Z1 = pca.transform(X1)
    pca_training_loss = np.linalg.norm(X1 - pca.inverse_transform(Z1))**2
    Z2 = pca.transform(X2)
    pca_validation_loss = np.linalg.norm(X2 - pca.inverse_transform(Z2))**2
    pca_64 = [pca_training_loss, pca_validation_loss]
    fig1,axes=plt.subplots(figsize=(6,6), nrows=1, ncols=1,squeeze=False)
    axes[0][0].plot(res8[2],'r-',label = 'Training Loss(K = 8)')
    axes[0][0].set_title("Error vs Iterations")
    axes[0][0].plot(res8[3],'g-',label = 'Validation Loss(K = 8)')
    axes[0][0].plot(res16[2],'r-.',label = 'Training Loss(K = 16)')
    axes[0][0].plot(res16[3],'g-.',label = 'Validation Loss(K = 16)')
    axes[0][0].plot(res64[2],'r--',label = 'Training Loss(K = 64)')
    axes[0][0].plot(res64[3],'g--',label = 'Validation Loss(K = 64)')
    plt.legend(loc=1, fontsize = 'x-small')
    with PdfPages('nmf_iter_all.pdf') as pdf:
        pdf.savefig(fig1)
    plt.close()
    # UNCOMMENT to visualize and plot basis vectors of NMF and PCA.
    #visualize(res[1], 16)
    #with PdfPages('NMF_8_early_log1.pdf') as pdf:
    #    pdf.savefig()
    #visualize(pca.components_, 16)
    #with PdfPages('PCA_1.pdf') as pdf:
    #    pdf.savefig()