cd "E:\document\AML\dataset\mnist"
import mnist_load_show as mnist
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

N=1000 #5000
x,y = mnist.read_mnist_training_data(N)
'''
def getPCA(x,component):
    #centroid = x.mean(1)
    X = x.T#-centroid
    eigvalue, eigvector = np.linalg.eig(np.cov(X))
    W = np.matrix(eigvector[:,(-eigvalue).argsort()[0:component]])
    Z = W.T.dot(X)
    return Z.T
'''

def getNeighbour(data,sigma):
    N,D=data.shape
    sigma_2 = sigma**2
    d_2 = np.matrix(np.zeros((N,N)))
    for i in range(N):
        d_2[i,:] = np.square(data[i,:]-data).sum(1).T
        d_2[i,i] = np.inf
    d_2=d_2/(2*sigma_2)
    j = np.exp(-d_2)
    s = j.sum(1)
    neighbour = j/(s-j.T)
    neighbour = np.maximum(neighbour,1e-10)
    return neighbour

def getGrad(z, p, q):
    N,D = z.shape
    grad = np.matrix(np.zeros((N,D)))
    p_q = p-q
    p_q_s = p_q+p_q.T
    for i in range(N):
        grad[i,:] = np.multiply(+z[i,:]-z,p_q_s[i,:].reshape((N,1))).sum(0)
    return 2*grad

def gradDescent(z,x,stepsize=0.05):
    N =  z.shape[0]
    pz = z
    p = getNeighbour(x,10000)
    q = getNeighbour(z,1)
    cost = [np.multiply(p,np.log(p/q)).sum()]
    while True:
        print(len(cost),cost[-1])
        qz = pz - stepsize * getGrad(pz, p, q)
        q = getNeighbour(qz,1)
        cost.append(np.multiply(p,np.log(p/q)).sum())
        if (abs(cost[-2]-cost[-1])<1e-3):
           return qz,cost
        else: pz=qz

pca = PCA(n_components=2)
z = pca.fit_transform(x)
z = z/np.std(z, 0)
z_sne,cost = gradDescent(z,x, stepsize=0.05)

with plt.style.context("default"):
    xp = np.squeeze(np.asarray(z[:,0].T))
    yp = np.squeeze(np.asarray(z[:,1].T))
    plt.scatter(x=xp,y=yp,c=y,marker="o")
    plt.title("Initial PCA")
    plt.show()
with plt.style.context("default"):
    xp = np.squeeze(np.asarray(z_sne[:,0].T))
    yp = np.squeeze(np.asarray(z_sne[:,1].T))
    plt.scatter(x=xp,y=yp,c=y,marker="o")
    plt.title("SNE")
    plt.show()
with plt.style.context("default"):
    xp = cost
    plt.plot(xp,"bo-")
    plt.title("Cost VS Iteration")
    plt.legend(["SNE"])
    plt.show()

cd "E:\document\AML\dataset\"
X = np.mat(np.loadtxt(open(".//exercise_4_3.csv","rb"),
	delimiter=",",skiprows=0)).T
	
def visualize(image,l):
    if image.ndim == 1:
        image = np.array([image])
    cols = int(np.ceil(np.sqrt(image.shape[0])))
    img_number = 0
    for row in range(0, cols):
        for col in range(0, cols):
            if img_number > image.shape[0] - 1:
                break
            else:
                ax = plt.subplot2grid((cols, cols), (row, col))
                ax.axes.axes.get_xaxis().set_visible(False)
                ax.axes.axes.get_yaxis().set_visible(False)
                imgplot = ax.imshow(image[img_number].reshape(l, l),\
                    cmap='Greys_r')
                imgplot.set_interpolation('nearest')
                ax.xaxis.set_ticks_position('top')
                ax.yaxis.set_ticks_position('left')
                img_number += 1
    plt.show()

def getNMF(X,K):
    D,N = X.shape #X-WH
    W = np.mat(np.random.rand(D,K))
    H = np.mat(np.random.rand(K,N))
    cost = [np.square(X-W.dot(H)).sum()]
    while True:
        H = np.multiply((W.T.dot(X))/(W.T.dot(W).dot(H)),H)
        W = np.multiply((X.dot(H.T))/(W.dot(H).dot(H.T)),W)
        cost.append(np.square(X-W.dot(H)).sum())
        if (cost[-2]-cost[-1]<1):
            return W,H,cost

xt = X.T
visualize(xt[:1,:],16)

K=8
pca = PCA(n_components=K)
pca.fit(xt)
Z8 = pca.fit_transform(xt)
X_pca8 = pca.inverse_transform(Z8)
cost_pca8 = np.square(xt-X_pca8).sum()
W8,H8,cost_nmf8 = getNMF(X,K)
visualize(pca.components_,16)
visualize(W8.T,16)

K=16
pca = PCA(n_components=K)
pca.fit(xt)
Z16 = pca.fit_transform(xt)
X_pca16 = pca.inverse_transform(Z16)
cost_pca16 = np.square(xt-X_pca16).sum()
W16,H16,cost_nmf16 = getNMF(X,K)
visualize(pca.components_,16)
visualize(W16.T,16)

K=64
pca = PCA(n_components=K)
pca.fit(xt)
Z64 = pca.fit_transform(xt)
X_pca64 = pca.inverse_transform(Z64)
cost_pca64 = np.square(xt-X_pca64).sum()
W64,H64,cost_nmf64 = getNMF(X,K)
visualize(pca.components_,16)
visualize(W64.T,16)