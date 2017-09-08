import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
cd "E:\document\AML\dataset"

data = np.mat(np.loadtxt(
	open(".//exercise_3_2.csv","rb"),
	delimiter=",",skiprows=0))

def get_adjmatrices(data,t=1,e=0.5,A=8):
    W = np.matrix(np.zeros((data.shape[0],data.shape[0])))
    for i in range(data.shape[0]):
        if(t==1):
            for j in range(data.shape[0]):
                d = np.sqrt(np.square(data[i,:]-data[j,:]).sum())
                if (d<=e): #e=0.5
                    W[i,j]=1
        elif(t==2):
            temp = np.array([])
            for j in range(i+1,data.shape[0]):
                temp = np.append(temp,
                    np.sqrt(np.square(data[i,:]-data[j,:]).sum()))
            closest = i+1+temp.argsort()[:A] #A=8
            for neighbour in closest:
                W[i,neighbour] = 1
                W[neighbour,i] = 1
    return W

def get_diagmatrix(W):
    D = np.matrix(np.zeros((W.shape[0],W.shape[0])))
    s = np.squeeze(np.array(W.sum(0)))
    for i in range(W.shape[0]):
        D[i,i] = s[i]
    return D

def get_eigen(L):
    value,vector = np.linalg.eig(L)
    return (value[value.argsort()],vector[:,value.argsort()])

e=0.5
A=8
M=4

W = [get_adjmatrices(data,t=1,e=e,A=A),get_adjmatrices(data,t=2,e=e,A=A)]
L = [get_diagmatrix(W[0])-W[0],get_diagmatrix(W[1])-W[1]]
eig = [get_eigen(L[0]),get_eigen(L[1])]
#eig[0][0][0] the 'first' 'eigenvalue' of 'first' type of W
'''
with plt.style.context("default"):
    plt.plot(eig[0][0],"b-")
    plt.plot(eig[1][0],"r-")
    plt.legend(["Distence < e","A nearest neighbours"])
    plt.title("Eigenvalue(e="+str(e)+",A="+str(A)+")")
    plt.show()
'''
with plt.style.context("default"):
    plt.plot(eig[0][1][:,:M],"-")
    plt.legend(["Eigenvector"+str(i+1) for i in range(M)])
    plt.title("Eigenvector(e="+str(e)+",A="+str(A)+") of W type 1")
    plt.show()

with plt.style.context("default"):
    plt.plot(eig[1][1][:,:M],"-")
    plt.legend(["Eigenvector"+str(i+1) for i in range(M)])
    plt.title("Eigenvector(e="+str(e)+",A="+str(A)+") of W type 2")
    plt.show()

with plt.style.context("default"):
    fig, axes = plt.subplots(nrows=M, ncols=M, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i in range(M):
        for j in range(M):
            if(i!=j):
                xp = np.squeeze(np.asarray(eig[0][1][:,i].T))
                yp = np.squeeze(np.asarray(eig[0][1][:,j].T))
                axes[i][j].scatter(x=xp,y=yp,marker="o")
    plt.show()

with plt.style.context("default"):
    fig, axes = plt.subplots(nrows=M, ncols=M, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i in range(M):
        for j in range(M):
            if(i!=j):
                xp = np.squeeze(np.asarray(eig[1][1][:,i].T))
                yp = np.squeeze(np.asarray(eig[1][1][:,j].T))
                axes[i][j].scatter(x=xp,y=yp,marker="o")
    plt.show()

with plt.style.context("default"):
    kms = KMeans(n_clusters=2)
    kms.fit(data)
    kms.labels_
    color=["b","r"]
    colors = [color[i] for i in kms.labels_]
    xp = np.squeeze(np.asarray(data[:,0].T))
    yp = np.squeeze(np.asarray(data[:,1].T))
    plt.scatter(x=xp,y=yp,c=colors,marker="o")
    plt.title("Standard K-means Clustering")
    plt.show()

with plt.style.context("default"):
    cluster=2
    kms = KMeans(n_clusters=cluster)
    kms.fit(eig[0][1][:,:M])
    kms.labels_
    color=["b","r"]
    colors = [color[i] for i in kms.labels_]
    xp = np.squeeze(np.asarray(data[:,0].T))
    yp = np.squeeze(np.asarray(data[:,1].T))
    plt.scatter(x=xp,y=yp,c=colors,marker="o")
    plt.title("Clustering (e="+str(e)+",A="+str(A)+" with W type 1)")
    plt.show()

with plt.style.context("default"):
    cluster=2
    kms = KMeans(n_clusters=cluster)
    kms.fit(eig[1][1][:,:M])
    kms.labels_
    color=["b","r"]
    colors = [color[i] for i in kms.labels_]
    xp = np.squeeze(np.asarray(data[:,0].T))
    yp = np.squeeze(np.asarray(data[:,1].T))
    plt.scatter(x=xp,y=yp,c=colors,marker="o")
    plt.title("Clustering (e="+str(e)+",A="+str(A)+" with W type 2)")
    plt.show()

