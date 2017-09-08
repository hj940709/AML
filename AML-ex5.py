cd "E:\document\AML\dataset\mnist"
#Change working directory
import mnist_load_show as mnist
import numpy as np
from matplotlib import pyplot as plt

x,y = mnist.read_mnist_training_data(10000)
trainN = 5000
train = x[:trainN,:]
tlabel = y[:trainN]
valid = x[trainN:,:]
vlabel = y[trainN:]

def getDecisionTree(data,label,full_label_list,simplified_feature_list):
    def getEntropy(label):
        c = np.unique(label)
        result = 0
        for i in c:
            p = (label==i).sum()/label.size
            result = result + p*np.log(p)
        return -result

    def getInformationGain(label,feature,threshold):
        L_label = label[data[:,feature]<=threshold]
        L = L_label.size/label.size*getEntropy(L_label)
        R_label = label[data[:,feature]>threshold]
        R = 0
        if(R_label.size!=0):
            R = R_label.size/label.size*getEntropy(R_label)
        return getEntropy(label)-L-R

    def classifier(data,label,full_label_list,feature,threshold):
        p = np.zeros((full_label_list.size,2))
        N = data.shape[0]
        for i in range(full_label_list.size):
            #left side
            L_label = label[data[:,feature]<=threshold]
            L = (L_label==full_label_list[i]).sum()/L_label.size
            R_label = label[data[:,feature]>threshold]
            R = 1/full_label_list.size
            if(R_label.size!=0):
                R = (R_label==full_label_list[i]).sum()/R_label.size
            p_i = np.array([L,R])
            p[i,:] = p_i
        return p

    selected_feature = np.random.randint(data.shape[1])
    possible_input = np.unique(data[:,selected_feature])
    gain = np.array([])
    for i in possible_input:
        gain = np.append(gain,getInformationGain(label,\
                selected_feature,i))
    threshold = gain.argmax()
    return (simplified_feature_list[selected_feature],threshold,\
    classifier(data,label,full_label_list,selected_feature,threshold))

def getRandomForest(data,label,M=2000,sample_size=100):
    N,D = data.shape
    full_label_list = np.unique(label)
    simplified_list = []
    for i in range(D):
        if(data[:,i]!=0).any():
            simplified_list.append(i)
    simplified_data = data[:,simplified_list]
    forest = []
    for i in range(M):
        choice = np.random.choice(N,sample_size, replace=False)
        sample = simplified_data[choice,:]
        slabel = label[choice]
        forest.append(getDecisionTree(sample,slabel,\
                                full_label_list,simplified_list))
    return forest

def getLoss(data,label,forest):
    M = len(forest)
    N = data.shape[0]
    y = np.zeros((N,M))
    c = np.unique(label).size
    for i in range(N):
        p = np.zeros(c)
        for j in range(M):
            p = p+forest[j][2][:,int(data[i,forest[j][0]]>forest[j][1])]
            y[i,j] = p.argmax()
    return [(y[:,i]!=label).sum()/label.size for i in range(M)]
    
forest = getRandomForest(train,tlabel)
train_loss = getLoss(train,tlabel,forest)
valid_loss = getLoss(valid,vlabel,forest)

with plt.style.context("default"):
	plt.plot(train_loss,"b-")
	plt.plot(valid_loss,"g-")
	plt.legend(['Training', 'Validation'])
	plt.title('Error Rate vs Iteration')
	plt.xlabel("Iteration")
	plt.ylabel("Classification Error Rate")
	plt.show()
