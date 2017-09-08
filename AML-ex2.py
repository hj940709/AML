import numpy as np
from matplotlib import pyplot as plt
cd "E:\document\AML\dataset"

train = np.loadtxt(
	open(".\\exercise_2_1_train.csv","rb"),
	delimiter=",",skiprows=0).T
test = np.loadtxt(
	open(".\\exercise_2_1_test.csv","rb"),
	delimiter=",",skiprows=0).T
theta = np.zeros(train.shape[0]-1)

def get_loss(data, theta, l=0):
	y = data[-1,]
	x = data[:-1,]
	N = x.shape[1]
	return np.power(theta.T.dot(x)-y,2).sum()/N+l*theta.T.dot(theta)

def get_gradient(data,theta,l=0):
	y = data[-1,]
	x = data[:-1,]
	N = x.shape[1]
	return np.multiply(theta.T.dot(x)-y,x).sum(1)*2/N+l*2*theta

def gradient_desent_ERM(data, init_theta,prop=0.1,step = 0.5):
	#ERM
	p = init_theta
	i=0
	N=data.shape[1]
	train_loss = np.array([get_loss(data[:,:int(N*prop)],p)])
	valid_loss = np.array([get_loss(data[:,int(N*prop):],p)])
	while True:
		i = i+1
		gradient = get_gradient(data[:,0:int(N*prop)],p)
		q = p - step*gradient
		train_loss = np.append(train_loss,get_loss(data[:,:int(N*prop)],q))
		valid_loss = np.append(valid_loss,get_loss(data[:,int(N*prop):],q))
		if np.square(gradient).sum()<1e-6:
			return [q,train_loss,valid_loss,i]
		else:
			p=q

def gradient_desent_ES(data, init_theta,prop=0.1,step = 0.5):
	#Early-stopping
	theta = init_theta
	i=0
	N=data.shape[1]
	p = get_loss(data[:,int(N*prop):],theta)
	train_loss = np.array([get_loss(data[:,0:int(N*prop)],theta)])
	valid_loss = np.array([p])
	while True:
		i = i+1
		theta = theta - step*get_gradient(data[:,0:int(N*prop)],theta)
		q = get_loss(data[:,int(N*prop):],theta)
		train_loss = np.append(train_loss,get_loss(data[:,0:int(N*prop)],theta))
		valid_loss = np.append(valid_loss,q)
		if q>=p: #new value for loss no longer decrease
			return [theta,train_loss,valid_loss,i]
		else:
			p = q

def gradient_desent_Reg(data, init_theta,l,prop=0.1,step = 0.5):
	#Regularization
	p = init_theta
	i=0
	N=data.shape[1]
	train_loss = np.array([get_loss(data[:,0:int(N*prop)],p)])
	valid_loss = np.array([get_loss(data[:,int(N*prop):],p)])
	while True:
		i = i+1
		gradient = get_gradient(data[:,0:int(N*prop)],p,l)
		q = p - step*gradient
		train_loss = np.append(train_loss,get_loss(data[:,0:int(N*prop)],q))
		valid_loss = np.append(valid_loss,get_loss(data[:,int(N*prop):],q))
		if np.square(gradient).sum()<1e-6:
			return [q,train_loss,valid_loss,i]
		else:
			p=q

l = np.array([0.001, 0.003, 0.01, 0.03, 0.1])
train_loss1 = np.array([])
train_loss2 = np.array([])
valid_loss1 = np.array([])
valid_loss2 = np.array([])
for i in range(0,l.size):
	estimated_theta,train_loss,valid_loss,iteration = gradient_desent_Reg(train,theta,l[i])
	train_loss1 = np.append(train_loss1,train_loss[-1])
	valid_loss1 = np.append(valid_loss1,valid_loss[-1])
	estimated_theta,train_loss,valid_loss,iteration = gradient_desent_Reg(train,theta,l[i],0.9)
	train_loss2 = np.append(train_loss2,train_loss[-1])
	valid_loss2 = np.append(valid_loss2,valid_loss[-1])

estimated_theta,train_loss,valid_loss,iteration = gradient_desent_ERM(train,theta)
estimated_theta,train_loss,valid_loss,iteration = gradient_desent_ES(train,theta)
estimated_theta,train_loss,valid_loss,iteration = gradient_desent_Reg(train,theta,l[valid_loss1.argmin()])

estimated_theta,train_loss,valid_loss,iteration = gradient_desent_ERM(train,theta,0.9)
estimated_theta,train_loss,valid_loss,iteration = gradient_desent_ES(train,theta,0.9)
estimated_theta,train_loss,valid_loss,iteration = gradient_desent_Reg(train,theta,l[valid_loss2.argmin()],0.9)
get_loss(test,estimated_theta)

with plt.style.context("default"):
	erm = gradient_desent_ERM(train,theta)
	es = gradient_desent_ES(train,theta)
	plt.plot(np.arange(erm[-1]+1),erm[1],"b-")
	plt.plot(np.arange(erm[-1]+1),erm[2],"g-")
	plt.plot(es[-1],es[1][-1],"bo")
	plt.plot(es[-1],es[2][-1],"go")
	plt.legend(['Training', 'Validation'])
	plt.title('Iteration vs Error (Train/Valid=10:90)')
	plt.xlabel("Iteration")
	plt.ylabel("Loss")
	plt.show()

with plt.style.context("default"):
	erm = gradient_desent_ERM(train,theta,0.9)
	es = gradient_desent_ES(train,theta,0.9)
	plt.plot(np.arange(erm[-1]+1),erm[1],"b-")
	plt.plot(np.arange(erm[-1]+1),erm[2],"g-")
	plt.plot(es[-1],erm[1][es[-1]],"bo")
	plt.plot(es[-1],erm[2][es[-1]],"go")
	plt.legend(['Training', 'Validation'])
	plt.title('Iteration vs Error (Train/Valid=90:10)')
	plt.xlabel("Iteration")
	plt.ylabel("Loss")
	plt.show()

with plt.style.context("default"):
	plt.plot(l, train_loss1,"bo-")
	plt.plot(l, train_loss2,"go-")
	plt.title("Train Loss vs Regularization Constant")
	plt.xlabel("Constants")
	plt.ylabel("Train Error")
	plt.legend(["Train/Valid=10/90", "Train/Valid=90/10"])
	plt.show()

with plt.style.context("default"):
	plt.plot(l, valid_loss1,"bo-")
	plt.plot(l, valid_loss2,"go-")
	plt.title("Validation Loss vs Regularization Constant")
	plt.xlabel("Constants")
	plt.ylabel("Validation Error")
	plt.legend(["Train/Valid=10/90", "Train/Valid=90/10"])
	plt.show()

data = np.loadtxt(
	open(".\\exercise_2_2.csv","rb"),
	delimiter=",",skiprows=0).T
theta = np.array([-0.3,1.5])

def gradient_desent(data, init_theta,step = 0.05):
	#ERM
	p = init_theta
	theta = np.mat(init_theta).reshape(2,1)
	loss = np.array([get_loss(data,p)])
	while True:
		gradient = get_gradient(data,p)
		q = p - step*gradient
		theta = np.column_stack((theta, q))
		loss = np.append(loss,get_loss(data,q))
		if (np.square(gradient).sum()<1e-6):
			return (theta,loss)
		else:
			p=q

def newton(data,init_theta):
	x = data[:-1,:]
	y = data[-1,:]
	N = x.shape[1]
	H = np.mat(x.dot(x.T)*2/N)
	p = init_theta
	theta = np.mat(init_theta).reshape(2,1)
	loss = np.array([get_loss(data,p)])
	while True:
		q = np.squeeze(np.asarray(p - H.I.dot(get_gradient(data,p))))
		theta = np.column_stack((theta, q))
		loss = np.append(loss,get_loss(data,q))
		if np.square(p-q).sum()<1e-6:
			return (theta,loss)
		else: p=q

def SGD(data,init_theta,sample_size=1,step = 0.01):
	p = init_theta
	N=data.shape[1]
	theta = np.mat(init_theta).reshape(2,1)
	loss = np.array([get_loss(data,p)])
	while True:
		sample = data
		if(sample_size!=0):
			sample = data[:,np.random.choice(data.shape[1],sample_size,replace=False)]
		gradient = get_gradient(sample,p)
		q = p - step*gradient
		theta = np.column_stack((theta, q))
		loss = np.append(loss,get_loss(data,q))
		if np.square(p-q).sum()<1e-8:
			return (theta,loss)
		else:
			p=q

def SGD_adaGrad(data,init_theta,sample_size=1):
	sample = data
	if(sample_size!=0):
		sample = data[:,np.random.choice(data.shape[1],sample_size,replace=False)]
	N = data.shape[1]
	p = init_theta
	g = np.mat(np.power(get_gradient(sample,p),2)).reshape(2,1)
	theta = np.mat(init_theta).reshape(2,1)
	loss = np.array([get_loss(data,p)])
	while True:
		sample = data
		if(sample_size!=0):
			sample = data[:,np.random.choice(data.shape[1],sample_size,replace=False)]
		s = g.sum(1)
		step = np.squeeze(np.asarray(1/(1+np.sqrt(s))))
		q = p - np.multiply(step,get_gradient(sample,p))
		theta = np.column_stack((theta, q))
		loss = np.append(loss,get_loss(data,q))
		if np.square(get_gradient(sample,q)).sum()<1e-6:
			return (theta,loss)
		else: 
		    p=q
		    g = np.column_stack((g, np.power(get_gradient(sample,p),2)))

I = np.arange(-2.0, 2, 0.05)
J = np.arange(-2.0, 2, 0.05)
Z = np.zeros((len(I),len(J)))
for i in range(len(I)):
	for j in range(len(J)):
		Z[i,j] = get_loss(data,np.array([I[i], J[j]]))
levels=np.arange(0.1,1.2,0.3)
I1, J1 = np.meshgrid(I, J)

np.random.seed(0)
gd,gd_loss = gradient_desent(data,theta)
n,n_loss = newton(data,theta)
sgd,sgd_loss = SGD(data,theta)
sgd1,sgd1_loss = SGD_adaGrad(data,theta)

with plt.style.context("default"):
	plt.plot(np.arange(gd_loss.size),gd_loss,"bo-")
	plt.plot(np.arange(n_loss.size),n_loss,"ro-")
	plt.plot(np.arange(sgd_loss.size),sgd_loss,"go-")
	plt.plot(np.arange(sgd1_loss.size),sgd1_loss,"co-")
	plt.legend(["GD", "Newton","SGD","SGD_adaGrad"])
	plt.title('Iteration vs Error')
	plt.xlabel("Iteration")
	plt.ylabel("Loss")
	plt.show()

with plt.style.context("default"):
	plt.contour(I1, J1, Z.T, levels)
	plt.plot(gd[0,:].T,gd[1,:].T,"bo-")
	plt.plot(n[0,:].T,n[1,:].T,"ro-")
	plt.plot(sgd[0,:].T,sgd[1,:].T,"go-")
	plt.plot(sgd1[0,:].T,sgd1[1,:].T,"co-")
	plt.legend(["GD", "Newton","SGD","SGD_adaGrad"])
	plt.show()


sgd,sgd_loss = SGD(data,theta,sample_size=50)
sgd1,sgd1_loss = SGD_adaGrad(data,theta,sample_size=50)

with plt.style.context("default"):
	plt.plot(np.arange(sgd_loss.size),sgd_loss,"go-")
	plt.plot(np.arange(sgd1_loss.size),sgd1_loss,"co-")
	plt.legend(["SGD with mini-batch","SGD_adaGrad with mini-batch"])
	plt.title('Iteration vs Error')
	plt.xlabel("Iteration")
	plt.ylabel("Loss")
	plt.show()

with plt.style.context("default"):
	plt.contour(I1, J1, Z.T, levels)
	plt.plot(sgd[0,:].T,sgd[1,:].T,"go-")
	plt.plot(sgd1[0,:].T,sgd1[1,:].T,"co-")
	plt.legend(["SGD with mini-batch","SGD_adaGrad with mini-batch"])
	plt.show()


