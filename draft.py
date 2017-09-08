import matplotlib.pyplot as plt
import numpy as np
import mnist_load_show as mnist
from sklearn.decomposition import PCA
def gt_stump_v2(data, one_y):
	n = data.shape[0]
	N = data.shape[1]
	ri = np.random.randint(N)
	thresh = 1e-10
	tot_counts = sum(one_y)
	tot_prob = (tot_counts / float(n))
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
			probs_left = (cl_counts / float(tot_l))
			E_left= -1*sum(probs_left*np.log(probs_left))
		if sum(cr_counts) == 0:
			tot_r = 0
			probs_right = np.ones(10)*0.1
			E_right= -1*sum(probs_right*np.log(probs_right))
		else:
			tot_r = sum(cr_counts)
			probs_right = (cr_counts / float(tot_r))
			E_right= -1*sum(probs_right*np.log(probs_right))
		IG = E_tot - (tot_l/(1.*n))*E_left - (tot_r/(1.*n))*E_right
		res.append([IG, ri, i, probs_left, probs_right])
		return res
x , y = mnist.read_mnist_training_data(10000)
gt_stump_v2(x, y)
tempvec = [(res[i][0], i) for i in range(len(res))]
tempvec.sort()
s1 = tempvec[-1]
stump = r1[s1[1]]