#!/usr/bin/env python

__author__ = "Dimitris Berberidis"
__copyright__ = "Copyright 2018"
__license__ = "GPL"

"""utils.py: Contains useful functions for handling graphs and labels. """


import numpy as np

def stochastic_from_adjacency(A):
	"""
	Convert CSR ajacency matrix to column stochastic
	"""
	degrees=np.array(A.sum(axis=0))
	degrees=degrees[0]
	A=sp.sparse.csc_matrix(A,dtype='float')
	for i in range(A.shape[1]):
		if degrees[i]>0.0:
				A[:,i]=A[:,i]/float(degrees[i])
	return sp.sparse.csr_matrix(A) 

def load_labels(filename):
	"""
	Read labels from file

	Labels must be stored in format bellow:

			<node_index> \t <label> \n
						.
						.
						.
			<node_index> \t <label> \n
	"""
	
	raw = np.genfromtxt(filename,dtype = int, delimiter='\t')

	num_nodes = np.max(raw[:,0]) +1
	num_class = np.max(raw[:,1]) +1

	one_hot = np.zeros([num_nodes,num_class])

	for i in range(raw.shape[0]):
		one_hot[ raw[i,0], raw[i,1] ] = 1

	return one_hot

def extract_seed_sets( OneHot, _indices ) :
	"""
	INPUT
			OneHot: np array with node labels in one-hot form
			_indices: indices of subset of nodes that we want to extract

	RETURNS
			listOlists: list of lists where the i-th list contains
					the seeds of i-th class (as found in _indices)
	"""
	num_class = OneHot.shape[1]
	listOlists = [[] for i in range(num_class)]
	for ind in _indices:
		for c in range(num_class):
			if OneHot[ind,c] ==1:
				listOlists[c].append(ind)
	return listOlists


if __name__ == "__main__":
    main()
    
