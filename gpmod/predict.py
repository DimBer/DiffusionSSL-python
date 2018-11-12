#!/usr/bin/env python

__author__ = "Dimitris Berberidis"
__copyright__ = "Copyright 2018"
__license__ = "GPL"


"""gpmod.py: Contains diffusion based methods for predicting labels of graph nodes. """


import numpy as np
from scipy import sparse as sp
from gpmod.utils import extract_seed_sets

class diffusion_SSL:

	avail_pred_types = ('hk', 'ppr', 'custom')

	# Predictor attributes initialized to empty/default

	def __init__(self):
		self.graph = sp.csr_matrix((1, 1), dtype=float)
		self.graph_dir = 'graphs'
		self.pred = np.array(1)
		self.num_classes = 0
		self.num_nodes = 0
		self.pred_type = 'ppr'
		self.class_weigh = False
		self.num_steps = 50
		self.hk_t = 15.0
		self.ppr_a = 0.98
		self.input_coefs = []

	# Parameter Modifiers - Graph Loader

	def SetNumClasses(self, num_classes ):
		"""
		Pre-define number of classes (OPTIONAL).

		Recommended for efficiency if running 
		multiple times for same number of classes 		
		"""
		self.num_classes = num_classes
		if self.num_nodes >0:
			self.pred = np.zeros((self.num_nodes,self.num_classes), dtype = float )	

	def SetNumSteps(self,num_steps):
		"""
		Change number of steps (DEFAULT = 50)		
		"""
		if(num_steps<1):
			print("Warning: steps must be >1")
			print(" ")
		else:
			self.num_steps = num_steps 

	def SetBalancing_ON(self):
		"""
		Activate class balancing 
		
		Each class diffusion ssums to 1
		Favors small classes (increases Macro F1 - decreases Micro F1 )
		"""
		self.class_weigh = False

	def SetBalancing_OFF(self):
		"""
		De-activate class balancing (DEFAULT is ON) 
		"""
		self.class_weigh = True

	def SetPredictorType(self, pred_type):
		"""
		Define type of diffusion:
			'hk' for Heat Kernel
			'ppr' for Personalized PageRank (DEFAULT)
			'custom' for customized diffusion  
		"""
		if pred_type not in diffusion_SSL.avail_pred_types:
			print("Warning: Predictor type must be in the following list:")
			print(diffusion_SSL.avail_pred_types)
			print("Current predictor type: ",self.pred_type)
			print(" ") 
		else:
			self.pred_type = pred_type

	def SetPPRcoef(self,x):
		"""
		change the value of Personalized PageRank coefficient ppr_a
		"""
		if x<=0.0 or x==1.0:
			print( "WARNING: PPR parameter must be in the range (0.0,1.0)")
			print( "Value was not changed" )
			print( "Use 'custom' diffusions" )
			print(" ")
		self.ppr_a  = x	

	def SetHKcoef(self,x):
		"""
		change the value of Heat Kernel coefficient hk_t
		"""
		if x<=0.0:
			print( "WARNING: HK parameter must be positive")
			print( "Value was not changed" )
			print( "Use 'custom' diffusions" )
			print(" ")
		self.hk_t  = x		

	def SetGraphDir(self,dirname):
		"""
		define directory where graphs are stored (DEFAULT is 'graphs')
		"""		
		self.graph_dir = dirname

	def InputCoefs(self, coefs):
		"""
		Input list of diffusion coefficients to run a customized diffusion
		"""

		coefs = np.array(coefs)

		if np.abs( 1.0 - np.sum(coefs)) > 1.0e-4:
			print("WARNING: input coefficients must sum to 1..")
			print("Normalizing..")
			print(" ")
			coefs = coefs / np.sum(coefs)			

		coefs = list(coefs)

		if len(coefs) != self.num_steps:
			print("WARNING: num_steps did not match length of input coefs")
			print("num_steps now set to ", len(coefs) )
			print(" ")
			self.num_steps = len(coefs)

		self.input_coefs = coefs 		

	def LoadGraph(self, graph_name):
		"""
		Loads normalized adjacency matrix A of graph named <graph_name>

		To use this function, the following must hold:
			1) A must be stored as a sp.sparse.csr_matrix() 
			2) A must be stored in directory defined by self.graph_dir (DEFAULT is 'graphs')
			3) filename must be in the format: <graph_name>_graph.mtx.npz

		_NOTE: Diffusions require column stochasticity
			   If A is not column-stochastic, it will be automatically normalized. 
		"""

		filename= self.graph_dir + '/'+graph_name+'_graph.mtx.npz'

		loader = np.load(filename)

		self.graph = sp.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
		 
		self.num_nodes = self.graph.get_shape()[0]

		testvec = np.ones([self.num_nodes,1]) / float(self.num_nodes) 
		if np.abs( 1.0 - np.sum(self.graph.dot(testvec))) > 1.0e-4: 
			print("WARNING: Adjacency matrix not column stochastic and will be normalized")
			print(" ")
			self.graph = stochastic_from_adjacency(self.graph)

		if self.num_classes > 0:
			self.pred = np.zeros((self.num_nodes,self.num_classes), dtype = float )	

	def reset(self):
		"""
		RESET all parameters (EXCEPT handle to graph, and graphs directory) to DEFAULT
		"""
		self.pred = np.array(1)
		self.num_classes = 0
		self.num_nodes = 0
		self.pred_type = 'ppr'
		self.class_weigh = False
		self.num_steps = 50
		self.hk_t = 15.0
		self.ppr_a = 0.98
		self.input_coefs = []		


	# Function that outputs predictions

	def SeedAndPredict(self,seedset):

		# report

		print("\nRunning predictions over graph..\n")
		print("Diffusion type: ", self.pred_type)
		print("Number of steps: ", self.num_steps)
		print("Number of nodes: ", self.num_nodes)
		print("Number of classes: ", self.num_classes)
		if self.class_weigh:
			print("Balancing: OFF")
		else:
			print("Balancing: ON")	 

		# checks

		if self.num_nodes == 0:
			print("ERROR: No graph loaded.")
			print(" ")
			return 0

		if self.num_classes != len(seedset):
			self.num_classes = len(seedset)
			self.pred = np.zeros((self.num_nodes,self.num_classes), dtype = float )
			print("Num_classes automatically set to ", self.num_classes )
			print("If running multiple times use -> SetNumClasses() for efficiency")
			print(" ")

		# prepare coeffiecients and seeds

		step = 1
		if self.pred_type == 'hk':
			hk_t = self.hk_t
			coef = [hk_t]
			while step < self.num_steps:
				coef.append( (hk_t/float(step)) * coef[-1])
				step += 1
		elif self.pred_type == 'ppr':		
			ppr_a = self.ppr_a
			coef = [ppr_a]
			while step < self.num_steps:
				coef.append( ppr_a * coef[-1])
				step += 1
		elif self.pred_type == 'custom':
			if self.input_coefs:
				coef = self.input_coefs
			else:
				print("WARNING: predictor type set to 'custom' but coefficients not defined...")
				print("Use .InputCoefs( ) to enter list of coefficients")
				print(" ")
		coef = list( np.array(coef) / np.sum(np.array(coef)))

		walks = np.empty_like(self.pred)
		walks.fill(0.0)
		for j,seeds in zip(range(self.num_classes),seedset):
			if seeds:
				walks[seeds,j] = 1.0 / float(len(seeds))	

		# run diffusion

		self.pred.fill(0.0)
		step = 0
		while step < self.num_steps:
			walks = self.graph.dot(walks)
			self.pred[:] = self.pred + coef[step]*walks 
			step += 1

		# class weiging if balancing is set to OFF
		if self.class_weigh:
			class_weights = np.zeros([self.num_classes,])
			for j,seeds in zip(range(self.num_classes),seedset):
				class_weights[j] = float(len(seeds))
			self.pred[:] = np.dot(self.pred, np.diag(class_weights))

		return self.pred


if __name__ == "__main__":
    main()
    
