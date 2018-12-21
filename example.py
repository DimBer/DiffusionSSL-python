from gpmod.predict import diffusion_SSL
from gpmod.utils import load_labels
from gpmod.utils import extract_seed_sets
import numpy as np


# create seeds by randomly sampling some of the true labels

y_true = load_labels('graphs/blogcat_label.lst')
num_seeds = 1000
N = y_true.shape[0]
seed_indices = np.random.choice(range(N), size= num_seeds, replace = False )
seedset = extract_seed_sets(y_true, seed_indices)


# configure diffusion_SSL() predictor and 
  
ssl = diffusion_SSL()
ssl.SetGraphDir('graphs') # NOTE_: this is already the default value 
ssl.LoadGraph('blogcat')
ssl.SetNumClasses(y_true.shape[1])

# run default Personalized PageRank

ssl.SetBalancing_ON()
#ssl.SetBalancing_OFF()

predictions = ssl.SeedAndPredict(seedset) 
# Get reference to soft label predictions attribute
# Run:  predictions = np.array(ssl.SeedAndPredict(seedset)) to obtain local copy..

print( "\nPersonalized PageRank predictions", predictions,"\n")

# switch to Heat Kernel

ssl.SetPredictorType('hk')

ssl.SeedAndPredict(seedset)

print( "\nHeat Kernel predictions", predictions,"\n")

# customized diffusion

coefficients = np.array([ 1.0 , 2.0 , 3.0, 1.5, 0.0, 0.0, 0.0, 1.0, 1.0 ,1.0 ])
coefficients = coefficients / np.sum(coefficients)

ssl.SetPredictorType('custom')
ssl.SetNumSteps(len(coefficients))
ssl.InputCoefs(list(coefficients))

ssl.SeedAndPredict(seedset)

print( "\ncustom diffusion predictions", predictions,"\n")

