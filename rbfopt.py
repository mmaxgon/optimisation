import numpy as np
import rbfopt
settings = rbfopt.RbfoptSettings(
	minlp_solver_path='C:/Program Files/bonmin/bin', 
	nlp_solver_path='C:/Program Files/ipopt/bin'
)

def obj_funct(x):
  return x[0]*x[1] - x[2]

bb = rbfopt.RbfoptUserBlackBox(
	dimension = 3, 
	var_lower = np.array([0] * 3), 
	var_upper = np.array([10] * 3),
	var_type = np.array(['R', 'I', 'R']), 
	obj_funct = obj_funct,
	obj_funct_noisy = None
)
settings = rbfopt.RbfoptSettings(max_evaluations=50)
alg = rbfopt.RbfoptAlgorithm(settings, bb)
val, x, itercount, evalcount, fast_evalcount = alg.optimize()
