import numpy as np
import rbfopt
settings = rbfopt.RbfoptSettings(
	max_iterations=1000,
	max_evaluations=50,
	algorithm="MSRSM", # MSRSM Gutmann
	global_search_method="solver", # genetic sampling solver
	minlp_solver_path='C:/Program Files/solvers/Bonmin/bonmin.exe', # если global_search_method="solver"
	nlp_solver_path='C:/Program Files/solvers/IPOPT/bin/ipopt.exe'  # если global_search_method="solver"
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
alg = rbfopt.RbfoptAlgorithm(settings, bb)
val, x, itercount, evalcount, fast_evalcount = alg.optimize()
