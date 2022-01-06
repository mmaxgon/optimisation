import time
import numpy as np
import cyipopt
import scipy as sp

class HS071:
	def objective(self, x):
		"""Returns the scalar value of the objective given x."""
		return x[0] * x[3] * np.sum(x[0:3]) + x[2]

	def gradient(self, x):
		"""Returns the gradient of the objective with respect to x."""
		return np.array([
			x[0]*x[3] + x[3]*np.sum(x[0:3]),
			x[0]*x[3],
			x[0]*x[3] + 1.0,
			x[0]*np.sum(x[0:3])
		])
	
	def constraints(self, x):
		"""Returns the constraints."""
		return np.array((np.prod(x), np.dot(x, x)))
	
	def jacobian(self, x):
		"""Returns the Jacobian of the constraints with respect to x."""
		return np.concatenate((np.prod(x)/x, 2*x))

	def hessianstructure(self):
		"""Returns the row and column indices for non-zero vales of the
		Hessian."""
		# NOTE: The default hessian structure is of a lower triangular matrix,
		# therefore this function is redundant. It is included as an example
		# for structure callback.
		return np.nonzero(np.tril(np.ones((4, 4))))

	def hessian(self, x, lagrange, obj_factor):
		"""Returns the non-zero values of the Hessian."""
		H = obj_factor*np.array((
			(2*x[3], 0, 0, 0),
			(x[3], 0, 0, 0),
			(x[3], 0, 0, 0),
			(2*x[0]+x[1]+x[2], x[0], x[0], 0)))
		H += lagrange[0]*np.array((
			(0, 0, 0, 0),
			(x[2]*x[3], 0, 0, 0),
			(x[1]*x[3], x[0]*x[3], 0, 0),
			(x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))
		H += lagrange[1]*2*np.eye(4)
		row, col = self.hessianstructure()
		return H[row, col]

	def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
				  d_norm, regularization_size, alpha_du, alpha_pr,
	ls_trials):
		"""Prints information at every Ipopt iteration."""
		msg = "Objective value at iteration #{:d} is - {:g}"
		print(msg.format(iter_count, obj_value))

# define the lower and upper bounds of 𝑥 and the constraints:
lb = [1.0, 1.0, 1.0, 1.0]
ub = [5.0, 5.0, 5.0, 5.0]
cl = [25.0, 40.0]
cu = [2.0e19, 40.0]
#  initial guess
x0 = [1.0, 5.0, 5.0, 1.0]

# Define the full problem using the cyipopt.Problem class
# The constructor of the cyipopt.Problem class requires:
# • n: the number of variables in the problem,
# • m: the number of constraints in the problem,
# • lb and ub: lower and upper bounds on the variables,
# • cl and cu: lower and upper bounds of the constraints.
# • problem_obj is an object whose methods implement objective, gradient, constraints,
# jacobian, and hessian of the problem.
nlp = cyipopt.Problem(
	n=len(x0),
	m=len(cl),
	problem_obj=HS071(),
	lb=lb,
	ub=ub,
	cl=cl,
	cu=cu,
)

# Setting optimization parameters
nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 1e-7)

x, info = nlp.solve(x0)
x
info
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
# Нужно задать значения gradient, jacobian
####################################################################
class gini:
	def objective(self, x):
		return np.sum(np.power(x, 2))

	def gradient(self, x):
		return sp.optimize.approx_fprime(x, self.objective, epsilon = 1e-8)
# 		return np.array(2 * np.array(x))
	
	def constraints(self, x):
		return np.array([
			np.sum(x), 
 			x[0] - np.floor(x[0]), 
			x[1] - np.floor(x[1]),
			x[2] - np.floor(x[2])
		])
	
	def jacobian(self, x):
		return sp.optimize.slsqp.approx_jacobian(x, self.constraints, epsilon = 1e-8)
		#return np.array([np.ones(len(x)), [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

x0 = [4, 4, 2, 0]

gi = gini()
gi.objective(x0)
gi.gradient(x0)
gi.jacobian(x0)

n = len(x0)
lb = [0] * n
ub = [10] * n
cl = [10, 0, 0, 0]
cu = [10, 0.1, 0.1, 0.1]

nlp = cyipopt.Problem(
	n = len(x0),
	m = len(cl),
	problem_obj = gini(),
	lb = lb,
	ub = ub,
	cl = cl,
	cu = cu,
)

nlp.add_option('jac_c_constant', 'yes')
nlp.add_option('jac_d_constant', 'no')
nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 1e-7)

start_time = time.time()
x, info = nlp.solve(x0)
print(time.time() - start_time)
x
info

gi.constraints(x)
gi.objective(x)
#######################################################################
def objective(x):
	return np.sum(np.power(x, 2))

def gradient(x):
 	return np.array(2 * np.array(x))

def hess(x):
 	return 2 * np.identity(len(x))

# >= 0
def constraints(x):
	return np.array([
		np.sum(x) - 10
	])

def constraints1(x):
	return np.array([
 		# np.floor(x[3]) - x[3] + 0.1,
 		np.floor(x[2]) - x[2] + 0.1,
		np.floor(x[1]) - x[1] + 0.1,
		np.floor(x[0]) - x[0] + 0.1
	])

x0 = [4, 4, 2, 0]
x = x0
n = len(x)
bounds = np.reshape([0, 10] * n, (n, 2))

# sp.optimize.show_options(solver = "minimize", method = "trust-constr")

start_time = time.time()
res = sp.optimize.minimize(
	method = "trust-constr", 
	options = {'gtol': 1e-4, "xtol": 1e-4},
	fun = objective,
	#jac = gradient,
	#hess = hess,
	bounds = bounds,
	constraints = [{"fun": constraints, "type": "eq"}, {"fun": constraints1, "type": "ineq"}],
	x0 = x0
)
print(time.time() - start_time)
x = res.x
res

res.constr_violation

objective(x0)
objective(res.x)
