import numpy as np
from gekko import GEKKO

# There are free solvers: 1: APOPT, 2: BPOPT, 3: IPOPT distributed with the public version of the software. There are additional solvers that are not included with the public version and require a commercial license. IPOPT is generally the best for problems with large numbers of degrees of freedom or when starting without a good initial guess. BPOPT has been found to be the best for systems biology applications. APOPT is generally the best when warm-starting from a prior solution or when the number of degrees of freedom (Number of Variables - Number of Equations) is less than 2000. APOPT is also the only solver that handles Mixed Integer problems. Use option 0 to compare all available solvers.
###############################################################################
m = GEKKO(remote = False) # Initialize gekko
m.options.SOLVER = 1  # APOPT is an MINLP solver

# optional solver settings with APOPT
m.solver_options = [
	'minlp_maximum_iterations 500', \
	# minlp iterations with integer solution
	'minlp_max_iter_with_int_sol 10', \
	# treat minlp as nlp
	'minlp_as_nlp 0', \
	 # nlp sub-problem max iterations
	'nlp_maximum_iterations 50', \
	# 1 = depth first, 2 = breadth first
	'minlp_branch_method 1', \
	# maximum deviation from whole number
	'minlp_integer_tol 0.05', \
	# covergence tolerance
	'minlp_gap_tol 0.01'
]

# Initialize variables
x1 = m.Var(value = 1, lb = 1, ub = 5)
x2 = m.Var(value = 5, lb = 1, ub = 5)

# Integer constraints for x3 and x4
x3 = m.Var(value = 5, lb = 1, ub = 5, integer = True)
x4 = m.Var(value = 1, lb=1, ub = 5, integer = True)

# Equations
m.Equation(x1 * x2 * x3 * x4 >= 25)
m.Equation(x1**2 + x2**2 + x3**2 + x4**2 == 40)

m.Obj(x1 * x4 * (x1 + x2 + x3) + x3) # Objective

m.solve(disp = True) # Solve

print('Results')
print('x1: ' + str(x1.value))
print('x2: ' + str(x2.value))
print('x3: ' + str(x3.value))
print('x4: ' + str(x4.value))
print('Objective: ' + str(m.options.objfcnval))

m.path

###############################################################################

import numpy as np
from gekko import GEKKO
m = GEKKO()

ni = 8
nj = 3

x = [[m.Var(lb = 0, integer = True) for j in range(nj)] for i in range(ni)]

s = 0
for i in range(ni):
    for j in range(nj):
        s += x[i][j]

m.Equation(s == 10)
m.Equations([x[2][j] + x[4][j] >= x[0][j] for j in range(nj)])
m.Equations([x[3][j] + x[5][j] >= x[1][j] for j in range(nj)])
for j in range(nj):
    x[6][j].upper = 15
    x[7][j].upper = 15

m.Equations([(m.sign3(x[6][j]) == m.sign3(x[2][j])) for j in range(nj)])
m.Equations([(m.sign3(x[7][j]) == m.sign3(x[3][j])) for j in range(nj)])

m.Obj(sum([m.tanh(x[i][0]) for i in range(ni)]))

m.options.SOLVER = 1

m.solve(disp = True)

print(x)

###############################################################################
# MINLP - дифференцируемая задача с нелинейными функциями, задаваемыми явно из GEKKO
import numpy as np
from gekko import GEKKO
	
m = GEKKO(remote = False)

m.options.SOLVER = 1
m.options.LINEAR = 0
# m.solver_options = [
# 	'minlp_maximum_iterations 10000',
# 	'minlp_max_iter_with_int_sol 500',
# 	'minlp_gap_tol 0.01',
# 	'nlp_maximum_iterations 500',
# 	'minlp_as_nlp 0',
# 	'minlp_interger_leaves = 0',
# 	'minlp_branch_method 1',
# 	'minlp_integer_tol 0.01',
# 	'minlp_print_level 2'
# ]

def con1(y):
	nn = len(y)
	res = sum([y[i] for i in range(nn)])
	return res

def goal1(y):
	nn = len(y)
	res = 0
	for i in range(nn):
		res += m.if2(-y[i], y[i] * m.log(y[i]), 0)
	# 	res = sum([(y[i] ** 2) for i in range(nn)])
	return res

n = 3
x = [m.Var(lb = 0, ub = 10, integer = True) for j in range(n)]

s = con1(x)
m.Equation(s == 10)

g = goal1(x)
m.Obj(g)

m.solve(disp = True)

print(x)


