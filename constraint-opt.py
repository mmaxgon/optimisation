# pip install --upgrade python-constraint
"""
https://github.com/python-constraint/python-constraint
http://labix.org/doc/constraint/
"""

# Find all (x,y) where x ∈ {1,2,3} and 0 <= y < 10, and x + y >= 5
import constraint

problem = constraint.Problem()

problem.addVariable('x', [1, 2, 3])
problem.addVariable('y', range(10))

def our_constraint(x, y):
    if x + y >= 5:
        return True

problem.addConstraint(our_constraint, ['x','y'])
problem.addConstraint(constraint.AllDifferentConstraint(), ['x','y'])

solutions = problem.getSolutions()

# Easier way to print and see all solutions
# for solution in solutions:
#    print(solution)

# Prettier way to print and see all solutions
length = len(solutions)
print("(x,y) ∈ {", end="")
for index, solution in enumerate(solutions):
    if index == length - 1:
        print("({},{})".format(solution['x'], solution['y']), end="")
    else:
        print("({},{}),".format(solution['x'], solution['y']), end="")
print("}")
#####################################################################################
problem = constraint.Problem()
problem.addVariable("b", [0, 1])
problem.addVariable("x", range(10))

def constr1(b, x):
	if b == 1:
		return x % 2 == 0
	else:
		return x % 2 == 1

problem.addConstraint(constr1, ["b", "x"])

solutions = problem.getSolutions()
