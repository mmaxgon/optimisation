"""
https://developers.google.com/optimization/reference/python/sat/python/cp_model
https://github.com/google/or-tools/blob/stable/ortools/sat/samples/
"""
from ortools.sat.python import cp_model

model = cp_model.CpModel()
 
num_vals = 3
x = model.NewIntVar(0, num_vals - 1, 'x')
y = model.NewIntVar(0, num_vals - 1, 'y')
z = model.NewIntVar(0, num_vals - 1, 'z')
# Adds an all-different constraint.
model.Add(x != y)

# Creates a solver and solves the model.
solver = cp_model.CpSolver()

# Sets a time limit of 10 seconds.
solver.parameters.max_time_in_seconds = 10.0
status = solver.Solve(model)

solver.Value(x)
solver.Value(y)
solver.Value(z)
#############################################################################
# Data
costs = [
	[90, 80, 75, 70],
	[35, 85, 55, 65],
	[125, 95, 90, 95],
	[45, 110, 95, 115],
	[50, 100, 90, 100],
]
num_workers = len(costs)
num_tasks = len(costs[0])

# Model
model = cp_model.CpModel()

# Decision Vars
x = []
for i in range(num_workers):
	t = []
	for j in range(num_tasks):
		t.append(model.NewBoolVar(f'x[{i},{j}]'))
	x.append(t)
		
# Constraints
# Each worker is assigned to at most one task.
for i in range(num_workers):
	model.Add(sum(x[i][j] for j in range(num_tasks)) <= 1)

# Each task is assigned to exactly one worker.
for j in range(num_tasks):
	model.Add(sum(x[i][j] for i in range(num_workers)) == 1)
  
# Objective
objective_terms = []
for i in range(num_workers):
	for j in range(num_tasks):
		objective_terms.append(costs[i][j] * x[i][j])
model.Minimize(sum(objective_terms))
	
# Solve
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Print solution.
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
	print(f'Total cost = {solver.ObjectiveValue()}')
	print()
	for i in range(num_workers):
		for j in range(num_tasks):
			if solver.BooleanValue(x[i][j]):
				print(f'Worker {i} assigned to task {j} Cost = {costs[i][j]}')
else:
	print('No solution found.')