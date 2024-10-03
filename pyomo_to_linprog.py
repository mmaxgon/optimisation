import pyomo.environ as pyomo

Demand = {
	'Lon': 125,  # London
	'Ber': 175,  # Berlin
	'Maa': 225,  # Maastricht
	'Ams': 250,  # Amsterdam
	'Utr': 225,  # Utrecht
	'Hag': 200  # The Hague
}

Supply = {
	'Arn': 600.3,  # Arnhem
	'Gou': 650.7  # Gouda
}

T = {
	('Lon', 'Arn'): 1000,
	('Lon', 'Gou'): 2.5,
	('Ber', 'Arn'): 2.5,
	('Ber', 'Gou'): 1000,
	('Maa', 'Arn'): 1.6,
	('Maa', 'Gou'): 2.0,
	('Ams', 'Arn'): 1.4,
	('Ams', 'Gou'): 1.0,
	('Utr', 'Arn'): 0.8,
	('Utr', 'Gou'): 1.0,
	('Hag', 'Arn'): 1.4,
	('Hag', 'Gou'): 0.8
}

# Define index sets
CUS = list(Demand.keys())
SRC = list(Supply.keys())

# Create an instance of the model
model = pyomo.ConcreteModel()

# Define the decision vars
model.x = pyomo.Var(CUS, SRC, domain=pyomo.NonNegativeIntegers, bounds=(0, 249))
model.y = pyomo.Var(CUS, SRC, domain=pyomo.Binary)

# Constraints
model.src = pyomo.ConstraintList()
for s in SRC:
	model.src.add(sum([model.x[c, s] for c in CUS]) <= Supply[s])

model.dmd = pyomo.ConstraintList()
for c in CUS:
	model.dmd.add(sum([model.x[c, s] for s in SRC]) == Demand[c])

# Define Objective
model.Cost = pyomo.Objective(
	expr=sum([T[c, s] * model.x[c, s] for c in CUS for s in SRC]),
	sense=pyomo.minimize
)

# Solve
results = pyomo.SolverFactory('glpk').solve(model)

# Solution
for c in CUS:
	for s in SRC:
		if pyomo.value(model.x[c, s]) > 0:
			print(c, s, model.x[c, s]())
############################################################################################################
# Перевод задачи в scipy.optimize.linprog
############################################################################################################

from pyomo.repn.standard_repn import generate_standard_repn
for c in model.dmd.keys():
	repn = generate_standard_repn(model.dmd[c].body)
	for var in repn.linear_vars:
		print(var)
	print(c, repn.linear_coefs)
	print(generate_standard_repn(model.dmd[c].lb).constant)
