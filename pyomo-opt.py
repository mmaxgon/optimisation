import pyomo.environ as pe

pe.SolverFactory('cbc').available()
pe.SolverFactory('glpk').available()
pe.SolverFactory('ipopt').available()
pe.SolverFactory('mindtpy').available()
pe.SolverFactory('symphony').available()
pe.SolverFactory('bonmin').available()
pe.SolverFactory('couenne').available()
pe.SolverFactory('shot').available()

Demand = {
   'Lon':   125,        # London
   'Ber':   175,        # Berlin
   'Maa':   225,        # Maastricht
   'Ams':   250,        # Amsterdam
   'Utr':   225,        # Utrecht
   'Hag':   200         # The Hague
}

Supply = {
   'Arn':   600.3,        # Arnhem
   'Gou':   650.7         # Gouda
}

T = {
    ('Lon','Arn'): 1000,
    ('Lon','Gou'): 2.5,
    ('Ber','Arn'): 2.5,
    ('Ber','Gou'): 1000,
    ('Maa','Arn'): 1.6,
    ('Maa','Gou'): 2.0,
    ('Ams','Arn'): 1.4,
    ('Ams','Gou'): 1.0,
    ('Utr','Arn'): 0.8,
    ('Utr','Gou'): 1.0,
    ('Hag','Arn'): 1.4,
    ('Hag','Gou'): 0.8
}

# Define index sets
CUS = list(Demand.keys())
SRC = list(Supply.keys())

# функция, возвращающая границы
def f_bounds(model, i, j):
	return (0, 249)
	
# функция ограничений на потребности, c - индекс
def constr_demand(model, c):
	return sum([model.x[c, s] for s in SRC]) == Demand[c]

#def constr_demand(model, c):
#	return sum([model.x[c, s] ** 2 for s in SRC]) == Demand[c]

# функция цели
def goal(model):
    # return sum([T[c,s] * model.x[c,s] for c in CUS for s in SRC])
    return sum([T[c,s] * model.x[c,s]**2 for c in CUS for s in SRC])

#####################################################################################
# Create an instance of the model
model = pe.ConcreteModel()
model.dual = pe.Suffix(direction = pe.Suffix.IMPORT)

# Define the decision 
model.x = pe.Var(CUS, SRC, domain = pe.NonNegativeIntegers, bounds = f_bounds)

# Define Objective
model.Cost = pe.Objective(
	#expr = sum([T[c,s] * model.x[c,s] for c in CUS for s in SRC]),
	rule = goal,
	sense = pe.minimize
)

# Constraints
model.src = pe.ConstraintList()
for s in SRC:
    model.src.add(sum([model.x[c,s] for c in CUS]) <= Supply[s])
        
model.dmd = pe.Constraint(CUS, rule = constr_demand)

# model.dmd = pe.ConstraintList()
# for c in CUS:
#     model.dmd.add(sum([model.x[c,s] for s in SRC]) == Demand[c])

model.write('c:\\temp\\problem.nl')
##################################################################################3
results = pe.SolverFactory('glpk').solve(model)
results = pe.SolverFactory('cbc').solve(model)

results = pe.SolverFactory('ipopt').solve(model)

results = pe.SolverFactory('bonmin').solve(model)
results = pe.SolverFactory('couenne').solve(model)
results = pe.SolverFactory('shot').solve(model, keepfiles=True)
results = pe.SolverFactory('mindtpy').solve(model, mip_solver='cbc', nlp_solver='ipopt') 

results = pe.SolverFactory('symphony').solve(model)

results.write()

# Solution
for c in CUS:
    for s in SRC:
        print(c, s, model.x[c, s]())
		
# Goal
print(model.Cost())

# Duals
model.dual.display()

model.x.domain = pe.Reals
# model.x.domain = pe.NonNegativeIntegers

for c in CUS:
  for s in SRC:
    model.x[c, s].fixed = False
    model.x[c, s].setlb(model.x[c, s]() - 1e-6)
    model.x[c, s].setub(model.x[c, s]() + 1e-6)
    print(c, s, model.x[c, s].fixed, model.x[c, s].domain, model.x[c, s].lb, model.x[c, s].ub)

results = pe.SolverFactory('cbc').solve(model)
model.dual.display()


for c in model.component_objects(pe.Constraint, active = True):
	print ("   Constraint", c)
	for index in c:
		dual = float(model.dual[c[index]])
#		if (dual != 0):
		print ("      ", index, dual)

#################################################################################
from pyomo.environ import *
model = ConcreteModel()

model.x = Var(bounds=(1.0,10.0),initialize=5.0)
model.y = Var(within=pe.Binary)
model.c1 = Constraint(expr=(model.x-4.0)**2 - model.x <= 50.0*(1-model.y))
model.c2 = Constraint(expr=model.x*log(model.x)+5.0 <= 50.0*(model.y))
model.objective = Objective(expr=model.x, sense=minimize)

res = SolverFactory('mindtpy').solve(model, mip_solver='glpk', nlp_solver='ipopt') 
res.write()

