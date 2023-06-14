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


# функция, возвращающая границы
def f_bounds(model, i, j):
	return (0, 249)


# функция ограничений на потребности, c - индекс
def constr_demand(model, c):
	return sum([model.x[c, s] for s in SRC]) == Demand[c]


# def constr_demand(model, c):
#	return sum([model.x[c, s] ** 2 for s in SRC]) == Demand[c]

# функция цели
def goal(model):
	return sum([T[c, s] * model.x[c, s] for c in CUS for s in SRC])
	# return sum([T[c,s] * model.x[c,s]**2 for c in CUS for s in SRC])


#####################################################################################
# Create an instance of the model
model = pe.ConcreteModel()
model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

# Define the decision 
model.x = pe.Var(CUS, SRC, domain=pe.NonNegativeIntegers, bounds=f_bounds)
model.y = pe.Var(CUS, SRC, domain=pe.Binary)

NBIG = 1e3
model.logical = pe.ConstraintList()
for c in CUS:
	for s in SRC:
		# model.y[c, s] == 0 => model.x <= 0
		model.logical.add(expr=-NBIG * model.y[c, s] + model.x[c, s] <= 0)
		# model.y[c, s] == 0 => model.x >= 0
		model.logical.add(expr=NBIG * model.y[c, s] + model.x[c, s] >= 0)
		# model.y[c, s] == 1 => model.x > 0
		model.logical.add(expr=NBIG * (1 - model.y[c, s]) + model.x[c, s] >= 1)

# Define Objective
model.Cost = pe.Objective(
	# expr = sum([T[c,s] * model.x[c,s] for c in CUS for s in SRC]),
	rule=goal,
	sense=pe.minimize
)

# Constraints
model.src = pe.ConstraintList()
for s in SRC:
	model.src.add(sum([model.x[c, s] for c in CUS]) <= Supply[s])

model.dmd = pe.Constraint(CUS, rule=constr_demand)

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
		print(c, s, model.x[c, s](), model.y[c, s]())

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

for c in model.component_objects(pe.Constraint, active=True):
	print("   Constraint", c)
	for index in c:
		dual = float(model.dual[c[index]])
		#		if (dual != 0):
		print("      ", index, dual)

#################################################################################
from pyomo.environ import *

model = ConcreteModel()

model.x = Var(bounds=(1.0, 10.0), initialize=5.0)
model.y = Var(within=pe.Binary)
model.c1 = Constraint(expr=(model.x - 4.0) ** 2 - model.x <= 50.0 * (1 - model.y))
model.c2 = Constraint(expr=model.x * log(model.x) + 5.0 <= 50.0 * (model.y))
model.objective = Objective(expr=model.x, sense=minimize)

res = SolverFactory('mindtpy').solve(model, mip_solver='glpk', nlp_solver='ipopt')
res.write()

########################################################################################
# piecewise
########################################################################################
import pyomo.environ as pyomo
from matplotlib import pyplot as plt

xdata = [1., 3., 6., 10.]
ydata = [6.,2.,8.,7.]

plt.plot(xdata, ydata)
plt.show()

############################
# Piecewise
model = pyomo.ConcreteModel()
model.X = pyomo.Var(domain=pyomo.Reals, bounds=(1, 10))
model.Y = pyomo.Var(domain=pyomo.Reals, bounds=(0, 100))

model.con = pyomo.Piecewise(
	model.Y,
	model.X,
	pw_pts=xdata,
	pw_constr_type='EQ',
	f_rule=ydata,
	pw_repn='CC'
)

model.obj = pyomo.Objective(expr=model.Y, sense=pyomo.maximize)

results = pyomo.SolverFactory('glpk').solve(model)
# results = pyomo.SolverFactory('cbc', executable="C:\\Program Files\\solvers\\CBC\\bin\\cbc.exe").solve(model)
# results = pyomo.SolverFactory('cplex').solve(model)

print(model.X.value, model.Y.value)

##########################
# Piecewise linear with NBIG
##########################
NBIG = 100
xdata = [1., 3., 6., 10.]
ydata = [6.,2.,8.,7.]
ix = range(len(xdata) - 1)
ix2 = range(len(ix) * 2)

plt.plot(xdata, ydata)
plt.show()

# Piecewise
model = pyomo.ConcreteModel()
model.X = pyomo.Var(domain=pyomo.Reals, bounds=(1, 10))
model.Y = pyomo.Var(domain=pyomo.Reals, bounds=(0, 100))

# флаг в каком из диапазонов находится X
model.b = pyomo.Var(ix, domain=pyomo.Binary)

# Пары l[2*i], l[2*i+1], в сумме дающие b[i]:
# Представляют собой коэффициенты выпуклой комбинации для соседних точек X и Y активного диапазона
# x[i] = l[2*i] * xdata[i] + l[2*i+1] * xdata[i+1],
# y[i] = l[2*i] * ydata[i] + l[2*i+1] * ydata[i+1]
model.l = pyomo.Var(ix2, domain=pyomo.NonNegativeReals, bounds=(0, 1))

# только один b[i] = 1, и в этом диапазоне находится x:
model.b_cons = pyomo.Constraint(expr=sum(model.b[i] for i in ix) == 1)
model.b_cons.pprint()

# x и y являются выпуклой комбинацией соседних точек с коэффициентами l[2*i] и l[2*i+1]
model.l_cons = pyomo.ConstraintList()
model.l_cons.clear()
# x и y являются выпуклой комбинацией всех точек с коэффициентами l[2*i] и l[2*i+1] (только 2 соседних ненулевые)
model.l_cons.add(expr=model.X == sum(xdata[i]*model.l[2*i] + xdata[i+1]*model.l[2*i+1] for i in ix))
model.l_cons.add(expr=model.Y == sum(ydata[i]*model.l[2*i] + ydata[i+1]*model.l[2*i+1] for i in ix))
for i in ix:
	# только 2 соседних коэффициента l[2*i] и l[2*i+1] отличны от 0 (дают в сумме 1) и определяются сотв. диапазоном b[i]
	expr = model.l[2*i] + model.l[2*i+1] == model.b[i]
	model.l_cons.add(expr)
model.l_cons.pprint()

model.obj = pyomo.Objective(expr=model.Y, sense=pyomo.maximize)
# model.obj = pyomo.Objective(expr=model.Y, sense=pyomo.minimize)

results = pyomo.SolverFactory('glpk').solve(model)
# results = pyomo.SolverFactory('cbc', executable="C:\\Program Files\\solvers\\CBC\\bin\\cbc.exe").solve(model)
# results = pyomo.SolverFactory('cplex').solve(model)

print([model.b[i].value for i in ix])
print([model.l[i].value for i in ix2])
print(model.X.value, model.Y.value)

##########################
# Экспорт задачи из pyomo
##########################
from pyomo.repn.standard_repn import generate_standard_repn
for c in CUS:
	repn = generate_standard_repn(model.dmd[c].body)
	for var in repn.linear_vars:
		print(var)
	print(c, repn.linear_coefs)
	print(generate_standard_repn(model.dmd[c].lb).constant)
