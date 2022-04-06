import mip as mip

m = mip.Model(name = "knapsacl", solver_name = mip.CBC)

p = [10, 13, 18, 31, 7, 15]
w = [11, 15, 20, 35, 10, 33]
c = 47
I = range(len(w))

x = [m.add_var(var_type = mip.BINARY) for i in I]

const1 = m.add_constr(mip.xsum(w[i] * x[i] for i in I) <= c)

obj = m.objective = mip.maximize(mip.xsum(p[i] * x[i] for i in I))

res = m.optimize()

selected = [i for i in I if x[i].x >= 0.99]
print("selected items: {}".format(selected))
############################################################################
import mip as mip

model_mip = mip.Model(name="MIP_POA", solver_name="CBC")

#  (‘B’) BINARY, (‘C’) CONTINUOUS and (‘I’) INTEGER
model_mip.y = [model_mip.add_var(name="y", lb=0., ub=10., var_type="C")]
model_mip.x = [model_mip.add_var(name="x{0}".format(i), lb=0., ub=10., var_type="C") for i in range(2)]
xvars = [model_mip.y[0], model_mip.x[0], model_mip.x[1]]

model_mip.lin_cons = model_mip.add_constr(8 * model_mip.y[0] + 14 * model_mip.x[0] + 7 * model_mip.x[1] - 56 == 0)

model_mip.objective = mip.minimize(model_mip.y[0])

res = model_mip.optimize()

if res.value == res.INFEASIBLE.value:
	print("No solution!")
else:
	if len(model_mip.objective.expr.values()) > 0:
		print(model_mip.objective_value)
	print([x.x for x in xvars])

model_mip.remove([model_mip.lin_cons])