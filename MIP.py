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

