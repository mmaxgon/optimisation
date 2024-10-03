import numpy as np
import torch
import copy

###################################################################
def max0(x, if_diff=True):
	if if_diff:
		res = torch.nn.functional.softplus(x, beta=1e3, threshold=1e-1)
	else:
		res = torch.relu(x)
	return res

###################################################################
n = 3 # число переменных
m = 2 # число ограничений-неравенств
k = 1 # число ограничений-равенств

x = torch.tensor(data=[0.]*n, dtype=torch.float, requires_grad=True)
print(x)

def obj(x):
	return -(1000 - x[0]**2 - 2*x[1]**2 - x[2]**2 - x[0]*x[1] - x[0]*x[2])
print(obj(x))

def ineq_cons_fun(x):
	return torch.stack((x[0]**2 + x[1]**2 + x[2]**2 - 25, x[1]**2 + x[2]**2 - 12))
print(ineq_cons_fun(x))

def eq_cons_fun(x):
	return torch.matmul(torch.tensor([8, 14, 7], dtype=torch.float), x) - 56
print(eq_cons_fun(x))

penalty = 1e1
v = torch.tensor([0.]*k)
u = torch.tensor([0.]*m)

def goal(x):
	return obj(x) + \
		(1/(2*penalty)) * \
		(
			torch.sum(max0(penalty * ineq_cons_fun(x) + u, if_diff=True)**2) +
			torch.sum((penalty * eq_cons_fun(x) + v)**2)
		)
print(goal(x))

def renew_lagrange(u, v, x):
	u = max0(u + penalty * ineq_cons_fun(x), if_diff=False).clone().detach()
	v = (v + penalty * eq_cons_fun(x)).clone().detach()
	return (u, v)

"""
Решение:
'x': array([0.60295854, 2.48721496, 2.33647461]), 'obj': -978.8963660834414
"""
# USUAL Adam
x = torch.tensor(data=[0.]*n, dtype=torch.float, requires_grad=True)
v = torch.tensor([0.]*k)
u = torch.tensor([0.]*m)
opt = torch.optim.Adam([x], lr=0.01)
# Нужен внешний цикл!
for i in range(int(1e3)):
	val = goal(x)
	print("goal: {0}, objective: {1}".format(val.data.item(), obj(x)))
	opt.zero_grad()
	val.backward()
	opt.step()
	print("data: {0}".format(x.data))
	print("gradient: {0}".format(x.grad))
(u, v) = renew_lagrange(u, v, x)
print("u: {0}, v: {1}".format(u, v))
print("eq_cons: {0} ineq_cons: {1}".format(eq_cons_fun(x), ineq_cons_fun(x)))
d = torch.norm(x.grad).data.item()
print("grad norm: {0}".format(d))
