import numpy as np
import torch
import copy
###################################################################
def do_armijo_step(x0, fx0, gradx0, goal_func):
	grad_norm = torch.norm(gradx0).data.item()
	alpha = 1
	xn = x0 - alpha * gradx0
	fxn = goal(xn).data.item()
	while np.isinf(fxn) or fx0 - fxn < 0.5 * alpha * grad_norm:
		alpha /= 2
		xn = x0 - alpha * gradx0
		fxn = goal(xn).data.item()
	return alpha
###################################################################
x = torch.tensor(data=[0., 0., 0.], dtype=torch.float, requires_grad=True)
print(x)

def obj(x):
	return -(1000 - x[0]**2 - 2*x[1]**2 - x[2]**2 - x[0]*x[1] - x[0]*x[2])
print(obj(x))

def non_lin_cons_fun(x):
	return torch.stack((x[0]**2 + x[1]**2 + x[2]**2 - 25, x[1]**2 + x[2]**2 - 12))
print(non_lin_cons_fun(x))

def lin_cons_fun(x):
	return torch.matmul(torch.tensor([8, 14, 7], dtype=torch.float), x) - 56
print(lin_cons_fun(x))

penalty = 1e3
def goal(x):
	return obj(x) + penalty * (torch.matmul(torch.ones(2), torch.relu(non_lin_cons_fun(x))**2) + lin_cons_fun(x)**2)
print(goal(x))

# CUSTOM Armijo
x = torch.tensor(data=[0., 0., 0.], dtype=torch.float, requires_grad=True)
for i in range(int(1e3)):
	x = torch.tensor(data=x.data, dtype=torch.float, requires_grad=True)
	print("data: {0}".format(x.data))
	val = goal(x)
	print("goal: {0}, objective: {1}".format(val.data.item(), obj(x)))
	x.grad = torch.zeros(x.shape)
	val.backward()
	print("gradient: {0}".format(x.grad))
	alpha = do_armijo_step(x0=copy.copy(x), fx0=val.data.item(), gradx0=copy.copy(x.grad), goal_func=goal)
	if alpha < 1e-9:
		break
	x = x - alpha * x.grad

# USUAL Adam
x = torch.tensor(data=[0., 0., 0.], dtype=torch.float, requires_grad=True)
opt = torch.optim.Adam([x], lr=0.1)
for i in range(int(1e3)):
	val = goal(x)
	print("goal: {0}, objective: {1}".format(val.data.item(), obj(x)))
	opt.zero_grad()
	val.backward()
	opt.step()
	print("data: {0}".format(x.data))
	print("gradient: {0}".format(x.grad))
