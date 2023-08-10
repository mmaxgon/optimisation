import numpy as np
import torch
import copy
###################################################################
def do_armijo_step(x0, fx0, gradx0, goal_func):
	grad_norm2 = (sum(gradx0 ** 2)).data.item()
	alpha = 1
	xn = x0 - alpha * gradx0
	fxn = goal_func(xn).data.item()
	while np.isinf(fxn) or fx0 - fxn < 0.5 * alpha * grad_norm2:
		alpha /= 2
		xn = x0 - alpha * gradx0
		fxn = goal(xn).data.item()
	return alpha
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
# CUSTOM Armijo
x = torch.tensor(data=[0.]*n, dtype=torch.float, requires_grad=True)
v = torch.tensor([0.]*k)
u = torch.tensor([0.]*m)
for i in range(int(1e3)):
	x = x.clone().detach().requires_grad_(True)
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
u, v = renew_lagrange(u, v, x)
print("u: {0}, v: {1}".format(u, v))
print("eq_cons: {0} ineq_cons: {1}".format(eq_cons_fun(x), ineq_cons_fun(x)))
d = torch.norm(x.grad).data.item()
print("grad norm: {0}".format(d))

# USUAL Adam
x = torch.tensor(data=[0.]*n, dtype=torch.float, requires_grad=True)
v = torch.tensor([0.]*k)
u = torch.tensor([0.]*m)
opt = torch.optim.Adam([x], lr=0.01)
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

##########################################################################
# Целочисленные ограничения
##########################################################################
"""
Кодируем целочисленные переменные в вектора длины их диапазона - one-hot-encoding соответствующего целого числа.
Для кодирования используем дифференцируемый вариант soft argmax - взвешиваем значения экспонентами.
Из векторов кодировок умножая на [0,1,2,3,..] получаем целые числа, которые попадают во все ограничения и функцию цели.

Оптимальное решение: 
'obj': -977.9375, 'x': [1.75, 2.0, 2.0]

Получается допустимое решение:
'obj': -976.4530, 'x': [2.5984, 2.0000, 1.0000]
"""

c = torch.tensor([0., 1., 2., 3])
y = torch.tensor([0.], requires_grad=True)
print(y)
z = torch.rand((2, 4), requires_grad=True)
print(z)

def collect_x(y, z):
	print(y, z)
	z_1 = torch.nn.functional.gumbel_softmax(z, dim=1, tau=1e-2, hard=False)
	# z_0 = torch.exp(10 * z)
	# z_1 = torch.matmul(torch.diag(1 / torch.sum(z_0, dim=1)), z_0)
	print(z_1)
	z_2 = torch.matmul(c, torch.t(z_1))
	print(z_2)
	x = torch.cat([y, z_2])
	print(x)
	return x

v = torch.tensor([0.]*k)
u = torch.tensor([0.]*m)
opt = torch.optim.Adam([y, z], lr=0.01)
for i in range(int(1e3)):
	x = collect_x(y, z)
	val = goal(x)
	print("goal: {0}, objective: {1}".format(val.data.item(), obj(x)))
	opt.zero_grad()
	val.backward()
	opt.step()
	print("data: {0}".format(x.data))
	print("gradient: {0}{1}".format(y.grad, z.grad))
u, v = renew_lagrange(u, v, x)
print("u: {0}, v: {1}".format(u, v))
print("eq_cons: {0} ineq_cons: {1}".format(eq_cons_fun(x), ineq_cons_fun(x)))
print("grad norm: {0}".format(torch.norm(torch.cat([y.grad, z.grad.flatten()]))))


