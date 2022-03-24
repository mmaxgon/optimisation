import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import pickle
###################################################################
x = torch.tensor(data=[0., 2., 2.], dtype=torch.float, requires_grad=True)
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
	return obj(x) + penalty * (torch.matmul(torch.ones(2), torch.relu(non_lin_cons_fun(x))) + torch.abs(lin_cons_fun(x)))
print(goal(x))

opt = torch.optim.Adam([x])

for i in range(int(1e3)):
	val = goal(x)
	print(val.data.item())
	opt.zero_grad()
	val.backward()
	opt.step()
	print(x.data)

###################################################################
# "теоретическая" функция зависимости объёма продаж от цены
prices = np.arange(1, 10, 0.1)
sales = -(prices - 5)**2 + 50
plt.plot(prices, sales)

# генерим "исторический" датасет со случайной ошибкой
np.random.seed(seed=12345)
x = np.random.choice(list(prices), 1000, replace=True)
y = -(x - 5)**2 + 50 + np.random.normal(loc=0.0, scale=1.0, size=1000)
plt.scatter(x, y)

###################################################################
# минимизируем ошибку прогноза

# создаём прогнозную нейросеть
class NNet(torch.nn.Module):
	def __init__(self):
		super(NNet, self).__init__()
		# коэффициенты при 1-ой и 2-ой степени полинома
		self.poly_coeff = torch.nn.Linear(in_features = 2, out_features = 1, bias = True)
	def forward(self, x):
		powers = torch.stack([x, x**2], dim = 1)
		poly = self.poly_coeff(powers)
		return poly
	
nnet = NNet()
# Оптимизируем по весам
opt = torch.optim.Adam(nnet.parameters())
loss_func = torch.nn.MSELoss()

batch_size = 10
num_epochs = 300
for epoch in range(num_epochs):
	for i in range(len(x)//batch_size):
		# входные переменные
		layer_x = torch.tensor([x[i] for i in range(i, i + batch_size)], dtype=torch.float32)
		# выходные переменные
		layer_y = torch.tensor([y[i] for i in range(i, i + batch_size)], dtype=torch.float32)
		# скрытые слои
		pred = nnet(layer_x).squeeze()
		# потери
		loss = loss_func(layer_y, pred)
		# обучаем
		print(loss.data.item())
		opt.zero_grad()
		loss.backward()
		opt.step()

layer_x = torch.tensor([x[i] for i in range(len(x))], dtype=torch.float32)
pred = nnet(layer_x).squeeze()
plt.scatter(layer_x.detach().numpy(), pred.detach().numpy(), color="blue")
plt.scatter(x, y, color="green", s=1)

###################################################################
# максимизируем продажи (по цене)

# переводим модель в режим скоринга (для корректной работы со всякими dropout-слоями)
nnet.eval()
# фиксируем веса нейросети
for par in nnet.parameters():
	par.requires_grad = False

# а цены меням
layer_x = torch.tensor([1.], requires_grad=True)
# Оптимизируем по цене!!!
opt = torch.optim.Adam([layer_x])

# ограничение x <= 4.5
def cons(x):
	return np.max([0, x - 4.5]) ** 2

# # целочисленное ограничение
# def cons_int(x):
# 	return torch.min(torch.ceil(x) - x, x - torch.floor(x)) ** 2

for epoch in range(10000):
	opt.zero_grad()
	# goal = -сумма продаж
	goal = -(nnet(layer_x).squeeze()) + 1e2 * cons(layer_x) 
		
	print(-goal.data.item())
	goal.backward()
	opt.step()

print(layer_x.detach().numpy()[0])

plt.plot(prices, sales, color="green")
plt.axvline(x = layer_x.detach().numpy()[0], color = "black", label = "Оптимум")
