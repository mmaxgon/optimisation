import matplotlib.pyplot as plt
import numpy as np
import torch
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
	return max(0, x - 4.5) ** 2

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
