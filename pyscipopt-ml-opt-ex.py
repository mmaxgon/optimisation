import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import pandas as pd
import numpy as np

from pyscipopt import Model
from pyscipopt_ml.add_predictor import add_predictor_constr

# Фиксируем random seeds для воспроизводимости
torch.manual_seed(42)
np.random.seed(42)

##################################################################################
# Подготовка данных для обучения прогнозной модели
##################################################################################

# Читаем датасет
df = pd.read_csv('car_sales.csv')
print(df)

# Будем использовать эти фичи -- характиристики машин
features = [
    'Vehicle_type',
    'Engine_size',
    'Horsepower',
    'Wheelbase',
    'Width',
    'Length',
    'Curb_weight',
    'Fuel_capacity',
    'Fuel_efficiency',
    'Power_perf_factor',
]

##################################################################################
# Прогнозная модель -- pytorch
##################################################################################

# Готовим датасет для обучения
# target
sales = torch.tensor(df['Sales_in_thousands'], dtype=torch.float32)
# predictors = features + price
df1 = df[features + ['Price_in_thousands']].copy()
# Дискретная переменная 1: 'Passenger', 0: 'Car'
df1['Vehicle_type'] = (df1['Vehicle_type'] == 'Passenger')
df1['Vehicle_type'] = df1['Vehicle_type'].astype("float")
# print(df1.dtypes)
X = torch.tensor(df1.to_numpy(), dtype=torch.float32)
# print(X)
print(X.shape)

# Создаём модель прогнозирования продаж (torch)
n_inputs = X.shape[1]
layers_sizes = (20, 20, 10)
reg_sales = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, layers_sizes[0]),
            torch.nn.Sigmoid(),
            torch.nn.Linear(layers_sizes[0], layers_sizes[1]),
            torch.nn.Sigmoid(),
            torch.nn.Linear(layers_sizes[1], layers_sizes[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(layers_sizes[2], 1),
        )
# print(reg_sales(X))

# Обучаем модель прогнозирования продаж (torch)
batch_size = X.shape[0]
n_epochs = 2000
n_train = X.shape[0]
n_batches = n_train // batch_size

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(reg_sales.parameters(), lr=0.001, weight_decay=0.0001)

# Early stopping параметры
best_loss = float('inf')
patience = 100
patience_counter = 0

epoch = 0
for epoch in range(n_epochs):
    # print(f"epoch: {epoch}")
    t_loss = 0
    i = 0
    for batch_num in range(n_batches):
        # print(f"batch_num: {batch_num}")
        batch_X = X[batch_num * batch_size: (batch_num + 1) * batch_size, :]
        batch_sales = sales[batch_num * batch_size: (batch_num + 1) * batch_size].flatten()

        sales_pred = reg_sales(batch_X).flatten()
        loss = criterion(sales_pred, batch_sales)
        i += 1
        t_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss = t_loss / i
    print(epoch_loss)

    # Early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

with torch.no_grad():
    final_loss = criterion(sales, reg_sales(X).flatten()).item()
print(f"finished: {final_loss}")

##################################################################################
# Обучение прогнозной модели закончилось -- приступаем к формированию оптимизационной модели
##################################################################################

model = Model()
model.redirectOutput()
model.setParam('limits/time', 300)  # 5 минут таймаут

# Numerical stability parameters
model.setParam('numerics/feastol', 1e-6)      # Feasibility tolerance
model.setParam('numerics/epsilon', 1e-7)      # Epsilon for zero comparisons
model.setParam('numerics/dualfeastol', 1e-6)  # Dual feasibility tolerance

# Цена нашего продукта
# Накладываем ограничения на исторические ценовые диапазоны
lb_price = df['Price_in_thousands'].quantile(0.1)
ub_price = df['Price_in_thousands'].quantile(0.9)
x_price = model.addVar(vtype='C', lb=lb_price, ub=ub_price, name='price')

# Продажи нашего продукта
# Накладываем ограничения на исторические диапазоны объёмов продаж -- обрезаем экстемальные прогнозы модели
lb_sales = df['Sales_in_thousands'].quantile(0.1)
ub_sales = df['Sales_in_thousands'].quantile(0.9)
x_sales = model.addVar(vtype='C', lb=lb_sales, ub=ub_sales, name='sales')

# Выручка
# В SCIP нельзя задавать нелинейные целевые функции,
# поэтому используем трюк с введением доп. переменной
x_revenue = model.addVar(vtype='C', lb=lb_price * lb_sales, ub=ub_price * ub_sales, name='revenue')
model.setObjective(x_revenue, 'maximize')
model.addCons(x_revenue <= x_price * x_sales, name='revenue')

# Создаём переменные решения для нашего продукта, соответствующие его фичам
x_features = []
for feature in features:
    if feature == 'Vehicle_type':
        x_features.append(model.addVar(vtype='B', name=feature))
    else:
        lb = df[feature].quantile(0.1)
        ub = df[feature].quantile(0.9)
        x_features.append(model.addVar(vtype='C', lb=lb, ub=ub, name=feature))

# Мы хотим ограничить по размеру новую машину
i_width = features.index('Width')
i_length = features.index('Length')
max_size = (df['Width'] * df['Length']).quantile(0.3)
model.addCons(x_features[i_width] * x_features[i_length] <= max_size, name='size')

"""
Создаём специальное ограничение, которое связывает целевую переменную ML-модели (прогноз продаж) с целевой переменной решения оптимизационной задачи
через переменные решения входных фичей
"""
# Ограничения, которые определяют переменную x_sales
add_predictor_constr(
    model, # оптимизационная модель
    reg_sales, # прогнозная модель
    np.array(x_features + [x_price]).reshape((1, -1)), # переменные решения, соотв. предикторам
    np.array([[x_sales]]), # переменная решения, соотв. прогнозу
    epsilon=0.001,  # Увеличиваем для лучшей сходимости
    unique_naming_prefix='model_sales_',
)

model.optimize()

# Проверяем статус решения
status = model.getStatus()
print(f"Optimization status: {status}")

if status != 'optimal' and status != 'feasible':
    print(f"Warning: Solution not optimal. Status: {status}")

print('=== Solution info ===')
print(f'Objective value: {model.getObjVal()}')
print(f'Gap: {model.getGap()}')

print('=== Наши переменные ===')
old_vars = []
for v in model.getVars():
    if 'model_' not in str(v):
        old_vars.append(v)
        print(v, v.vtype())
print(len(old_vars))

print('=== Переменные, которые добавил фреймворк ===')
new_vars = []
for v in model.getVars():
    if 'model_' in str(v):
        new_vars.append(v)
        print(v, v.vtype())
print(len(new_vars))

##################################################################################
# Посмотрим на данные и на наш результат
##################################################################################

df1 = pd.DataFrame(data={
    'revenue': df['Sales_in_thousands'] * df['Price_in_thousands'],
    'sales': df['Sales_in_thousands'],
    'price': df['Price_in_thousands'],
    'size': df['Width'] * df['Length'],
})
df1 = df1.sort_values(['revenue'], ascending=False)

entries = []
for i, feature in enumerate(features):
    if feature == 'Vehicle_type':
        label = ['Car', 'Passenger'][int(round(model.getVal(x_features[i])))]
        entries.append((feature, label, df[feature].min(), df[feature].max()))
    else:
        entries.append((feature, model.getVal(x_features[i]), df[feature].quantile(0.1), df[feature].quantile(0.9)))
our_features = pd.DataFrame(data=entries, columns=['feature', 'value', 'dataset_lb', 'dataset_ub'])
sales = model.getVal(x_sales)
price = model.getVal(x_price)
revenue = model.getVal(x_sales) * model.getVal(x_price)
size = model.getVal(x_features[features.index('Width')] * x_features[features.index('Length')])

print('=== Топ 10 продаж из датасета ===')
print(df1.head(10))
print()
print('=== Ожидаемые продажи нашей машины ===')
print(f'revenue {revenue:.2f}, sales {sales:.2f}, price {price:.2f}, size {size:.2f}')
print()
print('=== Параметры нашей машины ===')
print(our_features)

##################################################################################
# Другой подход -- парсинг формулы нейронки
##################################################################################

