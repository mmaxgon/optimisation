"""
MINLP Price Optimization with Polynomial Demand Forecast (Degree <= 2)
======================================================================

Вариант с полиномиальной моделью спроса (степень <= 2).

Вместо стандартных слоёв нейросети используется фиксированная полиномиальная формула:
    demand = W0 + W1*x + W2*x^2 + W3*x*y + ...

где x - входные переменные (цены, промо, контекст)
      W - обучаемые коэффициенты

Архитектура:
1. Функция генерации возвращает данные + все параметры для обучения и оптимизации
2. Обучение полиномиальной модели
3. Оптимизация (использует выгрузку формулы из обученной модели)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from pyscipopt import Model, quicksum
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from itertools import combinations_with_replacement


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NormalizationParams:
    """Параметры нормализации."""
    input_means: np.ndarray
    input_stds: np.ndarray
    output_max: np.ndarray


@dataclass
class ProblemParams:
    """Параметры задачи для оптимизатора."""
    n_products: int
    n_context_cont: int
    n_context_bin: int
    costs: np.ndarray
    price_ranges: List[Tuple[float, float]]
    max_demands: np.ndarray  # Для денормализации


# =============================================================================
# DATA GENERATION FUNCTION
# =============================================================================

def generate_training_data(n_products: int = 3,
                          n_context_cont: int = 2,
                          n_context_bin: int = 2,
                          n_samples: int = 2000,
                          seed: int = 42) -> Dict:
    """Генерирует обучающие данные и все необходимые параметры."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_demands = np.random.uniform(50, 200, n_products)
    costs = np.random.uniform(10, 50, n_products)

    price_ranges = []
    for i in range(n_products):
        min_price = costs[i] * 1.2
        max_price = costs[i] * 4
        price_ranges.append((min_price, max_price))

    price_elasticities = np.random.uniform(1.2, 2.5, n_products)
    cross_elasticities = np.random.uniform(0.1, 0.3, (n_products, n_products))
    np.fill_diagonal(cross_elasticities, 0)

    context_effects = {
        'continuous': np.random.uniform(-0.5, 0.5, (n_context_cont, n_products)),
        'binary': np.random.uniform(0.1, 0.5, (n_context_bin, n_products))
    }

    prices = np.zeros((n_samples, n_products))
    for i in range(n_products):
        min_p, max_p = price_ranges[i]
        prices[:, i] = np.random.uniform(min_p, max_p, n_samples)

    context_cont = np.random.uniform(-1, 1, (n_samples, n_context_cont))
    context_bin = np.random.randint(0, 2, (n_samples, n_context_bin)).astype(float)

    inputs = np.concatenate([prices, context_cont, context_bin], axis=1)

    demands = np.zeros((n_samples, n_products))
    for b in range(n_samples):
        avg_prices = np.array([(r[0] + r[1]) / 2 for r in price_ranges])
        price_ratios = prices[b] / avg_prices

        for i in range(n_products):
            demand = base_demands[i] * (price_ratios[i] ** (-price_elasticities[i]))
            for j in range(n_products):
                if i != j:
                    demand *= (price_ratios[j] ** cross_elasticities[i, j])
            for c in range(n_context_cont):
                effect = 1 + context_effects['continuous'][c, i] * context_cont[b, c]
                demand *= max(0.5, min(1.5, effect))
            for c in range(n_context_bin):
                if context_bin[b, c] > 0.5:
                    demand *= (1 + context_effects['binary'][c, i])
            demands[b, i] = max(0, demand)

    price_means = np.array([(r[0] + r[1]) / 2 for r in price_ranges])
    price_stds = np.array([(r[1] - r[0]) / 4 for r in price_ranges])

    input_means = np.concatenate([
        price_means,
        np.zeros(n_context_cont),
        np.zeros(n_context_bin)
    ])
    input_stds = np.concatenate([
        price_stds,
        np.ones(n_context_cont),
        np.ones(n_context_bin)
    ])

    max_demands = base_demands * 3

    return {
        'train_inputs': torch.FloatTensor(inputs),
        'train_outputs': torch.FloatTensor(demands),
        'norm_params': NormalizationParams(
            input_means=input_means,
            input_stds=input_stds,
            output_max=max_demands
        ),
        'problem_params': ProblemParams(
            n_products=n_products,
            n_context_cont=n_context_cont,
            n_context_bin=n_context_bin,
            costs=costs,
            price_ranges=price_ranges,
            max_demands=max_demands
        )
    }


# =============================================================================
# POLYNOMIAL DEMAND MODEL (WORKING VERSION)
# =============================================================================

class PolynomialDemandModel(nn.Module):
    """
    Полиномиальная модель спроса с ограничениями на квадратичные члены.

    Формула для каждого выхода i:
    y_i = bias[i]
          + sum(j) linear[i,j] * x_j
          + sum(j not in prices) quadratic[i,j] * x_j^2  # НЕТ квадратичных членов для цен!
          + sum(j<k) interaction[i,j,k] * x_j * x_k

    ВАЖНО: Квадратичные члены НЕ используются для цен (prices), только для контекста.
    Это предотвращает нереалистичный рост спроса при увеличении цены.
    """

    def __init__(self, input_dim: int, output_dim: int, n_prices: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_prices = n_prices

        self.linear = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)

        n_quad_vars = input_dim - n_prices
        self.quadratic = nn.Parameter(torch.randn(output_dim, n_quad_vars) * 0.01)

        n_pairs = input_dim * (input_dim - 1) // 2
        self.interaction = nn.Parameter(torch.randn(output_dim, n_pairs) * 0.01)

        self.bias = nn.Parameter(torch.randn(output_dim) * 0.1)

        self._pair_indices = [(j, k) for j in range(input_dim) for k in range(j + 1, input_dim)]
        self._quadratic_indices = list(range(n_prices, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        linear_term = torch.matmul(x, self.linear.T)

        x_quad = x[:, self._quadratic_indices] ** 2
        quadratic_term = torch.matmul(x_quad, self.quadratic.T)

        interaction_term = torch.zeros(batch_size, self.output_dim, device=x.device)
        for pair_idx, (j, k) in enumerate(self._pair_indices):
            interaction_term += (x[:, j] * x[:, k]).unsqueeze(1) * self.interaction[:, pair_idx].unsqueeze(0)

        output = self.bias + linear_term + quadratic_term + interaction_term
        return output

    def get_coefficients(self) -> Dict[str, np.ndarray]:
        return {
            'linear': self.linear.detach().numpy().copy(),
            'quadratic': self.quadratic.detach().numpy().copy(),
            'quadratic_indices': self._quadratic_indices,
            'interaction': self.interaction.detach().numpy().copy(),
            'bias': self.bias.detach().numpy().copy(),
            'pair_indices': self._pair_indices,
            'n_prices': self.n_prices
        }


# =============================================================================
# TRAINING
# =============================================================================

def train_demand_network(demand_net: nn.Module,
                         train_inputs: torch.Tensor,
                         train_outputs: torch.Tensor,
                         norm_params: NormalizationParams,
                         epochs: int = 500,
                         lr: float = 0.01) -> nn.Module:
    """Обучает модель."""
    print("="*70)
    print("ОБУЧЕНИЕ МОДЕЛИ СПРОСА")
    print("="*70 + "\n")

    inputs_norm = (train_inputs - torch.FloatTensor(norm_params.input_means)) / torch.FloatTensor(norm_params.input_stds)
    outputs_norm = train_outputs / torch.FloatTensor(norm_params.output_max)

    optimizer = torch.optim.Adam(demand_net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience = 100
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = demand_net(inputs_norm)
        loss = criterion(predictions, outputs_norm)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    print(f"\n[OK] Модель обучена (финальный loss: {loss.item():.6f})\n")
    return demand_net


# =============================================================================
# OPTIMIZATION WITH POLYNOMIAL FORMULA
# =============================================================================

class PriceOptimizerPolynomial:
    """МИП/МИНЛП оптимизатор цен с использованием полиномиальной модели спроса."""

    def __init__(self,
                 demand_net: PolynomialDemandModel,
                 norm_params: NormalizationParams,
                 problem_params: ProblemParams,
                 min_profit_margin: float = 0.20):
        self.demand_net = demand_net
        self.norm_params = norm_params
        self.problem_params = problem_params
        self.min_profit_margin = min_profit_margin
        self.coeffs = demand_net.get_coefficients()

    def _build_polynomial_expression(self, model: Model, input_vars: List,
                                     output_idx: int):
        expr = self.coeffs['bias'][output_idx]

        for j in range(len(input_vars)):
            expr += self.coeffs['linear'][output_idx, j] * input_vars[j]

        for quad_idx, var_idx in enumerate(self.coeffs['quadratic_indices']):
            expr += self.coeffs['quadratic'][output_idx, quad_idx] * (input_vars[var_idx] ** 2)

        for pair_idx, (j, k) in enumerate(self.coeffs['pair_indices']):
            expr += self.coeffs['interaction'][output_idx, pair_idx] * input_vars[j] * input_vars[k]

        return expr

    def optimize(self,
                context_cont: Optional[np.ndarray] = None,
                time_limit: int = 300) -> Dict:
        n_products = self.problem_params.n_products
        n_context_cont = self.problem_params.n_context_cont
        n_context_bin = self.problem_params.n_context_bin

        if context_cont is None:
            context_cont = np.zeros(n_context_cont)

        print("\n" + "="*70)
        print("МИНЛП ОПТИМИЗАЦИЯ ЦЕН (ПОЛИНОМИАЛЬНАЯ МОДЕЛЬ)")
        print("="*70 + "\n")

        model = Model("Price_Optimization_Polynomial")
        model.redirectOutput()
        model.setParam('limits/time', time_limit)
        model.setParam('display/verblevel', 2)
        model.setParam('numerics/feastol', 1e-6)
        model.setParam('numerics/epsilon', 1e-7)

        # Decision variables
        price_vars = []
        for i in range(n_products):
            min_p, max_p = self.problem_params.price_ranges[i]
            var = model.addVar(lb=min_p, ub=max_p, vtype='C', name=f"price_{i}")
            price_vars.append(var)

        context_bin_vars = []
        for i in range(n_context_bin):
            var = model.addVar(vtype='B', name=f"context_bin_{i}")
            context_bin_vars.append(var)

        model.addCons(quicksum(context_bin_vars) <= 1, name="max_one_promo")

        context_cont_vars = []
        for i in range(n_context_cont):
            var = model.addVar(lb=context_cont[i], ub=context_cont[i], vtype='C', name=f"context_cont_{i}")
            context_cont_vars.append(var)

        # Normalized variables
        input_norm_vars = []
        all_input_vars = price_vars + context_cont_vars + context_bin_vars

        for i, var in enumerate(all_input_vars):
            mean = self.norm_params.input_means[i]
            std = self.norm_params.input_stds[i]
            norm_var = model.addVar(vtype='C', name=f"input_norm_{i}")
            model.addCons(norm_var * std == var - mean, name=f"norm_{i}")
            input_norm_vars.append(norm_var)

        # Demand variables via polynomial expression
        print("Построение полиномиальных выражений спроса...")
        demand_norm_vars = []
        for i in range(n_products):
            demand_norm_var = model.addVar(lb=0, ub=1, vtype='C', name=f"demand_norm_{i}")
            poly_expr = self._build_polynomial_expression(model, input_norm_vars, i)
            model.addCons(demand_norm_var == poly_expr, name=f"demand_poly_{i}")
            demand_norm_vars.append(demand_norm_var)

        print(f"[OK] Построено {n_products} полиномиальных выражений\n")

        # Demand denormalization
        demand_vars = []
        for i in range(n_products):
            max_demand = self.norm_params.output_max[i]
            var = model.addVar(lb=0, ub=max_demand, vtype='C', name=f"demand_{i}")
            model.addCons(var == demand_norm_vars[i] * max_demand, name=f"denorm_{i}")
            demand_vars.append(var)

        # Objective and constraints
        revenue_expr = quicksum(price_vars[i] * demand_vars[i] for i in range(n_products))
        cost_expr = quicksum(self.problem_params.costs[i] * demand_vars[i] for i in range(n_products))
        profit_expr = revenue_expr - cost_expr

        model.addCons(profit_expr >= self.min_profit_margin * revenue_expr, name="min_profit_margin")

        max_revenue = sum(self.problem_params.price_ranges[i][1] * self.norm_params.output_max[i]
                         for i in range(n_products))
        revenue_var = model.addVar(lb=0, ub=max_revenue, vtype='C', name='total_revenue')
        model.addCons(revenue_var == revenue_expr, name='revenue_def')
        model.setObjective(revenue_var, sense='maximize')

        # Solve
        print("Информация о модели:")
        print(f"  Переменных: {model.getNVars()}")
        print(f"  Ограничений: {model.getNConss()}")
        print()

        print("Решение оптимизационной задачи...")
        status = model.optimize()

        return self._extract_results(model, price_vars, context_bin_vars, demand_vars, context_cont)

    def _extract_results(self, model, price_vars, context_bin_vars, demand_vars, context_cont) -> Dict:
        results = {
            'status': model.getStatus(),
            'optimal_prices': [],
            'demands': [],
            'revenues': [],
            'profits': [],
            'active_promos': [],
            'total_revenue': 0,
            'total_profit': 0,
            'profit_margin': 0
        }

        if model.getStatus() in ['optimal', 'best solution found', 'feasible']:
            print("\n" + "="*70)
            print("РЕШЕНИЕ НАЙДЕНО")
            print("="*70 + "\n")

            for i in range(self.problem_params.n_products):
                price = model.getVal(price_vars[i])
                demand = model.getVal(demand_vars[i])
                revenue = price * demand
                profit = (price - self.problem_params.costs[i]) * demand

                results['optimal_prices'].append(price)
                results['demands'].append(demand)
                results['revenues'].append(revenue)
                results['profits'].append(profit)

                print(f"Товар {i+1}:")
                print(f"  Цена: {price:.2f}")
                print(f"  Спрос: {demand:.2f}")
                print(f"  Выручка: {revenue:.2f}")
                print(f"  Прибыль: {profit:.2f}")
                print(f"  Себестоимость: {self.problem_params.costs[i]:.2f}\n")

            active_promos = []
            for p in range(self.problem_params.n_context_bin):
                if model.getVal(context_bin_vars[p]) > 0.5:
                    active_promos.append(p)
                    results['active_promos'].append(p)

            if active_promos:
                print(f"Активные промо-механики: {active_promos}\n")
            else:
                print("Активные промо-механики: нет\n")

            results['total_revenue'] = sum(results['revenues'])
            results['total_profit'] = sum(results['profits'])
            results['profit_margin'] = results['total_profit'] / results['total_revenue']

            print(f"ОБЩАЯ ВЫРУЧКА: {results['total_revenue']:.2f}")
            print(f"ОБЩАЯ ПРИБЫЛЬ: {results['total_profit']:.2f}")
            print(f"МАРЖА: {results['profit_margin'] * 100:.1f}%")
            print(f"Ограничение мин. маржи: {self.min_profit_margin * 100:.1f}%")
        else:
            print(f"\nСтатус решения: {model.getStatus()}")

        return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_results(optimizer, results: Dict, title_suffix: str = "Полиномиальная модель") -> None:
    """Визуализирует результаты."""

    n_products = optimizer.problem_params.n_products
    n_context_cont = optimizer.problem_params.n_context_cont
    n_context_bin = optimizer.problem_params.n_context_bin

    fig, axes = plt.subplots(1, n_products, figsize=(15, 4))
    if n_products == 1:
        axes = [axes]

    for i in range(n_products):
        ax = axes[i]
        min_p, max_p = optimizer.problem_params.price_ranges[i]

        prices_range = np.linspace(min_p, max_p, 30)
        demands_range = []

        for price in prices_range:
            inputs = torch.zeros(1, n_products + n_context_cont + n_context_bin)
            inputs[0, :n_products] = torch.FloatTensor(
                [(r[0] + r[1]) / 2 for r in optimizer.problem_params.price_ranges]
            )
            inputs[0, i] = price

            inputs_norm = (inputs - torch.FloatTensor(optimizer.norm_params.input_means)) / torch.FloatTensor(optimizer.norm_params.input_stds)
            with torch.no_grad():
                demand_norm = optimizer.demand_net(inputs_norm)
                demand = demand_norm * torch.FloatTensor(optimizer.norm_params.output_max)
            demands_range.append(demand[0, i].item())

        ax.plot(prices_range, demands_range, 'b-', linewidth=2, label='Спрос (модель)')

        if results['optimal_prices']:
            opt_price = results['optimal_prices'][i]
            opt_demand = results['demands'][i]
            ax.plot(opt_price, opt_demand, 'r*', markersize=15,
                   label=f'Оптимум: {opt_price:.2f}')

        ax.axvline(optimizer.problem_params.costs[i], color='g', linestyle='--',
                  label=f'Себестоимость: {optimizer.problem_params.costs[i]:.2f}')

        ax.set_xlabel('Цена', fontsize=11)
        ax.set_ylabel('Спрос', fontsize=11)
        ax.set_title(f'Товар {i+1}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Кривые спроса и оптимальные цены ({title_suffix})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = f"price_optimization_results_{'simple_nn' if 'NN' in title_suffix else 'polynomial'}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nГрафик сохранен: {filename}")
    plt.show()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Pipeline для полиномиальной модели."""
    n_products = 3
    n_context_cont = 2
    n_context_bin = 2

    print("="*70)
    print("PIPELINE: ГЕНЕРАЦИЯ ДАННЫХ -> ОБУЧЕНИЕ -> ОПТИМИЗАЦИЯ (ПОЛИНОМ)")
    print("="*70 + "\n")

    print("ШАГ 1: Генерация данных\n")

    data = generate_training_data(
        n_products=n_products,
        n_context_cont=n_context_cont,
        n_context_bin=n_context_bin,
        n_samples=3000
    )

    train_inputs = data['train_inputs']
    train_outputs = data['train_outputs']
    norm_params = data['norm_params']
    problem_params = data['problem_params']

    print(f"  Примеров: {train_inputs.shape[0]}")
    print(f"  Размерность входа: {train_inputs.shape[1]}")
    print(f"  Размерность выхода: {train_outputs.shape[1]}")
    print(f"\n  Параметры задачи:")
    print(f"    Товаров: {problem_params.n_products}")
    print(f"    Себестоимости: {problem_params.costs}")
    print(f"    Диапазоны цен: {problem_params.price_ranges}")
    print()

    print("ШАГ 2: Создание и обучение полиномиальной модели\n")

    input_dim = n_products + n_context_cont + n_context_bin

    n_pairs = input_dim * (input_dim - 1) // 2
    n_quad_vars = input_dim - n_products
    n_params = (input_dim + n_quad_vars + n_pairs) * n_products + n_products

    print(f"  Полиномиальная модель (степень <= 2):")
    print(f"    Размерность входа: {input_dim}")
    print(f"    Ценовых переменных (без квадратичных членов): {n_products}")
    print(f"    Неценовых переменных (с квадратичными членами): {n_quad_vars}")
    print(f"    Линейных коэффициентов: {input_dim * n_products}")
    print(f"    Квадратичных коэффициентов: {n_quad_vars * n_products}")
    print(f"    Коэффициентов взаимодействия: {n_pairs * n_products}")
    print(f"    Bias: {n_products}")
    print(f"    Всего параметров: {n_params}")
    print()

    demand_net = PolynomialDemandModel(input_dim=input_dim, output_dim=n_products, n_prices=n_products)

    demand_net = train_demand_network(
        demand_net=demand_net,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        norm_params=norm_params,
        epochs=500,
        lr=0.01
    )

    print("Валидация...")
    val_data = generate_training_data(
        n_products=n_products,
        n_context_cont=n_context_cont,
        n_context_bin=n_context_bin,
        n_samples=500
    )
    val_inputs_norm = (val_data['train_inputs'] - torch.FloatTensor(norm_params.input_means)) / torch.FloatTensor(norm_params.input_stds)
    val_outputs_norm = val_data['train_outputs'] / torch.FloatTensor(norm_params.output_max)

    with torch.no_grad():
        val_preds = demand_net(val_inputs_norm)
        mae = torch.mean(torch.abs(val_preds * torch.FloatTensor(norm_params.output_max) - val_data['train_outputs']))

    print(f"  MAE: {mae.item():.2f}\n")

    print("ШАГ 3: МИНЛП оптимизация (полиномиальная модель)\n")

    optimizer = PriceOptimizerPolynomial(
        demand_net=demand_net,
        norm_params=norm_params,
        problem_params=problem_params,
        min_profit_margin=0.15
    )

    results = optimizer.optimize(time_limit=120)

    if results['optimal_prices']:
        visualize_results(optimizer, results)


# =============================================================================
# ANALYSIS: SIMPLE NEURAL NETWORK WITH SIGMOID (FOR DEBUGGING)
# =============================================================================

class SimpleDemandNet(nn.Module):
    """
    Простая нейросеть с одним скрытым слоем и sigmoid активацией.

    Цель: проанализировать, почему не сходится оптимизация с нелинейными активациями.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 8):
        super().__init__()
        torch.manual_seed(42)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.sigmoid(self.fc1(x))
        y = self.fc2(h)
        return y

    def get_layer_info(self):
        return [
            {'type': 'linear', 'weights': self.fc1.weight.detach().numpy().copy(),
             'bias': self.fc1.bias.detach().numpy().copy(), 'input_dim': self.fc1.in_features, 'output_dim': self.fc1.out_features},
            {'type': 'activation', 'activation': 'sigmoid'},
            {'type': 'linear', 'weights': self.fc2.weight.detach().numpy().copy(),
             'bias': self.fc2.bias.detach().numpy().copy(), 'input_dim': self.fc2.in_features, 'output_dim': self.fc2.out_features}
        ]


class PriceOptimizerSimpleNN:
    """
    Оптимизатор для простой нейросети с sigmoid.
    Использует прямую формулу sigmoid через exp() для отладки.
    """
    def __init__(self, demand_net: SimpleDemandNet, norm_params, problem_params, min_profit_margin=0.15):
        self.demand_net = demand_net
        self.norm_params = norm_params
        self.problem_params = problem_params
        self.min_profit_margin = min_profit_margin
        self.nn_layers = demand_net.get_layer_info()

        print("[ANALYSIS] Информация о слоях:")
        for i, layer in enumerate(self.nn_layers):
            if layer['type'] == 'linear':
                print(f"  Layer {i}: Linear({layer['input_dim']} -> {layer['output_dim']})")
                print(f"    Weights: min={layer['weights'].min():.4f}, max={layer['weights'].max():.4f}")
                print(f"    Bias: min={layer['bias'].min():.4f}, max={layer['bias'].max():.4f}")
        print()

    def optimize(self, context_cont=None, time_limit=120):
        n_products = self.problem_params.n_products
        n_context_cont = self.problem_params.n_context_cont
        n_context_bin = self.problem_params.n_context_bin

        if context_cont is None:
            context_cont = np.zeros(n_context_cont)

        print("\n" + "="*70)
        print("ANALYSIS: SIMPLE NN + SIGMOID (ПРЯМАЯ ФОРМУЛА ЧЕРЕZ EXP)")
        print("="*70 + "\n")

        model = Model("Price_SimpleNN_Sigmoid")
        model.redirectOutput()
        model.setParam('limits/time', time_limit)
        model.setParam('display/verblevel', 1)

        # Variables
        price_vars = [model.addVar(lb=self.problem_params.price_ranges[i][0],
                                      ub=self.problem_params.price_ranges[i][1],
                                      vtype='C', name=f"price_{i}") for i in range(n_products)]
        context_bin_vars = [model.addVar(vtype='B', name=f"context_bin_{i}") for i in range(n_context_bin)]
        model.addCons(quicksum(context_bin_vars) <= 1)
        context_cont_vars = [model.addVar(lb=context_cont[i], ub=context_cont[i], vtype='C', name=f"context_cont_{i}") for i in range(n_context_cont)]

        # Normalized variables
        input_norm_vars = []
        all_vars = price_vars + context_cont_vars + context_bin_vars
        for i, var in enumerate(all_vars):
            mean, std = self.norm_params.input_means[i], self.norm_params.input_stds[i]
            norm_var = model.addVar(vtype='C', name=f"norm_{i}")
            model.addCons(norm_var * std == var - mean)
            input_norm_vars.append(norm_var)

        # Neural network with direct sigmoid formula
        print("[ANALYSIS] Используем ПРЯМУЮ формулу sigmoid(x) = 1 / (1 + exp(-x))")
        print("[ANALYSIS] Это может вызвать численные проблемы при больших |x|")
        print()

        # Layer 1: Linear
        layer1 = self.nn_layers[0]
        hidden_vars = []
        for j in range(layer1['output_dim']):
            h_var = model.addVar(vtype='C', name=f"hidden_{j}")
            expr = quicksum(layer1['weights'][j, i] * input_norm_vars[i] for i in range(len(input_norm_vars))) + layer1['bias'][j]
            model.addCons(h_var == expr, name=f"hidden_{j}_def")
            hidden_vars.append(h_var)

        # Sigmoid activation (PIECEWISE LINEAR APPROXIMATION)
        sigmoid_vars = []
        print("[ANALYSIS] Используем PIECEWISE LINEAR аппроксимацию sigmoid вместо exp()")

        for j, h_var in enumerate(hidden_vars):
            sig_var = model.addVar(lb=0, ub=1, vtype='C', name=f"sigmoid_{j}")

            # Piecewise linear approximation: 3 сегмента
            # [-∞, -3]: sigmoid ≈ 0
            # [-3, 3]: линейная аппроксимация
            # [3, ∞]: sigmoid ≈ 1

            # Бинарные переменные для выбора сегмента
            seg_low = model.addVar(vtype='B', name=f"seg_low_{j}")  # h < -3
            seg_mid = model.addVar(vtype='B', name=f"seg_mid_{j}")  # -3 <= h <= 3
            seg_high = model.addVar(vtype='B', name=f"seg_high_{j}") # h > 3

            model.addCons(seg_low + seg_mid + seg_high == 1, name=f"one_seg_{j}")

            # Ограничения на h_var для каждого сегмента
            M = 1000
            # seg_low = 1 => h <= -3
            model.addCons(h_var <= -3 + M * (1 - seg_low), name=f"h_low_{j}")
            # seg_mid = 1 => -3 <= h <= 3
            model.addCons(h_var >= -3 - M * (1 - seg_mid), name=f"h_mid_lb_{j}")
            model.addCons(h_var <= 3 + M * (1 - seg_mid), name=f"h_mid_ub_{j}")
            # seg_high = 1 => h >= 3
            model.addCons(h_var >= 3 - M * (1 - seg_high), name=f"h_high_{j}")

            # Значения sigmoid
            # sigmoid(-3) ≈ 0.047, sigmoid(0) = 0.5, sigmoid(3) ≈ 0.953
            # Линейная аппроксимация на [-3, 3]: slope = (0.953 - 0.047) / 6 ≈ 0.151
            slope = 0.151
            intercept = 0.5

            # sig_var = 0 при seg_low
            model.addCons(sig_var >= 0 - M * (1 - seg_low), name=f"sig_low_{j}")
            model.addCons(sig_var <= 0 + M * (1 - seg_low), name=f"sig_low_ub_{j}")

            # sig_var = slope * h + intercept при seg_mid
            model.addCons(sig_var >= slope * h_var + intercept - M * (1 - seg_mid), name=f"sig_mid_lb_{j}")
            model.addCons(sig_var <= slope * h_var + intercept + M * (1 - seg_mid), name=f"sig_mid_ub_{j}")

            # sig_var = 1 при seg_high
            model.addCons(sig_var >= 1 - M * (1 - seg_high), name=f"sig_high_{j}")
            model.addCons(sig_var <= 1 + M * (1 - seg_high), name=f"sig_high_ub_{j}")

            sigmoid_vars.append(sig_var)

        # Layer 2: Linear
        layer2 = self.nn_layers[2]
        demand_norm_vars = []
        for j in range(layer2['output_dim']):
            d_var = model.addVar(lb=0, ub=1, vtype='C', name=f"demand_norm_{j}")
            expr = quicksum(layer2['weights'][j, i] * sigmoid_vars[i] for i in range(len(sigmoid_vars))) + layer2['bias'][j]
            model.addCons(d_var == expr, name=f"demand_norm_{j}_def")
            demand_norm_vars.append(d_var)

        # Denormalize
        demand_vars = []
        for i in range(n_products):
            max_d = self.norm_params.output_max[i]
            d_var = model.addVar(lb=0, ub=max_d, vtype='C', name=f"demand_{i}")
            model.addCons(d_var == demand_norm_vars[i] * max_d)
            demand_vars.append(d_var)

        # Objective
        revenue_expr = quicksum(price_vars[i] * demand_vars[i] for i in range(n_products))
        cost_expr = quicksum(self.problem_params.costs[i] * demand_vars[i] for i in range(n_products))
        profit_expr = revenue_expr - cost_expr

        # [DIAGNOSTICS] Временно убираем ограничение на минимальную маржу
        # model.addCons(profit_expr >= self.min_profit_margin * revenue_expr)
        print("[DIAGNOSTICS] Ограничение на мин. маржу ОТКЛЮЧЕНО для теста")

        # [CRITICAL FIX] Используем вспомогательную переменную для нелинейной цели!
        max_revenue = sum(self.problem_params.price_ranges[i][1] * self.norm_params.output_max[i] for i in range(n_products))
        revenue_var = model.addVar(lb=0, ub=max_revenue, vtype='C', name='total_revenue')
        model.addCons(revenue_var == revenue_expr, name='revenue_def')
        model.setObjective(revenue_var, sense='maximize')

        print(f"  Variables: {model.getNVars()}, Constraints: {model.getNConss()}")
        print()
        print("[ANALYSIS] Решение...")

        status = model.optimize()
        print(f"\n[ANALYSIS] Статус: {model.getStatus()}")

        if model.getStatus() in ['optimal', 'best solution found', 'feasible']:
            print("\n[ANALYSIS] РЕШЕНИЕ НАЙДЕНО - sigmoid с прямой формулой РАБОТАЕТ!")
            for i in range(n_products):
                price = model.getVal(price_vars[i])
                demand = model.getVal(demand_vars[i])
                print(f"  Товар {i+1}: цена={price:.2f}, спрос={demand:.2f}")
        else:
            print("\n[ANALYSIS] НЕ СОШЛОСЬ - sigmoid с прямой формулой НЕ РАБОТАЕТ")
            print("[ANALYSIS] Возможные причины:")
            print("  1. Численная нестабильность при больших значениях exp()")
            print("  2. Нелинейное ограничение слишком сложное для solver")
            print("  3. Нужна аппроксимация sigmoid (piecewise linear)")

        return {'status': model.getStatus()}


def main_simple_nn_analysis():
    """Анализ проблемы сходимости с простой нейросетью + sigmoid."""
    n_products = 3
    n_context_cont = 2
    n_context_bin = 2

    print("="*70)
    print("АНАЛИЗ: ПРОСТАЯ НЕЙРОСЕТЬ + SIGMOID")
    print("="*70 + "\n")

    data = generate_training_data(n_products, n_context_cont, n_context_bin, n_samples=3000)

    input_dim = n_products + n_context_cont + n_context_bin
    hidden_dim = 8

    print(f"Архитектура: Linear({input_dim}->{hidden_dim}) -> Sigmoid -> Linear({hidden_dim}->{n_products})")
    print(f"Параметров: {input_dim * hidden_dim + hidden_dim + hidden_dim * n_products + n_products}")
    print()

    demand_net = SimpleDemandNet(input_dim, output_dim=n_products, hidden_dim=hidden_dim)
    demand_net = train_demand_network(demand_net, data['train_inputs'], data['train_outputs'],
                                       data['norm_params'], epochs=500, lr=0.01)

    optimizer = PriceOptimizerSimpleNN(demand_net, data['norm_params'], data['problem_params'])
    results = optimizer.optimize(time_limit=60)

    return results


if __name__ == "__main__":
    # ПОЛИНОМИАЛЬНАЯ МОДЕЛЬ (РАБОЧИЙ ВАРИАНТ)
    main()

    # РАСКОММЕНТИРУЙТЕ ДЛЯ АНАЛИЗА ПРОБЛЕМЫ SIGMOID:
    # main_simple_nn_analysis()
