"""
MINLP Price Optimization with Neural Network Demand Forecast
============================================================

Пример оптимизации цен на группу товаров с использованием pyscipopt-ml
для встраивания нейросети PyTorch в задачу MINLP.

Архитектура:
1. Функция генерации возвращает данные + все параметры для обучения и оптимизации
2. Обучение нейросети
3. Оптимизация (использует только данные, не классы)

pyscipopt-ml поддерживает только стандартные слои: Linear, ReLU, Sigmoid, Tanh, Identity
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from pyscipopt import Model, quicksum
from pyscipopt_ml.add_predictor import add_predictor_constr
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


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
    """
    Генерирует обучающие данные и все необходимые параметры.

    Returns:
        Dict с ключами:
            - 'train_inputs': torch.Tensor (ненормализованные входы)
            - 'train_outputs': torch.Tensor (ненормализованные выходы)
            - 'norm_params': NormalizationParams
            - 'problem_params': ProblemParams
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ======================================================================
    # Параметры товаров (для оптимизатора)
    # ======================================================================

    base_demands = np.random.uniform(50, 200, n_products)
    costs = np.random.uniform(10, 50, n_products)

    price_ranges = []
    for i in range(n_products):
        min_price = costs[i] * 1.2
        max_price = costs[i] * 4
        price_ranges.append((min_price, max_price))

    # Параметры для генерации спроса (случайные, воспроизводимые)
    price_elasticities = np.random.uniform(1.2, 2.5, n_products)
    cross_elasticities = np.random.uniform(0.1, 0.3, (n_products, n_products))
    np.fill_diagonal(cross_elasticities, 0)

    context_effects = {
        'continuous': np.random.uniform(-0.5, 0.5, (n_context_cont, n_products)),
        'binary': np.random.uniform(0.1, 0.5, (n_context_bin, n_products))
    }

    # ======================================================================
    # Генерация данных
    # ======================================================================

    # Цены
    prices = np.zeros((n_samples, n_products))
    for i in range(n_products):
        min_p, max_p = price_ranges[i]
        prices[:, i] = np.random.uniform(min_p, max_p, n_samples)

    # Контекст
    context_cont = np.random.uniform(-1, 1, (n_samples, n_context_cont))
    context_bin = np.random.randint(0, 2, (n_samples, n_context_bin)).astype(float)

    inputs = np.concatenate([prices, context_cont, context_bin], axis=1)

    # Спрос
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

    # ======================================================================
    # Параметры нормализации
    # ======================================================================

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

    # ======================================================================
    # Возвращаем всё как словарь
    # ======================================================================

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
# NEURAL NETWORK
# =============================================================================

class GroupDemandNet(nn.Sequential):
    """Нейросеть для прогнозирования спроса (работает с нормализованными данными)."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 activations: Optional[List[str]] = None):
        activation_map = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'identity': nn.Identity,
        }

        if activations is None:
            activations = ['relu'] * len(hidden_dims)
        elif len(activations) < len(hidden_dims):
            activations = activations + ['relu'] * (len(hidden_dims) - len(activations))

        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            act_name = activations[i].lower() if i < len(activations) else 'relu'
            act_cls = activation_map.get(act_name, nn.ReLU)
            layers.append(act_cls())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        super().__init__(*layers)


# =============================================================================
# TRAINING
# =============================================================================

def train_demand_network(demand_net: GroupDemandNet,
                         train_inputs: torch.Tensor,
                         train_outputs: torch.Tensor,
                         norm_params: NormalizationParams,
                         epochs: int = 200,
                         lr: float = 0.01) -> GroupDemandNet:
    """Обучает нейросеть."""
    print("="*70)
    print("ОБУЧЕНИЕ НЕЙРОСЕТИ")
    print("="*70 + "\n")

    # Нормализация для обучения
    inputs_norm = (train_inputs - torch.FloatTensor(norm_params.input_means)) / torch.FloatTensor(norm_params.input_stds)
    outputs_norm = train_outputs / torch.FloatTensor(norm_params.output_max)

    optimizer = torch.optim.Adam(demand_net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience = 50
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

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    print(f"\n[OK] Нейросеть обучена (финальный loss: {loss.item():.6f})\n")
    return demand_net


# =============================================================================
# OPTIMIZATION
# =============================================================================

class PriceOptimizerMINLP:
    """
    MINLP оптимизатор цен с использованием pyscipopt-ml.

    Принимает параметры задачи через структуру данных, не зависит от генератора.
    """

    def __init__(self,
                 demand_net: GroupDemandNet,
                 norm_params: NormalizationParams,
                 problem_params: ProblemParams,
                 min_profit_margin: float = 0.20):
        self.demand_net = demand_net
        self.norm_params = norm_params
        self.problem_params = problem_params
        self.min_profit_margin = min_profit_margin

    def optimize(self,
                context_cont: Optional[np.ndarray] = None,
                time_limit: int = 300) -> Dict:
        n_products = self.problem_params.n_products
        n_context_cont = self.problem_params.n_context_cont
        n_context_bin = self.problem_params.n_context_bin

        if context_cont is None:
            context_cont = np.zeros(n_context_cont)

        print("\n" + "="*70)
        print("MINLP ОПТИМИЗАЦИЯ ЦЕН С PYSCIPOPT-ML")
        print("="*70 + "\n")

        model = Model("Price_Optimization_NN")
        model.redirectOutput()
        model.setParam('limits/time', time_limit)
        model.setParam('display/verblevel', 2)
        model.setParam('numerics/feastol', 1e-6)
        model.setParam('numerics/epsilon', 1e-7)

        # ======================================================================
        # DECISION VARIABLES (ИСХОДНЫЙ МАСШТАБ)
        # ======================================================================

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

        # ======================================================================
        # NORMALIZED VARIABLES (через ограничения)
        # ======================================================================

        input_norm_vars = []
        all_input_vars = price_vars + context_cont_vars + context_bin_vars

        for i, var in enumerate(all_input_vars):
            mean = self.norm_params.input_means[i]
            std = self.norm_params.input_stds[i]
            norm_var = model.addVar(vtype='C', name=f"input_norm_{i}")
            model.addCons(norm_var == (var - mean) / std, name=f"norm_{i}")
            input_norm_vars.append(norm_var)

        demand_norm_vars = []
        for i in range(n_products):
            var = model.addVar(lb=0, ub=1, vtype='C', name=f"demand_norm_{i}")
            demand_norm_vars.append(var)

        # ======================================================================
        # NEURAL NETWORK EMBEDDING
        # ======================================================================

        print("Встраивание нейросети в модель...")
        add_predictor_constr(
            model,
            self.demand_net,
            np.array(input_norm_vars).reshape((1, -1)),
            np.array(demand_norm_vars).reshape((1, -1)),
            epsilon=0.001,
            unique_naming_prefix='nn_'
        )
        print("[OK] Нейросеть встроена\n")

        # ======================================================================
        # DEMAND DENORMALIZATION
        # ======================================================================

        demand_vars = []
        for i in range(n_products):
            max_demand = self.norm_params.output_max[i]
            var = model.addVar(lb=0, ub=max_demand, vtype='C', name=f"demand_{i}")
            model.addCons(var == demand_norm_vars[i] * max_demand, name=f"denorm_{i}")
            demand_vars.append(var)

        # ======================================================================
        # OBJECTIVE AND CONSTRAINTS
        # ======================================================================

        revenue_expr = quicksum(price_vars[i] * demand_vars[i] for i in range(n_products))
        cost_expr = quicksum(self.problem_params.costs[i] * demand_vars[i] for i in range(n_products))
        profit_expr = revenue_expr - cost_expr

        model.addCons(profit_expr >= self.min_profit_margin * revenue_expr, name="min_profit_margin")

        max_revenue = sum(self.problem_params.price_ranges[i][1] * self.norm_params.output_max[i]
                         for i in range(n_products))
        revenue_var = model.addVar(lb=0, ub=max_revenue, vtype='C', name='total_revenue')
        model.addCons(revenue_var == revenue_expr, name='revenue_def')
        model.setObjective(revenue_var, sense='maximize')

        # ======================================================================
        # SOLVE
        # ======================================================================

        print("Информация о модели:")
        print(f"  Переменных: {model.getNVars()}")
        print(f"  Ограничений: {model.getNConss()}")
        print()

        print("Решение оптимизационной задачи...")
        status = model.optimize()

        return self._extract_results(model, price_vars, context_bin_vars,
                                    demand_vars, context_cont)

    def _extract_results(self, model, price_vars, context_bin_vars,
                        demand_vars, context_cont) -> Dict:
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
# UTILITIES
# =============================================================================

def visualize_results(optimizer: PriceOptimizerMINLP, results: Dict) -> None:
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

        ax.plot(prices_range, demands_range, 'b-', linewidth=2, label='Спрос (нейросеть)')

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

    plt.suptitle('Кривые спроса и оптимальные цены', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('price_optimization_results.png', dpi=150, bbox_inches='tight')
    print("\nГрафик сохранен: price_optimization_results.png")
    plt.show()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Pipeline:
    1. generate_training_data() → Dict с данными и параметрами
    2. Обучение нейросети
    3. Оптимизация (использует только данные из словаря)
    """

    n_products = 3
    n_context_cont = 2
    n_context_bin = 2

    print("="*70)
    print("PIPELINE: ГЕНЕРАЦИЯ ДАННЫХ -> ОБУЧЕНИЕ -> ОПТИМИЗАЦИЯ")
    print("="*70 + "\n")

    # ======================================================================
    # STEP 1: Генерация данных (функция, возвращает словарь)
    # ======================================================================
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
    print(f"\n  Параметры нормализации:")
    print(f"    Input means: {norm_params.input_means}")
    print(f"    Input stds: {norm_params.input_stds}")
    print(f"    Output max: {norm_params.output_max}")
    print()

    # ======================================================================
    # STEP 2: Создание и обучение нейросети
    # ======================================================================
    print("ШАГ 2: Создание и обучение нейросети\n")

    input_dim = n_products + n_context_cont + n_context_bin
    demand_net = GroupDemandNet(
        input_dim=input_dim,
        output_dim=n_products,
        hidden_dims=[32, 16],
        activations=['sigmoid', 'relu']
    )

    print(f"  Архитектура (Sequential):")
    print(f"    Linear({input_dim} -> 32) -> Sigmoid -> Linear(32 -> 16) -> ReLU -> Linear(16 -> {n_products})")
    print()

    demand_net = train_demand_network(
        demand_net=demand_net,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        norm_params=norm_params,
        epochs=300,
        lr=0.01
    )

    # Валидация
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

    # ======================================================================
    # STEP 3: Оптимизация
    # ======================================================================
    print("ШАГ 3: MINLP оптимизация\n")

    optimizer = PriceOptimizerMINLP(
        demand_net=demand_net,
        norm_params=norm_params,
        problem_params=problem_params,
        min_profit_margin=0.15
    )

    results = optimizer.optimize(time_limit=120)

    # ======================================================================
    # STEP 4: Визуализация
    # ======================================================================
    if results['optimal_prices']:
        visualize_results(optimizer, results)


if __name__ == "__main__":
    main()
