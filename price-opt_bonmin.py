"""
MINLP Price Optimization with Neural Network Demand Forecast (BONMIN Solver)
====================================================================================

Вариант с использованием pyomo и bonmin solver вместо pyscipopt.

Преимущество bonmin: полная поддержка нелинейных функций (exp, sigmoid, etc.)
и нелинейных целей без дополнительных трюков.

Архитектура:
1. Функция генерации возвращает данные + все параметры для обучения и оптимизации
2. Обучение нейросети (PyTorch)
3. Оптимизация через pyomo + bonmin (встраивание формулы из обученной нейросети)

Сравнение:
- GroupDemandNet: нейросеть с линейными слоями + активации
- PolynomialDemandModel: полиномиальная модель (степень <= 2)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from itertools import combinations_with_replacement
import time

# Pyomo и bonmin solver
from pyomo.environ import (SolverFactory, SolverStatus, ConcreteModel, Var,
                            Objective, Constraint, Suffix, Reals, Binary,
                            maximize, NonNegativeReals, exp as pyo_exp, log,
                            RangeSet)


# =============================================================================
# DATA STRUCTURES (same as price-opt_ml.py)
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
    max_demands: np.ndarray


# =============================================================================
# DATA GENERATION FUNCTION (same as price-opt_ml.py)
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
# NEURAL NETWORK (same as price-opt_ml.py)
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
# POLYNOMIAL DEMAND MODEL (from price-opt_formula.py)
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
# TRAINING (same as price-opt_ml.py)
# =============================================================================

def train_demand_network(demand_net: nn.Module,
                         train_inputs: torch.Tensor,
                         train_outputs: torch.Tensor,
                         norm_params: NormalizationParams,
                         epochs: int = 200,
                         lr: float = 0.01) -> nn.Module:
    """Обучает нейросеть (работает с любой nn.Module, включая GroupDemandNet и PolynomialDemandModel)."""
    print("="*70)
    print("ОБУЧЕНИЕ НЕЙРОСЕТИ")
    print("="*70 + "\n")

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
# OPTIMIZATION WITH PYOMO + BONMIN
# =============================================================================

class PriceOptimizerBONMIN:
    """
    МИНЛП оптимизатор цен с использованием pyomo и bonmin solver.

    Bonmin (Basic Open-source Nonlinear Mixed INteger programming) поддерживает:
    - Полные нелинейные функции (exp, log, sin, cos, etc.)
    - Нелинейные цели напрямую
    - Составные нелинейные выражения
    """

    def __init__(self,
                 demand_net: GroupDemandNet,
                 norm_params: NormalizationParams,
                 problem_params: ProblemParams,
                 solver_path: str = r"C:\distr\solvers\bonmin.exe",
                 min_profit_margin: float = 0.20):
        self.demand_net = demand_net
        self.norm_params = norm_params
        self.problem_params = problem_params
        self.solver_path = solver_path
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
        print("МИНЛП ОПТИМИЗАЦИЯ ЦЕН (PYOMO + BONMIN)")
        print("="*70 + "\n")

        # ======================================================================
        # CREATE PYOMO MODEL
        # ======================================================================

        model = ConcreteModel()

        # Create index sets
        model.products = RangeSet(0, n_products - 1)
        model.context_bin_idx = RangeSet(0, n_context_bin - 1)
        model.context_cont_idx = RangeSet(0, n_context_cont - 1)
        model.input_idx = RangeSet(0, n_products + n_context_cont + n_context_bin - 1)

        # Decision variables: prices
        def price_bounds(model, i):
            min_p, max_p = self.problem_params.price_ranges[i]
            return (min_p, max_p)
        model.price = Var(model.products, within=Reals, bounds=price_bounds)

        # Context binary variables
        model.context_bin = Var(model.context_bin_idx, within=Binary)

        # Fixed context continuous variables
        def context_cont_bounds(model, i):
            return (context_cont[i], context_cont[i])
        model.context_cont = Var(model.context_cont_idx, within=Reals, bounds=context_cont_bounds)

        # Constraint: max one promo
        model.max_one_promo = Constraint(expr=sum(model.context_bin[i] for i in model.context_bin_idx) <= 1)

        # ======================================================================
        # NORMALIZED INPUT VARIABLES
        # ======================================================================

        # Helper to get all input variables in order
        def get_input_var(model, i):
            if i < n_products:
                return model.price[i]
            elif i < n_products + n_context_cont:
                return model.context_cont[i - n_products]
            else:
                return model.context_bin[i - n_products - n_context_cont]

        # Normalized input variables
        model.input_norm = Var(model.input_idx, within=Reals, bounds=(-10, 10))

        def normalization_rule(model, i):
            var = get_input_var(model, i)
            mean = self.norm_params.input_means[i]
            std = self.norm_params.input_stds[i]
            return model.input_norm[i] * std == var - mean
        model.normalization = Constraint(model.input_idx, rule=normalization_rule)

        # ======================================================================
        # NEURAL NETWORK EMBEDDING (FORMULA EXPORT)
        # ======================================================================

        print("Встраивание нейросети через pyomo...")

        # Получаем параметры всех слоёв
        layers_info = self._extract_network_layers()
        layer_vars = self._apply_network(model, model.input_norm, layers_info)

        print(f"[OK] Нейросеть встроена ({len(layers_info)} слоев)\n")

        # ======================================================================
        # DEMAND VARIABLES (DENORMALIZED)
        # ======================================================================

        def demand_bounds(model, i):
            return (0, self.norm_params.output_max[i])
        model.demand = Var(model.products, within=NonNegativeReals, bounds=demand_bounds)

        def denormalization_rule(model, i):
            return model.demand[i] == layer_vars[-1][i] * self.norm_params.output_max[i]
        model.denormalization = Constraint(model.products, rule=denormalization_rule)

        # ======================================================================
        # OBJECTIVE AND CONSTRAINTS
        # ======================================================================

        # Revenue expression
        def revenue_rule(model):
            return sum(model.price[i] * model.demand[i] for i in model.products)

        # Cost expression
        def cost_expr(model):
            return sum(self.problem_params.costs[i] * model.demand[i] for i in model.products)

        # Profit expression
        profit_expr = revenue_rule(model) - cost_expr(model)

        # Minimum profit margin constraint
        model.profit_margin = Constraint(expr=profit_expr >= self.min_profit_margin * revenue_rule(model))

        # Objective: maximize revenue
        model.objective = Objective(expr=revenue_rule(model), sense=maximize)

        # ======================================================================
        # SOLVE WITH BONMIN
        # ======================================================================

        print("Информация о модели:")
        print(f"  Переменных: {len(list(model.component_data_objects(Var)))}")
        print(f"  Ограничений: {len(list(model.component_data_objects(Constraint)))}")
        print()

        print(f"Решение через bonmin (solver: {self.solver_path})...")

        # Configure solver - bonmin uses ASL interface
        # Pass executable path directly to SolverFactory
        import pyomo.environ as pyo
        solver = SolverFactory('bonmin', executable=self.solver_path)
        solver.options['bonmin.time_limit'] = time_limit
        # Remove invalid option
        # solver.options['bonmin.max_iter'] = 10000

        # Solve with tee=True to see solver output
        result = solver.solve(model, tee=True)

        print(f"Статус solver: {result.solver.status}")
        print(f"Статус pyomo: {result.solver.termination_condition}")

        return self._extract_results(model, result)

    def _apply_network(self, model: ConcreteModel, input_vars, layers_info: List[Dict]) -> List:
        """Применяет все слои нейросети последовательно."""
        layer_outputs = [input_vars]  # Store outputs of each layer
        current_vars = input_vars

        for layer_idx, layer_info in enumerate(layers_info):
            if layer_info['type'] == 'linear':
                current_vars = self._apply_linear_layer(model, current_vars, layer_info, layer_idx)
            elif layer_info['type'] == 'activation':
                current_vars = self._apply_activation(model, current_vars, layer_info, layer_idx)
            layer_outputs.append(current_vars)

        return layer_outputs

    def _extract_network_layers(self) -> List[Dict]:
        """Извлекает параметры всех слоёв нейросети."""
        layers_info = []

        for layer in self.demand_net:
            if isinstance(layer, nn.Linear):
                layers_info.append({
                    'type': 'linear',
                    'weights': layer.weight.detach().numpy().copy(),
                    'bias': layer.bias.detach().numpy().copy(),
                    'input_dim': layer.in_features,
                    'output_dim': layer.out_features
                })
            elif isinstance(layer, nn.ReLU):
                layers_info.append({'type': 'activation', 'activation': 'relu'})
            elif isinstance(layer, nn.Sigmoid):
                layers_info.append({'type': 'activation', 'activation': 'sigmoid'})
            elif isinstance(layer, nn.Tanh):
                layers_info.append({'type': 'activation', 'activation': 'tanh'})
            elif isinstance(layer, nn.Identity):
                layers_info.append({'type': 'activation', 'activation': 'identity'})

        return layers_info

    def _apply_linear_layer(self, model: ConcreteModel, input_vars, layer_info: Dict, layer_idx: int):
        """Применяет линейный слой."""
        weights = layer_info['weights']
        bias = layer_info['bias']
        output_dim = layer_info['output_dim']
        input_dim = layer_info['input_dim']

        # Unique names for this layer
        idx_name = f'linear_{layer_idx}_output_idx'
        var_name = f'linear_{layer_idx}_output'
        con_name = f'linear_{layer_idx}_constraint'

        # Create indexed set for outputs
        setattr(model, idx_name, RangeSet(0, output_dim - 1))
        idx_set = getattr(model, idx_name)

        # Create output variables
        def linear_bounds(model, j):
            return (-100, 100)
        setattr(model, var_name, Var(idx_set, within=Reals, bounds=linear_bounds))
        output_var = getattr(model, var_name)

        # Create constraints for each output
        def make_linear_rule(var):
            def linear_rule(model, j):
                expr = sum(weights[j, i] * input_vars[i] for i in range(input_dim)) + bias[j]
                return var[j] == expr
            return linear_rule
        setattr(model, con_name, Constraint(idx_set, rule=make_linear_rule(output_var)))

        return [output_var[j] for j in range(output_dim)]

    def _apply_activation(self, model: ConcreteModel, input_vars, layer_info: Dict, layer_idx: int):
        """Применяет функцию активации через pyomo выражения."""
        activation = layer_info['activation']
        n_inputs = len(input_vars)

        if activation == 'identity':
            return input_vars

        # Unique names for this layer
        idx_name = f'act_{layer_idx}_{activation}_output_idx'
        var_name = f'act_{layer_idx}_{activation}_output'

        # Create indexed set for outputs
        setattr(model, idx_name, RangeSet(0, n_inputs - 1))
        idx_set = getattr(model, idx_name)

        if activation == 'sigmoid':
            # sigmoid(x) = 1 / (1 + exp(-x))
            def sigmoid_bounds(model, i):
                return (0, 1)
            setattr(model, var_name, Var(idx_set, within=Reals, bounds=sigmoid_bounds))
            output_var = getattr(model, var_name)

            con_name = f'act_{layer_idx}_sigmoid_constraint'
            def sigmoid_rule(model, i):
                return output_var[i] == 1.0 / (1.0 + pyo_exp(-input_vars[i]))
            setattr(model, con_name, Constraint(idx_set, rule=sigmoid_rule))

        elif activation == 'tanh':
            # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
            def tanh_bounds(model, i):
                return (-1, 1)
            setattr(model, var_name, Var(idx_set, within=Reals, bounds=tanh_bounds))
            output_var = getattr(model, var_name)

            con_name = f'act_{layer_idx}_tanh_constraint'
            def tanh_rule(model, i):
                exp_pos = pyo_exp(input_vars[i])
                exp_neg = pyo_exp(-input_vars[i])
                return output_var[i] == (exp_pos - exp_neg) / (exp_pos + exp_neg)
            setattr(model, con_name, Constraint(idx_set, rule=tanh_rule))

        elif activation == 'relu':
            # ReLU: max(0, x)
            def relu_bounds(model, i):
                return (0, 100)
            setattr(model, var_name, Var(idx_set, within=NonNegativeReals, bounds=relu_bounds))
            output_var = getattr(model, var_name)

            con1_name = f'act_{layer_idx}_relu_constraint1'
            con2_name = f'act_{layer_idx}_relu_constraint2'
            def relu_rule1(model, i):
                return output_var[i] >= input_vars[i]
            def relu_rule2(model, i):
                return output_var[i] >= 0
            setattr(model, con1_name, Constraint(idx_set, rule=relu_rule1))
            setattr(model, con2_name, Constraint(idx_set, rule=relu_rule2))

        output_var = getattr(model, var_name)
        return [output_var[i] for i in range(n_inputs)]

    def _extract_results(self, model, result) -> Dict:
        """Извлекает результаты из решенной модели."""
        results = {
            'status': 'unknown',
            'optimal_prices': [],
            'demands': [],
            'revenues': [],
            'profits': [],
            'active_promos': [],
            'total_revenue': 0,
            'total_profit': 0,
            'profit_margin': 0,
            'solver_status': str(result.solver.status),
            'termination_condition': str(result.solver.termination_condition)
        }

        # Check if solution was found
        try:
            # Try to extract values
            for i in range(self.problem_params.n_products):
                price = model.price[i].value
                demand = model.demand[i].value

                if price is None or demand is None:
                    raise ValueError("Variable values not set")

                revenue = price * demand
                profit = (price - self.problem_params.costs[i]) * demand

                results['optimal_prices'].append(float(price))
                results['demands'].append(float(demand))
                results['revenues'].append(float(revenue))
                results['profits'].append(float(profit))

            results['status'] = 'optimal'

            print("\n" + "="*70)
            print("РЕШЕНИЕ НАЙДЕНО (BONMIN)")
            print("="*70 + "\n")

            for i in range(self.problem_params.n_products):
                print(f"Товар {i+1}:")
                print(f"  Цена: {results['optimal_prices'][i]:.2f}")
                print(f"  Спрос: {results['demands'][i]:.2f}")
                print(f"  Выручка: {results['revenues'][i]:.2f}")
                print(f"  Прибыль: {results['profits'][i]:.2f}")
                print(f"  Себестоимость: {self.problem_params.costs[i]:.2f}\n")

            active_promos = []
            for p in range(self.problem_params.n_context_bin):
                val = model.context_bin[p].value
                if val is not None and val > 0.5:
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
        except Exception as e:
            print(f"\nОшибка при извлечении результатов: {e}")
            print(f"Статус: {results['solver_status']}")
            print(f"Termination: {results['termination_condition']}")

        return results


class PriceOptimizerPolynomialPyomo:
    """
    Универсальный МИНЛП оптимизатор для полиномиальной модели спроса.
    Работает с любым solver через pyomo (bonmin, couenne).
    """

    def __init__(self,
                 demand_net: PolynomialDemandModel,
                 norm_params: NormalizationParams,
                 problem_params: ProblemParams,
                 solver_name: str = "bonmin",
                 solver_path: str = r"C:\distr\solvers\bonmin.exe",
                 min_profit_margin: float = 0.20):
        self.demand_net = demand_net
        self.norm_params = norm_params
        self.problem_params = problem_params
        self.solver_name = solver_name
        self.solver_path = solver_path
        self.min_profit_margin = min_profit_margin
        self.coeffs = demand_net.get_coefficients()

    def optimize(self,
                context_cont: Optional[np.ndarray] = None,
                time_limit: int = 300) -> Dict:
        n_products = self.problem_params.n_products
        n_context_cont = self.problem_params.n_context_cont
        n_context_bin = self.problem_params.n_context_bin

        if context_cont is None:
            context_cont = np.zeros(n_context_cont)

        print("\n" + "="*70)
        print(f"МИНЛП ОПТИМИЗАЦИЯ ЦЕН (PYOMO + {self.solver_name.upper()}, ПОЛИНОМИАЛЬНАЯ МОДЕЛЬ)")
        print("="*70 + "\n")

        # ======================================================================
        # CREATE PYOMO MODEL
        # ======================================================================

        model = ConcreteModel()

        # Create index sets
        model.products = RangeSet(0, n_products - 1)
        model.context_bin_idx = RangeSet(0, n_context_bin - 1)
        model.context_cont_idx = RangeSet(0, n_context_cont - 1)
        model.input_idx = RangeSet(0, n_products + n_context_cont + n_context_bin - 1)

        # Decision variables: prices
        def price_bounds(model, i):
            min_p, max_p = self.problem_params.price_ranges[i]
            return (min_p, max_p)
        model.price = Var(model.products, within=Reals, bounds=price_bounds)

        # Context binary variables
        model.context_bin = Var(model.context_bin_idx, within=Binary)

        # Fixed context continuous variables
        def context_cont_bounds(model, i):
            return (context_cont[i], context_cont[i])
        model.context_cont = Var(model.context_cont_idx, within=Reals, bounds=context_cont_bounds)

        # Constraint: max one promo
        model.max_one_promo = Constraint(expr=sum(model.context_bin[i] for i in model.context_bin_idx) <= 1)

        # ======================================================================
        # NORMALIZED INPUT VARIABLES
        # ======================================================================

        # Helper to get all input variables in order
        def get_input_var(model, i):
            if i < n_products:
                return model.price[i]
            elif i < n_products + n_context_cont:
                return model.context_cont[i - n_products]
            else:
                return model.context_bin[i - n_products - n_context_cont]

        # Normalized input variables
        model.input_norm = Var(model.input_idx, within=Reals, bounds=(-10, 10))

        def normalization_rule(model, i):
            var = get_input_var(model, i)
            mean = self.norm_params.input_means[i]
            std = self.norm_params.input_stds[i]
            return model.input_norm[i] * std == var - mean
        model.normalization = Constraint(model.input_idx, rule=normalization_rule)

        # ======================================================================
        # POLYNOMIAL DEMAND MODEL (FORMULA)
        # ======================================================================

        print("Построение полиномиальных выражений спроса...")

        model.demand_norm = Var(model.products, within=Reals, bounds=(0, 1))

        def polynomial_demand_rule(model, i):
            """Строит полиномиальное выражение для продукта i."""
            expr = self.coeffs['bias'][i]

            # Linear terms
            for j in range(self.coeffs['linear'].shape[1]):
                expr += self.coeffs['linear'][i, j] * model.input_norm[j]

            # Quadratic terms (only for non-price variables)
            for quad_idx, var_idx in enumerate(self.coeffs['quadratic_indices']):
                expr += self.coeffs['quadratic'][i, quad_idx] * (model.input_norm[var_idx] ** 2)

            # Interaction terms
            for pair_idx, (j, k) in enumerate(self.coeffs['pair_indices']):
                expr += self.coeffs['interaction'][i, pair_idx] * model.input_norm[j] * model.input_norm[k]

            return model.demand_norm[i] == expr

        model.polynomial_demand = Constraint(model.products, rule=polynomial_demand_rule)

        print(f"[OK] Построено {n_products} полиномиальных выражений\n")

        # ======================================================================
        # DEMAND VARIABLES (DENORMALIZED)
        # ======================================================================

        def demand_bounds(model, i):
            return (0, self.norm_params.output_max[i])
        model.demand = Var(model.products, within=NonNegativeReals, bounds=demand_bounds)

        def denormalization_rule(model, i):
            return model.demand[i] == model.demand_norm[i] * self.norm_params.output_max[i]
        model.denormalization = Constraint(model.products, rule=denormalization_rule)

        # ======================================================================
        # OBJECTIVE AND CONSTRAINTS
        # ======================================================================

        # Revenue expression
        def revenue_rule(model):
            return sum(model.price[i] * model.demand[i] for i in model.products)

        # Cost expression
        def cost_expr(model):
            return sum(self.problem_params.costs[i] * model.demand[i] for i in model.products)

        # Profit expression
        profit_expr = revenue_rule(model) - cost_expr(model)

        # Minimum profit margin constraint
        model.profit_margin = Constraint(expr=profit_expr >= self.min_profit_margin * revenue_rule(model))

        # ======================================================================
        # OBJECTIVE: Need auxiliary variable for SCIP (nonlinear objective not supported)
        # ======================================================================
        max_revenue = sum(self.problem_params.price_ranges[i][1] * self.norm_params.output_max[i]
                         for i in range(n_products))

        if self.solver_name == "scip":
            # SCIP requires linear objective - use auxiliary variable
            model.revenue_var = Var(within=Reals, bounds=(0, max_revenue))
            model.revenue_def = Constraint(expr=model.revenue_var == revenue_rule(model))
            model.objective = Objective(expr=model.revenue_var, sense=maximize)
        else:
            # Bonmin and Couenne support nonlinear objectives directly
            model.objective = Objective(expr=revenue_rule(model), sense=maximize)

        # ======================================================================
        # SOLVE WITH SOLVER
        # ======================================================================

        print("Информация о модели:")
        print(f"  Переменных: {len(list(model.component_data_objects(Var)))}")
        print(f"  Ограничений: {len(list(model.component_data_objects(Constraint)))}")
        print()

        print(f"Решение через {self.solver_name} (solver: {self.solver_path})...")

        solver = SolverFactory(self.solver_name, executable=self.solver_path)

        # Different time limit options for different solvers
        if self.solver_name == "couenne":
            # Couenne uses 'max_time' instead of 'time_limit'
            solver.options['max_time'] = time_limit
        else:
            # Bonmin uses 'bonmin.time_limit'
            solver.options[f'{self.solver_name}.time_limit'] = time_limit

        # Solve with tee=False to suppress solver output
        try:
            result = solver.solve(model, tee=False)
        except Exception as e:
            print(f"Ошибка при решении: {e}")
            # Fallback: solve without time limit
            solver2 = SolverFactory(self.solver_name, executable=self.solver_path)
            result = solver2.solve(model, tee=False)

        print(f"Статус solver: {result.solver.status}")
        print(f"Статус pyomo: {result.solver.termination_condition}")

        return self._extract_results(model, result)

    def _extract_results(self, model, result) -> Dict:
        """Извлекает результаты из решенной модели."""
        results = {
            'status': 'unknown',
            'optimal_prices': [],
            'demands': [],
            'revenues': [],
            'profits': [],
            'active_promos': [],
            'total_revenue': 0,
            'total_profit': 0,
            'profit_margin': 0,
            'solver_status': str(result.solver.status),
            'termination_condition': str(result.solver.termination_condition)
        }

        try:
            for i in range(self.problem_params.n_products):
                price = model.price[i].value
                demand = model.demand[i].value

                if price is None or demand is None:
                    raise ValueError("Variable values not set")

                revenue = price * demand
                profit = (price - self.problem_params.costs[i]) * demand

                results['optimal_prices'].append(float(price))
                results['demands'].append(float(demand))
                results['revenues'].append(float(revenue))
                results['profits'].append(float(profit))

            results['status'] = 'optimal'

            print("\n" + "="*70)
            print(f"РЕШЕНИЕ НАЙДЕНО ({self.solver_name.upper()} + ПОЛИНОМ)")
            print("="*70 + "\n")

            for i in range(self.problem_params.n_products):
                print(f"Товар {i+1}:")
                print(f"  Цена: {results['optimal_prices'][i]:.2f}")
                print(f"  Спрос: {results['demands'][i]:.2f}")
                print(f"  Выручка: {results['revenues'][i]:.2f}")
                print(f"  Прибыль: {results['profits'][i]:.2f}")
                print(f"  Себестоимость: {self.problem_params.costs[i]:.2f}\n")

            active_promos = []
            for p in range(self.problem_params.n_context_bin):
                val = model.context_bin[p].value
                if val is not None and val > 0.5:
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
        except Exception as e:
            print(f"\nОшибка при извлечении результатов: {e}")
            print(f"Статус: {results['solver_status']}")
            print(f"Termination: {results['termination_condition']}")

        return results


def value(pyomo_var):
    """Получает значение переменной pyomo."""
    try:
        return pyo_value(pyomo_var)
    except:
        return None


def pyo_value(pyomo_expr):
    """Вычисляет значение выражения pyomo."""
    try:
        from pyomo.core.expr.numeric_expr import NumericValue
        if isinstance(pyomo_expr, Var):
            return pyomo_expr.value
        else:
            return pyomo_expr.expr()
    except:
        return None


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_results(optimizer: PriceOptimizerBONMIN, results: Dict) -> None:
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

    plt.suptitle('Кривые спроса и оптимальные цены (BONMIN)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('price_optimization_results_bonmin.png', dpi=150, bbox_inches='tight')
    print("\nГрафик сохранен: price_optimization_results_bonmin.png")
    plt.show()


# =============================================================================
# BENCHMARK: PYSCIPOPT vs BONMIN (POLYNOMIAL MODEL)
# =============================================================================

def benchmark_polynomial_model(bonmin_time_limit=120):
    """
    Сравнение производительности pyscipopt, bonmin, couenne
    на одной и той же полиномиальной модели спроса.
    """
    from pyscipopt import Model, quicksum

    n_products = 3
    n_context_cont = 2
    n_context_bin = 2

    print("="*70)
    print("БЕНЧМАРК: PYSCIP vs BONMIN vs COUENNE (ПОЛИНОМ)")
    print("="*70 + "\n")

    # ======================================================================
    # STEP 1: Генерация данных
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
    print()

    # ======================================================================
    # STEP 2: Обучение полиномиальной модели
    # ======================================================================
    print("ШАГ 2: Обучение полиномиальной модели\n")

    input_dim = n_products + n_context_cont + n_context_bin

    demand_net = PolynomialDemandModel(
        input_dim=input_dim,
        output_dim=n_products,
        n_prices=n_products
    )

    # Обучаем модель (используем train_demand_network, он работает с любой nn.Module)
    demand_net = train_demand_network(
        demand_net=demand_net,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        norm_params=norm_params,
        epochs=500,
        lr=0.01
    )

    # Валидация
    print("Валидация...")
    val_data = generate_training_data(
        n_products=n_products,
        n_context_cont=n_context_cont,
        n_context_bin=n_context_bin,
        n_samples=500,
        seed=123
    )
    val_inputs_norm = (val_data['train_inputs'] - torch.FloatTensor(norm_params.input_means)) / torch.FloatTensor(norm_params.input_stds)
    val_outputs_norm = val_data['train_outputs'] / torch.FloatTensor(norm_params.output_max)

    with torch.no_grad():
        val_preds = demand_net(val_inputs_norm)
        mae = torch.mean(torch.abs(val_preds * torch.FloatTensor(norm_params.output_max) - val_data['train_outputs']))

    print(f"  MAE: {mae.item():.2f}\n")

    # ======================================================================
    # STEP 3a: Оптимизация через PYSCIPOPT
    # ======================================================================
    print("\n" + "="*70)
    print("ТЕСТ 1: PYSCIPOPT + ПОЛИНОМ")
    print("="*70 + "\n")

    # Создаём модель pyscipopt с полиномиальными выражениями
    model_scip = Model("Price_Polynomial_SCIP")
    model_scip.redirectOutput()
    model_scip.setParam('limits/time', 120)
    model_scip.setParam('display/verblevel', 1)

    # Decision variables
    price_vars = []
    for i in range(n_products):
        min_p, max_p = problem_params.price_ranges[i]
        var = model_scip.addVar(lb=min_p, ub=max_p, vtype='C', name=f"price_{i}")
        price_vars.append(var)

    context_bin_vars = []
    for i in range(n_context_bin):
        var = model_scip.addVar(vtype='B', name=f"context_bin_{i}")
        context_bin_vars.append(var)

    model_scip.addCons(quicksum(context_bin_vars) <= 1, name="max_one_promo")

    context_cont = np.zeros(n_context_cont)
    context_cont_vars = []
    for i in range(n_context_cont):
        var = model_scip.addVar(lb=context_cont[i], ub=context_cont[i], vtype='C', name=f"context_cont_{i}")
        context_cont_vars.append(var)

    # Normalized variables
    input_norm_vars = []
    all_input_vars = price_vars + context_cont_vars + context_bin_vars

    for i, var in enumerate(all_input_vars):
        mean = norm_params.input_means[i]
        std = norm_params.input_stds[i]
        norm_var = model_scip.addVar(vtype='C', name=f"input_norm_{i}")
        model_scip.addCons(norm_var * std == var - mean, name=f"norm_{i}")
        input_norm_vars.append(norm_var)

    # Polynomial demand expressions
    coeffs = demand_net.get_coefficients()

    print("Построение полиномиальных выражений...")
    demand_norm_vars = []
    for i in range(n_products):
        demand_norm_var = model_scip.addVar(lb=0, ub=1, vtype='C', name=f"demand_norm_{i}")

        # Build polynomial expression
        expr = coeffs['bias'][i]

        # Linear terms
        for j in range(len(input_norm_vars)):
            expr += coeffs['linear'][i, j] * input_norm_vars[j]

        # Quadratic terms (only for non-price variables)
        for quad_idx, var_idx in enumerate(coeffs['quadratic_indices']):
            expr += coeffs['quadratic'][i, quad_idx] * (input_norm_vars[var_idx] ** 2)

        # Interaction terms
        for pair_idx, (j, k) in enumerate(coeffs['pair_indices']):
            expr += coeffs['interaction'][i, pair_idx] * input_norm_vars[j] * input_norm_vars[k]

        model_scip.addCons(demand_norm_var == expr, name=f"demand_poly_{i}")
        demand_norm_vars.append(demand_norm_var)

    print(f"[OK] Построено {n_products} полиномиальных выражений\n")

    # Demand denormalization
    demand_vars = []
    for i in range(n_products):
        max_demand = norm_params.output_max[i]
        var = model_scip.addVar(lb=0, ub=max_demand, vtype='C', name=f"demand_{i}")
        model_scip.addCons(var == demand_norm_vars[i] * max_demand, name=f"denorm_{i}")
        demand_vars.append(var)

    # Objective and constraints
    revenue_expr = quicksum(price_vars[i] * demand_vars[i] for i in range(n_products))
    cost_expr = quicksum(problem_params.costs[i] * demand_vars[i] for i in range(n_products))
    profit_expr = revenue_expr - cost_expr

    min_profit_margin = 0.15
    model_scip.addCons(profit_expr >= min_profit_margin * revenue_expr, name="min_profit_margin")

    max_revenue = sum(problem_params.price_ranges[i][1] * norm_params.output_max[i]
                     for i in range(n_products))
    revenue_var = model_scip.addVar(lb=0, ub=max_revenue, vtype='C', name='total_revenue')
    model_scip.addCons(revenue_var == revenue_expr, name='revenue_def')
    model_scip.setObjective(revenue_var, sense='maximize')

    print("Информация о модели (pyscipopt):")
    print(f"  Переменных: {model_scip.getNVars()}")
    print(f"  Ограничений: {model_scip.getNConss()}")
    print()

    print("Решение через pyscipopt...")
    start_time = time.time()
    status = model_scip.optimize()
    scip_time = time.time() - start_time

    scip_results = {
        'status': model_scip.getStatus(),
        'solve_time': scip_time,
        'n_vars': model_scip.getNVars(),
        'n_conss': model_scip.getNConss()
    }

    if model_scip.getStatus() in ['optimal', 'best solution found', 'feasible']:
        scip_results['total_revenue'] = 0
        scip_results['total_profit'] = 0
        scip_results['optimal_prices'] = []
        scip_results['demands'] = []

        for i in range(n_products):
            price = model_scip.getVal(price_vars[i])
            demand = model_scip.getVal(demand_vars[i])
            revenue = price * demand
            profit = (price - problem_params.costs[i]) * demand

            scip_results['optimal_prices'].append(price)
            scip_results['demands'].append(demand)
            scip_results['total_revenue'] += revenue
            scip_results['total_profit'] += profit

        scip_results['profit_margin'] = scip_results['total_profit'] / scip_results['total_revenue']

        print("\nРЕЗУЛЬТАТЫ PYSCIPOPT:")
        print(f"  Время решения: {scip_time:.2f} сек")
        print(f"  Выручка: {scip_results['total_revenue']:.2f}")
        print(f"  Прибыль: {scip_results['total_profit']:.2f}")
        print(f"  Маржа: {scip_results['profit_margin'] * 100:.1f}%")
        print(f"  Цены: {[f'{p:.2f}' for p in scip_results['optimal_prices']]}")
    else:
        print(f"\nPYSCIPOPT: решение не найдено, статус: {model_scip.getStatus()}")

    # ======================================================================
    # STEP 3b: Оптимизация через BONMIN
    # ======================================================================
    print("\n" + "="*70)
    print("ТЕСТ 2: BONMIN/PYOMO + ПОЛИНОМ")
    print("="*70 + "\n")

    optimizer_bonmin = PriceOptimizerPolynomialPyomo(
        demand_net=demand_net,
        norm_params=norm_params,
        problem_params=problem_params,
        solver_name="bonmin",
        solver_path=r"C:\distr\solvers\bonmin.exe",
        min_profit_margin=0.15
    )

    start_time = time.time()
    bonmin_results = optimizer_bonmin.optimize(time_limit=bonmin_time_limit)
    bonmin_time = time.time() - start_time

    bonmin_results['solve_time'] = bonmin_time

    # ======================================================================
    # STEP 3c: Оптимизация через COUENNE
    # ======================================================================
    print("\n" + "="*70)
    print("ТЕСТ 3: COUENNE/PYOMO + ПОЛИНОМ")
    print("="*70 + "\n")

    optimizer_couenne = PriceOptimizerPolynomialPyomo(
        demand_net=demand_net,
        norm_params=norm_params,
        problem_params=problem_params,
        solver_name="couenne",
        solver_path=r"C:\distr\solvers\couenne.exe",
        min_profit_margin=0.15
    )

    start_time = time.time()
    couenne_results = optimizer_couenne.optimize(time_limit=bonmin_time_limit)
    couenne_time = time.time() - start_time

    couenne_results['solve_time'] = couenne_time

    # ======================================================================
    # STEP 4: Сравнение результатов
    # ======================================================================
    print("\n" + "="*70)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*70 + "\n")

    print(f"{'Метрика':<30} {'PYSCIP':<12} {'BONMIN':<12} {'COUENNE':<12}")
    print("-" * 66)

    if scip_results['status'] in ['optimal', 'best solution found', 'feasible']:
        print(f"{'Статус':<30} {scip_results['status']:<12} {bonmin_results['status']:<12} {couenne_results['status']:<12}")
        print(f"{'Время (сек)':<30} {scip_results['solve_time']:<12.2f} {bonmin_results['solve_time']:<12.2f} {couenne_results['solve_time']:<12.2f}")
        print(f"{'Переменных (PYSCIP)':<30} {scip_results['n_vars']:<12} {'N/A':<12} {'N/A':<12}")
        print(f"{'Ограничений (PYSCIP)':<30} {scip_results['n_conss']:<12} {'N/A':<12} {'N/A':<12}")
        print(f"{'Выручка':<30} {scip_results['total_revenue']:<12.2f} {bonmin_results['total_revenue']:<12.2f} {couenne_results['total_revenue']:<12.2f}")
        print(f"{'Прибыль':<30} {scip_results['total_profit']:<12.2f} {bonmin_results['total_profit']:<12.2f} {couenne_results['total_profit']:<12.2f}")
        print(f"{'Маржа (%)':<30} {scip_results['profit_margin']*100:<12.1f} {bonmin_results['profit_margin']*100:<12.1f} {couenne_results['profit_margin']*100:<12.1f}")

        # Сравнение цен
        print(f"\n{'Оптимальные цены':<30}")
        for i in range(n_products):
            pscip = scip_results['optimal_prices'][i]
            bonmin = bonmin_results['optimal_prices'][i] if bonmin_results['optimal_prices'] else 0
            couenne = couenne_results['optimal_prices'][i] if couenne_results['optimal_prices'] else 0
            print(f"  Товар {i+1}: PYSCIP={pscip:.2f}, BONMIN={bonmin:.2f}, COU={couenne:.2f}")

        # Вывод о производительности
        print("\n" + "-" * 66)

        # Лучший solver по скорости
        times = [
            ('PYSCIP', scip_results['solve_time']),
            ('BONMIN', bonmin_results['solve_time']),
            ('COUENNE', couenne_results['solve_time'])
        ]
        fastest = min(times, key=lambda x: x[1])
        print(f"Самый быстрый: {fastest[0]} ({fastest[1]:.2f} сек)")

        # Лучший solver по выручке
        revenues = [
            ('PYSCIP', scip_results['total_revenue']),
            ('BONMIN', bonmin_results['total_revenue']),
            ('COUENNE', couenne_results['total_revenue'])
        ]
        best_revenue = max(revenues, key=lambda x: x[1])
        print(f"Лучшая выручка: {best_revenue[0]} ({best_revenue[1]:.2f})")
    else:
        print(f"PYSCIP не решил задачу")

    print()

    return scip_results, bonmin_results, couenne_results


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Pipeline:
    1. generate_training_data() → Dict с данными и параметрами
    2. Обучение нейросети
    3. Оптимизация через pyomo + bonmin (формула из нейросети)
    """

    n_products = 3
    n_context_cont = 2
    n_context_bin = 2

    print("="*70)
    print("PIPELINE: ГЕНЕРАЦИЯ ДАННЫХ -> ОБУЧЕНИЕ -> ОПТИМИЗАЦИЯ (BONMIN)")
    print("="*70 + "\n")

    # ======================================================================
    # STEP 1: Генерация данных
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
    # STEP 3: Оптимизация через pyomo + bonmin
    # ======================================================================
    print("ШАГ 3: МИНЛП оптимизация (PYOMO + BONMIN)\n")

    optimizer = PriceOptimizerBONMIN(
        demand_net=demand_net,
        norm_params=norm_params,
        problem_params=problem_params,
        solver_path=r"C:\distr\solvers\bonmin.exe",
        min_profit_margin=0.15
    )

    results = optimizer.optimize(time_limit=120)

    # ======================================================================
    # STEP 4: Визуализация
    # ======================================================================
    if results['optimal_prices']:
        visualize_results(optimizer, results)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        # Запуск бенчмарка сравнения pyscipopt vs bonmin
        benchmark_polynomial_model()
    else:
        # Обычный запуск с нейросетью
        main()
