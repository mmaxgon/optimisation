from mip import Model, maximize
import numpy as np
import time

def get_model_template(nclients=20):
    np.random.seed(987234)

    product_policy = 10 * np.ones((5, 5), dtype=int)
    channel_policy = np.array([
        [45, 20, 10, 6],
        [20, 22, 12, 7],
        [ 8, 12, 11, 6],
        [ 6,  5,  5, 9],
    ], dtype=int)
    nproducts = product_policy.shape[0]
    nchannels = channel_policy.shape[0]
    ntime = 30
    max_comms_per_client = 3
    max_total_comms = int(max_comms_per_client * nclients * 0.5)
    revenue = np.round(np.random.random(size=(nclients, nproducts, nchannels)), 2)

    model = Model()

    x = np.zeros((nclients, nproducts, nchannels, ntime), dtype=object)
    for k in range(nclients):
        for p in range(nproducts):
            for c in range(nchannels):
                for t in range(ntime):
                    x[k, p, c, t] = model.add_var(var_type='B')

    model.objective = maximize(np.sum(revenue * np.sum(x, axis=3)))

    for k in range(nclients):
        model.add_constr(np.sum(x[k, :, :, :]) <= max_comms_per_client)

    model.add_constr(np.sum(x) <= max_total_comms)

    return model, x, product_policy, channel_policy

def add_policy_1(model, x, product_policy, channel_policy):
    nclients, nproducts, nchannels, ntime = x.shape
    M = 1000
    for k in range(nclients):
        for t in range(ntime):
            model.add_constr(np.sum(x[k, :, :, t]) <= 1)
    for k in range(nclients):
        for c1 in range(nchannels):
            for c2 in range(nchannels):
                for t in range(ntime):
                    d = channel_policy[c1, c2]
                    model.add_constr(
                        M * np.sum(x[k, :, c1, t]) + np.sum(x[k, :, c2, t+1:t+d]) <= M
                    )
    for k in range(nclients):
        for p1 in range(nproducts):
            for p2 in range(nproducts):
                for t in range(ntime):
                    d = product_policy[p1, p2]
                    model.add_constr(
                        M * np.sum(x[k, p1, :, t]) + np.sum(x[k, p2, :, t+1:t+d]) <= M
                    )

def add_policy_2(model, x, product_policy, channel_policy):
    nclients, nproducts, nchannels, ntime = x.shape
    for k in range(nclients):
        for t in range(ntime):
            model.add_constr(np.sum(x[k, :, :, t]) <= 1)
        for c1 in range(nchannels):
            for c2 in range(nchannels):
                for t in range(ntime):
                    for p1 in range(nproducts):
                        for p2 in range(nproducts):
                            d = channel_policy[c1, c2]
                            for t2 in range(t + 1, min(t + d, ntime)):
                                model.add_constr(
                                    x[k, p1, c1, t] + x[k, p2, c2, t2] <= 1
                                )
        for p1 in range(nproducts):
            for p2 in range(nproducts):
                for t in range(ntime):
                    for c1 in range(nchannels):
                        for c2 in range(nchannels):
                            d = product_policy[p1, p2]
                            for t2 in range(t + 1, min(t + d, ntime)):
                                model.add_constr(
                                    x[k, p1, c1, t] + x[k, p2, c2, t2] <= 1
                                )
    model.clique_merge()

def add_policy_3(model, x, product_policy, channel_policy):
    nclients, nproducts, nchannels, ntime = x.shape
    merged_policy = transform_policy(product_policy, channel_policy, ntime)
    for k in range(nclients):
        for policy_cons in merged_policy:
            model.add_constr(sum(x[k, p, c, t] for p, c, t in policy_cons) <= 1)

def transform_policy(product_policy, channel_policy, ntime):
    nproducts = product_policy.shape[0]
    nchannels = channel_policy.shape[0]

    model = Model()

    x = np.zeros((nproducts, nchannels, ntime), dtype=object)
    for p in range(nproducts):
        for c in range(nchannels):
            for t in range(ntime):
                x[p, c, t] = model.add_var(var_type='B', name=f'x({p},{c},{t})')

    for t in range(ntime):
        model.add_constr(np.sum(x[:, :, t]) <= 1)
    for c1 in range(nchannels):
        for c2 in range(nchannels):
            for t in range(ntime):
                for p1 in range(nproducts):
                    for p2 in range(nproducts):
                        d = channel_policy[c1, c2]
                        for t2 in range(t + 1, min(t + d, ntime)):
                            model.add_constr(
                                x[p1, c1, t] + x[p2, c2, t2] <= 1
                            )
    for p1 in range(nproducts):
        for p2 in range(nproducts):
            for t in range(ntime):
                for c1 in range(nchannels):
                    for c2 in range(nchannels):
                        d = product_policy[p1, p2]
                        for t2 in range(t + 1, min(t + d, ntime)):
                            model.add_constr(
                                x[p1, c1, t] + x[p2, c2, t2] <= 1
                            )
    model.clique_merge()
    merged_policy = []
    for cons in model.constrs:
        policy_cons = []
        for v in cons.expr.expr:
            idx = tuple(map(int, v.name.split('(')[1].split(')')[0].split(',')))
            policy_cons.append(idx)
        merged_policy.append(policy_cons)
    return merged_policy

def example1():
    np.random.seed(987234)

    product_policy = 10 * np.ones((5, 5), dtype=int)
    channel_policy = np.array([
        [45, 20, 10, 6],
        [20, 22, 12, 7],
        [ 8, 12, 11, 6],
        [ 6,  5,  5, 9],
    ], dtype=int)
    nclients = 10
    nproducts = product_policy.shape[0]
    nchannels = channel_policy.shape[0]
    ntime = 30
    max_comms_per_client = 3
    max_total_comms = int(max_comms_per_client * nclients * 0.5)
    revenue = np.round(np.random.random(size=(nclients, nproducts, nchannels)), 2)

    model = Model()

    x = np.zeros((nclients, nproducts, nchannels, ntime), dtype=object)
    for k in range(nclients):
        for p in range(nproducts):
            for c in range(nchannels):
                for t in range(ntime):
                    x[k, p, c, t] = model.add_var(var_type='B')

    model.objective = maximize(np.sum(revenue * np.sum(x, axis=3)))

    for k in range(nclients):
        for t in range(ntime):
            model.add_constr(np.sum(x[k, :, :, t]) <= 1)

    for k in range(nclients):
        model.add_constr(np.sum(x[k, :, :, :]) <= max_comms_per_client)

    model.add_constr(np.sum(x) <= max_total_comms)

    add_policy_1(model, x, product_policy, channel_policy)

    model.max_seconds = 120
    model.max_mip_gap = 0.01
    model.preprocess = 0
    model.optimize()

def example2():
    product_policy = 10 * np.ones((5, 5), dtype=int)
    channel_policy = np.array([
        [45, 20, 10, 6],
        [20, 22, 12, 7],
        [ 8, 12, 11, 6],
        [ 6,  5,  5, 9],
    ], dtype=int)
    nproducts = product_policy.shape[0]
    nchannels = channel_policy.shape[0]
    ntime = 30

    model = Model()

    x = np.zeros((nproducts, nchannels, ntime), dtype=object)
    for p in range(nproducts):
        for c in range(nchannels):
            for t in range(ntime):
                x[p, c, t] = model.add_var(var_type='B', name=f'x({p},{c},{t})')

    for t in range(ntime):
        model.add_constr(np.sum(x[:, :, t]) <= 1)
    for c1 in range(nchannels):
        for c2 in range(nchannels):
            for t in range(ntime):
                for p1 in range(nproducts):
                    for p2 in range(nproducts):
                        d = channel_policy[c1, c2]
                        for t2 in range(t + 1, min(t + d, ntime)):
                            model.add_constr(
                                x[p1, c1, t] + x[p2, c2, t2] <= 1
                            )
    for p1 in range(nproducts):
        for p2 in range(nproducts):
            for t in range(ntime):
                for c1 in range(nchannels):
                    for c2 in range(nchannels):
                        d = product_policy[p1, p2]
                        for t2 in range(t + 1, min(t + d, ntime)):
                            model.add_constr(
                                x[p1, c1, t] + x[p2, c2, t2] <= 1
                            )

    nbefore = len(model.constrs)
    model.clique_merge()
    nafter = len(model.constrs)
    print(f'Число ограничений до    {nbefore}')
    print(f'Число ограничений после {nafter}')

    print(model.constrs[0])

    merged_policy = []
    for cons in model.constrs:
        policy_cons = []
        for v in cons.expr.expr:
            idx = tuple(map(int, v.name.split('(')[1].split(')')[0].split(',')))
            policy_cons.append(idx)
        merged_policy.append(policy_cons)




if __name__ == '__main__':
    t_solve = time.time()

    model, x, product_policy, channel_policy = get_model_template()
    add_policy_1(model, x, product_policy, channel_policy)
    ncons = len(model.constrs)

    model.max_seconds = 120
    model.max_mip_gap = 0.0
    model.preprocess = 0
    model.verbose = 0
    model.optimize()

    t_solve = time.time() - t_solve
    print(f'Число ограничений: {ncons}')
    print(f'Целевая функция  : {model.objective_value}')
    print(f'Время            : {t_solve:.2f} сек')