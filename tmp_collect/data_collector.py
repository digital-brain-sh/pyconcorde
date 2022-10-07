import os
import argparse
import time
import random
import pickle
import subprocess
import numpy as np
import ray

from ray.util import ActorPool
from concorde.tsp import TSPSolver


@ray.remote(num_cpus=0.1)
def generate_traj(seed: int, x_range, y_range, candidates, coords, scale: int = 200):
    random.seed(seed)
    idxes = tuple(random.sample(candidates, scale))
    selected = coords[idxes, :]

    xs = selected[:, 0] # * (x_range[1] - x_range[0]) + x_range[0]
    ys = selected[:, 1] # * (y_range[1] - y_range[0]) + y_range[0]

    solver = TSPSolver.from_data(xs, ys, norm="EUC_2D")
    solution = solver.solve(verbose=False, random_seed=seed)

    dataset = {
        'data': selected,
        'seq': solution.tour,
        'val': solution.optimal_value,
        'idxes': idxes
    }

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=200)

    args = parser.parse_args()

    with open('./coord.txt', 'r') as f:
        lines = f.readlines()

    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    # convert data to list
    data = []
    for line in lines:
        data.append(tuple(map(float, line.split())))
    data = np.asarray(data)
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()

    # norm it
    x_range = (x_min, x_max)
    y_range = (y_min, y_max)
    data[:, 0] = (data[:, 0] - x_min) / (x_max - x_min)
    data[:, 1] = (data[:, 1] - y_min) / (y_max - y_min)
    candidates = list(range(data.shape[0]))

    print("candidate node list:", len(candidates))
    print("x range:", data[:, 0].min(), data[:, 0].max())
    print("y range:", data[:, 1].min(), data[:, 1].max())

    actor_num = 50
    actors = [generate_traj.remote for _ in range(actor_num)]
    pool = ActorPool(actors)

    seeds = list(range(0, 100))
    for seed in seeds:
        dataset = {'data': [], 'seq': [], 'val': []}
        total_consum = 0

        for i in range(0, 10000, actor_num):
            tasks = pool.map(lambda a, v: a(v, x_range, y_range, candidates=candidates, coords=data), list(range(i, i + actor_num)))
            start = time.time()
            for res in tasks:
                for k, v in res.items():
                    if k == "idxes":
                        # print(v)
                        pass
                    else:
                        dataset[k].append(v)
            episode_consum = time.time() - start
            total_consum += episode_consum

            for file in os.listdir("./"):
                if file.endswith(".sol") or file.endswith(".res"):
                    os.remove(os.path.join("./", file))
            optim_val = np.mean(dataset['val'])
            print(f"{i + actor_num}/10000, time consum: {episode_consum}, total consum: {total_consum}, mean-optimal-value: {optim_val}")

        with open(f'./dataset/tsp{args.scale}_city_seed{seed}.pkl', 'wb') as f:
            pickle.dump(dataset, f)