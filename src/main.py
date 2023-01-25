from time import time
import subprocess as sub
import argparse
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.stats import multivariate_normal
from scipy.optimize import differential_evolution

from cec2013.cec2013 import *

from voxcraftevo.evo.algorithms import Solver
from voxcraftevo.listeners.listener import Listener
from voxcraftevo.utils.utilities import set_seed, get_bound
from voxcraftevo.evo.objectives import ObjectiveDict
from voxcraftevo.fitness.evaluation import FitnessFunction

__ACCURACIES__ = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]


def parse_args():
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--seed", default=0, type=int, help="seed for random number generation")
    parser.add_argument("--popsize", default=100, type=int, help="population size for the ea")
    parser.add_argument("--solver", default=None, type=str, help="ea")
    parser.add_argument("--elite", default=0.5, type=float, help="elite ratio for the ea")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="relative path to output dir")
    parser.add_argument("--fitness", default="fitness_score", type=str, help="fitness tag")
    parser.add_argument("--clustering", default=None, type=str, help="clustering algorithm")
    parser.add_argument("--problem", default=None, type=str, help="CEC2013 problem to test")
    parser.add_argument("--visualize", default=0, type=int, help="visualize evolution")
    return parser.parse_args()


class MyListener(Listener):

    def __init__(self, file_path: str, header: Iterable[str]):
        super().__init__(file_path, header)
        self.accuracies = __ACCURACIES__

    def listen(self, solver):
        with open(self._file, "a") as file:
            file.write(self._delimiter.join([str(solver.pop.gen), str(solver.elapsed_time()),
                                             str(solver.get_best_fitness())] +
                                            [str(self._get_recall(solver, accuracy=acc)) for acc in self.accuracies])
                       + "\n")

    def _get_recall(self, solver, accuracy):
        pop = np.array([ind.genotype for ind in solver.pop])
        if isinstance(solver.fitness_func.function, SoG):
            count = solver.fitness_func.function.how_many_goptima(pop, accuracy)
        else:
            count, _ = how_many_goptima(pop, solver.fitness_func.function, accuracy)
        return count / solver.fitness_func.function.get_no_goptima()


class VizListener(Listener):

    def __init__(self, file_path: str, header: Iterable[str]):
        super().__init__(file_path, header)
        self._inner_listener = MyListener(file_path=file_path, header=header)
        self.images = []
        os.system("rm -rf frames")
        os.makedirs("frames")

    def listen(self, solver) -> None:
        if solver.fitness_func.function.get_dimension() != 2:
            raise ValueError("Visualizing a non-bidimensional function")
        self._inner_listener.listen(solver=solver)
        x_axis = np.arange(solver.fitness_func.function.get_lbound(0),
                           solver.fitness_func.function.get_ubound(0), 0.05)
        y_axis = np.arange(solver.fitness_func.function.get_lbound(1),
                           solver.fitness_func.function.get_ubound(1), 0.05)
        x, y = np.meshgrid(x_axis, y_axis)
        results = np.array([[solver.fitness_func.function.evaluate(x=np.array([x[_j, _i], y[_j, _i]]))
                             for _i in range(x.shape[1])]
                            for _j in range(x.shape[0])])
        plt.pcolormesh(x_axis, y_axis, results, cmap="plasma")
        plt.scatter([ind.genotype[0] for ind in solver.pop], [ind.genotype[1] for ind in solver.pop], marker="x",
                    color="red")
        image = "frames/{}.png".format(solver.pop.gen)
        plt.savefig(image)
        self.images.append(image)
        plt.clf()

    def save_gif(self):
        with imageio.get_writer(self._file.replace("txt", "gif"), mode="I") as writer:
            for filename in self.images:
                image = imageio.imread(filename)
                writer.append_data(image)
        os.system("rm -rf frames")


class MyFitness(FitnessFunction):

    def __init__(self, function: CEC2013):
        self.objective_dict = ObjectiveDict()
        self.function = function

    def create_objectives_dict(self):
        self.objective_dict.add_objective(name="fitness_score", maximize=False,
                                          best_value=- self.function.get_fitness_goptima(),
                                          worst_value=float("inf"))
        return self.objective_dict

    def get_fitness(self, individual):
        fit = - self.function.evaluate(x=individual.genotype)
        if not isinstance(fit, float):
            fit = fit[0]
        return {"fitness_score": fit if fit is not None else float("inf")}


class SoG(object):

    def __init__(self, d, n):
        self.d = d
        self.n = n
        delta = abs(self.get_ubound(0) - self.get_lbound(0)) / self.n
        self.means = self._sample_means(delta=delta)
        self.h = np.zeros((self.n, self.d))
        alpha = delta / self.d
        smallest_idx, smallest_bandwidth = self._fill_h(alpha=alpha)
        self.root = differential_evolution(self.evaluate, bounds=[(self.means[smallest_idx, i] - smallest_bandwidth,
                                                                   self.means[smallest_idx, i] + smallest_bandwidth)
                                                                  for i in range(self.d)])

    def _sample_means(self, delta):
        means = np.zeros((self.n, self.d))
        for i in range(self.n):
            sample = np.random.uniform(low=self.get_lbound(0) + 0.1, high=self.get_ubound(0) - 0.1, size=(1, self.d))
            while any([np.linalg.norm(sample - m) < delta for m in means]):
                sample = np.random.uniform(low=self.get_lbound(0) + 0.1, high=self.get_ubound(0) - 0.1, size=(1, self.d))
            means[i] = sample
        return means

    def _fill_h(self, alpha):
        smallest_idx = -1
        smallest_bandwidth = float("inf")
        for i, m in enumerate(self.means):
            bandwidth = alpha * np.min([np.linalg.norm(other_m - m) for j, other_m in enumerate(self.means) if i != j])
            self.h[i] = np.full(shape=(self.d,), fill_value=bandwidth)
            smallest_idx = i if bandwidth < smallest_bandwidth else smallest_idx
            smallest_bandwidth = min(bandwidth, smallest_bandwidth)
        return smallest_idx, smallest_bandwidth

    def get_dimension(self):
        return self.d

    def get_lbound(self, i):
        return -1.0

    def get_ubound(self, i):
        return 1.0

    def evaluate(self, x):
        return np.sum([multivariate_normal.pdf(x, mean=self.means[i], cov=self.h[i]) for i in range(self.n)])

    def get_fitness_goptima(self):
        return self.root

    def get_no_goptima(self):
        return self.n

    def get_maxfes(self):
        return self.d * 1000

    def how_many_goptima(self, pop, accuracy):
        count = 0
        for m in self.means:
            best = min(pop, key=lambda x: np.linalg.norm(m - x))
            count += 1 if abs(self.evaluate(x=m) - self.evaluate(x=best)) < accuracy else 0
        return count


if __name__ == "__main__":
    arguments = parse_args()
    set_seed(arguments.seed)
    seed = arguments.seed
    if arguments.solver == "es":
        file_name = ".".join([arguments.solver, str(seed), str(arguments.popsize), arguments.problem, "txt"])
    else:
        file_name = ".".join([arguments.clustering, str(seed), str(arguments.popsize), str(arguments.elite),
                              arguments.problem, "txt"])
    if arguments.visualize == 1:
        listener = VizListener(file_path=file_name,
                               header=["iteration", "elapsed.time", "best.fitness"] +
                                      ["pr-{}".format(acc) for acc in __ACCURACIES__])
    else:
        listener = MyListener(file_path=file_name,
                              header=["iteration", "elapsed.time", "best.fitness"] +
                                     ["pr-{}".format(acc) for acc in __ACCURACIES__])
    pid = int(arguments.problem.split("-")[0])
    fitness = MyFitness(function=CEC2013(pid) if pid != 0 else SoG(int(arguments.problem.split("-")[1]),
                                                                   int(arguments.problem.split("-")[2])))
    number_of_params = fitness.function.get_dimension()
    pop_size = arguments.popsize
    gens = 200  # fitness.function.get_maxfes() // pop_size
    if arguments.solver == "es":
        evolver = Solver.create_solver(name="es",
                                       seed=seed,
                                       pop_size=pop_size,
                                       num_dims=number_of_params,
                                       genotype_factory="uniform_float",
                                       solution_mapper="direct",
                                       fitness_func=fitness,
                                       listener=listener,
                                       sigma=np.mean([math.floor(abs(fitness.function.get_ubound(i) -
                                                                     fitness.function.get_lbound(i)))
                                                      for i in range(fitness.function.get_dimension())]) * 0.1,
                                       sigma_decay=1.0 - 1.0 / gens,
                                       sigma_limit=0.0001,
                                       l_rate_init=0.02,
                                       l_rate_decay=1.0 - 1.0 / gens,
                                       l_rate_limit=0.001,
                                       range=get_bound(fitness.function),
                                       upper=2.0,
                                       lower=-1.0)
    else:
        evolver = Solver.create_solver(name="multimodal",
                                       seed=seed,
                                       pop_size=pop_size,
                                       num_dims=number_of_params,
                                       num_modes=fitness.function.get_no_goptima(),
                                       genotype_factory="uniform_float",
                                       solution_mapper="direct",
                                       fitness_func=fitness,
                                       listener=listener,
                                       elite_ratio=0.5,
                                       clustering=arguments.clustering,
                                       sigma=np.mean([math.floor(abs(fitness.function.get_ubound(i) -
                                                                     fitness.function.get_lbound(i)))
                                                      for i in range(fitness.function.get_dimension())]) * 0.1,
                                       sigma_decay=1.0 - 1.0 / gens,
                                       sigma_limit=0.0001,
                                       range=get_bound(fitness.function),
                                       upper=2.0,
                                       lower=-1.0)
    start_time = time()
    evolver.solve(max_gens=gens)
    if isinstance(listener, VizListener):
        listener.save_gif()
    sub.call("echo That took a total of {} seconds".format(time() - start_time), shell=True)
