import os
from time import time
import subprocess as sub
import argparse
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import imageio

from cec2013.cec2013 import *

from voxcraftevo.evo.algorithms import Solver
from voxcraftevo.listeners.listener import Listener
from voxcraftevo.utils.utilities import set_seed, get_bound
from voxcraftevo.evo.objectives import ObjectiveDict
from voxcraftevo.fitness.evaluation import FitnessFunction


def parse_args():
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--seed", default=0, type=int, help="seed for random number generation")
    parser.add_argument("--solver", default="kmeans", type=str, help="solver for the optimization")
    parser.add_argument("--gens", default=40, type=int, help="generations for the ea")
    parser.add_argument("--popsize", default=100, type=int, help="population size for the ea")
    parser.add_argument("--time", default=48, type=int, help="maximum hours for the ea")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="relative path to output dir")
    parser.add_argument("--fitness", default="fitness_score", type=str, help="fitness tag")
    parser.add_argument("--num_dims", default=10, type=int, help="number of problem dimensions")
    parser.add_argument("--num_targets", default=1, type=int, help="number of targets")
    parser.add_argument("--num_clusters", default=1, type=int, help="number of clusters")
    parser.add_argument("--clustering", default=None, type=str, help="clustering algorithm")
    parser.add_argument("--problem", default=None, type=int, help="CEC2013 problem to test")
    parser.add_argument("--visualize", default=0, type=int, help="visualize evolution")
    return parser.parse_args()


class MyListener(Listener):

    def __init__(self, file_path: str, header: Iterable[str], targets: Iterable[float]):
        super().__init__(file_path, header)
        self.targets = targets

    def listen(self, solver):
        with open(self._file, "a") as file:
            file.write(self._delimiter.join([str(solver.pop.gen), str(solver.elapsed_time()),
                                             str(solver.get_best_fitness()),
                                             str(solver.get_average_distance(self.targets)),
                                             "/".join([str(solver.get_best_distance(target))
                                                       for target in self.targets])]
                                            ) + "\n")


class VizListener(Listener):

    def __init__(self, file_path: str, header: Iterable[str], targets: Iterable[float]):
        super().__init__(file_path, header)
        self._inner_listener = MyListener(file_path=file_path, header=header, targets=targets)
        self.images = []
        os.system("rm -rf frames")
        os.makedirs("frames")

    def listen(self, solver) -> None:
        self._inner_listener.listen(solver=solver)
        r_min, r_max = -6.0, 6.0
        x_axis = np.arange(r_min, r_max, 0.05)
        y_axis = np.arange(r_min, r_max, 0.05)
        x, y = np.meshgrid(x_axis, y_axis)
        results = np.array([[solver.fitness_func.evaluate(np.array([x[i, j], y[i, j]])) for i in range(len(x_axis))]
                            for j in range(len(y_axis))])
        plt.pcolormesh(x_axis, y_axis, results, cmap="plasma")
        plt.scatter(2.0, 2.0, marker="o", color="white")
        plt.scatter(-2.0, -2.0, marker="o", color="white")
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
                                          tag="<{}>".format("fitness_score"),
                                          best_value=0.0, worst_value=float("inf"))
        return self.objective_dict

    def get_fitness(self, individual):
        return {"fitness_score": self.evaluate(x=individual.genotype)}

    def evaluate(self, x):
        return self.function.evaluate(x=x)


if __name__ == "__main__":
    arguments = parse_args()
    set_seed(arguments.seed)
    seed = arguments.seed
    if arguments.visualize == 1:
        listener = VizListener(file_path=".".join([str(arguments.clustering), str(seed), str(arguments.num_clusters),
                                                   str(arguments.num_dims), str(arguments.num_targets), "txt"]),
                               header=["iteration", "elapsed.time", "best.fitness", "avg.distance", "distances"],
                               targets=targets)
    else:
        listener = MyListener(file_path=".".join([str(arguments.clustering), str(seed), str(arguments.num_clusters),
                                                  str(arguments.num_dims), str(arguments.num_targets), "txt"]),
                              header=["iteration", "elapsed.time", "best.fitness", "avg.distance", "distances"],
                              targets=targets)
    fitness = MyFitness(function=CEC2013(arguments.problem))
    number_of_params = fitness.function.get_dimension()
    if arguments.solver == "es":
        evolver = Solver.create_solver(name="es", seed=seed, pop_size=arguments.popsize, num_dims=number_of_params,
                                       genotype_factory="uniform_float",
                                       solution_mapper="direct",
                                       fitness_func=fitness,
                                       listener=listener,
                                       sigma=0.3, sigma_decay=0.999, sigma_limit=0.01, l_rate_init=0.02,
                                       l_rate_decay=0.999, l_rate_limit=0.001,
                                       range=get_bound(fitness.function), upper=2.0, lower=-1.0)
    elif arguments.solver == "kmeans":
        evolver = Solver.create_solver(name="kmeans", seed=seed, pop_size=arguments.popsize, num_dims=number_of_params,
                                       num_modes=n_modes, genotype_factory="uniform_float",
                                       solution_mapper="direct",
                                       fitness_func=fitness,
                                       listener=listener, elite_ratio=0.5, clustering=arguments.clustering,
                                       sigma=0.3, sigma_decay=1.0 - 1.0 / arguments.gens, sigma_limit=0.01,
                                       l_rate_init=0.02, l_rate_decay=0.999, l_rate_limit=0.001,
                                       range=get_bound(fitness.function),
                                       upper=2.0, lower=-1.0)
    else:
        raise ValueError("Invalid solver name: {}".format(arguments.solver))
    start_time = time()
    evolver.solve(max_hours_runtime=arguments.time, max_gens=arguments.gens)
    if isinstance(listener, VizListener):
        listener.save_gif()
    sub.call("echo That took a total of {} seconds".format(time() - start_time), shell=True)
