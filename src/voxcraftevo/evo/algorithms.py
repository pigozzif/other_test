import abc
import time
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import subprocess as sub
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .operators.operator import GeneticOperator
from .optimizers import Adam
from .selection.filters import Filter
from .selection.selector import Selector
from ..fitness.evaluation import FitnessFunction
from ..listeners.listener import Listener
from ..representations.factory import GenotypeFactory
from ..representations.mapper import SolutionMapper
from ..representations.population import Population, Individual
from ..utils.utilities import weighted_random_by_dct, exp_decay

PLOT = True


class Solver(object):

    def __init__(self, seed: int, fitness_func: FitnessFunction, data_dir: str, hist_dir: str, pickle_dir: str,
                 output_dir: str, executables_dir: str, logs_dir: str):
        self.seed = seed
        self.fitness_func = fitness_func
        self.start_time = None
        self.best_so_far = None
        self.data_dir = data_dir

    def elapsed_time(self, units: str = "s") -> float:
        if self.start_time is None:
            self.start_time = time.time()
        s = time.time() - self.start_time
        if units == "s":
            return s
        elif units == "m":
            return s / 60.0
        elif units == "h":
            return s / 3600.0

    @abc.abstractmethod
    def solve(self, max_hours_runtime: int, max_gens: int, checkpoint_every: int, save_hist_every: int):
        pass

    @classmethod
    def create_solver(cls, name: str, **kwargs):
        if name == "ga":
            return GeneticAlgorithm(**kwargs)
        elif name == "nsgaii":
            return NSGAII(**kwargs)
        elif name == "es":
            return EvolutionaryStrategy(**kwargs)
        elif name == "kmeans":
            return KMeansStrategy(**kwargs)
        raise ValueError("Invalid solver name: {}".format(name))


class EvolutionarySolver(Solver):

    def __init__(self, seed, pop_size: int, genotype_factory: str, solution_mapper: str, fitness_func, remap: bool,
                 genetic_operators: Dict[str, float], data_dir, hist_dir, pickle_dir, output_dir, executables_dir,
                 logs_dir, listener: Listener, comparator: str, genotype_filter: str = None, **kwargs):
        super().__init__(seed, fitness_func, data_dir, hist_dir, pickle_dir, output_dir, executables_dir, logs_dir)
        self.pop_size = pop_size
        self.remap = remap
        self.continued_from_checkpoint = False
        self.pop = Population(pop_size=pop_size,
                              genotype_factory=GenotypeFactory.create_factory(name=genotype_factory,
                                                                              genotype_filter=Filter.create_filter(
                                                                                  genotype_filter), **kwargs),
                              solution_mapper=SolutionMapper.create_mapper(name=solution_mapper, **kwargs),
                              objectives_dict=self.fitness_func.create_objectives_dict(),
                              comparator=comparator)
        self.genetic_operators = {GeneticOperator.create_genetic_operator(name=k,
                                                                          genotype_filter=Filter.create_filter(
                                                                              genotype_filter), **kwargs):
                                      v for k, v in genetic_operators.items()}
        self.listener = listener

    def evaluate_individuals(self) -> None:
        to_evaluate = list(filter(lambda x: not x.evaluated, self.pop))
        for ind in to_evaluate:
            ind.fitness = self.fitness_func.get_fitness(individual=ind)
            ind.evaluated = True

    def solve(self, max_hours_runtime, max_gens, checkpoint_every, save_hist_every) -> None:
        self.start_time = time.time()

        if not self.continued_from_checkpoint:  # generation zero
            self.evaluate_individuals()
        self.best_so_far = self.get_best()
        # iterate until stop conditions met
        while self.pop.gen < max_gens and self.elapsed_time(units="h") <= max_hours_runtime:
            sub.call("rm -rf {}/*".format(self.data_dir), shell=True)

            # update population stats
            self.pop.gen += 1
            self.pop.update_ages()
            self.best_so_far = self.get_best()
            # update evolution
            self.listener.listen(solver=self)
            self.evolve()
        self.listener.listen(solver=self)
        sub.call("echo Saving history of run champ at generation {0}".format(self.pop.gen + 1), shell=True)

    @abc.abstractmethod
    def evolve(self):
        pass

    def get_best(self) -> Individual:
        return self.pop.get_best()


class GeneticAlgorithm(EvolutionarySolver):

    def __init__(self, seed, pop_size, genotype_factory, solution_mapper, survival_selector: str, parent_selector: str,
                 fitness_func, offspring_size: int, overlapping: bool, remap, genetic_operators, data_dir, hist_dir,
                 pickle_dir, output_dir, executables_dir, logs_dir, listener, **kwargs):
        super().__init__(seed=seed, pop_size=pop_size, genotype_factory=genotype_factory,
                         solution_mapper=solution_mapper, fitness_func=fitness_func, remap=remap,
                         genetic_operators=genetic_operators, data_dir=data_dir, hist_dir=hist_dir,
                         pickle_dir=pickle_dir, output_dir=output_dir, executables_dir=executables_dir,
                         logs_dir=logs_dir, listener=listener, comparator="lexicase", **kwargs)
        self.survival_selector = Selector.create_selector(name=survival_selector, **kwargs)
        self.parent_selector = Selector.create_selector(name=parent_selector, **kwargs)
        self.offspring_size = offspring_size
        self.overlapping = overlapping

    def build_offspring(self) -> list:
        children_genotypes = []
        while len(children_genotypes) < self.offspring_size:
            operator = weighted_random_by_dct(dct=self.genetic_operators)
            parents = [parent.genotype for parent in self.parent_selector.select(population=self.pop,
                                                                                 n=operator.get_arity())]
            children_genotypes.append(operator.apply(tuple(parents)))
        return children_genotypes

    def trim_population(self) -> None:
        while len(self.pop) > self.pop_size:
            self.pop.remove_individual(self.survival_selector.select(population=self.pop, n=1)[0])

    def evolve(self) -> None:
        # apply genetic operators
        if not self.overlapping:
            self.pop.clear()
        for child_genotype in self.build_offspring():
            self.pop.add_individual(genotype=child_genotype)
        # evaluate individuals
        self.evaluate_individuals()
        # apply selection
        self.trim_population()


class EvolutionaryStrategy(EvolutionarySolver):

    def __init__(self, seed, pop_size, genotype_factory, solution_mapper, sigma: float, sigma_decay: float,
                 sigma_limit: float, num_dims: int, l_rate_init: float, l_rate_decay: float, l_rate_limit: float,
                 fitness_func, data_dir, hist_dir, pickle_dir, output_dir, executables_dir, logs_dir, listener,
                 **kwargs):
        super().__init__(seed=seed, pop_size=pop_size, genotype_factory=genotype_factory,
                         solution_mapper=solution_mapper, fitness_func=fitness_func, remap=False,
                         genetic_operators={}, data_dir=data_dir, hist_dir=hist_dir,
                         pickle_dir=pickle_dir, output_dir=output_dir, executables_dir=executables_dir,
                         logs_dir=logs_dir, listener=listener, comparator="lexicase", **kwargs)
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.num_dims = num_dims
        self.optimizer = Adam(num_dims=num_dims, l_rate_init=l_rate_init, l_rate_decay=l_rate_decay,
                              l_rate_limit=l_rate_limit)
        self.mode = np.zeros(num_dims)
        self.best_fitness = float("inf")
        self.temp_best = None

    def build_offspring(self) -> list:
        z_plus = np.random.normal(loc=0.0, scale=self.sigma, size=(self.pop_size // 2, self.num_dims))
        z = np.concatenate([z_plus, -1.0 * z_plus])
        return [self.mode + x * self.sigma for x in z]

    def update_mode(self) -> None:
        noise = np.array([(x.genotype - self.mode) / self.sigma for x in self.pop])
        fitness = np.array([x.fitness["fitness_score"] for x in self.pop])
        self.best_fitness = np.min(fitness)
        theta_grad = (1.0 / (self.pop_size * self.sigma)) * np.dot(noise.T, fitness)
        self.mode = self.optimizer.optimize(mean=self.mode, t=self.pop.gen, theta_grad=theta_grad)

    def evolve(self) -> None:
        self.pop.clear()
        for child_genotype in self.build_offspring():
            self.pop.add_individual(genotype=child_genotype)
        self.evaluate_individuals()
        self.update_mode()
        self.temp_best = self.pop.get_best()
        self.sigma = exp_decay(self.sigma, self.sigma_decay, self.sigma_limit)

    def get_best_fitness(self) -> float:
        return self.best_fitness

    def get_best(self) -> Individual:
        return self.temp_best


class KMeansStrategy(EvolutionarySolver):

    def __init__(self, seed, pop_size, genotype_factory, solution_mapper, clustering: str, elite_ratio: float, sigma: float, sigma_decay: float,
                 sigma_limit: float, num_dims: int, l_rate_init: float, l_rate_decay: float, l_rate_limit: float,
                 num_modes: int, fitness_func, data_dir, hist_dir, pickle_dir, output_dir, executables_dir, logs_dir,
                 listener, **kwargs):
        super().__init__(seed=seed, pop_size=pop_size, genotype_factory=genotype_factory,
                         solution_mapper=solution_mapper, fitness_func=fitness_func, remap=False,
                         genetic_operators={}, data_dir=data_dir, hist_dir=hist_dir,
                         pickle_dir=pickle_dir, output_dir=output_dir, executables_dir=executables_dir,
                         logs_dir=logs_dir, listener=listener, comparator="lexicase", **kwargs)
        self.elite_ratio = elite_ratio
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.num_dims = num_dims
        self.optimizers = [Adam(num_dims=num_dims, l_rate_init=l_rate_init, l_rate_decay=l_rate_decay,
                                l_rate_limit=l_rate_limit) for _ in range(num_modes)]
        self.num_modes = num_modes
        self.temp_best = None
        self.best_fitness = float("inf")
        self.mixture = GaussianMixture(n_components=self.num_modes, covariance_type="diag", max_iter=1)
        self.mixture.weights_ = np.full(shape=(self.num_modes,), fill_value=1 / self.num_modes)
        self.mixture.means_ = np.zeros(shape=(self.num_modes, self.num_dims))
        self.mixture.covariances_ = np.full(shape=(self.num_modes, self.num_dims), fill_value=self.sigma)
        self.clustering = clustering
        if self.clustering == "kmeans":
            self.optimizer = KMeans(n_clusters=self.num_modes, random_state=seed)
        elif self.clustering != "em":
            raise ValueError("Invalid clustering method: {}".format(self.clustering))

    def build_offspring(self) -> List[Individual]:
        return self.mixture.sample(self.pop_size)[0]

    def update_modes(self) -> None:
        self.best_fitness = min([x.fitness["fitness_score"] for x in self.pop])
        if self.clustering == "kmeans":
            self.mixture.means_ = self.optimizer.fit([ind.genotype for ind in sorted(self.pop, key=lambda x: x.fitness["fitness_score"])[:int(self.elite_ratio * self.pop_size)]]).cluster_centers_
        else:
            self.mixture.fit([ind.genotype for ind in sorted(self.pop, key=lambda x: x.fitness["fitness_score"])[:int(self.elite_ratio * self.pop_size)]])
            self.mixture.weights_ = np.full(shape=(self.num_modes,), fill_value=1 / self.num_modes)
            self.mixture.covariances_ = np.full(shape=(self.num_modes, self.num_dims), fill_value=self.sigma)

    def evolve(self) -> None:
        self.pop.clear()
        for child in self.build_offspring():
            self.pop.add_individual(genotype=child)
        self.evaluate_individuals()
        self.update_modes()
        self.sigma = exp_decay(self.sigma, self.sigma_decay, self.sigma_limit)

    def get_best_fitness(self) -> float:
        return self.best_fitness

    def get_best_distance(self, target) -> float:
        return np.min([np.linalg.norm(ind.genotype - target) for ind in self.pop])

    def get_average_distance(self, targets) -> float:
        closest = []
        for target in targets:
            closest.append(self.get_best_distance(target=target))
        return sum(closest) / len(closest)


class NSGAII(EvolutionarySolver):

    def __init__(self, seed, pop_size, genotype_factory, solution_mapper, fitness_func, offspring_size: int, remap,
                 genetic_operators, data_dir, hist_dir, pickle_dir, output_dir, executables_dir, logs_dir,
                 listener, **kwargs):
        super().__init__(seed=seed, pop_size=pop_size, genotype_factory=genotype_factory,
                         solution_mapper=solution_mapper, fitness_func=fitness_func, remap=remap,
                         genetic_operators=genetic_operators, data_dir=data_dir, hist_dir=hist_dir,
                         pickle_dir=pickle_dir, output_dir=output_dir, executables_dir=executables_dir,
                         logs_dir=logs_dir, listener=listener, comparator="pareto", **kwargs)
        self.offspring_size = offspring_size
        self.fronts = {}
        self.dominates = {}
        self.dominated_by = {}
        self.crowding_distances = {}
        self.parent_selector = Selector.create_selector(name="tournament_crowded",
                                                        crowding_distances=self.crowding_distances, fronts=self.fronts,
                                                        **kwargs)
        self._fronts_to_plot = {}

    def fast_non_dominated_sort(self) -> None:
        self.fronts.clear()
        self.dominates.clear()
        self.dominated_by.clear()
        for p in self.pop:
            self.dominated_by[p.id] = 0
            for q in self.pop:
                if p.id == q.id:
                    continue
                elif p > q:
                    if p.id not in self.dominates:
                        self.dominates[p.id] = [q]
                    else:
                        self.dominates[p.id].append(q)
                elif p < q:
                    self.dominated_by[p.id] += 1
            if self.dominated_by[p.id] == 0:
                if 0 not in self.fronts:
                    self.fronts[0] = [p]
                else:
                    self.fronts[0].append(p)
        if not self.fronts:
            self.fronts[0] = [ind for ind in self.pop]
            return
        i = 0
        while len(self.fronts[i]):
            self.fronts[i + 1] = []
            for p in self.fronts[i]:
                for q in self.dominates.get(p.id, []):
                    self.dominated_by[q.id] -= 1
                    if self.dominated_by[q.id] == 0:
                        self.fronts[i + 1].append(q)
            i += 1
        self.fronts.pop(i)
        self.crowding_distances.clear()
        for front in self.fronts.values():
            self.crowding_distance_assignment(individuals=front)

    def crowding_distance_assignment(self, individuals: list) -> None:
        for individual in individuals:
            self.crowding_distances[individual.id] = 0.0
        for rank, goal in self.pop.objectives_dict.items():
            individuals.sort(key=lambda x: x.fitness[goal["name"]], reverse=goal["maximize"])
            self.crowding_distances[individuals[0].id] = float("inf")
            self.crowding_distances[individuals[len(individuals) - 1].id] = float("inf")
            for i in range(1, len(individuals) - 1):
                self.crowding_distances[individuals[i].id] += (individuals[i + 1].fitness[goal["name"]] -
                                                               individuals[i - 1].fitness[goal["name"]]) / \
                                                              (abs(goal["best_value"] - goal["worst_value"]))

    def build_offspring(self) -> list:
        children_genotypes = []
        while len(children_genotypes) < self.offspring_size:
            operator = weighted_random_by_dct(dct=self.genetic_operators)
            parents = [parent.genotype for parent in self.parent_selector.select(population=self.pop,
                                                                                 n=operator.get_arity())]
            children_genotypes.append(operator.apply(tuple(parents)))
        return children_genotypes

    def trim_population(self) -> None:
        self.fast_non_dominated_sort()
        i = 0
        n = 0
        while n + len(self.fronts[i]) <= self.pop_size:
            n += len(self.fronts[i])
            i += 1
        self.fronts[i].sort(key=lambda x: self.crowding_distances[x.id])
        for j in range(len(self.fronts[i]) - self.pop_size + n):
            self.pop.remove_individual(ind=self.fronts[i][j])
        i += 1
        while i in self.fronts:
            for ind in self.fronts[i]:
                self.pop.remove_individual(ind=ind)
            i += 1

    def evolve(self) -> None:
        if self.pop.gen == 1:
            self.fast_non_dominated_sort()
        for child_genotype in self.build_offspring():
            self.pop.add_individual(genotype=child_genotype)
        self.evaluate_individuals()
        self.trim_population()
        if not PLOT:
            return
        if self.pop.gen == 1:
            self._fronts_to_plot[self.pop.gen] = self.fronts[0]
        elif self.pop.gen == 40:
            self._fronts_to_plot[self.pop.gen] = self.fronts[0]
        elif self.pop.gen == 80:
            self._fronts_to_plot[self.pop.gen] = self.fronts[0]
            for color, (gen, front) in zip(["orange", "blue", "red"], self._fronts_to_plot.items()):
                loc = [float(ind.fitness["locomotion_score"]) for ind in front],
                sens = [float(ind.fitness["sensing_score"]) for ind in front]
                plt.scatter(loc, sens, color=color, alpha=0.5, label=str(gen))
            plt.scatter([self.pop.objectives_dict[0]["worst_value"], self.pop.objectives_dict[1]["best_value"]],
                        [self.pop.objectives_dict[0]["best_value"], self.pop.objectives_dict[1]["worst_value"]],
                        alpha=0.0)
            plt.xlabel("locomotion through the aperture (m)")
            plt.ylabel("affordance detection (% of timesteps correct)")
            plt.legend()
            plt.savefig("pareto_front_{}.png".format(self.seed))
            plt.clf()

    def get_best(self) -> Individual:
        if not self.fronts:
            self.fast_non_dominated_sort()
        return min(self.fronts[0], key=lambda x: self.get_distance_from_diagonal(individual=x,
                                                                                 objectives_dict=self.pop.objectives_dict))

    @staticmethod
    def get_distance_from_diagonal(individual: Individual, objectives_dict: dict) -> float:
        s = 0.0
        for goal in objectives_dict.values():
            obj = individual.fitness[goal["name"]] / abs(goal["best_value"] - goal["worst_value"])
            if s == 0.0:
                s = obj
            else:
                s -= obj
        return abs(s)
