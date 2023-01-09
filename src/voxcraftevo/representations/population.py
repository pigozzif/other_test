import abc
from functools import total_ordering
from typing import Iterable

import numpy as np

from voxcraftevo.representations.factory import GenotypeFactory
from voxcraftevo.evo.objectives import ObjectiveDict
from voxcraftevo.representations.mapper import SolutionMapper
from voxcraftevo.utils.utilities import dominates


@total_ordering
class Individual(object):

    def __init__(self, id: int, genotype, solution, comparator, fitness: dict = None, age: int = 0,
                 evaluated: bool = False):
        self.id = id
        self.genotype = genotype
        self.solution = solution
        self.comparator = comparator
        self.fitness = fitness
        self.age = age
        self.evaluated = evaluated

    def __str__(self):
        return "Individual[id={0},age={1},fitness={2}]".format(self.id, self.age, self.fitness)

    __repr__ = __str__

    def __eq__(self, other):
        return self.comparator.compare(ind1=self, ind2=other) == 0

    def __lt__(self, other):
        return self.comparator.compare(ind1=self, ind2=other) == -1

    def __gt__(self, other):
        return self.comparator.compare(ind1=self, ind2=other) == 1


class Comparator(object):

    def __init__(self, objective_dict: ObjectiveDict):
        self.objective_dict = objective_dict

    @abc.abstractmethod
    def compare(self, ind1: Individual, ind2: Individual) -> int:
        pass

    @classmethod
    def create_comparator(cls, name: str, objective_dict: ObjectiveDict):
        if name == "lexicase":
            return LexicaseComparator(objective_dict=objective_dict)
        elif name == "pareto":
            return ParetoComparator(objective_dict=objective_dict)
        raise ValueError("Invalid comparator name: {}".format(name))


class LexicaseComparator(Comparator):

    def compare(self, ind1, ind2):
        for rank in reversed(range(len(self.objective_dict))):
            goal = self.objective_dict[rank]
            d = dominates(ind1=ind1, ind2=ind2, attribute_name=goal["name"], maximize=goal["maximize"])
            if d == 1:
                return 1
            elif d == -1:
                return -1
        return 0


class ParetoComparator(Comparator):

    def compare(self, ind1, ind2):
        wins = [dominates(ind1=ind1, ind2=ind2, attribute_name=goal["name"], maximize=goal["maximize"])
                for rank, goal in self.objective_dict.items()]
        if all([w >= 0 for w in wins]):
            return 1
        elif any([w > 0 for w in wins]):
            return 0
        return -1


class Population(object):

    def __init__(self, pop_size: int, genotype_factory: GenotypeFactory, solution_mapper: SolutionMapper,
                 objectives_dict: ObjectiveDict, comparator: str):
        self.genotype_factory = genotype_factory
        self.solution_mapper = solution_mapper
        self.objectives_dict = objectives_dict
        self.comparator = Comparator.create_comparator(name=comparator, objective_dict=objectives_dict)
        self._individuals = []
        self._max_id = 0
        # init random population (generation 0)
        for g in self.genotype_factory.create_population(pop_size=pop_size):
            self.add_individual(g)
        self.gen = 0

    def __str__(self):
        return "Population[size={0},best={1}]".format(len(self), self.get_best())

    def __len__(self):
        return len(self._individuals)

    def __getitem__(self, item):
        return self._individuals[item]

    def __iter__(self):
        return iter(self._individuals)

    def __contains__(self, item):
        return any([ind.id == item.id for ind in self])

    def add_individual(self, genotype) -> Individual:
        self._individuals.append(Individual(id=self._max_id,
                                            genotype=genotype,
                                            solution=self.solution_mapper(genotype),
                                            comparator=self.comparator))
        self._max_id += 1
        return self._individuals[-1]

    def remove_individual(self, ind: Individual) -> None:
        for idx, individual in enumerate(self):
            if ind.id == individual.id:
                self._individuals.pop(idx)
                break

    def clear(self) -> None:
        self._individuals = []

    def update_ages(self) -> None:
        for ind in self:
            ind.age += 1

    def sort(self) -> None:
        self._individuals.sort(reverse=True)

    def get_best(self) -> Individual:
        self.sort()  # key=lambda x: x.fitness["sensing_score"])
        return self[0]

    def sample(self, n: int) -> Iterable[Individual]:
        return np.random.choice(self, size=n, replace=False)
