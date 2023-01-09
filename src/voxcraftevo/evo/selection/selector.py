import abc
from typing import List

from voxcraftevo.representations.population import Population
from voxcraftevo.representations.population import Individual


class Selector(object):

    def select(self, population: Population, n: int) -> List[Individual]:
        return [self.select_individual(population=population) for _ in range(n)]

    @abc.abstractmethod
    def select_individual(self, population: Population) -> Individual:
        pass

    @classmethod
    def create_selector(cls, name: str, **kwargs):
        if name == "worst":
            return WorstSelector()
        elif name == "tournament":
            return TournamentSelector(size=kwargs["tournament_size"])
        elif name == "tournament_crowded":
            return TournamentCrowdedSelector(size=kwargs["tournament_size"],
                                             crowding_distances=kwargs["crowding_distances"],
                                             fronts=kwargs["fronts"])
        raise ValueError("Invalid selector name: {}".format(name))


class WorstSelector(Selector):

    def select_individual(self, population):
        population.sort()
        return population[len(population) - 1]


class TournamentSelector(Selector):

    def __init__(self, size: int):
        self.size = size

    def select_individual(self, population):
        contenders = population.sample(n=self.size)
        return sorted(contenders, reverse=True)[0]


class TournamentCrowdedSelector(Selector):

    def __init__(self, size: int, crowding_distances: dict, fronts: dict):
        self.size = size
        self.crowding_distances = crowding_distances
        self.fronts = fronts

    def select_individual(self, population):
        contenders = population.sample(n=self.size)
        reverse_fronts = {}
        for individual in contenders:
            found = False
            for idx, front in self.fronts.items():
                if found:
                    break
                for ind in front:
                    if individual.id == ind.id:
                        reverse_fronts[individual.id] = idx
                        found = True
                        break
        best_front_idx = reverse_fronts[min(contenders, key=lambda x: reverse_fronts[x.id]).id]
        best_front = list(filter(lambda x: reverse_fronts[x.id] == best_front_idx, contenders))
        best = max(best_front, key=lambda x: self.crowding_distances[x.id])
        return best
