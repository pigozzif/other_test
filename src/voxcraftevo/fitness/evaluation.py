import abc

from voxcraftevo.evo.objectives import ObjectiveDict
from voxcraftevo.representations.population import Individual


class FitnessFunction(object):

    @abc.abstractmethod
    def create_objectives_dict(self) -> ObjectiveDict:
        pass

    @abc.abstractmethod
    def get_fitness(self, individual: Individual) -> dict:
        pass
