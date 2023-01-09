import abc
from typing import Tuple, List

from voxcraftevo.evo.objectives import ObjectiveDict
from voxcraftevo.representations.population import Individual


class FitnessFunction(object):

    @staticmethod
    def parse_fitness_from_xml(root, bot_id: str, fitness_tag: str, worst_value: float) -> float:
        detail = root.findall("detail/")
        for d in detail:
            if d.tag == bot_id:
                return float(d.findall(fitness_tag)[0].text)
        return worst_value

    @staticmethod
    def parse_fitness_from_history(root, fitness_tag: str, worst_value: float) -> float:
        with open(root, "r") as file:
            for line in file:
                if line.startswith(fitness_tag):
                    try:
                        return float("".join(c for c in line.split(":")[1].strip() if c.isdigit() or c == ".").strip("."))
                    except:
                        return worst_value
        return worst_value

    @staticmethod
    def parse_pos(root, bot_id: str, tag: str) -> Tuple[float, float, float]:
        detail = root.findall("detail/")
        for d in detail:
            if d.tag == bot_id:
                center = d.findall(tag)[0]
                return float(center.findall("x")[0].text), float(center.findall("y")[0].text), float(
                    center.findall("z")[0].text)
        raise IndexError

    @abc.abstractmethod
    def create_objectives_dict(self) -> ObjectiveDict:
        pass

    @abc.abstractmethod
    def create_vxa(self, directory: str) -> None:
        pass

    @abc.abstractmethod
    def create_vxd(self, ind: Individual, directory: str, record_history: bool) -> None:
        pass

    @abc.abstractmethod
    def get_fitness(self, individual: Individual) -> dict:
        pass

    @abc.abstractmethod
    def save_histories(self, individual: Individual, input_directory: str, output_directory: str,
                       executables_directory: str) -> None:
        pass
