import abc


class SolutionMapper(object):

    @abc.abstractmethod
    def __call__(self, genotype):
        pass

    @classmethod
    def create_mapper(cls, name: str, **kwargs):
        if name == "direct":
            return DirectMapper()
        raise ValueError("Invalid mapper name: {}".format(name))


class DirectMapper(SolutionMapper):  # a lambda would work, but non-trivial to pickle

    def __call__(self, genotype):
        return genotype
