import abc


class Filter(object):

    @abc.abstractmethod
    def __call__(self, item):
        pass

    @classmethod
    def create_filter(cls, name: str):
        if name is None:
            return NoneFilter()
        raise ValueError("Invalid filter: {}".format(name))


class NoneFilter(Filter):

    def __call__(self, item):
        return True
