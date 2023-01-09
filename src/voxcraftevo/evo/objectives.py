from ..utils.utilities import xml_format


class ObjectiveDict(dict):
    """A dictionary describing the objectives for optimization. See self.add_objective()."""

    def __init__(self):
        super(ObjectiveDict, self).__init__()
        self._max_rank = 0

    def add_objective(self, name: str, maximize: bool, tag: str, best_value: float, worst_value: float, meta_func=None) -> None:
        """Add an optimization objective to the dictionary.
        Objectives must be added in order of importance, however fitness is fixed to be the most important.
        The keys of an ObjectiveDict correspond to the objective's rank or importance. The ranks are set via the order
        in which objectives are added (fitness will auto-correct to rank 0).
        For each rank key, starting with 0, the corresponding value is another dictionary with three components:
        name, maximized, tag.
        Parameters
        ----------
        name : str
            The associated individual-level attribute name
        maximize : bool
            Whether superior individuals maximized (True) or minimize (False) the objective.
        tag : str or None
            The tag used in parsing the resulting output from a VoxCraft simulation.
            If this is None then the attribute is calculated outside of VoxCraft (in Python only).
        best_value: float
        worst_value: float
        meta_func : function
            This is applied as a function of the objective value and individual, i.e. func(val, ind)
        """
        for value in self.values():
            if name in value:
                raise ValueError("Adding alread-present objective: {}".format(name))

        curr_rank = self._max_rank

        # if fitness is not added first, shift every other objective "down" in importance
        if name == "fitness_score" and self._max_rank > 0:
            curr_rank = 0  # change the key to rank 0
            for rank in reversed(range(len(self))):
                self[rank + 1] = self[rank]

        super(ObjectiveDict, self).__setitem__(curr_rank, {"name": name,
                                                           "maximize": maximize,
                                                           "tag": xml_format(tag) if tag is not None else None,
                                                           "best_value": best_value,
                                                           "worst_value": worst_value,
                                                           "meta_func": meta_func
                                                           })

        self._max_rank += 1
