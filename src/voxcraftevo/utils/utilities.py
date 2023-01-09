import random

import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def weighted_random_by_dct(dct):
    rand_val = random.random()
    total = 0
    for k, v in dct.items():
        total += v
        if rand_val <= total:
            return k
    raise RuntimeError("Could not sample from dictionary")


def xml_format(tag):
    """Ensures that tag is encapsulated inside angle brackets."""
    if tag[0] != "<":
        tag = "<" + tag
    if tag[-1:] != ">":
        tag += ">"
    return tag


def dominates(ind1, ind2, attribute_name, maximize):
    """Returns 1 if ind1 dominates ind2 in a shared attribute, -1 if ind2 dominates ind1, 0 otherwise."""
    if ind1.fitness[attribute_name] > ind2.fitness[attribute_name]:
        ans = 1
    elif ind1.fitness[attribute_name] < ind2.fitness[attribute_name]:
        ans = -1
    else:
        ans = 0
    return ans if maximize else -ans


def exp_decay(param, param_decay, param_limit):
    """Exponentially decay parameter and clip by minimal value."""
    param = param * param_decay
    return max(param, param_limit)
