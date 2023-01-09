import abc

import numpy as np

from voxcraftevo.utils.utilities import exp_decay


class Optimizer(abc.ABC):

    @abc.abstractmethod
    def optimize(self, **kwargs):
        pass

    @classmethod
    def create_optimizer(cls, name, **kwargs):
        if name == "adam":
            return Adam(**kwargs)
        raise ValueError("Invalid optimizer name: {}".format(name))


class Adam(Optimizer):

    def __init__(self, num_dims, l_rate_init, l_rate_decay, l_rate_limit, beta_1=0.99, beta_2=0.999, eps=1e-8):
        self.num_dims = num_dims
        self.l_rate = l_rate_init
        self.l_rate_decay = l_rate_decay
        self.l_rate_limit = l_rate_limit
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.m = np.zeros(self.num_dims)
        self.v = np.zeros(self.num_dims)

    def optimize(self, mean, t, theta_grad):
        self.m = (1 - self.beta_1) * theta_grad + self.beta_1 * self.m
        self.v = (1 - self.beta_2) * (theta_grad ** 2) + self.beta_2 * self.v
        m_hat = self.m / (1 - self.beta_1 ** (t + 1))
        v_hat = self.v / (1 - self.beta_2 ** (t + 1))
        new_mean = mean - self.l_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        self.l_rate = exp_decay(self.l_rate, self.l_rate_decay, self.l_rate_limit)
        return new_mean
