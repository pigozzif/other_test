import logging
from typing import Union, Optional
import numpy as np
import jax
import jax.numpy as jnp

from evojax.algo.base import NEAlgorithm, QualityDiversityMethod
from evojax.util import create_logger
from evojax.task.base import BDExtractor, TaskState

from evosax_support import FitnessShaper


class iAMaLGaM(NEAlgorithm):
    """A wrapper around evosax's iAMaLGaM.
    Implementation: https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/indep_iamalgam.py
    Reference: Bosman et al. (2013) - https://tinyurl.com/y9fcccx2
    """

    def __init__(
            self,
            param_size: int,
            pop_size: int,
            elite_ratio: float = 0.35,
            full_covariance: bool = False,
            eta_sigma: Optional[float] = None,
            eta_shift: Optional[float] = None,
            init_stdev: float = 0.01,
            decay_stdev: float = 0.999,
            limit_stdev: float = 0.001,
            w_decay: float = 0.0,
            seed: int = 0,
            logger: Optional[logging.Logger] = None,
            bd_extractor: BDExtractor = None
    ):
        """Initialization function.
        Args:
            param_size - Parameter size.
            pop_size - Population size.
            elite_ratio - Population elite fraction used for mean update.
            full_covariance - Whether to estimate full covariance or only diag.
            eta_sigma - Lrate for covariance (use default if not provided).
            eta_shift - Lrate for mean shift (use default if not provided).
            init_stdev - Initial scale of Gaussian perturbation.
            decay_stdev - Multiplicative scale decay between tell iterations.
            limit_stdev - Smallest scale (clipping limit).
            w_decay - L2 weight regularization coefficient.
            seed - Random seed for parameters sampling.
            logger - Logger.
        """

        # Set up object variables.

        if logger is None:
            self.logger = create_logger(name="iAMaLGaM")
        else:
            self.logger = logger

        self.param_size = param_size
        self.pop_size = abs(pop_size)
        self.rand_key = jax.random.PRNGKey(seed=seed)

        # Instantiate evosax's iAMaLGaM - choice between full cov & diagonal
        if full_covariance:
            self.es = evosax.Full_iAMaLGaM(
                popsize=pop_size, num_dims=param_size, elite_ratio=elite_ratio
            )
        else:
            self.es = evosax.Indep_iAMaLGaM(
                popsize=pop_size, num_dims=param_size, elite_ratio=elite_ratio
            )

        # Set hyperparameters according to provided inputs
        self.es_params = self.es.default_params.replace(
            sigma_init=init_stdev,
            sigma_decay=decay_stdev,
            sigma_limit=limit_stdev,
            init_min=0.0,
            init_max=0.0,
        )

        # Only replace learning rates for mean shift and sigma if provided!
        if eta_shift is not None:
            self.es_params = self.es_params.replace(eta_shift=eta_shift)
        if eta_sigma is not None:
            self.es_params = self.es_params.replace(eta_sigma=eta_sigma)

        # Initialize the evolution strategy state
        self.rand_key, init_key = jax.random.split(self.rand_key)
        self.es_state = self.es.initialize(init_key, self.es_params)

        # By default evojax assumes maximization of fitness score!
        # Evosax, on the other hand, minimizes!
        self.fit_shaper = FitnessShaper(w_decay=w_decay, maximize=True)
        if bd_extractor is not None:
            self.qd_aux = InnerQDAux(bd_extractor, param_size, pop_size)

    def ask(self) -> jnp.ndarray:
        self.rand_key, ask_key = jax.random.split(self.rand_key)
        self.params, self.es_state = self.es.ask(
            ask_key, self.es_state, self.es_params
        )
        return self.params

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        # Reshape fitness to conform with evosax minimization
        fit_re = self.fit_shaper.apply(self.params, fitness)
        self.es_state = self.es.tell(
            self.params, fit_re, self.es_state, self.es_params
        )
        if hasattr(self, "qd_aux"):
            self.qd_aux.tell(fitness)

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp.array(self.es_state.mean, copy=True)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        self.es_state = self.es_state.replace(
            best_member=jnp.array(params, copy=True),
            mean=jnp.array(params, copy=True),
        )


class CMA_ES(NEAlgorithm):
    """A wrapper around evosax's CMA-ES.
    Implementation: https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/cma_es.py
    Reference: Hansen & Ostermeier (2008) - http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf
    """

    def __init__(
            self,
            param_size: int,
            pop_size: int,
            elite_ratio: float = 0.5,
            init_stdev: float = 0.1,
            w_decay: float = 0.0,
            seed: int = 0,
            logger: logging.Logger = None,
            bd_extractor: BDExtractor = None
    ):
        """Initialization function.
        Args:
            param_size - Parameter size.
            pop_size - Population size.
            elite_ratio - Population elite fraction used for gradient estimate.
            init_stdev - Initial scale of istropic part of covariance.
            w_decay - L2 weight regularization coefficient.
            seed - Random seed for parameters sampling.
            logger - Logger.
        """

        self.param_size = param_size
        self.pop_size = abs(pop_size)
        self.elite_ratio = elite_ratio
        self.rand_key = jax.random.PRNGKey(seed=seed)

        # Instantiate evosax's ARS strategy
        self.es = evosax.CMA_ES(
            popsize=pop_size,
            num_dims=param_size,
            elite_ratio=elite_ratio,
        )

        # Set hyperparameters according to provided inputs
        self.es_params = self.es.default_params.replace(sigma_init=init_stdev)

        # Initialize the evolution strategy state
        self.rand_key, init_key = jax.random.split(self.rand_key)
        self.es_state = self.es.initialize(init_key, self.es_params)

        # By default evojax assumes maximization of fitness score!
        # Evosax, on the other hand, minimizes!
        self.fit_shaper = FitnessShaper(w_decay=w_decay, maximize=True)
        if bd_extractor is not None:
            self.qd_aux = InnerQDAux(bd_extractor, param_size, pop_size)

    def ask(self) -> jnp.ndarray:
        self.rand_key, ask_key = jax.random.split(self.rand_key)
        self.params, self.es_state = self.es.ask(
            ask_key, self.es_state, self.es_params
        )
        return self.params

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        # Reshape fitness to conform with evosax minimization
        fit_re = self.fit_shaper.apply(self.params, fitness)
        self.es_state = self.es.tell(
            self.params, fit_re, self.es_state, self.es_params
        )
        if hasattr(self, "qd_aux"):
            self.qd_aux.tell(fitness)

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp.array(self.es_state.mean, copy=True)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        self.es_state = self.es_state.replace(
            best_member=jnp.array(params, copy=True),
            mean=jnp.array(params, copy=True),
        )


class ARS(NEAlgorithm):
    """A wrapper around evosax's Augmented Random Search.
    Implementation: https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/ars.py
    Reference: Mania et al. (2018) - https://arxiv.org/pdf/1803.07055.pdf
    NOTE: More details on the optimizer configuration can be found here
    https://github.com/RobertTLange/evosax/blob/main/evosax/utils/optimizer.py
    """

    def __init__(
        self,
        param_size: int,
        pop_size: int,
        elite_ratio: float = 0.2,
        optimizer: str = "clipup",
        optimizer_config: dict = {
            "lrate_init": 0.15,  # Initial learning rate
            "lrate_decay": 0.999,  # Multiplicative decay factor
            "lrate_limit": 0.05,  # Smallest possible lrate
            "max_speed": 0.3,  # Max. clipping velocity
            "momentum": 0.9,  # Momentum coefficient
        },
        init_stdev: float = 0.01,
        decay_stdev: float = 0.999,
        limit_stdev: float = 0.001,
        w_decay: float = 0.0,
        seed: int = 0,
        logger: Optional[logging.Logger] = None,
        bd_extractor: BDExtractor = None
    ):
        """Initialization function.
        Args:
            param_size - Parameter size.
            pop_size - Population size.
            elite_ratio - Population elite fraction used for gradient estimate.
            optimizer - Optimizer name ("sgd", "adam", "rmsprop", "clipup").
            optimizer_config - Configuration of optimizer hyperparameters.
            init_stdev - Initial scale of Gaussian perturbation.
            decay_stdev - Multiplicative scale decay between tell iterations.
            limit_stdev - Smallest scale (clipping limit).
            w_decay - L2 weight regularization coefficient.
            seed - Random seed for parameters sampling.
            logger - Logger.
        """

        # Set up object variables.

        if logger is None:
            self.logger = create_logger(name="ARS")
        else:
            self.logger = logger

        self.param_size = param_size
        self.pop_size = abs(pop_size)
        self.elite_ratio = elite_ratio
        self.rand_key = jax.random.PRNGKey(seed=seed)

        # Instantiate evosax's ARS strategy
        self.es = evosax.ARS(
            popsize=pop_size,
            num_dims=param_size,
            elite_ratio=elite_ratio,
            opt_name=optimizer,
        )

        # Set hyperparameters according to provided inputs
        self.es_params = self.es.default_params.replace(
            sigma_init=init_stdev,
            sigma_decay=decay_stdev,
            sigma_limit=limit_stdev,
            init_min=0.0,
            init_max=0.0,
        )

        # Update optimizer-specific parameters of Adam
        self.es_params = self.es_params.replace(
            opt_params=self.es_params.opt_params.replace(**optimizer_config)
        )

        # Initialize the evolution strategy state
        self.rand_key, init_key = jax.random.split(self.rand_key)
        self.es_state = self.es.initialize(init_key, self.es_params)

        # By default evojax assumes maximization of fitness score!
        # Evosax, on the other hand, minimizes!
        self.fit_shaper = FitnessShaper(w_decay=w_decay, maximize=True)
        if bd_extractor is not None:
            self.qd_aux = InnerQDAux(bd_extractor, param_size, pop_size)

    def ask(self) -> jnp.ndarray:
        self.rand_key, ask_key = jax.random.split(self.rand_key)
        self.params, self.es_state = self.es.ask(
            ask_key, self.es_state, self.es_params
        )
        return self.params

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        # Reshape fitness to conform with evosax minimization
        fit_re = self.fit_shaper.apply(self.params, fitness)
        self.es_state = self.es.tell(
            self.params, fit_re, self.es_state, self.es_params
        )
        if hasattr(self, "qd_aux"):
            self.qd_aux.tell(fitness)

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp.array(self.es_state.mean, copy=True)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        self.es_state = self.es_state.replace(
            best_member=jnp.array(params, copy=True),
            mean=jnp.array(params, copy=True),
        )


class InnerQDAux(QualityDiversityMethod):

    def __init__(self,
                 bd_extractor: BDExtractor,
                 param_size: int,
                 pop_size: int):
        self.bd_names = [x[0] for x in bd_extractor.bd_spec]
        self.bd_n_bins = [x[1] for x in bd_extractor.bd_spec]
        self.params_lattice = jnp.zeros((np.prod(self.bd_n_bins), param_size))
        self.fitness_lattice = -float("inf") * jnp.ones(np.prod(self.bd_n_bins))
        self.occupancy_lattice = jnp.zeros(np.prod(self.bd_n_bins), dtype=jnp.int32)
        self.population = None
        self.bin_idx = jnp.zeros(pop_size, dtype=jnp.int32)

        def get_bin_idx(task_state):
            bd_idx = [task_state.__dict__[name].astype(int) for name in self.bd_names]
            return jnp.ravel_multi_index(bd_idx, self.bd_n_bins, mode='clip')

        self._get_bin_idx = jax.jit(jax.vmap(get_bin_idx))

        def update_fitness_and_param(target_bin, bin_idx, fitness, fitness_lattice, param, param_lattice):
            best_ix = jnp.where(bin_idx == target_bin, fitness, fitness_lattice.min()).argmax()
            best_fitness = fitness[best_ix]
            new_fitness_lattice = jnp.where(
                best_fitness > fitness_lattice[target_bin],
                best_fitness, fitness_lattice[target_bin])
            new_param_lattice = jnp.where(
                best_fitness > fitness_lattice[target_bin],
                param[best_ix], param_lattice[target_bin])
            return new_fitness_lattice, new_param_lattice

        self._update_lattices = jax.jit(jax.vmap(update_fitness_and_param, in_axes=(0, None, None, None, None, None)))

    def observe_bd(self, task_state: TaskState) -> None:
        self.bin_idx = self._get_bin_idx(task_state)

    def ask(self) -> jnp.ndarray:
        raise NotImplementedError()

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]):
        unique_bins = jnp.unique(self.bin_idx)
        fitness_lattice, params_lattice = self._update_lattices(unique_bins, self.bin_idx,
                                                                fitness, self.fitness_lattice,
                                                                self.population, self.params_lattice)
        self.occupancy_lattice = self.occupancy_lattice.at[unique_bins].set(1)
        self.fitness_lattice = self.fitness_lattice.at[unique_bins].set(fitness_lattice)
        self.params_lattice = self.params_lattice.at[unique_bins].set(params_lattice)

    def set_population(self, pop):
        self.population = pop
