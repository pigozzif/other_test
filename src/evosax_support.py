import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Optional, Union, Dict
from flax import struct
from functools import partial
from evosax.utils import get_best_fitness_member, ParameterReshaper, FitnessShaper


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: float
    best_member: chex.Array
    best_fitness: float
    gen_counter: int


@struct.dataclass
class EvoParams:
    sigma_init: float = 0.03
    sigma_decay: float = 0.999
    sigma_limit: float = 0.01
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class Strategy(object):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """Base Class for an Evolution Strategy."""
        self.popsize = popsize

        # Setup optional parameter reshaper
        self.use_param_reshaper = pholder_params is not None
        if self.use_param_reshaper:
            self.param_reshaper = ParameterReshaper(pholder_params)
            self.num_dims = self.param_reshaper.total_params
        else:
            self.num_dims = num_dims
        assert (
            self.num_dims is not None
        ), "Provide either num_dims or pholder_params to strategy."

        # Setup optional fitness shaper
        self.fitness_shaper = FitnessShaper(**fitness_kwargs)

    @property
    def default_params(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        params = self.params_strategy
        return params

    @partial(jax.jit, static_argnums=(0,))
    def initialize(
        self, rng: chex.PRNGKey, params: Optional[EvoParams] = None
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Initialize strategy based on strategy-specific initialize method
        state = self.initialize_strategy(rng, params)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(
        self,
        rng: chex.PRNGKey,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ) -> Tuple[Union[chex.Array, chex.ArrayTree], EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Generate proposal based on strategy-specific ask method
        x, state = self.ask_strategy(rng, state, params)
        # Clip proposal candidates into allowed range
        x_clipped = jnp.clip(jnp.squeeze(x), params.clip_min, params.clip_max)

        # Reshape parameters into pytrees
        if self.use_param_reshaper:
            x_out = self.param_reshaper.reshape(x_clipped)
        else:
            x_out = x_clipped
        return x_out, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: Union[chex.Array, chex.ArrayTree],
        fitness: chex.Array,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ) -> chex.ArrayTree:
        """`tell` performance data for strategy state update."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Flatten params if using param reshaper for ES update
        if self.use_param_reshaper:
            x = self.param_reshaper.flatten(x)

        # Perform fitness reshaping inside of strategy tell call (if desired)
        fitness_re = self.fitness_shaper.apply(x, fitness)

        # Update the search state based on strategy-specific update
        state = self.tell_strategy(x, fitness_re, state, params)

        # Check if there is a new best member & update trackers
        best_member, best_fitness = get_best_fitness_member(
            x, fitness, state, self.fitness_shaper.maximize
        )
        return state.replace(
            best_member=best_member,
            best_fitness=best_fitness,
            gen_counter=state.gen_counter + 1,
        )

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """Search-specific `initialize` method. Returns initial state."""
        raise NotImplementedError

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """Search-specific `ask` request. Returns proposals & updated state."""
        raise NotImplementedError

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """Search-specific `tell` update. Returns updated state."""
        raise NotImplementedError

    def get_eval_params(self, state: EvoState):
        """Return reshaped parameters to evaluate."""
        if self.use_param_reshaper:
            x_out = self.param_reshaper.reshape_single(state.mean)
        else:
            x_out = state.mean
        return x_out


def exp_decay(
    param: chex.Array, param_decay: chex.Array, param_limit: chex.Array
) -> chex.Array:
    """Exponentially decay parameter & clip by minimal value."""
    param = param * param_decay
    param = jnp.maximum(param, param_limit)
    return param


@struct.dataclass
class OptState:
    lrate: float
    m: chex.Array
    v: Optional[chex.Array] = None
    n: Optional[chex.Array] = None
    last_grads: Optional[chex.Array] = None
    gen_counter: int = 0


@struct.dataclass
class OptParams:
    lrate_init: float = 0.01
    lrate_decay: float = 0.999
    lrate_limit: float = 0.001
    momentum: Optional[float] = None
    beta_1: Optional[float] = None
    beta_2: Optional[float] = None
    beta_3: Optional[float] = None
    eps: Optional[float] = None
    max_speed: Optional[float] = None


class Optimizer(object):
    def __init__(self, num_dims: int):
        """Simple JAX-Compatible Optimizer Class."""
        self.num_dims = num_dims

    @property
    def default_params(self) -> OptParams:
        """Return shared and optimizer-specific default parameters."""
        return OptParams(**self.params_opt)

    def initialize(self, params: OptParams) -> OptState:
        """Initialize the optimizer state."""
        return self.initialize_opt(params)

    def step(
        self,
        mean: chex.Array,
        grads: chex.Array,
        state: OptState,
        params: OptParams,
    ) -> [chex.Array, OptState]:
        """Perform a gradient-based update step."""
        return self.step_opt(mean, grads, state, params)

    def update(self, state: OptState, params: OptParams) -> OptState:
        """Exponentially decay the learning rate if desired."""
        lrate = exp_decay(state.lrate, params.lrate_decay, params.lrate_limit)
        return state.replace(lrate=lrate)

    @property
    def params_opt(self) -> OptParams:
        """Optimizer-specific hyperparameters."""
        raise NotImplementedError

    def initialize_opt(self, params: OptParams) -> OptState:
        """Optimizer-specific initialization of optimizer state."""
        raise NotImplementedError

    def step_opt(
        self,
        mean: chex.Array,
        grads: chex.Array,
        state: OptState,
        params: OptParams,
    ) -> Tuple[chex.Array, OptState]:
        """Optimizer-specific step to update parameter estimates."""
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, num_dims: int):
        """Simple JAX-Compatible SGD + Momentum optimizer."""
        super().__init__(num_dims)
        self.opt_name = "sgd"

    @property
    def params_opt(self) -> Dict[str, float]:
        """Return default SGD+Momentum parameters."""
        return {
            "momentum": 0.0,
        }

    def initialize_opt(self, params: OptParams) -> OptState:
        """Initialize the momentum trace of the optimizer."""
        return OptState(m=jnp.zeros(self.num_dims), lrate=params.lrate_init)

    def step_opt(
        self,
        mean: chex.Array,
        grads: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> Tuple[chex.Array, OptState]:
        """Perform a simple SGD + Momentum step."""
        m = grads + params.momentum * state.m
        mean_new = mean - state.lrate * state.m
        return mean_new, state.replace(m=m, gen_counter=state.gen_counter + 1)


class Adam(Optimizer):
    def __init__(self, num_dims: int):
        """JAX-Compatible Adam Optimizer (Kingma & Ba, 2015)
        Reference: https://arxiv.org/abs/1412.6980"""
        super().__init__(num_dims)
        self.opt_name = "adam"

    @property
    def params_opt(self) -> Dict[str, float]:
        """Return default Adam parameters."""
        return {
            "beta_1": 0.99,
            "beta_2": 0.999,
            "eps": 1e-8,
        }

    def initialize_opt(self, params: OptParams) -> OptState:
        """Initialize the m, v trace of the optimizer."""
        return OptState(
            m=jnp.zeros(self.num_dims),
            v=jnp.zeros(self.num_dims),
            lrate=params.lrate_init,
        )

    def step_opt(
        self,
        mean: chex.Array,
        grads: chex.Array,
        state: OptState,
        params: OptParams,
    ) -> Tuple[chex.Array, OptState]:
        """Perform a simple Adam GD step."""
        m = (1 - params.beta_1) * grads + params.beta_1 * state.m
        v = (1 - params.beta_2) * (grads ** 2) + params.beta_2 * state.v
        mhat = m / (1 - params.beta_1 ** (state.gen_counter + 1))
        vhat = v / (1 - params.beta_2 ** (state.gen_counter + 1))
        mean_new = mean - state.lrate * mhat / (jnp.sqrt(vhat) + params.eps)
        return mean_new, state.replace(
            m=m, v=v, gen_counter=state.gen_counter + 1
        )


GradientOptimizer = {
    "sgd": SGD,
    "adam": Adam
}
