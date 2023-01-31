# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train an agent to solve the classic CartPole swing up task.
Example command to run this script:
# Train in a harder setup.
python train_cartpole.py --gpu-id=0
# Train in an easy setup.
python train_cartpole.py --gpu-id=0 --easy
# Train a permutation invariant agent in a harder setup.
python train_cartpole.py --gpu-id=0 --pi --max-iter=20000 --pop-size=256 \
--center-lr=0.037 \
--std-lr=0.078 \
--init-std=0.082
# Train a permutation invariant agent in a harder setup with CMA-ES.
python train_cartpole.py --gpu-id=0 --pi --max-iter=20000 --pop-size=256 --cma
"""

import argparse
import os
import time

import logging
import numpy as np

from evojax.algo import ARS, MAPElites
from evojax.policy import MLPPolicy
from evojax.task.brax_task import BraxTask, AntBDExtractor, BDExtractorState
from evojax.task.base import BDExtractor, TaskState
from evojax.task.cartpole import CartPoleSwingUp
from evojax.algo.base import NEAlgorithm, QualityDiversityMethod
from evojax.util import create_logger, save_lattices

from evojax import ObsNormalizer
from evojax import SimManager

import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Optional, Union
from flax import struct

from evosax_support import GradientOptimizer, Strategy, OptParams, OptState, exp_decay, FitnessShaper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hidden-size', type=int, default=64, help='Policy hidden size.')
    parser.add_argument(
        '--num-tests', type=int, default=100, help='Number of test rollouts.')
    parser.add_argument(
        '--max-iter', type=int, default=1000, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=10, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=20, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=0, help='Random seed for training.')
    parser.add_argument(
        '--center-lr', type=float, default=0.05, help='Center learning rate.')
    parser.add_argument(
        '--std-lr', type=float, default=0.1, help='Std learning rate.')
    parser.add_argument(
        '--init-std', type=float, default=0.1, help='Initial std.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--solver', type=str, default="pgpe", help='Solver for training.')
    parser.add_argument(
        '--task', type=str, default="cartpole_hard", help='Task.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: float
    opt_state: OptState
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    opt_params: OptParams
    sigma_init: float = 0.04
    sigma_decay: float = 0.999
    sigma_limit: float = 0.01
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class InnerMyES(Strategy):
    def __init__(
            self,
            popsize: int,
            num_dims: Optional[int] = None,
            pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
            opt_name: str = "adam",
            lrate_init: float = 0.05,
            lrate_decay: float = 1.0,
            lrate_limit: float = 0.001,
            sigma_init: float = 0.03,
            sigma_decay: float = 1.0,
            sigma_limit: float = 0.01,
            is_openes: bool = False,
            **fitness_kwargs: Union[bool, int, float]
    ):
        super().__init__(popsize, num_dims, pholder_params, **fitness_kwargs)
        assert not self.popsize & 1, "Population size must be even"
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup", "adan"]
        self.pop_size = popsize
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.strategy_name = "OpenES"

        # Set core kwargs es_params (lrate/sigma schedules)
        self.lrate_init = lrate_init
        self.lrate_decay = lrate_decay
        self.lrate_limit = lrate_limit
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.is_openes = is_openes

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        opt_params = self.optimizer.default_params.replace(
            lrate_init=self.lrate_init,
            lrate_decay=self.lrate_decay,
            lrate_limit=self.lrate_limit,
        )
        return EvoParams(
            opt_params=opt_params,
            sigma_init=self.sigma_init,
            sigma_decay=self.sigma_decay,
            sigma_limit=self.sigma_limit,
        )

    def initialize_strategy(
            self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            mean=initialization,
            sigma=params.sigma_init,
            opt_state=self.optimizer.initialize(params.opt_params),
            best_member=initialization,
        )
        return state

    def ask_strategy(
            self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        # Antithetic sampling of noise
        z_plus = jax.random.normal(
            rng,
            (int(self.popsize / 2), self.num_dims),
        )
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = state.mean + state.sigma * z
        self.rng = rng
        return x, state

    def tell_strategy(
            self,
            x: chex.Array,
            fitness: chex.Array,
            state: EvoState,
            params: EvoParams,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        # Reconstruct noise from last mean/std estimates
        noise = (x - state.mean) / state.sigma
        if self.is_openes:
            theta_grad = (
                    1.0 / (self.popsize * state.sigma) * jnp.dot(noise.T, fitness)
            )
        else:
            theta_grad = (
                    (1.0 / (self.popsize * state.sigma) * jnp.dot(noise.T, fitness))
                    + np.random.normal(loc=0.0, scale=state.sigma * jnp.sqrt(state.opt_state.lrate),
                                       size=state.mean.shape)
            )

        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = self.optimizer.step(
            state.mean, theta_grad, state.opt_state, params.opt_params
        )

        opt_state = self.optimizer.update(opt_state, params.opt_params)
        sigma = exp_decay(state.sigma, params.sigma_decay, params.sigma_limit)
        return state.replace(mean=mean, sigma=sigma, opt_state=opt_state)


class MyES(NEAlgorithm):

    def __init__(
            self,
            param_size: int,
            pop_size: int,
            optimizer: str = "adam",
            optimizer_config: dict = {
                "lrate_init": 0.01,  # Initial learning rate
                "lrate_decay": 0.999,  # Multiplicative decay factor
                "lrate_limit": 0.001,  # Smallest possible lrate
                "beta_1": 0.99,  # beta_1 Adam
                "beta_2": 0.999,  # beta_2 Adam
                "eps": 1e-8,  # eps constant Adam denominator
            },
            init_stdev: float = 0.01,
            decay_stdev: float = 0.999,
            limit_stdev: float = 0.001,
            w_decay: float = 0.0,
            seed: int = 0,
            logger: logging.Logger = None,
            inner_es: Strategy = None,
            bd_extractor: BDExtractor = None
    ):
        # Delayed importing of evosax

        if logger is None:
            self.logger = create_logger(name="MyES")
        else:
            self.logger = logger

        self.param_size = param_size
        self.pop_size = abs(pop_size)
        self.rand_key = jax.random.PRNGKey(seed=seed)

        # Instantiate evosax's Open ES strategy
        self.es = inner_es

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
        self.fit_shaper = FitnessShaper(
            centered_rank=True, z_score=False, w_decay=w_decay, maximize=True
        )

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


class HalfCheetahBDExtractor(BDExtractor):
    """Behavior descriptor extractor for the HaldCheetah locomotion task."""

    def __init__(self, name, logger):
        # Each BD represents the quantized foot-ground contact time ratio.
        if name == "halfcheetah":
            bd_spec = [
                ('feet1_contact', 10),
                ('feet2_contact', 10),
                ('feet3_contact', 10)
            ]
        else:
            bd_spec = [
                ('feet1_contact', 10),
                ('feet2_contact', 10),
                ('feet3_contact', 10),
                ('feet4_contact', 10),
                ('feet5_contact', 10)
            ]
        bd_state_spec = [
            ('feet_contact_times', jnp.ndarray),
            ('step_cnt', jnp.int32),
            ('valid_mask', jnp.int32),
        ]
        super(HalfCheetahBDExtractor, self).__init__(bd_spec, bd_state_spec, BDExtractorState)
        self._logger = logger

    def init_state(self, extended_task_state):
        return {
            'feet_contact_times': jnp.zeros_like(
                extended_task_state.feet_contact, dtype=jnp.int32),
            'step_cnt': jnp.zeros((), dtype=jnp.int32),
            'valid_mask': jnp.ones((), dtype=jnp.int32),
        }

    def update(self, extended_task_state, action, reward, done):
        valid_mask = (
                extended_task_state.valid_mask * (1 - done)).astype(jnp.int32)
        return extended_task_state.replace(
            feet_contact_times=(
                    extended_task_state.feet_contact_times +
                    extended_task_state.feet_contact * valid_mask),
            step_cnt=extended_task_state.step_cnt + valid_mask,
            valid_mask=valid_mask)

    def summarize(self, extended_task_state):
        feet_contact_ratio = (
                extended_task_state.feet_contact_times /
                extended_task_state.step_cnt[..., None]).mean(axis=1)
        bds = {bd[0]: (feet_contact_ratio[:, i] * bd[1]).astype(jnp.int32)
               for i, bd in enumerate(self.bd_spec)}
        return extended_task_state.replace(**bds)


class FileListener(object):

    def __init__(self, file_name, header):
        self.file_name = file_name
        self.header = header
        with open(file_name, "w") as file:
            file.write(";".join(header) + "\n")

    def listen(self, **kwargs):
        with open(self.file_name, "a") as file:
            file.write(";".join([str(kwargs.get(col, None)) for col in self.header]) + "\n")


def create_solver(config, num_params, lrate_init=0.01, init_stdev=0.04):
    if config.solver == "ars":
        return ARS(
            param_size=num_params,
            pop_size=1024,
            elite_ratio=0.05,
            init_stdev=0.05,
            decay_stdev=0.999,
            limit_stdev=0.001,
            optimizer="adam",
            optimizer_config={"lrate_init": 0.01, "lrate_decay": 0.999, "lrate_limit": 0.001, "momentum": 0.0},
            seed=config.seed
        )
    elif config.solver == "openes":
        return MyES(
            param_size=num_params,
            pop_size=256,
            init_stdev=0.04,
            decay_stdev=0.999,
            limit_stdev=0.001,
            optimizer="adam",
            optimizer_config={"lrate_init": 0.01, "lrate_decay": 0.999, "lrate_limit": 0.005, "momentum": 0.0},
            seed=config.seed,
            inner_es=InnerMyES(
                popsize=256,
                num_dims=num_params,
                opt_name="adam",
                is_openes=True
            ),
            bd_extractor=AntBDExtractor(logger=None) if config.task == "ant" else HalfCheetahBDExtractor(name=config.task, logger=None)
        )
    elif config.solver == "noise":
        return MyES(
            param_size=num_params,
            pop_size=256,
            init_stdev=init_stdev,
            decay_stdev=0.999,
            limit_stdev=0.001,
            optimizer="adam",
            optimizer_config={"lrate_init": lrate_init, "lrate_decay": 0.999, "lrate_limit": 0.001, "momentum": 0.0},
            seed=config.seed,
            inner_es=InnerMyES(
                popsize=256,
                num_dims=num_params,
                opt_name="adam",
                is_openes=False
            ),
            bd_extractor=AntBDExtractor(logger=None) if config.task == "ant" else HalfCheetahBDExtractor(name=config.task, logger=None)
        )
    elif config.solver == "me":
        return MAPElites(
            pop_size=1024,
            param_size=num_params,
            bd_extractor=AntBDExtractor(logger=None) if config.task == "ant" else HalfCheetahBDExtractor(name=config.task, logger=None),
            iso_sigma=0.05,
            line_sigma=0.3,
            seed=config.seed
        )
    raise ValueError("Invalid solver name: {}".format(config.solver))


def create_task(config):
    task_name = config.task
    if task_name.startswith("cartpole_"):
        train_task = CartPoleSwingUp(test=False, harder="hard" in task_name)
        test_task = CartPoleSwingUp(test=True, harder="easy" not in task_name)
    elif task_name == "ant":
        bd_extractor = AntBDExtractor(logger=None)
        train_task = BraxTask(env_name="ant", bd_extractor=bd_extractor, test=False)
        test_task = BraxTask(env_name="ant", bd_extractor=bd_extractor, test=True)
    elif task_name == "halfcheetah":
        bd_extractor = HalfCheetahBDExtractor(name="halfcheetah", logger=None)
        train_task = BraxTask(env_name="halfcheetah", bd_extractor=bd_extractor, test=False)
        test_task = BraxTask(env_name="halfcheetah", bd_extractor=bd_extractor, test=True)
    elif task_name == "humanoid":
        bd_extractor = HalfCheetahBDExtractor(name="humanoid", logger=None)
        train_task = BraxTask(env_name="humanoid", bd_extractor=bd_extractor, test=False)
        test_task = BraxTask(env_name="humanoid", bd_extractor=bd_extractor, test=True)
    else:
        raise ValueError("Invalid task name: {}".format(task_name))
    return train_task, test_task


def train(sim_mgr, file_name, solver, max_iters, num_tests, test_interval, is_qd):
    print("Start training {} for {} iterations.".format(file_name, max_iters))
    header = ["iteration", "evals", "elapsed.time", "best.fitness", "avg.test", "std.test"]
    if is_qd:
        header += ["coverage", "qd.score"]
    listener = FileListener(file_name, header)
    start_time = time.perf_counter()
    best_fitness = float("-inf")

    for train_iters in range(max_iters):

        # Training.
        params = solver.ask()
        train_scores, bds = sim_mgr.eval_params(params=params, test=False)
        if np.max(train_scores) >= best_fitness:
            best_fitness = np.max(train_scores)
        if is_qd:
            if isinstance(solver, MAPElites):
                solver.observe_bd(bds)
            else:
                solver.qd_aux.set_population(params)
                solver.qd_aux.observe_bd(bds)
        solver.tell(fitness=train_scores)

        # Test periodically.
        if train_iters > 0 and train_iters % test_interval == 0:
            best_params = solver.best_params
            test_scores = np.array(sim_mgr.eval_params(params=best_params, test=True)[0])
            score_avg = np.mean(test_scores)
            score_std = np.std(test_scores)
            line = {"iteration": train_iters, "evals": train_iters * solver.pop_size,
                    "elapsed.time": time.perf_counter() - start_time,
                    "best.fitness": best_fitness, "avg.test": score_avg, "std.test": score_std}
            if is_qd:
                if isinstance(solver, MAPElites):
                    line["coverage"] = np.mean(solver.occupancy_lattice)
                    line["qd.score"] = np.mean(solver.fitness_lattice[solver.fitness_lattice != float("-inf")])
                else:
                    line["coverage"] = np.mean(solver.qd_aux.occupancy_lattice)
                    line["qd.score"] = np.mean(
                        solver.qd_aux.fitness_lattice[solver.qd_aux.fitness_lattice != float("-inf")])
            listener.listen(**line)
            if train_iters % 100 == 0:
                print("Iter={0}, #tests={1}, score.avg={2:.2f}, score.std={3:.2f}".format(
                    train_iters, num_tests, score_avg, score_std))
            if train_iters % 50 == 0 and is_qd:
                if isinstance(solver, MAPElites):
                    save_lattices(log_dir="/".join(file_name.split("/")[:-1]),
                                  file_name=file_name.split("/")[-1].replace("txt",
                                                                             "{}.qd_lattices".format(train_iters)),
                                  fitness_lattice=solver.fitness_lattice,
                                  params_lattice=jnp.empty(()),
                                  occupancy_lattice=solver.occupancy_lattice)
                else:
                    save_lattices(log_dir="/".join(file_name.split("/")[:-1]),
                                  file_name=file_name.split("/")[-1].replace("txt",
                                                                             "{}.qd_lattices".format(train_iters)),
                                  fitness_lattice=solver.qd_aux.fitness_lattice,
                                  params_lattice=jnp.empty(()),
                                  occupancy_lattice=solver.qd_aux.occupancy_lattice)

    # Final test.
    best_params = solver.best_params
    scores = np.array(sim_mgr.eval_params(params=best_params, test=True)[0])
    score_avg = np.mean(scores)
    score_std = np.std(scores)
    print("Iter={0}, #tests={1}, score.avg={2:.2f}, score.std={3:.2f}".format(
        train_iters, num_tests, score_avg, score_std))
    if is_qd:
        if isinstance(solver, MAPElites):
            save_lattices(log_dir="/".join(file_name.split("/")[:-1]),
                          file_name=file_name.split("/")[-1].replace("txt", "qd_lattices"),
                          fitness_lattice=solver.fitness_lattice,
                          params_lattice=jnp.empty(()),
                          occupancy_lattice=solver.occupancy_lattice)
        else:
            save_lattices(log_dir="/".join(file_name.split("/")[:-1]),
                          file_name=file_name.split("/")[-1].replace("txt", "qd_lattices"),
                          fitness_lattice=solver.qd_aux.fitness_lattice,
                          params_lattice=jnp.empty(()),
                          occupancy_lattice=solver.qd_aux.occupancy_lattice)
    print("time cost: {}s".format(time.perf_counter() - start_time))


def main(config, max_iter, num_dims=2, lrate_init=None, init_stdev=None, is_qd=False):
    logs_dir = "./output/{}/".format(config.task)
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    train_task, test_task = create_task(config)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        hidden_dims=[config.hidden_size] * num_dims,
        output_dim=train_task.act_shape[0],
    )
    if lrate_init is None or init_stdev is None:
        solver = create_solver(config, policy.num_params)
    else:
        solver = create_solver(config, policy.num_params, lrate_init, init_stdev)

    # Train.
    sim_mgr = SimManager(
        n_repeats=1,
        test_n_repeats=1,
        pop_size=solver.pop_size,
        n_evaluations=config.num_tests,
        policy_net=policy,
        train_vec_task=train_task,
        valid_vec_task=test_task,
        seed=config.seed,
        obs_normalizer=ObsNormalizer(obs_shape=train_task.obs_shape) if "cartpole" not in config.task else None
    )
    if lrate_init is None or init_stdev is None:
        file_name = os.path.join(logs_dir, ".".join([config.solver, str(config.seed), "txt"]))
    else:
        file_name = os.path.join(logs_dir, ".".join(
            [config.solver, str(config.seed), str(num_dims), str(lrate_init).split(".")[1],
             str(init_stdev).split(".")[1], "txt"]))
    train(sim_mgr, file_name, solver, max_iter, config.num_tests, config.test_interval, is_qd=is_qd)

    # Generate a GIF to visualize the policy.
    if "cartpole" not in config.task:
        return
    best_params = solver.best_params[None, :]
    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    act_fn = jax.jit(policy.get_actions)
    rollout_key = jax.random.PRNGKey(seed=0)[None, :]

    images = []
    task_s = task_reset_fn(rollout_key)
    policy_s = policy_reset_fn(task_s)
    images.append(CartPoleSwingUp.render(task_s, 0))
    done = False
    step = 0
    while not done:
        act, policy_s = act_fn(task_s, best_params, policy_s)
        task_s, r, d = step_fn(task_s, act)
        step += 1
        done = bool(d[0])
        if step % 5 == 0:
            images.append(
                CartPoleSwingUp.render(task_s, 0))

    gif_file = file_name.replace("txt", "gif")
    images[0].save(
        gif_file, save_all=True, append_images=images[1:], duration=40, loop=0)


if __name__ == "__main__":
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu_id
    main(configs, 490 if configs.solver not in ["me", "ars"] else 123, is_qd=True)
