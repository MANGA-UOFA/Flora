from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
from optax._src import base, clipping, transform

from src.optimizers.optax.utils import scale_by_learning_rate, scale_by_factored_rms, scale_by_sign

ScalarOrSchedule = Union[float, jax.Array, base.Schedule]
MaskOrFn = Optional[Union[Any, Callable[[base.Params], Any]]]


def adafactor(
    learning_rate: Optional[ScalarOrSchedule] = None,
    min_dim_size_to_factor: int = 128,
    decay_rate: float = 0.8,
    decay_offset: int = 0,
    multiply_by_parameter_scale: float = True,
    clipping_threshold: Optional[float] = 1.0,
    momentum: Optional[float] = None,
    dtype_momentum: Any = jnp.float32,
    weight_decay_rate: Optional[float] = None,
    eps: float = 1e-30,
    factored: bool = True,
    weight_decay_mask: MaskOrFn = None,
    sign: bool = False,
) -> base.GradientTransformation:
    tx = []

    # momentum
    if momentum is not None:
        tx.append(transform.ema(momentum, debias=False, accumulator_dtype=dtype_momentum))

    # main transformation
    if sign:
        tx.append(scale_by_sign())
    else:
        tx.append(scale_by_factored_rms(factored, decay_rate, decay_offset, min_dim_size_to_factor, eps))

    # transformation (all can be disabled via adafactor's constructor args).
    if clipping_threshold is not None:
        tx.append(clipping.clip_by_block_rms(clipping_threshold))
    if learning_rate is not None:
        tx.append(scale_by_learning_rate(learning_rate, flip_sign=False))
    if multiply_by_parameter_scale:
        tx.append(transform.scale_by_param_block_rms())
    if weight_decay_rate is not None:
        tx.append(transform.add_decayed_weights(weight_decay_rate, mask=weight_decay_mask))
    # In gradient "descent" we follow the negative gradient.
    tx.append(transform.scale(-1))

    init_fns = [t.init for t in tx]
    update_fns = [t.update for t in tx]

    def init_fn(params):
        return tuple(fn(params) for fn in init_fns)

    def update_fn(grads, state, params=None, query_only=False, **extra_args):
        if len(update_fns) != len(state):
            raise ValueError(
                "The number of updates and states has to be the same in " "chain! Make sure you have called init first!"
            )

        new_state = []

        if momentum is not None:
            updates, new_s = update_fns[0](grads, state[0], params, **extra_args)  # TODO: query_only
            new_state.append(new_s)
            updates, new_s = update_fns[1](grads, state[1], params, updates, query_only, **extra_args)
            new_state.append(new_s)
            next_idx = 2

        else:
            updates, new_s = update_fns[0](grads, state[0], params, grads, query_only, **extra_args)
            new_state.append(new_s)
            next_idx = 1

        for s, fn in zip(state[next_idx:], update_fns[next_idx:]):
            updates, new_s = fn(updates, s, params, **extra_args)
            new_state.append(new_s)
        return updates, tuple(new_state)

    return base.GradientTransformationExtraArgs(init_fn, update_fn)
