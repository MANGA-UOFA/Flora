from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import base, clipping, transform, utils
import numpy as np

from src.optimizers.optax.utils import (
    NaiveDecomposition,
    TwoSideRandomDecomposition,
    next_rng_key,
    random_orthogonal,
    random_split_like_tree,
    scale_by_factored_rms,
    scale_by_learning_rate,
)


def random_generate(key, shape, dtype):
    orth = False
    if orth:
        return random_orthogonal(key, shape, dtype)
    else:
        return jax.random.normal(key, shape, dtype=dtype) / jnp.sqrt(min(shape))


def identity() -> base.GradientTransformation:
    """Stateless identity transformation that leaves input gradients untouched.

    This function passes through the *gradient updates* unchanged.

    Note, this should not to be confused with `set_to_zero`, which maps the input
    updates to zero - which is the transform required for the *model parameters*
    to be left unchanged when the updates are applied to them.

    Returns:
      A `GradientTransformation` object.
    """

    def init_fn(_):
        return base.EmptyState()

    def update_fn(grads, state, params, updates):
        del grads, params
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)


def flora(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    tau: int = 4,
    seed: int = 0,
    kappa: int = 1000,
    clipping_threshold: Optional[float] = 1.0,
    multiply_by_parameter_scale: bool = True,
    weight_decay: Optional[optax.ScalarOrSchedule] = None,
    eps: float = 1e-30,
    rng_only: bool = False,
    min_dim_size_to_factor: int = 128,
    factorized_second_moment: bool = True,
    side: str = "auto",
) -> optax.GradientTransformation:
    tx = [
        scale_by_flora(
            factored=tau is not None,
            beta=b1,
            tau=tau,
            seed=seed,
            kappa=kappa,
            rng_only=rng_only,
            min_dim_size_to_factor=min_dim_size_to_factor,
            side=side,
        )
    ]
    tx.append(
        scale_by_factored_rms(
            factored=factorized_second_moment,
            decay_rate=b2,
            epsilon=eps,
            min_dim_size_to_factor=min_dim_size_to_factor,
        )
    )
    if clipping_threshold is not None:
        tx.append(clipping.clip_by_block_rms(clipping_threshold))
    tx.append(scale_by_learning_rate(learning_rate, flip_sign=False))
    if multiply_by_parameter_scale:
        tx.append(transform.scale_by_param_block_rms())
    if weight_decay is not None:
        tx.append(transform.add_decayed_weights(weight_decay))
    tx.append(transform.scale(-1.0))

    init_fns = [t.init for t in tx]
    update_fns = [t.update for t in tx]

    def init_fn(params):
        return tuple(fn(params) for fn in init_fns)

    def update_fn(grads, state, params=None, **extra_args):
        if len(update_fns) != len(state):
            raise ValueError(
                "The number of updates and states has to be the same in "
                "chain! Make sure you have called init first!"
            )

        new_state = []

        updates, new_s = update_fns[0](grads, state[0], params, **extra_args)
        new_state.append(new_s)

        updates, new_s = update_fns[1](grads, state[1], params, updates, **extra_args)
        new_state.append(new_s)

        for s, fn in zip(state[2:], update_fns[2:]):
            updates, new_s = fn(updates, s, params, **extra_args)
            new_state.append(new_s)
        return updates, tuple(new_state)

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


class ScaleByFloraState(NamedTuple):
    """State for the Flora algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    decomposition: chex.ArrayTree
    rng: chex.PRNGKey


def scale_by_flora(
    factored: bool = True,
    beta: Optional[float] = None,
    tau: int = 4,
    seed: int = 0,
    kappa: int = 1000,
    rng_only: bool = False,
    min_dim_size_to_factor: int = 128,
    side: str = "auto",
) -> optax.GradientTransformation:
    def should_factorize(params):
        if factored is False:
            return False
        if params.ndim != 2:
            return False
        if max(params.shape) > min(params.shape) * 16:
            # do not factorize embeddings
            return False
        return min(params.shape) >= min_dim_size_to_factor
        return True

    mu_dtype = utils.canonicalize_dtype(jnp.float32)

    def init_fn(params):
        rng = jax.random.PRNGKey(seed)
        prngkey_tree = random_split_like_tree(rng, params)

        def _init_layer(params, key):
            if should_factorize(params):
                ind, outd = params.shape
                l_key, r_key = jax.random.split(key, 2)
                otau = itau = tau
                if side == "auto":
                    _side = "right" if outd > ind else "left"
                else:
                    _side = side
                return TwoSideRandomDecomposition(
                    r_data=jnp.zeros((outd, otau), dtype=mu_dtype)
                    if _side != "left"
                    else None,
                    r_proj=(
                        r_key
                        if rng_only
                        else random_generate(r_key, (otau, ind), mu_dtype)
                    )
                    if _side != "left"
                    else None,
                    l_data=jnp.zeros((itau, ind), dtype=mu_dtype)
                    if _side != "right"
                    else None,
                    l_proj=(
                        l_key
                        if rng_only
                        else random_generate(l_key, (outd, itau), mu_dtype)
                    )
                    if _side != "right"
                    else None,
                )
            else:
                return NaiveDecomposition(
                    data=jnp.zeros_like(params, dtype=mu_dtype),
                )

        return ScaleByFloraState(
            count=jnp.zeros([], jnp.int32),
            decomposition=jax.tree_map(_init_layer, params, prngkey_tree),
            rng=rng,
        )

    @jax.remat
    def update_state(grads, state, params=None):
        prngkey_tree = random_split_like_tree(state.rng, grads)
        grads = jax.tree_map(lambda x: x.astype(mu_dtype), grads)

        def _naive_fn(grad, dcomp, key):
            return NaiveDecomposition(data=beta * dcomp.data + (1 - beta) * grad)

        def _full_layer_fn(grad, dcomp, key):
            if not should_factorize(grad):
                return _naive_fn(grad, dcomp, key)
            shape = np.array(grad.shape)
            ind, outd = shape
            otau = itau = tau

            if side == "auto":
                _side = "right" if outd > ind else "left"
            else:
                _side = side

            l_key, r_key = jax.random.split(key, 2)
            ind, outd = grad.shape

            if _side != "right":
                l_proj = (
                    random_generate(dcomp.l_proj, (outd, tau), mu_dtype)
                    if rng_only
                    else dcomp.l_proj
                )
                l_data = dcomp.l_data
                new_l_key = next_rng_key(dcomp.l_proj) if rng_only else l_key
                new_l_proj = random_generate(new_l_key, (outd, itau), mu_dtype)
                his_l = jnp.linalg.multi_dot([new_l_proj.T, l_proj, l_data])
                new_l_data = beta * his_l + (1 - beta) * (new_l_proj.T @ grad.T)

            if _side != "left":
                r_proj = (
                    random_generate(dcomp.r_proj, (tau, ind), mu_dtype)
                    if rng_only
                    else dcomp.r_proj
                )
                r_data = dcomp.r_data
                new_r_key = next_rng_key(dcomp.r_proj) if rng_only else r_key
                new_r_proj = random_generate(new_r_key, (otau, ind), mu_dtype)
                his_r = jnp.linalg.multi_dot([r_data, r_proj, new_r_proj.T])
                new_r_data = beta * his_r + (1 - beta) * (grad.T @ new_r_proj.T)

            return TwoSideRandomDecomposition(
                l_data=new_l_data if _side != "right" else None,
                l_proj=(new_l_key if rng_only else new_l_proj)
                if _side != "right"
                else None,
                r_data=new_r_data if _side != "left" else None,
                r_proj=(new_r_key if rng_only else new_r_proj)
                if _side != "left"
                else None,
            )

        def _partial_layer_fn(grad, dcomp, key):
            if not should_factorize(grad):
                return _naive_fn(grad, dcomp, key)
            shape = np.array(grad.shape)
            ind, outd = shape

            if side == "auto":
                _side = "right" if outd > ind else "left"
            else:
                _side = side

            if _side != "right":
                l_data = dcomp.l_data
                l_proj = (
                    random_generate(dcomp.l_proj, (outd, tau), mu_dtype)
                    if rng_only
                    else dcomp.l_proj
                )

            if _side != "left":
                r_data = dcomp.r_data
                r_proj = (
                    random_generate(dcomp.r_proj, (tau, ind), mu_dtype)
                    if rng_only
                    else dcomp.r_proj
                )

            return TwoSideRandomDecomposition(
                l_data=beta * l_data + (1 - beta) * (l_proj.T @ grad.T)
                if _side != "right"
                else None,
                l_proj=(dcomp.l_proj if rng_only else dcomp.l_proj)
                if _side != "right"
                else None,
                r_data=beta * r_data + (1 - beta) * (grad.T @ r_proj.T)
                if _side != "left"
                else None,
                r_proj=(dcomp.r_proj if rng_only else dcomp.r_proj)
                if _side != "left"
                else None,
            )

        rng = next_rng_key(state.rng)
        decomposition = jax.lax.cond(
            jnp.mod(state.count, kappa) == 0,
            lambda: jax.tree_map(
                _full_layer_fn, grads, state.decomposition, prngkey_tree
            ),
            lambda: jax.tree_map(
                _partial_layer_fn, grads, state.decomposition, prngkey_tree
            ),
        )

        return ScaleByFloraState(
            count=state.count + 1,
            decomposition=decomposition,
            rng=rng,
        )

    @jax.remat
    def query(grads, state, params=None):
        def _naive_fn(dcomp, g):
            return dcomp.data

        def _layer_fn(grads, dcomp):
            if not should_factorize(grads):
                return _naive_fn(dcomp, grads)

            shape = np.array(grads.shape)
            (ind, outd) = shape

            if side == "auto":
                _side = "right" if outd > ind else "left"
            else:
                _side = side

            if _side == "left":
                l_proj = (
                    random_generate(dcomp.l_proj, (outd, tau), mu_dtype)
                    if rng_only
                    else dcomp.l_proj
                )
                l_data = dcomp.l_data
                return (l_proj @ l_data).T
            elif _side == "right":
                r_proj = (
                    random_generate(dcomp.r_proj, (tau, ind), mu_dtype)
                    if rng_only
                    else dcomp.r_proj
                )
                r_data = dcomp.r_data
                return (r_data @ r_proj).T
            else:
                r_proj = (
                    random_generate(dcomp.r_proj, (tau, ind), mu_dtype)
                    if rng_only
                    else dcomp.r_proj
                )
                l_proj = (
                    random_generate(dcomp.l_proj, (outd, tau), mu_dtype)
                    if rng_only
                    else dcomp.l_proj
                )
                l_data = dcomp.l_data
                r_data = dcomp.r_data
                return (r_data @ r_proj + l_proj @ l_data).T / 2

        return jax.tree_map(_layer_fn, grads, state.decomposition)

    @jax.remat
    def update_fn(grads, state, params=None):
        del params

        updates = query(grads, state)
        updates = jax.tree_map(lambda m, g: m * beta + g * (1 - beta), updates, grads)
        new_state = update_state(grads, state)

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_interp(
    weight: float = 0.9,
    transform_fn: Optional[callable] = None,
) -> optax.GradientTransformation:
    def init_fn(params):
        return base.EmptyState()

    def update_fn(grads, state, params=None, updates=None):
        updates = jax.tree_map(
            lambda g, u: (1 - weight) * g + weight * u, grads, updates
        )
        if transform_fn is not None:
            updates = jax.tree_map(transform_fn, updates)
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)
