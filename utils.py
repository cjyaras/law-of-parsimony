import jax
import jax.numpy as jnp

from functools import partial
from jax import vmap, jit

@partial(vmap, in_axes=[None, 0])
def sensing_operator(output_mat, sensing_mat):
    return jnp.mean(output_mat * sensing_mat)

def update_weights(weights, gradient, step_size, factors):
    if factors:
        inner_weights = jax.tree_map(lambda p, g: p - step_size * g, weights[1:-1], gradient[1:-1])
        first_factor = weights[0] - 1e-1 * step_size * gradient[0]
        last_factor = weights[-1] - 1e-1 * step_size * gradient[-1]

        return [first_factor] + inner_weights + [last_factor]
    else:
        return jax.tree_map(lambda p, g: p - step_size * g, weights, gradient)

def svd(A):
    U, s, VT = jnp.linalg.svd(A)
    return U, s, VT.T
