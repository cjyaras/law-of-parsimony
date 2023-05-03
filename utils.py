import jax.numpy as jnp

from functools import partial
from jax import vmap

@partial(vmap, in_axes=[None, 0])
def sensing_operator(output_mat, sensing_mat):
    return jnp.mean(output_mat * sensing_mat)

def svd(A):
    U, s, VT = jnp.linalg.svd(A)
    return U, s, VT.T
