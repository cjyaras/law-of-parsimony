import jax.numpy as jnp

def svd(A, full_matrices=True):
    U, s, VT = jnp.linalg.svd(A, full_matrices=full_matrices)
    return U, s, VT.T

def compose(f, g):
    def h(x):
        return f(g(x))
    return h