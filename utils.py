import jax.numpy as jnp

def svd(A):
    U, s, VT = jnp.linalg.svd(A)
    return U, s, VT.T

def compose(f, g):
    def h(x):
        return f(g(x))
    return h
