import jax.random as random
import jax.numpy as jnp

from math import prod

def generate_data(key, shape, rank=None, orth=False):
    m, n = shape
    d = jnp.maximum(m, n)

    assert rank is None or orth is False, "Data cannot be low rank and orthogonal"

    if orth:
        mat = random.orthogonal(key=key, n=d)
        if m > n:
            mat = mat[:, :n]
        else:
            mat = mat[:m, :]
    elif rank is not None:
        key1, key2 = random.split(key)
        mat = (random.normal(key=key1, shape=(m, rank)) @
            random.normal(key=key2, shape=(rank, n))) / rank
    else:
        mat = random.normal(key=key, shape=shape)
        
    return mat

def generate_observation_matrix(key, percent_observed, shape):
    n_entries = prod(shape)
    n_observations = int(n_entries * percent_observed)
    indices = random.choice(key=key, a=jnp.arange(n_entries), shape=(n_observations,), replace=False)
    return jnp.zeros(n_entries, dtype=float).at[indices].set(1).reshape(*shape)

def generate_sensing_matrices(key, n_measurements, shape):
    return random.normal(key=key, shape=(n_measurements, shape[0], shape[1]))