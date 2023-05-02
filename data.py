import jax.random as random
import jax.numpy as jnp

from math import prod

def generate_data(key, shape, rank=None):
    if rank is None:
        mat = random.normal(key=key, shape=shape)
    else:
        key1, key2 = random.split(key)
        mat = (random.normal(key=key1, shape=(shape[0], rank)) @
            random.normal(key=key2, shape=(rank, shape[1]))) / rank
    return mat

def generate_observation_matrix(key, percent_observed, shape):
    n_entries = prod(shape)
    n_observations = int(n_entries * percent_observed)
    indices = random.choice(key=key, a=jnp.arange(n_entries), shape=(n_observations,), replace=False)
    return jnp.zeros(n_entries, dtype=bool).at[indices].set(True).reshape(*shape)

def generate_sensing_matrices(key, n_measurements, shape):
    return random.normal(key=key, shape=(n_measurements, shape[0], shape[1]))