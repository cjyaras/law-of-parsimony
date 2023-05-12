import jax.random as random
import jax.numpy as jnp
from utils import svd

from math import prod

def generate_data(key, shape, rank=None):
    mat = random.normal(key=key, shape=shape)
    if rank is not None:
        U, s, V = svd(mat)
        mat = U[:, :rank] @ jnp.diag(s[:rank]) @ V[:, :rank].T
    return mat

def generate_label_matrix(labels):
    return 

def generate_observation_matrix(key, percent_observed, shape):
    n_entries = prod(shape)
    n_observations = int(n_entries * percent_observed)
    indices = random.choice(key=key, a=jnp.arange(n_entries), shape=(n_observations,), replace=False)
    return jnp.zeros(n_entries, dtype=float).at[indices].set(1).reshape(*shape)