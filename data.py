from math import prod

import jax.numpy as jnp
import jax.random as random
from jax.nn import one_hot

from utils import svd


def generate_orthogonal_input(key, input_dim, total_samples):
    assert input_dim <= total_samples
    return random.orthogonal(key, n=total_samples)[:input_dim, :]

def generate_data(key, shape, rank=None):
    mat = random.normal(key=key, shape=shape)
    if rank is not None:
        U, s, V = svd(mat)
        mat = U[:, :rank] @ jnp.diag(s[:rank]) @ V[:, :rank].T
    return mat

def generate_labels_and_target(num_classes, num_samples_per_class):
    labels = jnp.ravel(jnp.array([num_samples_per_class*[k] for k in range(num_classes)]))
    target = one_hot(labels, num_classes).T
    return labels, target

def generate_observation_matrix(key, percent_observed, shape):
    n_entries = prod(shape)
    n_observations = int(n_entries * percent_observed)
    indices = random.choice(key=key, a=jnp.arange(n_entries), shape=(n_observations,), replace=False)
    return jnp.zeros(n_entries, dtype=float).at[indices].set(1).reshape(*shape)