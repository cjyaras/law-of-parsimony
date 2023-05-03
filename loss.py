import jax.numpy as jnp

from utils import sensing_operator

def create_l2_loss(target):
    def loss_fn(output):
        residual = output - target
        return 1/2 * jnp.mean(residual**2)
    return loss_fn

def create_mc_loss(target, observed_entries):
    mask = jnp.array(observed_entries, dtype=float)
    def loss_fn(output):
        residual = mask * (output - target)
        return 1/2 * jnp.mean(residual**2)
    return loss_fn

def create_gs_loss(sensing_target, sensing_matrices):
    def loss_fn(output):
        residual = sensing_operator(output, sensing_matrices) - sensing_target
        return 1/2 * jnp.mean(residual**2)
    return loss_fn