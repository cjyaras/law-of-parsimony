import jax.numpy as jnp

from utils import sensing_operator

def create_l2_loss(target, reduction="mean"):
    def loss_fn(output):
        residual = output - target
        if reduction == "mean":
            return 1/2 * jnp.mean(residual**2)
        elif reduction == "sum":
            return 1/2 * jnp.sum(residual**2)
        else:
            raise ValueError("Reduction type not implemented")
    return loss_fn

def create_mc_loss(target, observed_entries, reduction="mean"):
    mask = jnp.array(observed_entries, dtype=float)
    def loss_fn(output):
        residual = mask * (output - target)
        if reduction == "mean":
            return 1/2 * jnp.mean(residual**2)
        elif reduction == "sum":
            return 1/2 * jnp.sum(residual**2)
        else:
            raise ValueError("Reduction type not implemented")
    return loss_fn

def create_gs_loss(sensing_target, sensing_matrices, reduction="mean"):
    def loss_fn(output):
        residual = sensing_operator(output, sensing_matrices) - sensing_target
        if reduction == "mean":
            return 1/2 * jnp.mean(residual**2)
        elif reduction == "sum":
            return 1/2 * jnp.sum(residual**2)
        else:
            raise ValueError("Reduction type not implemented")
    return loss_fn