import jax.numpy as jnp

def create_l2_loss(target, input_data, reduction="mean"):

    def loss_fn(output):
        if input_data is not None:
            residual = output @ input_data - target
        else:
            residual = output - target

        if reduction == "mean":
            return 1/2 * jnp.mean(residual**2)
        elif reduction == "sum":
            return 1/2 * jnp.sum(residual**2)
        else:
            raise ValueError("Reduction type not implemented")
        
    return loss_fn

def create_mc_loss(target, mask, reduction="mean"):

    def loss_fn(output):
        residual = mask * (output - target)
        if reduction == "mean":
            return 1/2 * jnp.mean(residual**2)
        elif reduction == "sum":
            return 1/2 * jnp.sum(residual**2)
        else:
            raise ValueError("Reduction type not implemented")
        
    return loss_fn