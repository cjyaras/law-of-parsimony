import jax.numpy as jnp

def create_loss(target, input_data=None, mask=None, reduction="mean"):

    def loss_fn(output):
        if input_data is not None:
            residual = output @ input_data - target
        else:
            residual = output - target

        if mask is not None:
            residual *= mask

        if reduction == "mean":
            return 1/2 * jnp.mean(residual**2)
        elif reduction == "sum":
            return 1/2 * jnp.sum(residual**2)
        else:
            raise ValueError("Reduction type not implemented")
        
    return loss_fn