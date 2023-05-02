import jax.numpy as jnp

from jax import value_and_grad
from tqdm.auto import tqdm
from time import time

from utils import update_weights

def train(init_weights, loss_fn, network_fn, num_iters, step_size, factors=False, save_weights=False):

    # Run update once to compile
    loss, gradient = value_and_grad(lambda w: loss_fn(network_fn(w)))(init_weights)
    update_weights(init_weights, gradient, step_size, factors)

    loss_list = [loss]
    time_list = [0.]

    weights_list = [init_weights]
    weights = init_weights

    iterator = tqdm(range(num_iters))

    start_time = time()
    for _ in iterator:
        loss, gradient = value_and_grad(lambda w: loss_fn(network_fn(w)))(weights)
        weights = update_weights(weights, gradient, step_size, factors)
        loss_list.append(loss)
        time_list.append(time() - start_time)
        if save_weights:
            weights_list.append(weights)

    if save_weights:
        return weights_list, jnp.array(loss_list), jnp.array(time_list)
    else:
        return weights, jnp.array(loss_list), jnp.array(time_list)