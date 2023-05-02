import jax.numpy as jnp

from jax import value_and_grad, grad
from jax.lax import fori_loop
from tqdm.auto import tqdm
from time import time

from utils import update_weights

n_inner_loops = 100

def train(init_weights, loss_fn, network_fn, num_iters, step_size, factors=False, save_weights=False):

    e2e_loss_fn = lambda w: loss_fn(network_fn(w))

    # Define fun body in lax.fori_loop
    def body_fun(_, w):
        g = grad(e2e_loss_fn)(w)
        return update_weights(w, g, step_size, factors)
    
    # Run once to compile
    fori_loop(0, n_inner_loops, body_fun, init_weights)

    loss = e2e_loss_fn(init_weights)
    loss_list = [loss]
    time_list = [0.]

    weights_list = [init_weights]
    weights = init_weights

    pbar = tqdm(range(num_iters))

    start_time = time()
    for _ in pbar:

        pbar.set_description(f"Loss: {loss:0.3e}")
        weights = fori_loop(0, n_inner_loops, body_fun, weights)
        loss = e2e_loss_fn(weights)
        # loss, gradient = value_and_grad(lambda w: loss_fn(network_fn(w)))(weights)
        # weights = update_weights(weights, gradient, step_size, factors)
        loss_list.append(loss)
        time_list.append(time() - start_time)
        if save_weights:
            weights_list.append(weights)

    if save_weights:
        return weights_list, jnp.array(loss_list), jnp.array(time_list)
    else:
        return weights, jnp.array(loss_list), jnp.array(time_list)