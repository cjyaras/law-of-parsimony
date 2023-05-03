import jax.numpy as jnp
import jax

from jax import grad
from jax.lax import fori_loop
from tqdm.auto import tqdm
from time import time

def update_weights(weights, gradient, step_size, factors):
    if factors:
        inner_weights = jax.tree_map(lambda p, g: p - step_size * g, weights[1:-1], gradient[1:-1])
        first_factor = weights[0] - 1e-1 * step_size * gradient[0]
        last_factor = weights[-1] - 1e-1 * step_size * gradient[-1]
        return [first_factor] + inner_weights + [last_factor]
    else:
        return jax.tree_map(lambda p, g: p - step_size * g, weights, gradient)

def train(init_weights, e2e_loss_fn, n_epochs, step_size, n_inner_loops=100, factors=False, save_weights=False):

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

    num_iters = n_epochs // n_inner_loops
    pbar = tqdm(range(num_iters))

    start_time = time()
    for _ in pbar:
        pbar.set_description(f"Loss: {loss:0.2e}")
        weights = fori_loop(0, n_inner_loops, body_fun, weights)
        loss = e2e_loss_fn(weights)
        loss_list.append(loss)
        time_list.append(time() - start_time)
        if save_weights:
            weights_list.append(weights)

    if save_weights:
        return weights_list, jnp.array(loss_list), jnp.array(time_list)
    else:
        return weights, jnp.array(loss_list), jnp.array(time_list)