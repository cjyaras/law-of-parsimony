import jax.numpy as jnp
import jax

from jax import grad
from jax.lax import fori_loop
from tqdm.auto import tqdm
from time import time

def _update_weights(weights, gradient, step_size, weight_decay):
    wd = weight_decay
    return jax.tree_map(lambda p, g, s: (1 - s * wd) * p - s * g, weights, gradient, step_size)

def train(init_weights, train_e2e_loss_fn, n_outer_loops, step_size, weight_decay=0, test_e2e_loss_fn=None, tol=0, n_inner_loops=100, save_weights=False):

    if type(step_size) is not list:
        step_size = len(init_weights) * [step_size]

    # Define fun body in lax.fori_loop
    def body_fun(_, w):
        g = grad(train_e2e_loss_fn)(w)
        return _update_weights(w, g, step_size, weight_decay)
    
    # Run once to compile
    fori_loop(0, n_inner_loops, body_fun, init_weights)

    train_loss = train_e2e_loss_fn(init_weights)
    train_loss_list = [train_loss]
    
    if test_e2e_loss_fn is not None:
        test_loss = test_e2e_loss_fn(init_weights)
        test_loss_list = [test_loss]

    time_list = [0.]
    weights = init_weights
    if save_weights:
        weights_list = [weights]

    pbar = tqdm(range(n_outer_loops))

    start_time = time()
    for _ in pbar:
        if test_e2e_loss_fn is not None:
            pbar.set_description(f"Train loss: {train_loss:0.2e}, test loss: {test_loss:0.2e}")
        else:
            pbar.set_description(f"Train loss: {train_loss:0.2e}")

        weights = fori_loop(0, n_inner_loops, body_fun, weights)
        train_loss = train_e2e_loss_fn(weights)
        train_loss_list.append(train_loss)

        if test_e2e_loss_fn is not None:
            test_loss = test_e2e_loss_fn(weights)
            test_loss_list.append(test_loss)

        if save_weights:
            weights_list.append(weights)

        time_list.append(time() - start_time)

        if train_loss < tol:
            break

    result_dict = {
        'train_loss': jnp.array(train_loss_list),
        'time': jnp.array(time_list),
        'final_weights': weights,
    }

    if test_e2e_loss_fn is not None:
        result_dict['test_loss'] = jnp.array(test_loss_list)

    if save_weights:
        result_dict['weights'] = weights_list

    return result_dict