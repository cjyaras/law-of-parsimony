import jax.numpy as jnp
import jax

from jax import grad
from jax.lax import fori_loop
from tqdm.auto import tqdm
from time import time

def _update_weights(weights, gradient, step_size, precond):
    if precond:
        return jax.tree_map(lambda p, g, s: p - s * g @ jnp.linalg.pinv(p.T @ p), weights, gradient, step_size)
    else:
        return jax.tree_map(lambda p, g, s: p - s * g, weights, gradient, step_size)
    
def compute_dlr(step_size, depth, prop):
    step_sizes = depth * [step_size]
    step_sizes[0] *= prop
    step_sizes[-1] *= prop

    return step_sizes

def train(init_weights, train_e2e_loss_fn, n_epochs, step_size, precond=False, test_e2e_loss_fn=None, tol=0, n_inner_loops=100):

    if type(step_size) is float:
        step_size = len(init_weights) * [step_size]

    # Define fun body in lax.fori_loop
    def body_fun(_, w):
        g = grad(train_e2e_loss_fn)(w)
        return _update_weights(w, g, step_size, precond)
    
    # Run once to compile
    fori_loop(0, n_inner_loops, body_fun, init_weights)

    train_loss = train_e2e_loss_fn(init_weights)
    train_loss_list = [train_loss]
    
    if test_e2e_loss_fn is not None:
        test_loss = test_e2e_loss_fn(init_weights)
        test_loss_list = [test_loss]

    time_list = [0.]
    weights = init_weights

    num_iters = n_epochs // n_inner_loops
    pbar = tqdm(range(num_iters))

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

        time_list.append(time() - start_time)

        if train_loss < tol:
            break

    result_dict = {
        'train_loss': jnp.array(train_loss_list),
        'time': jnp.array(time_list),
        'final_weights': weights
    }

    if test_e2e_loss_fn is not None:
        result_dict['test_loss'] = jnp.array(test_loss_list)

    return result_dict