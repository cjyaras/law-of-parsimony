import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import grad
from jax.nn import relu

from utils import compose, svd


def _init_weight_orth(key, shape, init_scale):
    m, n = shape
    d = int(jnp.maximum(m, n))
    weight = init_scale * random.orthogonal(key=key, n=d)
    if m > n:
        weight = weight[:, :n]
    else:
        weight = weight[:m, :]
    return weight


def _init_weight_norm(key, shape, init_scale):
    m, n = shape
    d = jnp.minimum(m, n)

    weight = init_scale * random.normal(key=key, shape=shape)
    # Correction factor to have the same norm as orth init
    correct_factor = jnp.sqrt(d) / jnp.sqrt(m * n)

    return correct_factor * weight


def _init_weight_unif(key, shape, init_scale):
    m, n = shape
    d = jnp.minimum(m, n)

    weight = random.uniform(key=key, shape=shape, minval=-init_scale, maxval=init_scale)
    # Correction factor to have the same norm as orth init
    correct_factor = jnp.sqrt(3) * jnp.sqrt(d) / jnp.sqrt(m * n)

    return correct_factor * weight


def init_net(key, input_dim, output_dim, width, depth, init_scale, init_type="orth"):
    if init_type == "orth":
        init_func = _init_weight_orth
    elif init_type == "norm":
        init_func = _init_weight_norm
    elif init_type == "unif":
        init_func = _init_weight_unif
    else:
        raise ValueError("Init type not recognized")

    keys = random.split(key, num=depth)

    weights = [init_func(keys[0], (width, input_dim), init_scale)]
    for i in range(depth - 2):
        weights.append(init_func(keys[i + 1], (width, width), init_scale))
    weights.append(init_func(keys[-1], (output_dim, width), init_scale))

    return weights


def compute_end_to_end(weights):
    product = weights[0]
    for w in weights[1:]:
        product = w @ product
    return product


def compute_outputs(weights, input_data, nonlinear=False):
    intermediates = [input_data]
    output = weights[0] @ input_data
    if nonlinear:
        output = relu(output)
    intermediates.append(output)

    for w in weights[1:-1]:
        output = w @ output
        if nonlinear:
            output = relu(output)
        intermediates.append(output)

    output = weights[-1] @ output
    return output, intermediates


def compute_factor(init_weights, network_fn, loss_fn, grad_rank):
    e2e_loss_fn = compose(loss_fn, network_fn)

    width = init_weights[0].shape[0]
    init_scale = jnp.linalg.norm(init_weights[0]) / jnp.sqrt(width)

    grad_W1_t0 = grad(e2e_loss_fn)(init_weights)[0]
    Ugrad, _, Vgrad = svd(grad_W1_t0)
    Va = init_weights[0].T @ Ugrad[:, grad_rank:] / init_scale
    Vb = Vgrad[:, grad_rank:]
    V0 = Va @ svd(jnp.concatenate([Va, -Vb], axis=1))[2][: Va.shape[1], width:]
    V = svd(V0)[0][:, ::-1]

    return V[:, : 2 * grad_rank]


def compute_factor_v2(init_weights, network_fn, loss_fn, grad_rank):
    e2e_loss_fn = compose(loss_fn, network_fn)

    grad_W1_t0 = grad(e2e_loss_fn)(init_weights)[0]
    v1 = np.linalg.svd(grad_W1_t0, full_matrices=False)[2].T[:, :grad_rank]
    v2 = np.linalg.svd(grad_W1_t0.T @ init_weights[0], full_matrices=False)[2].T[
        :, :grad_rank
    ]
    V1_1 = np.linalg.svd(np.concatenate([v1, v2], axis=1), full_matrices=False).U
    return V1_1


def compute_factor_v3(init_weights, network_fn, loss_fn, grad_rank):
    e2e_loss_fn = compose(loss_fn, network_fn)

    grad_W1_t0 = grad(e2e_loss_fn)(init_weights)[0]
    A = grad_W1_t0
    B = grad_W1_t0.T @ init_weights[0]
    V1_1 = np.linalg.svd(np.concatenate([A, B], axis=0), full_matrices=True)[2].T[
        :, : 2 * grad_rank
    ]
    return V1_1


def compress_network(init_weights, V1_1, grad_rank):
    width = init_weights[0].shape[0]
    depth = len(init_weights)
    init_scale = jnp.linalg.norm(init_weights[0]) / jnp.sqrt(width)

    UL_1 = compute_end_to_end(init_weights) @ V1_1 / (init_scale**depth)

    compressed_init_weights = [V1_1.T]
    compressed_init_weights += [
        init_scale * jnp.eye(2 * grad_rank) for _ in range(depth)
    ]
    compressed_init_weights += [UL_1]

    return compressed_init_weights
