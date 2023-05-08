import jax.numpy as jnp
import jax.random as random
from jax import grad
from utils import svd

def _init_weight_orth(key, shape, init_scale):
    m, n = shape
    d = jnp.maximum(m, n)
    weight = init_scale * random.orthogonal(key=key, n=d)
    if m > n:
        weight = weight[:, :n]
    else:
        weight = weight[:m, :]
    return weight

def init_net_orth(key, input_dim, output_dim, width, depth, init_scale):
    keys = random.split(key, num=depth)

    weights = [_init_weight_orth(keys[0], (width, input_dim), init_scale)]
    for i in range(depth-2):
        weights.append(_init_weight_orth(keys[i+1], (width, width), init_scale))
    weights.append(_init_weight_orth(keys[-1], (output_dim, width), init_scale))

    return weights

def init_net_spec(target, width, depth):
    U, s, V = svd(target)
    sqrtS = jnp.diag(jnp.sqrt(s[:width]))

    weights = [sqrtS @ V[:, :width].T]
    for _ in range(depth-2):
        weights.append(jnp.eye(width))
    weights.append(U[:, :width] @ sqrtS)

    return weights

def compute_end_to_end(weights):
    product = weights[0]
    for w in weights[1:]:
        product = w @ product
    return product

def compute_factor(init_weights, e2e_loss_fn, grad_rank):

    width = init_weights[0].shape[0]
    init_scale = jnp.linalg.norm(init_weights[0]) / jnp.sqrt(width)

    grad_W1_t0 = grad(e2e_loss_fn)(init_weights)[0]
    Ugrad, _, Vgrad = svd(grad_W1_t0)
    Va = init_weights[0].T @ Ugrad[:, grad_rank:] / init_scale
    Vb = Vgrad[:, grad_rank:]
    V0 = Va @ svd(jnp.concatenate([Va, -Vb], axis=1))[2][:Va.shape[1], width:]
    V = svd(V0)[0][:, ::-1]

    return V

def compress_network(init_weights, V, grad_rank):

    width = init_weights[0].shape[0]
    depth = len(init_weights)
    init_scale = jnp.linalg.norm(init_weights[0]) / jnp.sqrt(width)

    V1_1 = V[:, :2*grad_rank]
    UL_1 = compute_end_to_end(init_weights) @ V1_1 / (init_scale ** depth)

    compressed_init_weights = [V1_1.T]
    compressed_init_weights += [
        init_scale * jnp.eye(2*grad_rank) for _ in range(depth)
    ]
    compressed_init_weights += [UL_1]

    return compressed_init_weights