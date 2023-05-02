import jax.numpy as jnp
import jax.random as random
from jax import jit, grad
from utils import svd

def _init_weight(key, shape, init_type, init_scale):
    m, n = shape
    d = jnp.maximum(m, n)

    if init_type == "normal":
        # There is a sqrt(d) factor since we want initializations to have same size (Frobenius norm) in expectation
        weight = init_scale * random.normal(key=key, shape=shape) / jnp.sqrt(d)

    elif init_type == "orth":
        weight = init_scale * random.orthogonal(key=key, n=d)
        if m > n:
            weight = weight[:, :n]
        else:
            weight = weight[:m, :]

    else:
        raise ValueError(f"{init_type} initialization not implemented.")
    
    return weight

def init_net(key, input_dim, output_dim, width, depth, init_type, init_scale):
    if depth == 1:
        return [_init_weight(key, (output_dim, input_dim), init_type, init_scale)]
    
    keys = random.split(key, num=depth)
    weights = [_init_weight(keys[0], (width, input_dim), init_type, init_scale)]
    for i in range(depth-2):
        weights.append(_init_weight(keys[i+1], (width, width), init_type, init_scale))
    weights.append(_init_weight(keys[-1], (output_dim, width), init_type, init_scale))

    return weights

def create_network(pre=None, post=None):

    def network_fn(weights):
        product = weights[0]

        for w in weights[1:]:
            product = w @ product
        if pre is not None:
            product = product @ pre
        if post is not None:
            product = post @ product

        return product
    
    return jit(network_fn)

def compute_end_to_end(weights):
    product = weights[0]

    for w in weights[1:]:
        product = w @ product

    return product

def compute_prefactor(init_weights, loss_fn, network_fn, grad_rank):

    width = init_weights[0].shape[0]
    init_scale = jnp.linalg.norm(init_weights[0]) / jnp.sqrt(width)

    grad_W1_t0 = grad(lambda w: loss_fn(network_fn(w)))(init_weights)[0]
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

    compressed_initial_weights = [
        init_scale * jnp.eye(2*grad_rank) for _ in range(depth)
    ]

    return compressed_initial_weights, V1_1, UL_1