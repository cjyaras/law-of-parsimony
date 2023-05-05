import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import jax.numpy as jnp

from jax.random import PRNGKey, split
from jax import config
config.update("jax_enable_x64", True)

from data import generate_data
from loss import create_l2_loss
from network import init_net, compute_end_to_end, compute_factor
from solver import train
from utils import compose

key = PRNGKey(0)

## First orthogonal data
input_dim = 20
output_dim = 2
n_samples = 1000
depth = 3
init_type = "orth"
init_scale = 1

key, subkey = split(key)
target = generate_data(key=subkey, shape=(output_dim, n_samples))

key, subkey = split(key)
input_data = generate_data(key=subkey, shape=(input_dim, n_samples), orth=True)

key, subkey = split(key)
init_weights = init_net(key=subkey, input_dim=input_dim, output_dim=output_dim, width=input_dim, depth=depth, init_type="orth", init_scale=init_scale)

l2_loss_fn = create_l2_loss(target, input_data=input_data)
e2e_loss_fn = compose(l2_loss_fn, compute_end_to_end)

n_epochs = 1000
step_size = 1e-1
result = train(
    init_weights=init_weights,
    train_e2e_loss_fn=e2e_loss_fn,
    n_epochs=n_epochs,
    step_size=step_size
)

V1 = compute_factor(init_weights=init_weights, e2e_loss_fn=e2e_loss_fn, grad_rank=output_dim)
U1 = init_weights[0] @ V1 / init_scale

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
axes[0].imshow(jnp.log10(jnp.abs(U1.T @ init_weights[0] @ V1)), cmap='YlGn', vmax=1, vmin=-5)
axes[0].set_title('Before Training', fontsize=15)
axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))

pcm = axes[1].imshow(jnp.log10(jnp.abs(U1.T @ result['final_weights'][0] @ V1)), cmap='YlGn', vmax=1, vmin=-5)
axes[1].set_title('After Training', fontsize=15)
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

fig.colorbar(pcm, ax=axes.ravel().tolist())
plt.savefig('figs/clf_thm_whitened.png', dpi=300, bbox_inches='tight')

## Now with orthogonal data
input_dim = 20
output_dim = 2
n_samples = 1000
depth = 3
init_type = "orth"
init_scale = 1

key, subkey = split(key)
target = generate_data(key=subkey, shape=(output_dim, n_samples))

key, subkey = split(key)
input_data = generate_data(key=subkey, shape=(input_dim, n_samples), orth=False)

key, subkey = split(key)
init_weights = init_net(key=subkey, input_dim=input_dim, output_dim=output_dim, width=input_dim, depth=depth, init_type="orth", init_scale=init_scale)

l2_loss_fn = create_l2_loss(target, input_data=input_data)
e2e_loss_fn = compose(l2_loss_fn, compute_end_to_end)

n_epochs = 1000
step_size = 1e-1
result = train(
    init_weights=init_weights,
    train_e2e_loss_fn=e2e_loss_fn,
    n_epochs=n_epochs,
    step_size=step_size
)

V1 = compute_factor(init_weights=init_weights, e2e_loss_fn=e2e_loss_fn, grad_rank=output_dim)
U1 = init_weights[0] @ V1 / init_scale

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
axes[0].imshow(jnp.log10(jnp.abs(U1.T @ init_weights[0] @ V1)), cmap='YlGn', vmax=1, vmin=-2)
axes[0].set_title('Before Training', fontsize=15)
axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))

pcm = axes[1].imshow(jnp.log10(jnp.abs(U1.T @ result['final_weights'][0] @ V1)), cmap='YlGn', vmax=1, vmin=-2)
axes[1].set_title('After Training', fontsize=15)
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

fig.colorbar(pcm, ax=axes.ravel().tolist())
plt.savefig('figs/clf_thm_nonwhitened.png', dpi=300, bbox_inches='tight')