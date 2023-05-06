import matplotlib.pyplot as plt
from jax.random import PRNGKey, split

from data import generate_data, generate_observation_matrix
from loss import create_mc_loss
from network import init_net, compute_end_to_end, compute_factor, compress_network
from solver import train, compute_dlr
from utils import compose

key = PRNGKey(0)

# Generate data
r = 10
d = 2000
depth = 3
init_type = "orth"
init_scale = 1e-3

key, subkey = split(key)
target = generate_data(key=subkey, shape=(d, d), rank=r)

key, subkey = split(key)
init_weights = init_net(key=subkey, input_dim=d, output_dim=d, width=d, depth=depth, init_type="orth", init_scale=init_scale)

key, subkey = split(key)
percent_observed = 0.2
mask = generate_observation_matrix(key=subkey, percent_observed=percent_observed, shape=(d, d))

train_mc_loss_fn = create_mc_loss(target, mask)
test_mc_loss_fn = create_mc_loss(target, 1 - mask)
train_e2e_loss_fn = compose(train_mc_loss_fn, compute_end_to_end)
test_e2e_loss_fn = compose(test_mc_loss_fn, compute_end_to_end)

tol = 1e-12
step_size = 5e3

n_epochs = 250000
original_step_size = step_size
# original_result = train(
#     init_weights=init_weights,
#     train_e2e_loss_fn=train_e2e_loss_fn,
#     n_epochs=n_epochs,
#     step_size=original_step_size,
#     test_e2e_loss_fn=test_e2e_loss_fn,
#     tol=tol
# )

V = compute_factor(init_weights=init_weights, e2e_loss_fn=train_e2e_loss_fn, grad_rank=r)
comp_init_weights = compress_network(init_weights, V, r)
comp_step_size = compute_dlr(step_size=step_size, depth=len(comp_init_weights), prop=0.01)
print(comp_step_size)
comp_result = train(
    init_weights=comp_init_weights,
    train_e2e_loss_fn=train_e2e_loss_fn,
    n_epochs=n_epochs,
    step_size=comp_step_size,
    test_e2e_loss_fn=test_e2e_loss_fn,
    tol=tol
)

# Narrow, width=r
key, subkey = split(key)
narrow_r_init_weights = init_net(key=subkey, input_dim=d, output_dim=d, width=r, depth=depth, init_type="orth", init_scale=init_scale)
narrow_r_step_size = step_size
narrow_r_result = train(
    init_weights=narrow_r_init_weights,
    train_e2e_loss_fn=train_e2e_loss_fn,
    n_epochs=n_epochs,
    step_size=narrow_r_step_size,
    test_e2e_loss_fn=test_e2e_loss_fn,
    tol=tol
)

# Narrow, width=2r
key, subkey = split(key)
narrow_2r_init_weights = init_net(key=subkey, input_dim=d, output_dim=d, width=2*r, depth=depth, init_type="orth", init_scale=init_scale)
narrow_2r_step_size = step_size
narrow_2r_result = train(
    init_weights=narrow_2r_init_weights,
    train_e2e_loss_fn=train_e2e_loss_fn,
    n_epochs=n_epochs,
    step_size=narrow_2r_step_size,
    test_e2e_loss_fn=test_e2e_loss_fn,
    tol=tol
)

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

axes[0].plot(original_result['train_loss'], linewidth=6, linestyle='--', label='Original')
axes[0].plot(comp_result['train_loss'], linewidth=3, label='CompDMF')
axes[0].plot(narrow_r_result['train_loss'], linewidth=3, label='Width=$r$')
axes[0].plot(narrow_2r_result['train_loss'], linewidth=3, label='Width=$2r$')
axes[0].set_xlabel('Iteration (x100)', fontsize=14)
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# axes[0].locator_params(axis='y', nbins=6)
axes[0].set_ylabel('Train Loss', fontsize=14)
axes[0].legend(fontsize=14)

axes[1].plot(original_result['time'], original_result['train_loss'], linewidth=4, label='Original')
axes[1].plot(comp_result['time'], comp_result['train_loss'], linewidth=4, label='CompDMF')
axes[1].plot(narrow_r_result['time'], narrow_r_result['train_loss'], linewidth=4, label='Width=$r$')
axes[1].plot(narrow_2r_result['time'], narrow_2r_result['train_loss'], linewidth=4, label='Width=$2r$')
axes[1].set_xlabel('Time (s)', fontsize=14)
axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# axes[1].locator_params(axis='y', nbins=6)
axes[1].set_ylabel('Train Loss', fontsize=14)
axes[1].legend(fontsize=14)

plt.savefig('figs/comp_v_orig.png', dpi=300, bbox_inches='tight')