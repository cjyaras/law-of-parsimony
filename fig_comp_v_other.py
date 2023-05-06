import matplotlib.pyplot as plt
from jax.random import PRNGKey, split

from data import generate_data, generate_observation_matrix
from loss import create_mc_loss
from network import init_net, compute_end_to_end, compute_factor, compress_network
from solver import train, compute_dlr
from utils import compose

key = PRNGKey(0)

# Generate data
r = 100
d = 5000
depth = 3
init_type = "orth"
init_scale = 1e-2
tol = 1e-12

key, subkey = split(key)
target = generate_data(key=subkey, shape=(d, d), rank=r)

key, subkey = split(key)
init_weights = init_net(key=subkey, input_dim=d, output_dim=d, width=d, depth=depth, init_type="orth", init_scale=init_scale)

key, subkey = split(key)
percent_observed = 0.20
mask = generate_observation_matrix(key=subkey, percent_observed=percent_observed, shape=(d, d))

train_mc_loss_fn = create_mc_loss(target, mask)
test_mc_loss_fn = create_mc_loss(target, 1 - mask)
train_e2e_loss_fn = compose(train_mc_loss_fn, compute_end_to_end)
test_e2e_loss_fn = compose(test_mc_loss_fn, compute_end_to_end)

# CompDMF
n_epochs = 500000
V = compute_factor(init_weights=init_weights, e2e_loss_fn=train_e2e_loss_fn, grad_rank=r)
comp_init_weights = compress_network(init_weights, V, r)
comp_step_size = compute_dlr(step_size=1e5, depth=len(comp_init_weights), prop=0.01)
print(comp_step_size)
comp_result = train(
    init_weights=comp_init_weights,
    train_e2e_loss_fn=train_e2e_loss_fn,
    n_epochs=n_epochs,
    step_size=comp_step_size,
    test_e2e_loss_fn=test_e2e_loss_fn,
    tol=tol
)