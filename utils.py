import jax.numpy as jnp
from tqdm import tqdm
from scipy.linalg import svdvals

def svd(A, full_matrices=True):
    U, s, VT = jnp.linalg.svd(A, full_matrices=full_matrices)
    return U, s, VT.T

def compose(f, g):
    def h(x):
        return f(g(x))
    return h

def compute_angle(U, V):
    if len(U.shape) < 2:
        U = U.reshape(-1, 1)
    if len(V.shape) < 2:
        V = V.reshape(-1, 1)
    return jnp.arccos(jnp.clip(jnp.linalg.norm(jnp.dot(U.T, V), ord=2), 0, 1))

def compute_svd_series(weights, layer, rank):
    input_dim = weights[0][0].shape[1]
    m = input_dim - 2 * rank
    
    Uinit, _, Vinit = svd(weights[1][layer])

    right_series = [[ *(input_dim * [0]) ]]
    left_series = [[ *(input_dim * [0]) ]]
    sval_series = [svdvals(weights[0][0])]

    for w in tqdm(weights[1:]):
        Ut, st, Vt = svd(w[layer])
        sval_series.append(st)

        Ut_top = Ut[:, :rank]
        Ut_mid = Ut[:, rank:-rank]
        Ut_bot = Ut[:, -rank:]

        Uinit_top = Uinit[:, :rank]
        Uinit_mid = Uinit[:, rank:-rank]
        Uinit_bot = Uinit[:, -rank:]

        Vt_top = Vt[:, :rank]
        Vt_mid = Vt[:, rank:-rank]
        Vt_bot = Vt[:, -rank:]

        Vinit_top = Vinit[:, :rank]
        Vinit_mid = Vinit[:, rank:-rank]
        Vinit_bot = Vinit[:, -rank:]

        right = []

        for k in range(rank):
            right.append(compute_angle(Vt_top[:, k], Vinit_top[:, k]))

        # right += rank * [compute_angle(Vt_top, Vinit_top)]

        # for k in range(m):
        #     right.append(compute_angle(Vt_mid[:, k], Vinit_mid[:, k]))

        right += m * [compute_angle(Vt_mid, Vinit_mid)]

        for k in range(rank):
            right.append(compute_angle(Vt_bot[:, k], Vinit_bot[:, k]))

        # right += rank * [compute_angle(Vt_bot, Vinit_bot)]

        right_series.append(right)

        left = []

        for k in range(rank):
            left.append(compute_angle(Ut_top[:, k], Uinit_top[:, k]))

        # left += rank * [compute_angle(Ut_top, Uinit_top)]

        # for k in range(m):
        #     left.append(compute_angle(Ut_mid[:, k], Uinit_mid[:, k]))

        left += m * [compute_angle(Ut_mid, Uinit_mid)]

        for k in range(rank):
            left.append(compute_angle(Ut_bot[:, k], Uinit_bot[:, k]))

        # left += rank * [compute_angle(Ut_bot, Uinit_bot)]

        left_series.append(left)

    sval_series = jnp.array(sval_series)
    right_series = jnp.array(right_series)
    left_series = jnp.array(left_series)
    
    return (sval_series, right_series, left_series)