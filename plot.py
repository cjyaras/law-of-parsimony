import jax.numpy as jnp
from scipy.interpolate import interp1d

def interpolate_points(arr):
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]
    
    n = jnp.arange(arr.shape[0])
    
    x_spline = interp1d(n, x, kind='cubic')
    y_spline = interp1d(n, y, kind='cubic')
    z_spline = interp1d(n, z, kind='cubic')
    
    n_ = jnp.linspace(n.min(), n.max(), 500)
    interp_arr = jnp.concatenate([
        x_spline(n_).reshape(-1, 1),
        y_spline(n_).reshape(-1, 1),
        z_spline(n_).reshape(-1, 1)
    ], axis=1)
    
    return interp_arr