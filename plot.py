import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.ticker import MaxNLocator

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

def plot_sv_series(ax, series, color='viridis', spec_step=2):
    n_time_indices, n_sval_indices = series.shape
    time_indices = jnp.arange(n_time_indices)
    sval_indices = jnp.arange(n_sval_indices)

    spectrum_verts = []

    for idx in time_indices[::spec_step]:
        spectrum_verts.append([
            (0, jnp.min(series)-0.05), *zip(sval_indices, series[idx, :]), (n_sval_indices, jnp.min(series)-0.05)
        ])

    path_verts = []

    for idx in sval_indices:
        path_verts.append([
            *zip(time_indices, series[:, idx])
        ])

    spectrum_poly = PolyCollection(spectrum_verts)
    spectrum_poly.set_alpha(0.8)
    spectrum_poly.set_facecolor(plt.colormaps[color](jnp.linspace(0, 0.7, len(spectrum_verts))))
    spectrum_poly.set_edgecolor('black')

    path_line = LineCollection(path_verts)
    path_line.set_linewidth(1)
    path_line.set_edgecolor('black')
    
    ax.set_box_aspect(aspect=None, zoom=0.85)

    ax.add_collection3d(spectrum_poly, zs=time_indices[::spec_step], zdir='y')
    ax.add_collection3d(path_line, zs=sval_indices, zdir='x')

    ax.set_xlim(0, n_sval_indices)
    ax.set_ylim(0, n_time_indices)
    ax.set_zlim(jnp.min(series)-0.1, jnp.max(series)+0.1)

    elev = 30
    azim = -50
    roll = 0
    ax.view_init(elev, azim, roll)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))