import jax.numpy as jnp

def compute_feature_collapse(features, labels):
    num_classes = jnp.max(labels) + 1
    global_mean = jnp.mean(features, axis=1, keepdims=True)
    class_means = jnp.concatenate(
        [jnp.mean(features[:, labels==k], axis=1, keepdims=True) for k in range(num_classes)], 
        axis=1
    )
    features_class_centered = features - class_means[:, labels]
    class_means_global_centered = class_means - global_mean
    sigma_W = (features_class_centered @ features_class_centered.T) / len(labels)
    sigma_B = (class_means_global_centered @ class_means_global_centered.T) / num_classes
    
    return jnp.trace(sigma_W) / jnp.trace(sigma_B)

def best_fit_line(feature_collapse_per_layer):
    layer_indices = jnp.arange(len(feature_collapse_per_layer), dtype=float)
    p = jnp.polyfit(layer_indices, jnp.log(feature_collapse_per_layer), deg=1)
    return jnp.exp(p[0] * layer_indices + p[1])