import os

import jax
import numpy as np
import matplotlib.pyplot as plt


def load_matching_outputs(data_dir, prefix):
    matching_paths = sorted(data_dir.glob(f"{prefix}*.npz"))
    if not matching_paths:
        raise FileNotFoundError(
            f"No .npz files found in {data_dir} matching prefix {prefix!r}"
        )

    outputs = {}
    concat_buffers = {}

    for path in matching_paths:
        with np.load(path, allow_pickle=True) as file_outputs:
            for key in file_outputs.files:
                value = file_outputs[key]
                if value.ndim == 0:
                    outputs.setdefault(key, value)
                else:
                    concat_buffers.setdefault(key, []).append(value)

    for key, values in concat_buffers.items():
        outputs[key] = concatenate_with_common_axis1(values, key)

    return outputs


def get_scalar(outputs, key, default=None):
    if key not in outputs:
        return default
    value = outputs[key]
    if getattr(value, "shape", None) == ():
        return value.item()
    return value


def weighted_mean(values, weights):
    weights = np.asarray(weights)
    if np.all(weights == 0):
        return 0.0
    return float(np.average(values, weights=weights))


def concatenate_with_common_axis1(values, key):
    try:
        return np.concatenate(values, axis=0)
    except ValueError as exc:
        concat_error = exc

    if not values or any(value.ndim < 2 for value in values):
        raise concat_error

    min_axis1 = min(value.shape[1] for value in values)
    trimmed_values = [value[:, :min_axis1, ...] for value in values]
    reference_shape = trimmed_values[0].shape[1:]
    if any(value.shape[1:] != reference_shape for value in trimmed_values):
        raise ValueError(
            f"Cannot concatenate {key!r}: shapes still differ after truncating "
            f"axis 1 to {min_axis1}: {[value.shape for value in values]}"
        )
    return np.concatenate(trimmed_values, axis=0)


def save_fig(fig, name, subfolder=None, fmts=["png"], tight=True):
    folder = "figures"
    if subfolder is not None:
        folder = f"{folder}/{subfolder}"

    # create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    if tight:
        fig.tight_layout()

    for fmt in fmts:
        fig.savefig(f"{folder}/{name}.{fmt}", bbox_inches="tight", dpi=300)

    plt.close(fig)


def jax_key_to_np_rng(key):
    return np.random.default_rng(np.asarray(jax.random.key_data(key)))
