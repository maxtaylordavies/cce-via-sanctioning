import os
import numpy as np


def weighted_mean(values, weights):
    weights = np.asarray(weights)
    if np.all(weights == 0):
        return 0.0
    return float(np.average(values, weights=weights))


def save_fig(fig, filename, subfolder=None, fmts=["png"], tight=True):
    folder = "figures"
    if subfolder is not None:
        folder = f"{folder}/{subfolder}"

    # create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    if tight:
        fig.tight_layout()

    for fmt in fmts:
        fig.savefig(f"{folder}/{filename}.{fmt}", bbox_inches="tight")
