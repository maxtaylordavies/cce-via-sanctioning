import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils import save_fig
from utils import get_eta_star_fn, compute_rate_optimal_eta

sns.set_theme(style="whitegrid")

N = 100
phi = 0.1
turnover_rate = 0.01
preservation_rate = 1 / N
v_innov_fn = b_imit_fn = lambda D: D
c_innov_fn = lambda D: 0.0
c_imit_fn = lambda D: 0.0
beta = 1.0
learning_rate = 0.5
pi_0 = 0.5

p_success_fns = {
    "$p_{success}(D) = 0.1$": lambda D: 0.1,
    "$p_{success}(D) = \\exp(-D / 5)$": lambda D: jnp.exp(-D / 5),
    "$p_{success}(D) = 1/(D+1)$": lambda D: 1 / (D + 1),
}

fig, axs = plt.subplots(
    1, len(p_success_fns), figsize=(5 * len(p_success_fns), 5), sharey=True
)

for i, (p_success_label, p_success_fn) in enumerate(p_success_fns.items()):
    eta_star_fn = get_eta_star_fn(
        v_innov_fn=v_innov_fn,
        b_imit_fn=b_imit_fn,
        p_success_fn=p_success_fn,
        c_innov_fn=c_innov_fn,
        c_imit_fn=c_imit_fn,
        beta=beta,
    )

    D_values = jnp.arange(0, 60, 1)
    eta_star_values, _, ps = jax.vmap(eta_star_fn)(D_values)
    print(ps)
    eta_target_values = jax.vmap(compute_rate_optimal_eta, in_axes=(0, None))(ps, phi)

    sns.lineplot(x=D_values, y=eta_star_values, ax=axs[i], label="$\\eta^*$")
    sns.lineplot(x=D_values, y=eta_target_values, ax=axs[i], label="$\\eta_{target}$")
    axs[i].set_title(p_success_label)

plt.show()
