from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["n_die"])
def step(key, hps, n_die, mut_rate):
    key_repr, key_mut = jax.random.split(key)

    # find the indices of the n_die lowest hps
    death_indices = jnp.argsort(hps)[:n_die]

    # sample n_die agents to reproduce, proportional to their hp
    p = hps.at[death_indices].set(0)
    p /= p.sum()
    repr_indices = jax.random.choice(
        key_repr, len(hps), shape=(n_die,), p=p, replace=True
    )

    # create new hps for the offspring by mutating the parents' hps
    new_hps = hps[repr_indices] + jax.random.normal(key_mut, shape=(n_die,)) * mut_rate
    return hps.at[death_indices].set(new_hps)


key = jax.random.PRNGKey(0)
n_prey = 1000
n_die = 100
mut_rate = 0.01
hps = jax.random.uniform(key, shape=(n_prey,))

for t in range(100):
    key, subkey = jax.random.split(key)
    hps = step(subkey, hps, n_die, mut_rate)
    print(f"Generation {t}: mean hp = {hps.mean():.4f}, max hp = {hps.max():.4f}")
