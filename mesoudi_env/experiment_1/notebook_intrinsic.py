from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

ROLE_INNOVATE, ROLE_IMITATE = 0, 1
L = 100
MAX_TOTAL_L = 5000


@jax.jit
def compute_trait_payoffs(all_traits, b):
    return all_traits.sum(axis=1) * b


@jax.jit
def innovate(key, traits, curr_L, p_i):
    idx_key, flip_key = jax.random.split(key)
    idx = jax.random.randint(idx_key, (), 0, curr_L)
    success = (traits[idx] == 0) & jax.random.bernoulli(flip_key, p_i)
    new_val = jnp.where(success, 1, traits[idx])
    return traits.at[idx].set(new_val), success.astype(int)


@jax.jit
def choose_demonstrator(key, can_imitate, agent_idx):
    weights = jnp.where(can_imitate[agent_idx], 1.0, 0.0)

    fallback_mask = jnp.arange(can_imitate.shape[0]) != agent_idx
    fallback_weights = jnp.where(fallback_mask, 1.0, 0.0)
    weights = jnp.where(weights.sum() > 0, weights, fallback_weights)
    logits = jnp.where(weights > 0, jnp.log(weights), -jnp.inf)
    return jax.random.categorical(key, logits)


@jax.jit
def imitate(key, all_traits, can_imitate, agent_idx, p_c):
    demonstrator_key, copy_key = jax.random.split(key)
    demonstrator_idx = choose_demonstrator(demonstrator_key, can_imitate, agent_idx)
    copied_traits = all_traits[demonstrator_idx] & jax.random.bernoulli(
        copy_key, p_c, shape=all_traits.shape[1:]
    )
    new = all_traits[agent_idx] | copied_traits
    n_changed = (new & ~all_traits[agent_idx]).sum()
    return new, n_changed


@jax.jit
def update_traits(
    key,
    all_traits,
    all_roles,
    can_imitate,
    curr_L,
    p_i,
    p_c,
):
    def per_agent(key_, agent_idx):
        def do_innovate(_):
            new_traits, success = innovate(key_, all_traits[agent_idx], curr_L, p_i)
            return new_traits, success, 0  # traits, # innovated, # imitated

        def do_imitate(_):
            new_traits, n_copied = imitate(
                key_, all_traits, can_imitate, agent_idx, p_c
            )
            return new_traits, 0, n_copied  # traits, # innovated, # imitated

        return jax.lax.cond(
            all_roles[agent_idx] == ROLE_INNOVATE, do_innovate, do_imitate, operand=None
        )

    keys = jax.random.split(key, all_traits.shape[0])
    return jax.vmap(per_agent)(keys, jnp.arange(all_traits.shape[0]))


@partial(jax.jit, static_argnames=("grid_length", "T"))
def run_simulation_loop(
    key,
    grid_length,
    T,
    prestige_gain,
    p_i=1.0,
    p_c=1.0,
    b=0.2,
    innov_cost=0.1,
    p_d=0.001,
    init_q=1.0,
    choice_beta=0.1,
    learning_rate=0.1,
    imit_dist_threshold=100,
    prestige_decay=0.01,
    prestige_value=1.0,
):
    # Compute pairwise toroidal distances between agents for imitation.
    N = grid_length**2
    agent_idxs = jnp.arange(N)
    agent_locs = jnp.stack(
        [
            agent_idxs // grid_length,  # row index
            agent_idxs % grid_length,  # column index
        ]
    ).T
    row_diffs = jnp.abs(agent_locs[:, None, 0] - agent_locs[None, :, 0])
    col_diffs = jnp.abs(agent_locs[:, None, 1] - agent_locs[None, :, 1])
    torus_row_dists = jnp.minimum(row_diffs, grid_length - row_diffs)
    torus_col_dists = jnp.minimum(col_diffs, grid_length - col_diffs)
    agent_dists = torus_row_dists + torus_col_dists
    neighbours_mask = ((agent_dists > 0) & (agent_dists <= imit_dist_threshold)).astype(
        jnp.bool_
    )

    def body_fn(carry, t):
        key, curr_L, all_traits, all_q_vals, all_prestiges = carry

        # get new keys
        key, death_key, role_key, update_key = jax.random.split(key, 4)

        # random deaths
        deaths = jax.random.bernoulli(death_key, p_d, shape=(N,))
        all_traits = jnp.where(deaths[:, None], 0, all_traits)
        all_q_vals = jnp.where(deaths[:, None], init_q, all_q_vals)
        all_prestiges = jnp.where(deaths, 0, all_prestiges)
        all_prestiges = all_prestiges * (1.0 - prestige_decay)

        # agents select roles
        role_probs = jax.nn.softmax(all_q_vals / choice_beta, axis=1)
        all_roles = jax.random.categorical(role_key, jnp.log(role_probs), axis=1)

        # compute current material payoffs
        curr_trait_payoffs = compute_trait_payoffs(all_traits, b)

        # update traits + prestige and compute new material payoffs
        new_all_traits, n_innovated, n_imitated = update_traits(
            update_key, all_traits, all_roles, neighbours_mask, curr_L, p_i, p_c
        )
        prestige_changes = prestige_gain * n_innovated.astype(jnp.float32)
        new_all_prestiges = all_prestiges + prestige_changes
        new_trait_payoffs = compute_trait_payoffs(new_all_traits, b)

        # compute role rewards and costs
        rewards = new_trait_payoffs - curr_trait_payoffs
        rewards -= jnp.where(all_roles == ROLE_INNOVATE, innov_cost, 0)
        rewards += prestige_value * prestige_changes

        # update q vals
        rpe = rewards - all_q_vals[jnp.arange(N), all_roles]
        new_all_q_vals = all_q_vals.at[jnp.arange(N), all_roles].add(
            learning_rate * rpe
        )

        # compute some metrics for logging
        total_unique_traits_known = (new_all_traits.sum(axis=0) > 0).sum()
        most_traits_known = new_all_traits.sum(axis=1).max()
        mean_traits_known = new_all_traits.sum(axis=1).mean()

        # determine whether to unlock new space of traits
        # mask = jnp.arange(MAX_TOTAL_L)
        # mask = (mask >= curr_L - L) & (mask < curr_L)
        # mean_prop_known = (all_traits * mask.reshape(1, -1)).sum(axis=1).mean() / L

        mean_prop_known = mean_traits_known / curr_L
        unlock = (mean_prop_known >= 0.9) & (curr_L < MAX_TOTAL_L)
        new_L = jnp.where(unlock, curr_L + L, curr_L)

        return (key, new_L, new_all_traits, new_all_q_vals, new_all_prestiges), (
            mean_traits_known,
            most_traits_known,
            total_unique_traits_known,
            role_probs.mean(axis=0),
            all_roles,
            n_innovated.sum(),
            n_imitated.sum(),
            new_all_prestiges.mean(),
            new_all_prestiges.max(),
        )

    carry = (
        key,  # key
        L,  # curr_L
        jnp.zeros((N, MAX_TOTAL_L), dtype=jnp.int8),  # all_traits
        jnp.full((N, 2), init_q, dtype=jnp.float32),  # all_q_vals
        jnp.zeros((N,), dtype=jnp.float32),  # all_prestiges
    )

    _, metrics = jax.lax.scan(body_fn, carry, jnp.arange(T))
    metrics = list(metrics)
    metrics[5] = jnp.cumsum(metrics[5])  # cumulative number innovated
    metrics[6] = jnp.cumsum(metrics[6])  # cumulative number imitated
    return metrics


def main():
    seeds = list(range(10))
    grid_length, T = 10, int(4e3)
    prestige_gain_vals = 10 * jnp.linspace(0.0, 1.0, 30)
    prestige_decay = 0.01
    prestige_value = 1.0

    def run_with_prestige_gain(key, prestige_gain):
        return jax.block_until_ready(
            run_simulation_loop(
                key,
                grid_length,
                T,
                prestige_gain=prestige_gain,
                prestige_decay=prestige_decay,
                prestige_value=prestige_value,
            )
        )

    all_mean_traits = []
    all_most_traits = []
    all_total_unique_traits = []
    all_role_probs = []
    all_agent_roles = []
    all_n_innovated = []
    all_n_imitated = []
    all_mean_prestige = []
    all_max_prestige = []

    for seed in tqdm(seeds):
        key = jax.random.PRNGKey(seed)
        (
            mean_traits,
            most_traits,
            total_unique_traits,
            role_probs,
            agent_roles,
            n_innov,
            n_imit,
            mean_prestige,
            max_prestige,
        ) = jax.vmap(run_with_prestige_gain, in_axes=(None, 0))(key, prestige_gain_vals)

        all_mean_traits.append(np.asarray(mean_traits))
        all_most_traits.append(np.asarray(most_traits))
        all_total_unique_traits.append(np.asarray(total_unique_traits))
        all_role_probs.append(np.asarray(role_probs))
        all_agent_roles.append(np.asarray(agent_roles))
        all_n_innovated.append(np.asarray(n_innov))
        all_n_imitated.append(np.asarray(n_imit))
        all_mean_prestige.append(np.asarray(mean_prestige))
        all_max_prestige.append(np.asarray(max_prestige))

    simulation_outputs = {
        "prestige_gains": np.asarray(prestige_gain_vals),
        "seeds": np.asarray(seeds),
        "T": np.int32(T),
        "grid_length": np.int32(grid_length),
        "max_total_l": np.int32(MAX_TOTAL_L),
        "prestige_decay": np.float32(prestige_decay),
        "prestige_value": np.float32(prestige_value),
        "role_innovate": np.int32(ROLE_INNOVATE),
        "role_imitate": np.int32(ROLE_IMITATE),
        "mean_traits": np.stack(all_mean_traits, axis=0),
        "most_traits": np.stack(all_most_traits, axis=0),
        "total_unique_traits": np.stack(all_total_unique_traits, axis=0),
        "role_probs": np.stack(all_role_probs, axis=0),
        "agent_roles": np.stack(all_agent_roles, axis=0),
        "n_innovated": np.stack(all_n_innovated, axis=0),
        "n_imitated": np.stack(all_n_imitated, axis=0),
        "mean_prestige": np.stack(all_mean_prestige, axis=0),
        "max_prestige": np.stack(all_max_prestige, axis=0),
    }

    output_path = f"simulation_outputs_{seeds[0]}-{seeds[-1]}.npz"
    np.savez(output_path, **simulation_outputs)


if __name__ == "__main__":
    main()
