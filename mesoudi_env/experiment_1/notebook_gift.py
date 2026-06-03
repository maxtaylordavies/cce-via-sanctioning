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
def choose_demonstrator(
    key,
    can_imitate,
    all_prestiges,
    agent_idx,
    prestige_bias,
    demonstrator_prestige_baseline,
):
    candidate_mask = can_imitate[agent_idx]
    prestige_scores = (
        demonstrator_prestige_baseline + jnp.maximum(all_prestiges, 0.0)
    ) ** prestige_bias
    weights = jnp.where(candidate_mask, prestige_scores, 0.0)

    fallback_mask = jnp.arange(all_prestiges.shape[0]) != agent_idx
    fallback_weights = jnp.where(fallback_mask, 1.0, 0.0)
    weights = jnp.where(weights.sum() > 0, weights, fallback_weights)
    logits = jnp.where(weights > 0, jnp.log(weights), -jnp.inf)
    return jax.random.categorical(key, logits)


@jax.jit
def compute_gift(
    demonstrator_prestige,
    gift_rate,
    gift_base,
    gift_exponent,
    gift_cap,
):
    raw_gift = gift_base + gift_rate * (
        jnp.maximum(demonstrator_prestige, 0.0) ** gift_exponent
    )
    return jnp.minimum(raw_gift, gift_cap)


@jax.jit
def imitate(
    key,
    all_traits,
    all_prestiges,
    can_imitate,
    agent_idx,
    p_c,
    prestige_bias,
    demonstrator_prestige_baseline,
    gift_rate,
    gift_base,
    gift_exponent,
    gift_cap,
):
    demonstrator_key, copy_key = jax.random.split(key)
    demonstrator_idx = choose_demonstrator(
        demonstrator_key,
        can_imitate,
        all_prestiges,
        agent_idx,
        prestige_bias,
        demonstrator_prestige_baseline,
    )
    copied_traits = all_traits[demonstrator_idx] & jax.random.bernoulli(
        copy_key, p_c, shape=all_traits.shape[1:]
    )
    new = all_traits[agent_idx] | copied_traits
    n_changed = (new & ~all_traits[agent_idx]).sum()
    gift = compute_gift(
        all_prestiges[demonstrator_idx],
        gift_rate,
        gift_base,
        gift_exponent,
        gift_cap,
    )
    return new, n_changed, demonstrator_idx, gift


@jax.jit
def update_traits_and_gifts(
    key,
    all_traits,
    all_prestiges,
    all_roles,
    can_imitate,
    curr_L,
    p_i,
    p_c,
    prestige_bias,
    demonstrator_prestige_baseline,
    gift_rate,
    gift_base,
    gift_exponent,
    gift_cap,
):
    def per_agent(key_, agent_idx):
        def do_innovate(_):
            new_traits, success = innovate(key_, all_traits[agent_idx], curr_L, p_i)
            return (
                new_traits,
                success,
                0,
                agent_idx,
                jnp.asarray(0.0, dtype=jnp.float32),
            )

        def do_imitate(_):
            new_traits, n_copied, demonstrator_idx, gift = imitate(
                key_,
                all_traits,
                all_prestiges,
                can_imitate,
                agent_idx,
                p_c,
                prestige_bias,
                demonstrator_prestige_baseline,
                gift_rate,
                gift_base,
                gift_exponent,
                gift_cap,
            )
            return new_traits, 0, n_copied, demonstrator_idx, gift

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
    innov_cost=0.2,
    p_d=0.001,
    init_q=1.0,
    choice_beta=0.1,
    learning_rate=0.1,
    imit_dist_threshold=1,
    prestige_decay=0.01,
    prestige_value=0.0,
    prestige_bias=1.0,
    demonstrator_prestige_baseline=1.0,
    gift_rate=0.01,
    gift_base=0.0,
    gift_exponent=1.0,
    gift_cap=jnp.inf,
    eligibility_trace_decay=0.8,
    eligibility_discount=1.0,
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
        key, curr_L, all_traits, all_q_vals, all_prestiges, all_eligibilities = carry

        # get new keys
        key, death_key, role_key, update_key = jax.random.split(key, 4)

        # random deaths
        deaths = jax.random.bernoulli(death_key, p_d, shape=(N,))
        all_traits = jnp.where(deaths[:, None], 0, all_traits)
        all_q_vals = jnp.where(deaths[:, None], init_q, all_q_vals)
        all_prestiges = jnp.where(deaths, 0, all_prestiges)
        all_eligibilities = jnp.where(deaths[:, None], 0.0, all_eligibilities)
        all_prestiges = all_prestiges * (1.0 - prestige_decay)

        # agents select roles
        role_probs = jax.nn.softmax(all_q_vals / choice_beta, axis=1)
        all_roles = jax.random.categorical(role_key, jnp.log(role_probs), axis=1)

        # compute current material payoffs
        curr_trait_payoffs = compute_trait_payoffs(all_traits, b)

        # update traits, prestige, and deference transfers
        (
            new_all_traits,
            n_innovated,
            n_imitated,
            demonstrator_idxs,
            gifts_paid,
        ) = update_traits_and_gifts(
            update_key,
            all_traits,
            all_prestiges,
            all_roles,
            neighbours_mask,
            curr_L,
            p_i,
            p_c,
            prestige_bias,
            demonstrator_prestige_baseline,
            gift_rate,
            gift_base,
            gift_exponent,
            gift_cap,
        )
        prestige_changes = prestige_gain * n_innovated.astype(jnp.float32)
        new_all_prestiges = all_prestiges + prestige_changes
        new_trait_payoffs = compute_trait_payoffs(new_all_traits, b)

        incoming_gifts = (
            jnp.zeros((N,), dtype=jnp.float32).at[demonstrator_idxs].add(gifts_paid)
        )
        transfer_rewards = incoming_gifts - gifts_paid

        # compute role rewards and costs
        rewards = new_trait_payoffs - curr_trait_payoffs
        rewards += transfer_rewards
        rewards -= jnp.where(all_roles == ROLE_INNOVATE, innov_cost, 0)
        rewards += prestige_value * prestige_changes

        # update q vals using an eligibility trace over recent role choices.
        rpe = rewards - all_q_vals[jnp.arange(N), all_roles]
        decayed_eligibilities = (
            eligibility_discount * eligibility_trace_decay * all_eligibilities
        )
        role_eligibilities = jax.nn.one_hot(all_roles, 2, dtype=all_eligibilities.dtype)
        new_all_eligibilities = decayed_eligibilities + role_eligibilities
        new_all_q_vals = all_q_vals + learning_rate * (
            rpe[:, None] * new_all_eligibilities
        )

        # compute some metrics for logging
        total_unique_traits_known = (new_all_traits.sum(axis=0) > 0).sum()
        most_traits_known = new_all_traits.sum(axis=1).max()
        mean_traits_known = new_all_traits.sum(axis=1).mean()
        total_gifts_paid = gifts_paid.sum()
        max_gift_income = incoming_gifts.max()

        # determine whether to unlock new space of traits
        mean_prop_known = mean_traits_known / curr_L
        unlock = (mean_prop_known >= 0.9) & (curr_L < MAX_TOTAL_L)
        new_L = jnp.where(unlock, curr_L + L, curr_L)

        return (
            key,
            new_L,
            new_all_traits,
            new_all_q_vals,
            new_all_prestiges,
            new_all_eligibilities,
        ), (
            mean_traits_known,
            most_traits_known,
            total_unique_traits_known,
            role_probs.mean(axis=0),
            n_innovated.sum(),
            n_imitated.sum(),
            new_all_prestiges.mean(),
            new_all_prestiges.max(),
            total_gifts_paid,
            max_gift_income,
        )

    carry = (
        key,  # key
        L,  # curr_L
        jnp.zeros((N, MAX_TOTAL_L), dtype=jnp.int8),  # all_traits
        jnp.full((N, 2), init_q, dtype=jnp.float32),  # all_q_vals
        jnp.zeros((N,), dtype=jnp.float32),  # all_prestiges
        jnp.zeros((N, 2), dtype=jnp.float32),  # all_eligibilities
    )

    _, metrics = jax.lax.scan(body_fn, carry, jnp.arange(T))
    metrics = list(metrics)
    metrics[4] = jnp.cumsum(metrics[4])  # cumulative number innovated
    metrics[5] = jnp.cumsum(metrics[5])  # cumulative number imitated
    metrics[8] = jnp.cumsum(metrics[8])  # cumulative gift transfers
    return metrics


def main():
    seeds = list(range(5))
    grid_length, T = 10, int(2e3)
    prestige_gain_vals = jnp.linspace(0.0, 10.0, 21)
    prestige_decay = 0.01
    prestige_value = 0.0
    prestige_bias = 1.0
    demonstrator_prestige_baseline = 1.0
    gift_rate = 0.01
    gift_base = 0.0
    gift_exponent = 1.0
    gift_cap = np.float32(np.inf)
    eligibility_trace_decay = 0.8
    eligibility_discount = 1.0

    def run_with_prestige_gain(key, prestige_gain):
        return jax.block_until_ready(
            run_simulation_loop(
                key,
                grid_length,
                T,
                prestige_gain=prestige_gain,
                prestige_decay=prestige_decay,
                prestige_value=prestige_value,
                prestige_bias=prestige_bias,
                demonstrator_prestige_baseline=demonstrator_prestige_baseline,
                gift_rate=gift_rate,
                gift_base=gift_base,
                gift_exponent=gift_exponent,
                gift_cap=gift_cap,
                eligibility_trace_decay=eligibility_trace_decay,
                eligibility_discount=eligibility_discount,
            )
        )

    all_mean_traits = []
    all_most_traits = []
    all_total_unique_traits = []
    all_role_probs = []
    all_n_innovated = []
    all_n_imitated = []
    all_mean_prestige = []
    all_max_prestige = []
    all_total_gifts = []
    all_max_gift_income = []

    for seed in tqdm(seeds):
        key = jax.random.PRNGKey(seed)
        (
            mean_traits,
            most_traits,
            total_unique_traits,
            role_probs,
            n_innov,
            n_imit,
            mean_prestige,
            max_prestige,
            total_gifts,
            max_gift_income,
        ) = jax.vmap(run_with_prestige_gain, in_axes=(None, 0))(key, prestige_gain_vals)

        all_mean_traits.append(np.asarray(mean_traits))
        all_most_traits.append(np.asarray(most_traits))
        all_total_unique_traits.append(np.asarray(total_unique_traits))
        all_role_probs.append(np.asarray(role_probs))
        all_n_innovated.append(np.asarray(n_innov))
        all_n_imitated.append(np.asarray(n_imit))
        all_mean_prestige.append(np.asarray(mean_prestige))
        all_max_prestige.append(np.asarray(max_prestige))
        all_total_gifts.append(np.asarray(total_gifts))
        all_max_gift_income.append(np.asarray(max_gift_income))

    simulation_outputs = {
        "prestige_gains": np.asarray(prestige_gain_vals),
        "seeds": np.asarray(seeds),
        "T": np.int32(T),
        "grid_length": np.int32(grid_length),
        "max_total_l": np.int32(MAX_TOTAL_L),
        "prestige_decay": np.float32(prestige_decay),
        "prestige_value": np.float32(prestige_value),
        "prestige_bias": np.float32(prestige_bias),
        "demonstrator_prestige_baseline": np.float32(demonstrator_prestige_baseline),
        "gift_rate": np.float32(gift_rate),
        "gift_base": np.float32(gift_base),
        "gift_exponent": np.float32(gift_exponent),
        "gift_cap": np.float32(gift_cap),
        "eligibility_trace_decay": np.float32(eligibility_trace_decay),
        "eligibility_discount": np.float32(eligibility_discount),
        "role_innovate": np.int32(ROLE_INNOVATE),
        "role_imitate": np.int32(ROLE_IMITATE),
        "mean_traits": np.stack(all_mean_traits, axis=0),
        "most_traits": np.stack(all_most_traits, axis=0),
        "total_unique_traits": np.stack(all_total_unique_traits, axis=0),
        "role_probs": np.stack(all_role_probs, axis=0),
        "n_innovated": np.stack(all_n_innovated, axis=0),
        "n_imitated": np.stack(all_n_imitated, axis=0),
        "mean_prestige": np.stack(all_mean_prestige, axis=0),
        "max_prestige": np.stack(all_max_prestige, axis=0),
        "total_gifts": np.stack(all_total_gifts, axis=0),
        "max_gift_income": np.stack(all_max_gift_income, axis=0),
    }

    output_path = f"simulation_outputs_gift_{seeds[0]}-{seeds[-1]}.npz"
    np.savez(output_path, **simulation_outputs)


if __name__ == "__main__":
    main()
