from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

ROLE_INNOVATE, ROLE_IMITATE = 0, 1
L = 100
MAX_TOTAL_L = 10000


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
    candidate_mask,
    all_prestiges,
    agent_idx,
    prestige_bias,
    demonstrator_prestige_baseline,
):
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


@partial(jax.jit, static_argnames=("grid_length", "T"))
def run_simulation_loop(
    key,
    grid_length,
    T,
    run_cgs=True,
    disconnect_group_traits=False,
    run_cgs_every=200,
    cgs_mut_std=0.05,
    max_n_groups=9,
    tournament_size=3,
    p_i=1.0,
    p_c=1.0,
    b=0.2,
    innov_cost=0.2,
    p_d=0.001,
    init_q=1.0,
    choice_beta=0.1,
    learning_rate=0.1,
    imit_dist_threshold=100,
    prestige_decay=0.01,
    prestige_value=0.0,
    prestige_bias=1.0,
    demonstrator_prestige_baseline=1.0,
    gift_rate=0.01,
    gift_base=0.0,
    gift_exponent=1.0,
    gift_cap=jnp.inf,
):
    # In the top-down regime, every group-selection event replaces all 9 groups
    # with fresh descendants, so we budget for one initial cohort plus at most T
    # subsequent cohorts.
    max_group_instances = max_n_groups * (T + 1)
    EMPTY_GROUP_INSTANCE_ID = jnp.int32(-1)

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

    # initialise group tiling
    cells_per_side = grid_length // 3
    base_group_grid = (
        3 * (jnp.arange(grid_length)[:, None] // cells_per_side)
        + (jnp.arange(grid_length)[None, :] // cells_per_side)
    ).astype(jnp.int32)

    def _group_average_values(grid, flat_values):
        group_labels = grid.reshape(-1)
        totals = jnp.bincount(
            group_labels,
            weights=flat_values,
            length=max_n_groups,
        )
        counts = jnp.bincount(group_labels, length=max_n_groups)
        return totals / jnp.maximum(counts, 1)

    def _reseed_groups(
        key,
        t,
        group_norm_vals,
        group_scores,
        group_instance_ids_by_label,
        next_group_instance_id,
        group_parent_instance_ids,
        group_birth_timesteps,
    ):
        # Top-down group selection: run one tournament per offspring group,
        # choose a winning parent within each tournament, then mutate each
        # offspring independently from its selected parent's norm value.
        tournament_key, tie_key, mut_key = jax.random.split(key, 3)
        group_ids = jnp.arange(max_n_groups, dtype=jnp.int32)

        tournament_keys = jax.random.split(tournament_key, max_n_groups)

        def sample_tournament_groups(single_key):
            return jax.random.choice(
                single_key,
                group_ids,
                shape=(tournament_size,),
                replace=False,
            )

        tournament_group_ids = jax.vmap(sample_tournament_groups)(tournament_keys)
        tie_break = 1e-6 * jax.random.uniform(
            tie_key, shape=(max_n_groups, tournament_size)
        )
        winner_positions = jnp.argmax(
            group_scores[tournament_group_ids] + tie_break, axis=1
        )
        winner_groups = tournament_group_ids[jnp.arange(max_n_groups), winner_positions]

        parent_instance_ids = group_instance_ids_by_label[winner_groups]
        winner_norm_vals = group_norm_vals[winner_groups]
        next_group_norm_vals = winner_norm_vals + (
            jax.random.normal(mut_key, shape=(max_n_groups,)) * cgs_mut_std
        )
        # next_group_norm_vals = jnp.maximum(next_group_norm_vals, 0.0)
        new_group_instance_ids_by_label = next_group_instance_id + jnp.arange(
            max_n_groups, dtype=jnp.int32
        )
        next_group_parent_instance_ids = group_parent_instance_ids.at[
            new_group_instance_ids_by_label
        ].set(parent_instance_ids)
        next_group_birth_timesteps = group_birth_timesteps.at[
            new_group_instance_ids_by_label
        ].set(t)
        return (
            base_group_grid,
            next_group_norm_vals,
            new_group_instance_ids_by_label,
            next_group_instance_id + max_n_groups,
            next_group_parent_instance_ids,
            next_group_birth_timesteps,
        )

    def _apply_group_prestige_gain_change_to_q_vals(
        q_vals,
        old_grid,
        old_group_norm_vals,
        old_group_instance_ids_by_label,
        new_grid,
        new_group_norm_vals,
        new_group_instance_ids_by_label,
    ):
        old_group_labels = old_grid.reshape(-1)
        new_group_labels = new_grid.reshape(-1)
        old_group_instances = old_group_instance_ids_by_label[old_group_labels]
        new_group_instances = new_group_instance_ids_by_label[new_group_labels]
        group_changed = old_group_instances != new_group_instances
        delta_prestige_gain = (
            new_group_norm_vals[new_group_labels]
            - old_group_norm_vals[old_group_labels]
        )

        q_delta = jnp.where(
            group_changed,
            delta_prestige_gain,
            0.0,
        )
        q_delta = jnp.where(disconnect_group_traits, 0.0, q_delta)
        new_q_vals = q_vals.at[:, ROLE_INNOVATE].add(q_delta)
        return jnp.where(disconnect_group_traits, q_vals, new_q_vals)

    @jax.jit
    def can_imitate(agent_idx, group_labels_grid):
        # agent can imitate anyone else in their group
        row = agent_idx // grid_length
        col = agent_idx % grid_length
        group = group_labels_grid[row, col]
        return (group_labels_grid == group).reshape(-1) & neighbours_mask[agent_idx]

    @jax.jit
    def update_traits_and_gifts(
        key,
        all_traits,
        all_prestiges,
        all_roles,
        group_labels_grid,
        curr_Ls,
    ):
        group_labels_1d = group_labels_grid.reshape(-1)

        def per_agent(key_, agent_idx):
            group_idx = group_labels_1d[agent_idx]

            def do_innovate(_):
                new_traits, success = innovate(
                    key_, all_traits[agent_idx], curr_Ls[group_idx], p_i
                )
                return (
                    new_traits,
                    success,
                    0,
                    agent_idx,
                    jnp.asarray(0.0, dtype=jnp.float32),
                )

            def do_imitate(_):
                imit_mask = can_imitate(agent_idx, group_labels_grid)
                new_traits, n_copied, demonstrator_idx, gift = imitate(
                    key_,
                    all_traits,
                    all_prestiges,
                    imit_mask,
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
                all_roles[agent_idx] == ROLE_INNOVATE,
                do_innovate,
                do_imitate,
                operand=None,
            )

        keys = jax.random.split(key, all_traits.shape[0])
        return jax.vmap(per_agent)(keys, jnp.arange(all_traits.shape[0]))

    def body_fn(carry, t):
        (
            key,
            curr_Ls,
            all_traits,
            all_q_vals,
            all_prestiges,
            group_norm_values,
            group_labels_grid,
            group_instance_ids_by_label,
            next_group_instance_id,
            group_parent_instance_ids,
            group_birth_timesteps,
            traits_gained_since_cgs,
        ) = carry

        # get new keys
        key, death_key, role_key, update_key, cgs_key = jax.random.split(key, 5)

        # random deaths
        deaths = jax.random.bernoulli(death_key, p_d, shape=(N,))
        all_traits = jnp.where(deaths[:, None], 0, all_traits)
        all_q_vals = jnp.where(deaths[:, None], init_q, all_q_vals)
        all_prestiges = jnp.where(deaths, 0, all_prestiges)
        all_prestiges = all_prestiges * (1.0 - prestige_decay)
        traits_gained_since_cgs = jnp.where(deaths, 0.0, traits_gained_since_cgs)

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
            group_labels_grid,
            curr_Ls,
        )
        group_labels_1d = group_labels_grid.reshape(-1)
        prestige_gain_by_agent = group_norm_values[group_labels_1d]
        prestige_gain_by_agent = jnp.where(
            disconnect_group_traits, 0.0, prestige_gain_by_agent
        )
        prestige_changes = prestige_gain_by_agent * n_innovated.astype(jnp.float32)
        new_all_prestiges = all_prestiges + prestige_changes
        new_trait_payoffs = compute_trait_payoffs(new_all_traits, b)
        updated_traits_gained_since_cgs = traits_gained_since_cgs + (
            n_innovated + n_imitated
        ).astype(jnp.float32)

        incoming_gifts = (
            jnp.zeros((N,), dtype=jnp.float32).at[demonstrator_idxs].add(gifts_paid)
        )

        direct_rewards = new_trait_payoffs - curr_trait_payoffs
        direct_rewards -= gifts_paid
        direct_rewards -= jnp.where(all_roles == ROLE_INNOVATE, innov_cost, 0.0)
        direct_rewards += prestige_value * prestige_changes

        direct_rpe = direct_rewards - all_q_vals[jnp.arange(N), all_roles]
        new_all_q_vals = all_q_vals.at[jnp.arange(N), all_roles].add(
            learning_rate * direct_rpe
        )
        gift_income_rpe = jnp.where(
            incoming_gifts > 0.0,
            incoming_gifts - new_all_q_vals[:, ROLE_INNOVATE],
            0.0,
        )
        new_all_q_vals = new_all_q_vals.at[:, ROLE_INNOVATE].add(
            learning_rate * gift_income_rpe
        )

        # compute some metrics for logging
        per_agent_traits_known = new_all_traits.sum(axis=1)
        mean_traits_known = per_agent_traits_known.mean()
        total_gifts_paid = gifts_paid.sum()
        max_gift_income = incoming_gifts.max()

        # determine whether to unlock new space of traits
        group_num_traits_known = _group_average_values(
            group_labels_grid, per_agent_traits_known
        )
        group_new_traits_gained = _group_average_values(
            group_labels_grid, updated_traits_gained_since_cgs
        )
        group_trait_gain_scores = group_new_traits_gained / curr_Ls.astype(jnp.float32)
        group_prop_traits_known = group_num_traits_known / curr_Ls
        unlock = (group_prop_traits_known >= 0.9) & (curr_Ls < MAX_TOTAL_L)
        new_Ls = jnp.where(unlock, curr_Ls + L, curr_Ls)

        # Every `run_cgs_every` steps, do tournament selection and mutation at the group level
        should_run_group_selection = run_cgs & (t % run_cgs_every == 0) & (t > 0)
        old_group_labels_grid = group_labels_grid
        old_group_norm_values = group_norm_values
        old_group_instance_ids_by_label = group_instance_ids_by_label
        (
            group_labels_grid,
            group_norm_values,
            group_instance_ids_by_label,
            next_group_instance_id,
            group_parent_instance_ids,
            group_birth_timesteps,
        ) = jax.lax.cond(
            should_run_group_selection,
            lambda args: _reseed_groups(
                args[0], args[1], args[3], args[4], args[5], args[6], args[7], args[8]
            ),
            lambda args: (args[2], args[3], args[5], args[6], args[7], args[8]),
            (
                cgs_key,
                t,
                group_labels_grid,
                group_norm_values,
                group_trait_gain_scores,
                # group_num_traits_known,
                group_instance_ids_by_label,
                next_group_instance_id,
                group_parent_instance_ids,
                group_birth_timesteps,
            ),
        )
        new_all_q_vals = _apply_group_prestige_gain_change_to_q_vals(
            new_all_q_vals,
            old_group_labels_grid,
            old_group_norm_values,
            old_group_instance_ids_by_label,
            group_labels_grid,
            group_norm_values,
            group_instance_ids_by_label,
        )
        next_traits_gained_since_cgs = jnp.where(
            should_run_group_selection,
            jnp.zeros_like(updated_traits_gained_since_cgs),
            updated_traits_gained_since_cgs,
        )

        return (
            key,
            new_Ls,
            new_all_traits,
            new_all_q_vals,
            new_all_prestiges,
            group_norm_values,
            group_labels_grid,
            group_instance_ids_by_label,
            next_group_instance_id,
            group_parent_instance_ids,
            group_birth_timesteps,
            next_traits_gained_since_cgs,
        ), (
            mean_traits_known,
            group_norm_values,
            group_labels_grid,
            group_instance_ids_by_label,
            role_probs.mean(axis=0),
            all_roles,
            n_innovated.sum(),
            n_imitated.sum(),
            new_all_prestiges.mean(),
            new_all_prestiges.max(),
            total_gifts_paid,
            max_gift_income,
        )

    init_norm_key = jax.random.fold_in(key, 0)
    group_norm_values = (
        jax.random.normal(init_norm_key, shape=(max_n_groups,)) * cgs_mut_std
    )
    # group_norm_values = jnp.maximum(
    #     jax.random.normal(init_norm_key, shape=(max_n_groups,)) * cgs_mut_std,
    #     0.0,
    # )
    group_norm_values = jnp.where(
        run_cgs, group_norm_values, jnp.zeros_like(group_norm_values)
    )
    group_instance_ids_by_label = jnp.arange(max_n_groups, dtype=jnp.int32)
    next_group_instance_id = jnp.int32(max_n_groups)
    group_parent_instance_ids = jnp.full(
        max_group_instances, EMPTY_GROUP_INSTANCE_ID, dtype=jnp.int32
    )
    group_birth_timesteps = (
        jnp.full(max_group_instances, -1, dtype=jnp.int32).at[:max_n_groups].set(0)
    )

    carry = (
        key,  # key
        jnp.full(max_n_groups, L, dtype=jnp.int32),  # curr_Ls
        jnp.zeros((N, MAX_TOTAL_L), dtype=jnp.int8),  # all_traits
        jnp.full((N, 2), init_q, dtype=jnp.float32),  # all_q_vals
        jnp.zeros((N,), dtype=jnp.float32),  # all_prestiges
        group_norm_values,
        base_group_grid,
        group_instance_ids_by_label,
        next_group_instance_id,
        group_parent_instance_ids,
        group_birth_timesteps,
        jnp.zeros(N, dtype=jnp.float32),  # traits_gained_since_cgs
    )

    carry, metrics = jax.lax.scan(body_fn, carry, jnp.arange(T))

    metrics = list(metrics)
    metrics[6] = jnp.cumsum(metrics[6])  # cumulative number innovated
    metrics[7] = jnp.cumsum(metrics[7])  # cumulative number imitated
    metrics[10] = jnp.cumsum(metrics[10])  # cumulative gift transfers
    return (*metrics, carry[8], carry[9], carry[10])


def main():
    seeds = list(range(10))
    grid_length, T = 30, int(1e4)
    run_cgs_every = 100
    cgs_mut_std = 0.1
    prestige_decay = 0.01
    prestige_value = 0.0
    prestige_bias = 1.0
    demonstrator_prestige_baseline = 1.0
    gift_rate = 0.01
    gift_base = 0.0
    gift_exponent = 1.0
    gift_cap = np.float32(np.inf)

    all_mean_traits_known = []
    all_group_norm_values = []
    all_group_labels_grids = []
    all_group_instance_ids_by_label_history = []
    all_role_probs = []
    all_agent_roles = []
    all_n_innovated = []
    all_n_imitated = []
    all_mean_prestige = []
    all_max_prestige = []
    all_total_gifts = []
    all_max_gift_income = []
    all_group_lineage_arrays = []
    all_final_next_group_instance_ids = []
    for seed in tqdm(seeds):
        key = jax.random.PRNGKey(seed)

        (
            mean_traits_known,
            group_norm_values,
            group_labels_grid,
            group_instance_ids_by_label_history,
            role_probs,
            agent_roles,
            n_innovated,
            n_imitated,
            mean_prestige,
            max_prestige,
            total_gifts,
            max_gift_income,
            final_next_group_instance_id,
            final_group_parent_instance_ids,
            final_group_birth_timesteps,
        ) = jax.block_until_ready(
            run_simulation_loop(
                key,
                grid_length,
                T,
                run_cgs=True,
                disconnect_group_traits=False,
                run_cgs_every=run_cgs_every,
                cgs_mut_std=cgs_mut_std,
                prestige_decay=prestige_decay,
                prestige_value=prestige_value,
                prestige_bias=prestige_bias,
                demonstrator_prestige_baseline=demonstrator_prestige_baseline,
                gift_rate=gift_rate,
                gift_base=gift_base,
                gift_exponent=gift_exponent,
                gift_cap=gift_cap,
            )
        )

        all_mean_traits_known.append(np.asarray(mean_traits_known))
        all_group_norm_values.append(np.asarray(group_norm_values))
        all_group_labels_grids.append(np.asarray(group_labels_grid))
        all_group_instance_ids_by_label_history.append(
            np.asarray(group_instance_ids_by_label_history)
        )
        all_role_probs.append(np.asarray(role_probs))
        all_agent_roles.append(np.asarray(agent_roles))
        all_n_innovated.append(np.asarray(n_innovated))
        all_n_imitated.append(np.asarray(n_imitated))
        all_mean_prestige.append(np.asarray(mean_prestige))
        all_max_prestige.append(np.asarray(max_prestige))
        all_total_gifts.append(np.asarray(total_gifts))
        all_max_gift_income.append(np.asarray(max_gift_income))
        all_group_lineage_arrays.append(
            np.stack(
                [
                    np.asarray(final_group_parent_instance_ids),
                    np.asarray(final_group_birth_timesteps),
                ],
                axis=1,
            )
        )
        all_final_next_group_instance_ids.append(
            np.asarray(final_next_group_instance_id)
        )

    simulation_outputs = {
        "seeds": np.asarray(seeds),
        "T": np.int32(T),
        "grid_length": np.int32(grid_length),
        "max_total_l": np.int32(MAX_TOTAL_L),
        "run_cgs_every": np.int32(run_cgs_every),
        "cgs_mut_std": np.float32(cgs_mut_std),
        "prestige_decay": np.float32(prestige_decay),
        "prestige_value": np.float32(prestige_value),
        "prestige_bias": np.float32(prestige_bias),
        "demonstrator_prestige_baseline": np.float32(demonstrator_prestige_baseline),
        "gift_rate": np.float32(gift_rate),
        "gift_base": np.float32(gift_base),
        "gift_exponent": np.float32(gift_exponent),
        "gift_cap": np.float32(gift_cap),
        "learning_rule": np.asarray("chosen_role_plus_gift_income_to_innovate"),
        "group_norm_kind": np.asarray("prestige_gain"),
        "role_innovate": np.int32(ROLE_INNOVATE),
        "role_imitate": np.int32(ROLE_IMITATE),
        "mean_traits_known": np.stack(all_mean_traits_known, axis=0),
        "group_norm_values": np.stack(all_group_norm_values, axis=0),
        "group_labels_grids": np.stack(all_group_labels_grids, axis=0),
        "group_instance_ids_by_label_history": np.stack(
            all_group_instance_ids_by_label_history, axis=0
        ),
        "role_probs": np.stack(all_role_probs, axis=0),
        "agent_roles": np.stack(all_agent_roles, axis=0),
        "n_innovated": np.stack(all_n_innovated, axis=0),
        "n_imitated": np.stack(all_n_imitated, axis=0),
        "mean_prestige": np.stack(all_mean_prestige, axis=0),
        "max_prestige": np.stack(all_max_prestige, axis=0),
        "total_gifts": np.stack(all_total_gifts, axis=0),
        "max_gift_income": np.stack(all_max_gift_income, axis=0),
        "group_lineage_arrays": np.stack(all_group_lineage_arrays, axis=0),
        "final_next_group_instance_ids": np.stack(
            all_final_next_group_instance_ids, axis=0
        ),
    }
    np.savez(f"real_{seeds[0]}-{seeds[-1]}.npz", **simulation_outputs)


if __name__ == "__main__":
    main()
