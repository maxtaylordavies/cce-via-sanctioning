from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

ROLE_INNOVATE, ROLE_IMITATE = 0, 1
L = 100
MAX_TOTAL_L = 15000
GROUP_SWITCH_BUFFER = 0.0


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
def choose_demonstrator(key, candidate_mask, agent_idx):
    weights = jnp.where(candidate_mask, 1.0, 0.0)

    fallback_mask = jnp.arange(candidate_mask.shape[0]) != agent_idx
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


def _adjacent_mask(mask):
    # 4-neighbourhood on a torus: rolling wraps the grid at the edges.
    up = jnp.roll(mask, 1, axis=0)
    down = jnp.roll(mask, -1, axis=0)
    left = jnp.roll(mask, 1, axis=1)
    right = jnp.roll(mask, -1, axis=1)
    return up | down | left | right


@jax.jit
def _next_grid(key, grid, cell_scores):
    # Synchronous proposal: each cell compares itself to its four neighbours and
    # only switches if a neighbour beats its own recent score by a meaningful
    # buffer, which dampens brittle group changes caused by tiny fluctuations.
    neighbour_groups = jnp.stack(
        [
            jnp.roll(grid, 1, axis=0),
            jnp.roll(grid, -1, axis=0),
            jnp.roll(grid, 1, axis=1),
            jnp.roll(grid, -1, axis=1),
        ],
        axis=-1,
    )
    neighbour_scores = jnp.stack(
        [
            jnp.roll(cell_scores, 1, axis=0),
            jnp.roll(cell_scores, -1, axis=0),
            jnp.roll(cell_scores, 1, axis=1),
            jnp.roll(cell_scores, -1, axis=1),
        ],
        axis=-1,
    )
    # Tiny noise breaks ties between equally good neighbours without changing
    # the main dynamics.
    tie_breakers = 1e-3 * jax.random.uniform(key, shape=neighbour_scores.shape)
    best_neighbour_idx = jnp.argmax(
        neighbour_scores.astype(jnp.float32) + tie_breakers,
        axis=-1,
    )
    best_neighbour_groups = jnp.take_along_axis(
        neighbour_groups, best_neighbour_idx[..., None], axis=-1
    ).squeeze(axis=-1)
    best_neighbour_scores = jnp.take_along_axis(
        neighbour_scores, best_neighbour_idx[..., None], axis=-1
    ).squeeze(axis=-1)
    should_switch = best_neighbour_scores >= (cell_scores + GROUP_SWITCH_BUFFER)
    return jnp.where(should_switch, best_neighbour_groups, grid)


def _torus_distance_grid(seed_row, seed_col, grid_size):
    # Distance-to-seed on the wrapped grid, used when creating split daughters.
    rows, cols = jnp.meshgrid(
        jnp.arange(grid_size), jnp.arange(grid_size), indexing="ij"
    )
    row_distance = jnp.minimum(
        jnp.abs(rows - seed_row), grid_size - jnp.abs(rows - seed_row)
    )
    col_distance = jnp.minimum(
        jnp.abs(cols - seed_col), grid_size - jnp.abs(cols - seed_col)
    )
    return row_distance + col_distance


def _assign_split_regions(mask, seed_a_idx, seed_b_idx, priority):
    # Grow two connected regions outward from the split seeds. Any unresolved
    # cells at the end are assigned by toroidal distance as a fallback.
    seed_a_mask = (
        jnp.reshape(jax.nn.one_hot(seed_a_idx, mask.size, dtype=bool), mask.shape)
        & mask
    )
    seed_b_mask = (
        jnp.reshape(jax.nn.one_hot(seed_b_idx, mask.size, dtype=bool), mask.shape)
        & mask
    )
    owners = jnp.full(mask.shape, -1, dtype=jnp.int32)
    owners = jnp.where(seed_a_mask, 0, owners)
    owners = jnp.where(seed_b_mask, 1, owners)

    def grow(_, current):
        frontier_a = _adjacent_mask(current == 0) & mask & (current == -1)
        frontier_b = _adjacent_mask(current == 1) & mask & (current == -1)
        assign_a = frontier_a & (~frontier_b | (priority < 0.5))
        assign_b = frontier_b & (~frontier_a | (priority >= 0.5))

        updated = jnp.where(assign_a, 0, current)
        updated = jnp.where(assign_b, 1, updated)
        return updated

    owners = jax.lax.fori_loop(0, mask.size, grow, owners)

    seed_a_row = seed_a_idx // mask.shape[0]
    seed_a_col = seed_a_idx % mask.shape[0]
    seed_b_row = seed_b_idx // mask.shape[0]
    seed_b_col = seed_b_idx % mask.shape[0]
    dist_to_a = _torus_distance_grid(seed_a_row, seed_a_col, mask.shape[0])
    dist_to_b = _torus_distance_grid(seed_b_row, seed_b_col, mask.shape[0])
    fallback_to_b = mask & (owners == -1) & (dist_to_b < dist_to_a)
    owners = jnp.where(fallback_to_b, 1, owners)
    owners = jnp.where(mask & (owners == -1), 0, owners)
    return owners


@partial(jax.jit, static_argnames=("grid_length", "T", "max_n_groups"))
def run_simulation_loop(
    key,
    grid_length,
    T,
    run_cgs=True,
    disconnect_group_traits=False,
    run_cgs_every=25,
    cgs_mut_std=0.05,
    max_n_groups=10,
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
    prestige_value=1.0,
):
    # At most one split can happen per timestep, and each split now creates two
    # new descendant instances from one parent, so we budget for 1 + 2T total
    # historical group instances.
    max_group_instances = (2 * T) + 1
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
        jnp.bool
    )

    def _group_average_values(grid, flat_values):
        group_labels = grid.reshape(-1)
        totals = jnp.bincount(
            group_labels,
            weights=flat_values,
            length=max_n_groups,
        )
        counts = jnp.bincount(group_labels, length=max_n_groups)
        return totals / jnp.maximum(counts, 1)

    def _group_sizes(grid):
        # Count how many cells currently belong to each possible group label.
        return jnp.bincount(grid.reshape(-1), length=max_n_groups)

    def _refresh_group_instance_ids(grid, group_instance_ids_by_label):
        occupied = _group_sizes(grid) > 0
        return jnp.where(occupied, group_instance_ids_by_label, EMPTY_GROUP_INSTANCE_ID)

    def _maybe_split_group(
        key,
        t,
        grid,
        group_norm_vals,
        group_instance_ids_by_label,
        next_group_instance_id,
        group_parent_instance_ids,
        group_birth_timesteps,
    ):
        # MVP split rule: at most one sufficiently large connected group can split per
        # timestep, and the daughter group reuses one currently inactive label.
        split_event_key, split_key = jax.random.split(key)
        sizes = _group_sizes(grid)
        occupied = sizes > 0

        # 1. Calculate the base probability for every group based on its size
        total_cells = grid.shape[0] * grid.shape[1]
        size_ratio = sizes / total_cells

        # Using a power law (alpha = 3.0 or 4.0 is a good starting point)
        split_exponent = 2.0
        p_splits = jnp.power(size_ratio, split_exponent)

        # Ensure groups of size 1 cannot split (probability 0)
        p_splits = jnp.where(sizes <= 1, 0.0, p_splits)

        # 2. Roll a loaded die for every group simultaneously
        wants_to_split = jax.random.bernoulli(split_event_key, p_splits) & occupied

        inactive_groups = ~occupied
        should_split = wants_to_split.any() & inactive_groups.any()

        def split_once(args):
            (
                current_grid,
                current_norm_vals,
                current_group_instance_ids_by_label,
                current_next_group_instance_id,
                current_group_parent_instance_ids,
                current_group_birth_timesteps,
            ) = args
            (
                parent_key,
                child_key,
                seed_a_key,
                seed_b_key,
                priority_key,
                mut_key,
            ) = jax.random.split(split_key, 6)
            # Pick the parent! If multiple groups want to split on the same tick,
            # default to splitting the largest one to relieve the most scalar stress.
            parent_group = jnp.argmax(jnp.where(wants_to_split, sizes, -1))

            child_scores = jax.random.uniform(child_key, shape=(max_n_groups,))
            child_group = jnp.argmax(jnp.where(inactive_groups, child_scores, -1.0))

            parent_mask = current_grid == parent_group
            mask_flat = parent_mask.reshape(-1)

            # Pick one random seed, then a second seed that is as far away as
            # possible so the two daughter regions separate cleanly.
            seed_a_scores = jax.random.uniform(seed_a_key, shape=(parent_mask.size,))
            seed_a_idx = jnp.argmax(jnp.where(mask_flat, seed_a_scores, -1.0))

            seed_a_row = seed_a_idx // current_grid.shape[0]
            seed_a_col = seed_a_idx % current_grid.shape[0]
            distance_scores = _torus_distance_grid(
                seed_a_row, seed_a_col, current_grid.shape[0]
            ).reshape(-1)
            seed_b_tie = 1e-3 * jax.random.uniform(
                seed_b_key, shape=(parent_mask.size,)
            )
            valid_seed_b = mask_flat & (jnp.arange(parent_mask.size) != seed_a_idx)
            seed_b_idx = jnp.argmax(
                jnp.where(
                    valid_seed_b, distance_scores.astype(jnp.float32) + seed_b_tie, -1.0
                )
            )

            priority = jax.random.uniform(priority_key, shape=parent_mask.shape)
            owners = _assign_split_regions(
                parent_mask, seed_a_idx, seed_b_idx, priority
            )
            child_mask = owners == 1
            split_grid = jnp.where(child_mask, child_group, current_grid)

            # Parent and child both inherit mutated copies of the parent's trait.
            base_norm_val = current_norm_vals[parent_group]
            noise = jax.random.normal(mut_key, shape=(2,)) * cgs_mut_std
            split_vals = current_norm_vals.at[parent_group].set(
                base_norm_val + noise[0]
            )
            split_vals = split_vals.at[child_group].set(base_norm_val + noise[1])
            ancestor_instance_id = current_group_instance_ids_by_label[parent_group]
            parent_descendant_instance_id = current_next_group_instance_id
            child_instance_id = current_next_group_instance_id + 1
            split_group_instance_ids_by_label = current_group_instance_ids_by_label.at[
                parent_group
            ].set(parent_descendant_instance_id)
            split_group_instance_ids_by_label = split_group_instance_ids_by_label.at[
                child_group
            ].set(child_instance_id)
            split_group_parent_instance_ids = current_group_parent_instance_ids.at[
                parent_descendant_instance_id
            ].set(ancestor_instance_id)
            split_group_parent_instance_ids = split_group_parent_instance_ids.at[
                child_instance_id
            ].set(ancestor_instance_id)
            split_group_birth_timesteps = current_group_birth_timesteps.at[
                parent_descendant_instance_id
            ].set(t)
            split_group_birth_timesteps = split_group_birth_timesteps.at[
                child_instance_id
            ].set(t)
            return (
                split_grid,
                split_vals,
                split_group_instance_ids_by_label,
                current_next_group_instance_id + 2,
                split_group_parent_instance_ids,
                split_group_birth_timesteps,
            )

        return jax.lax.cond(
            should_split,
            split_once,
            lambda args: args,
            (
                grid,
                group_norm_vals,
                group_instance_ids_by_label,
                next_group_instance_id,
                group_parent_instance_ids,
                group_birth_timesteps,
            ),
        )

    def step_cgs(
        key,
        t,
        grid,
        group_norm_vals,
        cell_scores,
        group_instance_ids_by_label,
        next_group_instance_id,
        group_parent_instance_ids,
        group_birth_timesteps,
    ):
        key, step_key, split_key = jax.random.split(key, 3)
        next_grid = _next_grid(step_key, grid, cell_scores)
        next_group_norm_vals = group_norm_vals
        next_group_instance_ids_by_label = _refresh_group_instance_ids(
            next_grid, group_instance_ids_by_label
        )
        (
            next_grid,
            next_group_norm_vals,
            next_group_instance_ids_by_label,
            next_group_instance_id,
            next_group_parent_instance_ids,
            next_group_birth_timesteps,
        ) = _maybe_split_group(
            split_key,
            t,
            next_grid,
            next_group_norm_vals,
            next_group_instance_ids_by_label,
            next_group_instance_id,
            group_parent_instance_ids,
            group_birth_timesteps,
        )
        return (
            next_grid,
            next_group_norm_vals,
            next_group_instance_ids_by_label,
            next_group_instance_id,
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
        q_delta = jnp.where(group_changed, prestige_value * delta_prestige_gain, 0.0)
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
    def update_traits(key, all_traits, all_roles, group_labels_grid, curr_Ls):
        group_labels_1d = group_labels_grid.reshape(-1)

        def per_agent(key_, agent_idx):
            group_idx = group_labels_1d[agent_idx]

            def do_innovate(_):
                new_traits, success = innovate(
                    key_, all_traits[agent_idx], curr_Ls[group_idx], p_i
                )
                return new_traits, success, 0  # traits, # innovated, # imitated

            def do_imitate(_):
                imit_mask = can_imitate(agent_idx, group_labels_grid)
                new_traits, n_copied = imitate(
                    key_, all_traits, imit_mask, agent_idx, p_c
                )
                return new_traits, 0, n_copied  # traits, # innovated, # imitated

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

        # compute current payoffs
        curr_trait_payoffs = compute_trait_payoffs(all_traits, b)

        # update traits and compute new payoffs
        new_all_traits, n_innovated, n_imitated = update_traits(
            update_key, all_traits, all_roles, group_labels_grid, curr_Ls
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

        # compute role rewards and costs
        rewards = new_trait_payoffs - curr_trait_payoffs
        rewards -= jnp.where(all_roles == ROLE_INNOVATE, innov_cost, 0.0)
        rewards += prestige_value * prestige_changes

        # update q vals
        rpe = rewards - all_q_vals[jnp.arange(N), all_roles]
        new_all_q_vals = all_q_vals.at[jnp.arange(N), all_roles].add(
            learning_rate * rpe
        )

        # compute some metrics for logging
        total_unique_traits_known = (new_all_traits.sum(axis=0) > 0).sum()
        per_agent_traits_known = new_all_traits.sum(axis=1)
        most_traits_known, mean_traits_known = (
            per_agent_traits_known.max(),
            per_agent_traits_known.mean(),
        )

        # determine whether to unlock new space of traits
        # mask = jnp.arange(MAX_TOTAL_L)
        # mask = (mask >= curr_L - L) & (mask < curr_L)
        # mean_prop_known = (all_traits * mask.reshape(1, -1)).sum(axis=1).mean() / L

        group_num_traits_known = _group_average_values(
            group_labels_grid, per_agent_traits_known
        )
        group_prop_traits_known = group_num_traits_known / curr_Ls
        unlock = (group_prop_traits_known >= 0.9) & (curr_Ls < MAX_TOTAL_L)
        new_Ls = jnp.where(unlock, curr_Ls + L, curr_Ls)

        # Every `run_cgs_every` steps, do tournament selection and mutation at the group level
        should_run_cgs = run_cgs & (t % run_cgs_every == 0) & (t > 0)
        agent_frontiers = curr_Ls[group_labels_1d].astype(jnp.float32)
        traits_gained_scores = updated_traits_gained_since_cgs / jnp.maximum(
            agent_frontiers, 1.0
        )
        traits_gained_grid = traits_gained_scores.reshape(grid_length, grid_length)
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
            should_run_cgs,
            lambda args: step_cgs(*args),
            lambda args: (
                args[2],
                args[3],
                args[5],
                args[6],
                args[7],
                args[8],
            ),
            (
                cgs_key,
                t,
                group_labels_grid,
                group_norm_values,
                traits_gained_grid,
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
            should_run_cgs,
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
            per_agent_traits_known,
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
            total_unique_traits_known,
        )

    group_norm_values = jnp.zeros(max_n_groups, dtype=jnp.float32)
    group_labels_grid = jnp.zeros((grid_length, grid_length), dtype=jnp.int32)
    group_instance_ids_by_label = (
        jnp.full(max_n_groups, EMPTY_GROUP_INSTANCE_ID, dtype=jnp.int32).at[0].set(0)
    )
    next_group_instance_id = jnp.int32(1)
    group_parent_instance_ids = jnp.full(
        max_group_instances, EMPTY_GROUP_INSTANCE_ID, dtype=jnp.int32
    )
    group_birth_timesteps = (
        jnp.full(max_group_instances, -1, dtype=jnp.int32).at[0].set(0)
    )

    carry = (
        key,  # key
        jnp.full(max_n_groups, L, dtype=jnp.int32),  # curr_Ls
        jnp.zeros((N, MAX_TOTAL_L), dtype=jnp.int8),  # all_traits
        jnp.full((N, 2), init_q, dtype=jnp.float32),  # all_q_vals
        jnp.zeros((N,), dtype=jnp.float32),  # all_prestiges
        group_norm_values,
        group_labels_grid,
        group_instance_ids_by_label,
        next_group_instance_id,
        group_parent_instance_ids,
        group_birth_timesteps,
        jnp.zeros(N, dtype=jnp.float32),  # traits_gained_since_cgs
    )

    carry, metrics = jax.lax.scan(body_fn, carry, jnp.arange(T))
    metrics = list(metrics)
    metrics[7] = jnp.cumsum(metrics[7])  # cumulative number innovated
    metrics[8] = jnp.cumsum(metrics[8])  # cumulative number imitated

    return (*metrics, carry[8], carry[9], carry[10])


def main():
    seeds = list(range(5, 10))
    grid_length, T = 30, int(5e3)
    prestige_decay = 0.01
    prestige_value = 1.0
    run_cgs_every = 25
    cgs_mut_std = 0.1

    all_traits_known = []
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
    all_total_unique_traits = []
    all_group_lineage_arrays = []
    all_final_next_group_instance_ids = []
    for seed in tqdm(seeds):
        key = jax.random.PRNGKey(seed)

        (
            traits_known,
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
            total_unique_traits,
            final_next_group_instance_id,
            final_group_parent_instance_ids,
            final_group_birth_timesteps,
        ) = jax.block_until_ready(
            run_simulation_loop(
                key,
                grid_length,
                T,
                run_cgs=True,
                run_cgs_every=run_cgs_every,
                cgs_mut_std=cgs_mut_std,
                disconnect_group_traits=False,
                prestige_decay=prestige_decay,
                prestige_value=prestige_value,
            )
        )

        all_traits_known.append(np.asarray(traits_known))
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
        all_total_unique_traits.append(np.asarray(total_unique_traits))
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
        "prestige_decay": np.float32(prestige_decay),
        "prestige_value": np.float32(prestige_value),
        "group_norm_kind": np.asarray("prestige_gain"),
        "role_innovate": np.int32(ROLE_INNOVATE),
        "role_imitate": np.int32(ROLE_IMITATE),
        "traits_known": np.stack(all_traits_known, axis=0),
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
        "total_unique_traits": np.stack(all_total_unique_traits, axis=0),
        "group_lineage_arrays": np.stack(all_group_lineage_arrays, axis=0),
        "final_next_group_instance_ids": np.stack(
            all_final_next_group_instance_ids, axis=0
        ),
    }
    np.savez(
        f"real_{seeds[0]}-{seeds[-1]}.npz",
        **simulation_outputs,
    )


if __name__ == "__main__":
    main()
