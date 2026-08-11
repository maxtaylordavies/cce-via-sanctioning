from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

ROLE_INNOVATE, ROLE_IMITATE = 0, 1

p_max = 100
l_max = 100
GROUP_SWITCH_BUFFER = 0.0


def get_increment(l):
    tmp = 0
    for j in range(1, l + 1):
        tmp += 0.95 ** (l - j)
    tmp *= 0.05 / (1 - (0.95**l_max))
    return tmp * p_max


increments = jnp.array([0] + [get_increment(l) for l in range(1, l_max + 1)])


def exploit(agent_arm_levels, agent_arm_payoff_estimates, full_rewards):
    arm_idx = jnp.argmax(agent_arm_payoff_estimates)
    arm_level = agent_arm_levels[arm_idx]
    payoff = full_rewards[arm_idx, arm_level]
    return payoff, agent_arm_payoff_estimates.at[arm_idx].set(payoff)


def explore(key, agent_arm_levels, agent_arm_payoff_estimates, full_rewards):
    unknown_mask = agent_arm_levels == 0
    n_unknown = unknown_mask.sum()

    def learn_new_arm():
        p_arm = unknown_mask / n_unknown
        arm_idx = jax.random.choice(key, full_rewards.shape[0], p=p_arm)
        payoff = full_rewards[arm_idx, 1]
        return (
            agent_arm_levels.at[arm_idx].set(1),
            agent_arm_payoff_estimates.at[arm_idx].set(payoff),
            True,
        )

    def do_nothing():
        return agent_arm_levels, agent_arm_payoff_estimates, False

    return jax.lax.cond(n_unknown > 0, learn_new_arm, do_nothing)


def refine(key, agent_arm_levels, agent_arm_payoff_estimates, full_rewards):
    eligible_mask = (agent_arm_levels > 0) & (agent_arm_levels < l_max)
    n_eligible = eligible_mask.sum()

    def refine_weighted_arm():
        arm_weights = jnp.where(
            eligible_mask, jnp.maximum(agent_arm_payoff_estimates, 0.0), 0.0
        )
        total_arm_weight = arm_weights.sum()

        def sample_weighted_arm():
            p_arm = arm_weights / total_arm_weight
            return jax.random.choice(key, full_rewards.shape[0], p=p_arm)

        def sample_uniform_eligible_arm():
            p_arm = eligible_mask / n_eligible
            return jax.random.choice(key, full_rewards.shape[0], p=p_arm)

        arm_idx = jax.lax.cond(
            total_arm_weight > 0,
            sample_weighted_arm,
            sample_uniform_eligible_arm,
        )

        # update the arm level and payoff estimate for the selected arm
        arm_level = agent_arm_levels[arm_idx]
        payoff = full_rewards[arm_idx, arm_level + 1]
        return (
            agent_arm_levels.at[arm_idx].set(arm_level + 1),
            agent_arm_payoff_estimates.at[arm_idx].set(payoff),
            True,
        )

    def do_nothing():
        return agent_arm_levels, agent_arm_payoff_estimates, False

    return jax.lax.cond(n_eligible > 0, refine_weighted_arm, do_nothing)


def innovate(key, agent_arm_levels, agent_arm_payoff_estimates, full_rewards):
    keys = jax.random.split(key, 2)
    op = jax.random.bernoulli(keys[0])
    return jax.lax.cond(
        op,
        lambda: explore(
            keys[1], agent_arm_levels, agent_arm_payoff_estimates, full_rewards
        ),
        lambda: refine(
            keys[1], agent_arm_levels, agent_arm_payoff_estimates, full_rewards
        ),
    )


@jax.jit
def choose_demonstrator(
    key,
    candidate_mask,
    prestiges,
    agent_idx,
    prestige_bias,
    demonstrator_prestige_baseline,
):
    prestige_scores = (
        demonstrator_prestige_baseline + jnp.maximum(prestiges, 0.0)
    ) ** prestige_bias
    weights = jnp.where(candidate_mask, prestige_scores, 0.0)

    fallback_mask = jnp.arange(prestiges.shape[0]) != agent_idx
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


def imitate(
    key,
    all_agent_arm_levels,
    all_agent_arm_payoff_estimates,
    can_imitate,
    prestiges,
    agent_idx,
    prestige_bias,
    demonstrator_prestige_baseline,
    gift_rate,
    gift_base,
    gift_exponent,
    gift_cap,
):
    demonstrator_key, arm_key = jax.random.split(key)
    demonstrator_idx = choose_demonstrator(
        demonstrator_key,
        can_imitate,
        prestiges,
        agent_idx,
        prestige_bias,
        demonstrator_prestige_baseline,
    )

    demonstrator_arm_levels = all_agent_arm_levels[demonstrator_idx]
    demonstrator_payoff_estimates = all_agent_arm_payoff_estimates[demonstrator_idx]
    arm_weights = jnp.where(
        demonstrator_arm_levels > 0,
        jnp.maximum(demonstrator_payoff_estimates, 0.0),
        0.0,
    )
    total_arm_weight = arm_weights.sum()

    def sample_weighted_arm():
        p_arm = arm_weights / total_arm_weight
        return jax.random.choice(arm_key, all_agent_arm_levels.shape[1], p=p_arm)

    def sample_uniform_arm():
        return jax.random.randint(arm_key, (), 0, all_agent_arm_levels.shape[1])

    arm_idx = jax.lax.cond(
        total_arm_weight > 0,
        sample_weighted_arm,
        sample_uniform_arm,
    )

    new_level = all_agent_arm_levels[demonstrator_idx, arm_idx]
    new_payoff = all_agent_arm_payoff_estimates[demonstrator_idx, arm_idx]
    current_level = all_agent_arm_levels[agent_idx, arm_idx]
    accept = new_level > current_level
    gift = compute_gift(
        prestiges[demonstrator_idx],
        gift_rate,
        gift_base,
        gift_exponent,
        gift_cap,
    )
    return jax.lax.cond(
        accept,
        lambda: (
            all_agent_arm_levels[agent_idx].at[arm_idx].set(new_level),
            all_agent_arm_payoff_estimates[agent_idx].at[arm_idx].set(new_payoff),
            True,
            demonstrator_idx,
            gift,
        ),
        lambda: (
            all_agent_arm_levels[agent_idx],
            all_agent_arm_payoff_estimates[agent_idx],
            False,
            demonstrator_idx,
            gift,
        ),
    )


@partial(jax.jit, static_argnames=("n_arms",))
def sample_arm_rewards(key, n_arms):
    rewards = jax.random.exponential(key, shape=(n_arms,))
    rewards = jnp.ceil(rewards**2)

    full_rewards = jnp.zeros((n_arms, 1 + l_max))
    return full_rewards.at[:, 1:].set(rewards[:, None] + increments[:l_max][None, :])


def _adjacent_mask(mask):
    # 4-neighbourhood on a torus: rolling wraps the grid at the edges.
    up = jnp.roll(mask, 1, axis=0)
    down = jnp.roll(mask, -1, axis=0)
    left = jnp.roll(mask, 1, axis=1)
    right = jnp.roll(mask, -1, axis=1)
    return up | down | left | right


@jax.jit
def _next_grid(key, grid, cell_scores):
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


@partial(jax.jit, static_argnames=("grid_length", "n_arms", "T", "max_n_groups"))
def run_simulation_loop(
    key,
    grid_length,
    n_arms,
    T,
    run_cgs=True,
    disconnect_group_traits=False,
    run_cgs_every=25,
    cgs_mut_std=0.05,
    max_n_groups=10,
    innov_cost=2.0,
    p_death=0.001,
    p_change=0.01,
    choice_beta=0.1,
    imit_dist_threshold=100,
    learning_rate=0.1,
    init_q=1.0,
    prestige_decay=0.01,
    prestige_value=0.0,
    prestige_bias=1.0,
    demonstrator_prestige_baseline=1.0,
    gift_rate=0.01,
    gift_base=0.0,
    gift_exponent=1.0,
    gift_cap=jnp.inf,
):
    n_agents = grid_length**2

    # At most one split can happen per timestep, and each split creates two new
    # descendant instances from one parent.
    max_group_instances = (2 * T) + 1
    EMPTY_GROUP_INSTANCE_ID = jnp.int32(-1)

    # Compute pairwise toroidal distances between agents for imitation.
    agent_idxs = jnp.arange(n_agents)
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
        split_event_key, split_key = jax.random.split(key)
        sizes = _group_sizes(grid)
        occupied = sizes > 0

        total_cells = grid.shape[0] * grid.shape[1]
        size_ratio = sizes / total_cells
        p_splits = jnp.power(size_ratio, 2.0)
        p_splits = jnp.where(sizes <= 1, 0.0, p_splits)

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
                child_key,
                seed_a_key,
                seed_b_key,
                priority_key,
                mut_key,
            ) = jax.random.split(split_key, 5)

            parent_group = jnp.argmax(jnp.where(wants_to_split, sizes, -1))

            child_scores = jax.random.uniform(child_key, shape=(max_n_groups,))
            child_group = jnp.argmax(jnp.where(inactive_groups, child_scores, -1.0))

            parent_mask = current_grid == parent_group
            mask_flat = parent_mask.reshape(-1)

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
        q_delta = jnp.where(group_changed, delta_prestige_gain, 0.0)
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
    def update_arm_knowledge_and_gifts(
        key,
        all_agent_arm_levels,
        all_agent_arm_payoff_estimates,
        all_roles,
        group_labels_grid,
        full_rewards,
        prestiges,
    ):
        def per_agent(key_, agent_idx):
            def do_innovate(_):
                new_levels, new_payoffs, success = innovate(
                    key_,
                    all_agent_arm_levels[agent_idx],
                    all_agent_arm_payoff_estimates[agent_idx],
                    full_rewards,
                )
                return (
                    new_levels,
                    new_payoffs,
                    success.astype(int),
                    0,
                    agent_idx,
                    jnp.asarray(0.0, dtype=jnp.float32),
                )

            def do_imitate(_):
                imit_mask = can_imitate(agent_idx, group_labels_grid)
                new_levels, new_payoffs, success, demonstrator_idx, gift = imitate(
                    key_,
                    all_agent_arm_levels,
                    all_agent_arm_payoff_estimates,
                    imit_mask,
                    prestiges,
                    agent_idx,
                    prestige_bias,
                    demonstrator_prestige_baseline,
                    gift_rate,
                    gift_base,
                    gift_exponent,
                    gift_cap,
                )
                return (
                    new_levels,
                    new_payoffs,
                    0,
                    success.astype(int),
                    demonstrator_idx,
                    gift,
                )

            return jax.lax.cond(
                all_roles[agent_idx] == ROLE_INNOVATE,
                do_innovate,
                do_imitate,
                operand=None,
            )

        keys = jax.random.split(key, all_agent_arm_levels.shape[0])
        new_levels, new_payoffs, innovated, imitated, demonstrator_idxs, gifts_paid = (
            jax.vmap(per_agent)(keys, jnp.arange(all_agent_arm_levels.shape[0]))
        )
        return (
            new_levels,
            new_payoffs,
            innovated,
            imitated,
            demonstrator_idxs,
            gifts_paid,
        )

    vmapped_exploit = jax.vmap(exploit, in_axes=(0, 0, None))

    def body_fn(carry, t):
        (
            key,
            full_rewards,
            agent_arm_levels,
            agent_arm_payoff_estimates,
            q_vals,
            prestiges,
            group_norm_values,
            group_labels_grid,
            group_instance_ids_by_label,
            next_group_instance_id,
            group_parent_instance_ids,
            group_birth_timesteps,
            payoff_gains_since_cgs,
        ) = carry

        # get new keys
        key, change_key, death_key, role_key, update_key, cgs_key = jax.random.split(
            key, 6
        )

        # maybe reset environment rewards
        full_rewards = jax.lax.cond(
            jax.random.bernoulli(change_key, p=p_change),
            lambda: sample_arm_rewards(key, n_arms),
            lambda: full_rewards,
        )

        # random deaths
        deaths = jax.random.bernoulli(death_key, p=p_death, shape=(n_agents,))
        agent_arm_levels = jnp.where(deaths[:, None], 0, agent_arm_levels)
        agent_arm_payoff_estimates = jnp.where(
            deaths[:, None], 0.0, agent_arm_payoff_estimates
        )
        prestiges = jnp.where(deaths, 0.0, prestiges)
        prestiges = prestiges * (1.0 - prestige_decay)
        payoff_gains_since_cgs = jnp.where(deaths, 0.0, payoff_gains_since_cgs)

        # compute payoffs from exploiting current knowledge
        curr_payoffs, agent_arm_payoff_estimates = vmapped_exploit(
            agent_arm_levels, agent_arm_payoff_estimates, full_rewards
        )

        # agents select roles
        role_probs = jax.nn.softmax(q_vals / choice_beta, axis=1)
        roles = jax.random.categorical(role_key, jnp.log(role_probs), axis=1)

        # update knowledge based on roles and compute new prospective payoffs
        (
            new_agent_arm_levels,
            new_agent_arm_payoff_estimates,
            n_innov,
            n_imit,
            demonstrator_idxs,
            gifts_paid,
        ) = update_arm_knowledge_and_gifts(
            update_key,
            agent_arm_levels,
            agent_arm_payoff_estimates,
            roles,
            group_labels_grid,
            full_rewards,
            prestiges,
        )
        gifts_paid = jnp.where(disconnect_group_traits, 0.0, gifts_paid)
        new_payoffs, _ = vmapped_exploit(
            new_agent_arm_levels, new_agent_arm_payoff_estimates, full_rewards
        )
        updated_payoff_gains_since_cgs = payoff_gains_since_cgs + jnp.maximum(
            new_payoffs - curr_payoffs, 0.0
        )
        group_labels_1d = group_labels_grid.reshape(-1)
        prestige_gain_by_agent = group_norm_values[group_labels_1d]
        prestige_gain_by_agent = jnp.where(
            disconnect_group_traits, 0.0, prestige_gain_by_agent
        )
        prestige_changes = prestige_gain_by_agent * n_innov.astype(jnp.float32)
        new_prestiges = prestiges + prestige_changes
        incoming_gifts = (
            jnp.zeros((n_agents,), dtype=jnp.float32)
            .at[demonstrator_idxs]
            .add(gifts_paid)
        )

        direct_rewards = new_payoffs - curr_payoffs
        direct_rewards -= gifts_paid
        direct_rewards -= jnp.where(roles == ROLE_INNOVATE, innov_cost, 0.0)
        direct_rewards += prestige_value * prestige_changes

        direct_rpe = direct_rewards - q_vals[jnp.arange(n_agents), roles]
        new_all_q_vals = q_vals.at[jnp.arange(n_agents), roles].add(
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

        # Every `run_cgs_every` steps, update group boundaries from local agent gains.
        should_run_cgs = run_cgs & (t % run_cgs_every == 0) & (t > 0)
        old_group_labels_grid = group_labels_grid
        old_group_norm_values = group_norm_values
        old_group_instance_ids_by_label = group_instance_ids_by_label
        payoff_gain_grid = updated_payoff_gains_since_cgs.reshape(
            grid_length, grid_length
        )
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
                payoff_gain_grid,
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
        next_payoff_gains_since_cgs = jnp.where(
            should_run_cgs,
            jnp.zeros_like(updated_payoff_gains_since_cgs),
            updated_payoff_gains_since_cgs,
        )

        # compute some metrics for logging
        mean_payoff = curr_payoffs.mean()
        mean_avg_level = agent_arm_levels.mean()
        mean_max_level = agent_arm_levels.max(axis=1).mean()
        mean_prestige = new_prestiges.mean()
        max_prestige = new_prestiges.max()
        total_gifts = gifts_paid.sum()
        max_gift_income = incoming_gifts.max()

        return (
            key,
            full_rewards,
            new_agent_arm_levels,
            new_agent_arm_payoff_estimates,
            new_all_q_vals,
            new_prestiges,
            group_norm_values,
            group_labels_grid,
            group_instance_ids_by_label,
            next_group_instance_id,
            group_parent_instance_ids,
            group_birth_timesteps,
            next_payoff_gains_since_cgs,
        ), (
            mean_payoff,
            mean_avg_level,
            mean_max_level,
            group_norm_values,
            group_labels_grid,
            group_instance_ids_by_label,
            role_probs.mean(axis=0),
            roles,
            n_innov.sum(),
            n_imit.sum(),
            mean_prestige,
            max_prestige,
            total_gifts,
            max_gift_income,
        )

    full_rewards = sample_arm_rewards(key, n_arms)

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
        key,
        full_rewards,
        jnp.zeros((n_agents, n_arms), dtype=jnp.int32),  # agent_arm_levels
        jnp.zeros((n_agents, n_arms), dtype=jnp.float32),  # agent_arm_payoff_estimates
        jnp.full((n_agents, 2), init_q, dtype=jnp.float32),  # q_vals
        jnp.zeros((n_agents,), dtype=jnp.float32),  # prestiges
        group_norm_values,
        group_labels_grid,
        group_instance_ids_by_label,
        next_group_instance_id,
        group_parent_instance_ids,
        group_birth_timesteps,
        jnp.zeros(n_agents, dtype=jnp.float32),  # payoff_gains_since_cgs
    )

    carry, metrics = jax.lax.scan(body_fn, carry, jnp.arange(T))
    return (
        metrics[0] / full_rewards.max(),
        *metrics[1:],
        carry[9],
        carry[10],
        carry[11],
    )


def main():
    seeds = list(range(5, 10))
    grid_length, n_arms, T = 30, 100, int(2e3)
    prestige_decay = 0.01
    prestige_value = 0.0
    prestige_bias = 1.0
    demonstrator_prestige_baseline = 1.0
    gift_rate = 0.01
    gift_base = 0.0
    gift_exponent = 1.0
    gift_cap = np.float32(np.inf)

    all_prop_payoffs = []
    all_mean_avg_levels = []
    all_mean_max_levels = []
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
            prop_payoffs,
            mean_avg_levels,
            mean_max_levels,
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
                n_arms,
                T,
                run_cgs=True,
                disconnect_group_traits=False,
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

        all_prop_payoffs.append(np.asarray(prop_payoffs))
        all_mean_avg_levels.append(np.asarray(mean_avg_levels))
        all_mean_max_levels.append(np.asarray(mean_max_levels))
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
        "n_arms": np.int32(n_arms),
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
        "payoffs": np.stack(all_prop_payoffs, axis=0),
        "avg_levels": np.stack(all_mean_avg_levels, axis=0),
        "max_levels": np.stack(all_mean_max_levels, axis=0),
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
