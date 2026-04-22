from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from src.grammar import (
    PAD,
    MAX_LIBRARY_SIZE,
    MAX_RECIPE_LEN,
    MAX_PLANT_LEN,
    MAX_RULE_LEN,
    MAX_COMPLEXITY_LEVEL,
    GOAL_PLANT,
    NUM_RULES_IN_INITIAL_LIBRARY,
    EMPTY_RECIPE_ID,
    initial_recipe_ids,
    pregenerate_plants,
    atomic_rules,
    initial_library,
)

ATOMIC_TARGETS = atomic_rules[:, 0, :]
ATOMIC_REPLACEMENTS = atomic_rules[:, 1, :]
ATOMIC_TARGET_LENGTHS = jnp.sum(ATOMIC_TARGETS != PAD, axis=1).astype(jnp.int32)
ATOMIC_REPLACEMENT_LENGTHS = jnp.sum(ATOMIC_REPLACEMENTS != PAD, axis=1).astype(
    jnp.int32
)
PLANT_POSITIONS = jnp.arange(MAX_PLANT_LEN, dtype=jnp.int32)
RULE_OFFSETS = jnp.arange(MAX_RULE_LEN, dtype=jnp.int32)

MAX_ENERGY = 500
CHOICE_BETA = 0.01
ROLE_INNOVATE, ROLE_IMITATE = 0, 1
OP_PROBS = jnp.array(
    [
        0.4,
        0.4,
        0.2,
    ]
)  # probabilities for add, delete, combine operations during innovation
OP_THRESHOLDS = jnp.cumsum(OP_PROBS)


@jax.jit
def choose_innov_op(key):
    x = jax.random.uniform(key)
    return jnp.sum(x > OP_THRESHOLDS)


@jax.jit
def get_acceptance_prob(delta):
    p_min, p_max, tau = 0.05, 0.95, 0.5
    return p_min + (p_max - p_min) * jax.nn.sigmoid(delta / tau)


@partial(jax.jit, static_argnames=["n"])
def sample_levels(key, energy, n):
    avg_level = (energy / MAX_ENERGY) * MAX_COMPLEXITY_LEVEL
    lo = jnp.floor(avg_level)
    p = jnp.array([1 - (avg_level - lo), avg_level - lo])
    levels = jax.random.choice(key, jnp.array([lo, lo + 1]), p=p, shape=(n,))
    return jnp.clip(levels, 1, MAX_COMPLEXITY_LEVEL).astype(jnp.int32)


@jax.jit
def apply_rule_idx(plant, rule_idx):
    positions = PLANT_POSITIONS
    offsets = RULE_OFFSETS

    target = ATOMIC_TARGETS[rule_idx]
    replacement = ATOMIC_REPLACEMENTS[rule_idx]
    target_len = ATOMIC_TARGET_LENGTHS[rule_idx]
    replacement_len = ATOMIC_REPLACEMENT_LENGTHS[rule_idx]

    plant_len = jnp.sum(plant != PAD)
    can_start = positions <= (plant_len - target_len)

    idx_matrix = positions[:, None] + offsets[None, :]
    safe_idx_matrix = jnp.clip(idx_matrix, 0, MAX_PLANT_LEN - 1)
    plant_windows = plant[safe_idx_matrix]

    active_target_mask = offsets < target_len
    token_match = plant_windows == target[None, :]
    window_match = jnp.all(
        jnp.where(active_target_mask[None, :], token_match, True),
        axis=1,
    )
    matches = can_start & window_match & (target_len > 0)

    has_match = jnp.any(matches)
    first_match = jnp.argmax(matches.astype(jnp.int32))
    delta = replacement_len - target_len

    in_prefix = positions < first_match
    in_replacement = (positions >= first_match) & (
        positions < (first_match + replacement_len)
    )

    src_tail = positions - delta
    src_idx = jnp.where(in_prefix, positions, src_tail)
    src_idx = jnp.clip(src_idx, 0, MAX_PLANT_LEN - 1)
    copied = plant[src_idx]

    repl_idx = jnp.clip(positions - first_match, 0, MAX_RULE_LEN - 1)
    repl_vals = replacement[repl_idx]

    updated = jnp.where(in_replacement, repl_vals, copied)

    new_len = jnp.clip(plant_len + delta, 0, MAX_PLANT_LEN)
    updated = jnp.where(positions < new_len, updated, PAD)

    return jnp.where(has_match, updated, plant)


@jax.jit
def apply_recipe(plant, recipe):
    recipe_len = jnp.sum(recipe != PAD)

    def body(i, current_plant):
        return apply_rule_idx(current_plant, recipe[i])

    return jax.lax.fori_loop(0, recipe_len, body, plant)


def plant_value(level):
    return level


def foraging_cost(level):
    return level / 2


@jax.jit
def evaluate_recipe(plants, levels, recipe, rule_cost=0.05):
    processed = jax.vmap(lambda plant: apply_recipe(plant, recipe))(plants)

    successes = jnp.all(processed == GOAL_PLANT, axis=1)

    yields = successes.astype(jnp.float32) * plant_value(levels)

    recipe_len = jnp.sum(recipe != PAD)
    yields -= rule_cost * recipe_len

    return jnp.maximum(yields, 0.0)


@jax.jit
def evaluate_library(plants, levels, library, rule_cost=0.01):
    """
    Evaluates a library of recipes on a batch of plants.

    Args:
        plants: jnp.ndarray[int32] shape (batch_size, MAX_PLANT_LEN), PAD-terminated.
        levels: jnp.ndarray[int32] shape (batch_size,), complexity levels for each plant.
        library: jnp.ndarray[int32] shape (num_recipes, MAX_RECIPE_LEN), where each recipe is a padded sequence of rule pointers.

    Returns:
        average yield across batch
    """
    # scores has shape (n_recipes, n_plants)
    scores = jax.vmap(
        lambda recipe: evaluate_recipe(plants, levels, recipe, rule_cost)
    )(library)
    best_recipe_idx = scores.mean(axis=1).argmax()
    per_plant_max = scores.max(
        axis=0
    )  # best yield achieved for each plant across all recipes
    avg_yield = per_plant_max.mean()  # average of best yields across the batch
    return avg_yield, per_plant_max, best_recipe_idx


@jax.jit
def get_recipe_length(recipe):
    return jnp.sum(recipe != PAD)


@jax.jit
def get_library_size(library):
    return jax.vmap(get_recipe_length)(library).sum()


@jax.jit
def get_diff_size(library_1, library_2):
    return (library_2 != library_1).astype(jnp.int32).sum()


@jax.jit
def get_num_recipes(library):
    return jnp.sum(jnp.any(library != PAD, axis=1))


@jax.jit
def get_library_entropy(library):
    num_rules = atomic_rules.shape[0]
    transitions_from = library[:, :-1].reshape(-1)
    transitions_to = library[:, 1:].reshape(-1)
    transition_counts = jnp.zeros((num_rules, num_rules), dtype=jnp.int32)
    transition_counts = transition_counts.at[transitions_from, transitions_to].add(1)

    # exclude the first row of transition_counts (corresponding to PAD)
    transition_counts = transition_counts[1:]

    p = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    p = jnp.nan_to_num(p)  # replace NaNs from zero rows with zeros
    return -jnp.sum(p * jnp.log(p + 1e-10))


@jax.jit
def add_rule(key, recipe):
    key_rule, key_insert = jax.random.split(key)

    # sample an atomic rule to add
    atomic_rule_idx = jax.random.randint(key_rule, (), 1, atomic_rules.shape[0])

    # find the first empty slot in the recipe
    num_rules = get_recipe_length(recipe)
    recipe_full = num_rules >= MAX_RECIPE_LEN
    insert_idx = num_rules % MAX_RECIPE_LEN

    # if the recipe is full, insert at a random index (overwriting an existing rule)
    overwrite_idx = jax.random.randint(key_insert, (), 0, num_rules)
    insert_idx = jnp.where(recipe_full, overwrite_idx, insert_idx)
    return recipe.at[insert_idx].set(atomic_rule_idx)


@jax.jit
def delete_rule(key, recipe):
    # find the number of rules in the recipe
    num_rules = get_recipe_length(recipe)
    has_rules = num_rules > 0

    # safe sampling bounds even when empty recipe
    maxval = jnp.maximum(num_rules, 1)
    pos_idx = jax.random.randint(key, (), 0, maxval)

    positions = jnp.arange(MAX_RECIPE_LEN, dtype=jnp.int32)

    # For positions >= rule_idx, shift source left by one (drop rule_idx)
    src_idx = jnp.where(positions < pos_idx, positions, positions + 1)
    src_idx = jnp.clip(src_idx, 0, MAX_RECIPE_LEN - 1)

    shifted = recipe[src_idx]

    # New logical length after deletion
    new_len = jnp.maximum(num_rules - 1, 0)

    # Zero out trailing rows beyond new length
    valid_rows = positions < new_len
    new_recipe = jnp.where(valid_rows, shifted, 0)

    # If no rules existed, keep recipe unchanged
    return jnp.where(has_rules, new_recipe, recipe)


@jax.jit
def combine_recipes(recipe_1, recipe_2):
    len_1 = get_recipe_length(recipe_1)
    len_2 = get_recipe_length(recipe_2)

    positions = jnp.arange(MAX_RECIPE_LEN, dtype=jnp.int32)

    # For each output row i:
    # - take recipe_1[i] when i < len_1
    # - else take recipe_2[i - len_1]
    use_first = positions < len_1
    src_idx_first = positions
    src_idx_second = jnp.clip(positions - len_1, 0, MAX_RECIPE_LEN - 1)

    gathered_first = recipe_1[src_idx_first]
    gathered_second = recipe_2[src_idx_second]

    combined = jnp.where(use_first, gathered_first, gathered_second)

    # Logical concatenated length, clipped to capacity
    total_len = jnp.minimum(len_1 + len_2, MAX_RECIPE_LEN)
    valid_rows = positions < total_len

    # Zero-pad remainder
    return jnp.where(valid_rows, combined, 0)


@jax.jit
def innovate(key, library, recipe_ages, recipe_ids):
    key_op, key_recipe = jax.random.split(key)

    # select which type of operation to perform: add a rule, delete a rule, or combine two recipes
    op = choose_innov_op(key_op)

    # sample recipe(s) from the library to modify
    num_recipes = get_num_recipes(library)
    recipe_idxs = jax.random.randint(
        key_recipe, shape=(2,), minval=0, maxval=num_recipes
    )
    recipe_1, recipe_2 = library[recipe_idxs]

    # perform the selected operation
    new_recipe = jax.lax.switch(
        op,
        [
            lambda _: add_rule(key, recipe_1),
            lambda _: delete_rule(key, recipe_1),
            lambda _: combine_recipes(recipe_1, recipe_2),
        ],
        operand=None,
    )

    # store the new recipe
    overwrite_idx = jnp.where(
        num_recipes < MAX_LIBRARY_SIZE, num_recipes, recipe_ages.argmax()
    )
    insert_idx = jnp.where(op == 2, overwrite_idx, recipe_idxs[0])
    parent_1_id = recipe_ids[recipe_idxs[0]]
    parent_2_id = jnp.where(op == 2, recipe_ids[recipe_idxs[1]], EMPTY_RECIPE_ID)
    return library.at[insert_idx].set(new_recipe), insert_idx, parent_1_id, parent_2_id


@partial(jax.jit, static_argnames=["n_agents"])
def imitate_recipe(
    key,
    libraries,
    recipe_ids,
    can_imitate,
    agent_idx,
    n_agents,
    best_recipe_idxs,
    recipe_ages,
):
    key_agent, _ = jax.random.split(key)

    # select random neighbour to imitate from
    p = can_imitate / can_imitate.sum()
    demonstrator_idx = jax.random.choice(key_agent, n_agents, p=p)

    # select the recipe that contributed most to the demonstrator's yield in the most recent batch
    recipe_idx = best_recipe_idxs[demonstrator_idx]

    # determine where to insert the imitated recipe in the imitator's library
    num_recipes = get_num_recipes(libraries[agent_idx])
    insert_idx = jnp.where(
        num_recipes < MAX_LIBRARY_SIZE, num_recipes, recipe_ages[agent_idx].argmax()
    )

    # insert the imitated recipe and return the updated library
    return (
        libraries[agent_idx]
        .at[insert_idx]
        .set(libraries[demonstrator_idx, recipe_idx]),
        insert_idx,
        recipe_ids[demonstrator_idx, recipe_idx],
    )


def _adjacent_mask(mask):
    # 4-neighbourhood on a torus: rolling wraps the grid at the edges.
    up = jnp.roll(mask, 1, axis=0)
    down = jnp.roll(mask, -1, axis=0)
    left = jnp.roll(mask, 1, axis=1)
    right = jnp.roll(mask, -1, axis=1)
    return up | down | left | right


@jax.jit
def _next_grid(key, grid, cell_yields):
    # Synchronous proposal: each cell compares itself to its four neighbours and
    # only switches if a neighbour beats its own yield EMA by a meaningful
    # buffer, which dampens brittle tribe changes caused by tiny fluctuations.
    neighbour_groups = jnp.stack(
        [
            jnp.roll(grid, 1, axis=0),
            jnp.roll(grid, -1, axis=0),
            jnp.roll(grid, 1, axis=1),
            jnp.roll(grid, -1, axis=1),
        ],
        axis=-1,
    )
    neighbour_yields = jnp.stack(
        [
            jnp.roll(cell_yields, 1, axis=0),
            jnp.roll(cell_yields, -1, axis=0),
            jnp.roll(cell_yields, 1, axis=1),
            jnp.roll(cell_yields, -1, axis=1),
        ],
        axis=-1,
    )
    # Tiny noise breaks ties between equally good neighbours without changing
    # the main dynamics.
    tie_breakers = 1e-3 * jax.random.uniform(key, shape=neighbour_yields.shape)
    best_neighbour_idx = jnp.argmax(
        neighbour_yields.astype(jnp.float32) + tie_breakers,
        axis=-1,
    )
    best_neighbour_groups = jnp.take_along_axis(
        neighbour_groups, best_neighbour_idx[..., None], axis=-1
    ).squeeze(axis=-1)
    best_neighbour_yields = jnp.take_along_axis(
        neighbour_yields, best_neighbour_idx[..., None], axis=-1
    ).squeeze(axis=-1)
    should_switch = best_neighbour_yields >= (cell_yields + CA_SWITCH_YIELD_EMA_BUFFER)
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


@partial(jax.jit, static_argnames=("T", "grid_length"))
def run_simulation_loop(
    key,
    plants,
    grid_length,
    T,
    run_ca=True,
    final_phase=500,
    n_forage=5,
    n_innov_attempts=3,
    innov_cost=0.5,
    run_ca_every=25,
    norm_mut_std=0.05,
    max_n_groups=20,
    imit_dist_threshold=1,
    learning_rate=0.1,
    p_death=0.001,
    yield_ema_alpha=0.05,
):
    n_agents = grid_length**2
    max_recipe_ids = NUM_RULES_IN_INITIAL_LIBRARY + (T * n_agents)

    # At most one split can happen per timestep, and each split now creates two
    # new descendant instances from one parent, so we budget for 1 + 2T total
    # historical group instances.
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

    plants_per_level = plants.shape[1]

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
        # connected = jax.vmap(lambda group_id: _mask_is_connected(grid == group_id))(
        #     jnp.arange(max_n_groups)
        # )

        # 1. Calculate the base probability for every group based on its size
        total_cells = grid.shape[0] * grid.shape[1]
        size_ratio = sizes / total_cells

        # Using a power law (alpha = 3.0 or 4.0 is a good starting point)
        split_exponent = 3.0
        p_splits = jnp.power(size_ratio, split_exponent)

        # Ensure groups of size 1 cannot split (probability 0)
        p_splits = jnp.where(sizes <= 1, 0.0, p_splits)

        # 2. Roll a loaded die for every group simultaneously
        wants_to_split = jax.random.bernoulli(split_event_key, p_splits) & occupied

        inactive_groups = ~occupied
        should_split = wants_to_split.any() & inactive_groups.any()

        # eligible_parents = occupied & connected & (sizes >= split_size_threshold)
        # inactive_groups = ~occupied
        # should_split = (
        #     eligible_parents.any()
        #     & inactive_groups.any()
        #     & jax.random.bernoulli(split_event_key, group_split_prob)
        # )

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
            # parent_scores = jax.random.uniform(parent_key, shape=(max_n_groups,))
            # parent_group = jnp.argmax(jnp.where(eligible_parents, parent_scores, -1.0))

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
            noise = jax.random.normal(mut_key, shape=(2,)) * norm_mut_std
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

    def step_ca(
        key,
        t,
        grid,
        group_norm_vals,
        yields,
        group_instance_ids_by_label,
        next_group_instance_id,
        group_parent_instance_ids,
        group_birth_timesteps,
    ):
        key, step_key, split_key = jax.random.split(key, 3)
        next_grid = _next_grid(step_key, grid, yields)
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

    @jax.jit
    def forage(key, energy):
        level_key, plant_key = jax.random.split(key)
        levels = sample_levels(level_key, energy, n_forage)
        plant_idxs = jax.random.randint(
            plant_key, shape=(n_forage,), minval=0, maxval=plants_per_level
        )
        plants_ = plants[levels, plant_idxs]
        return plants_, levels

    # (keys, energies) -> foraged_plants, levels
    vmapped_forage = jax.vmap(forage, in_axes=(0, 0))

    # (foraged_plant, level, libraries) -> avg yields, per-plant max yields, best recipe indices
    vmapped_eval_library = jax.vmap(evaluate_library, in_axes=(0, 0, 0))

    @jax.jit
    def can_imitate(agent_idx, group_labels_grid):
        # agent can imitate anyone else in their group
        row = agent_idx // grid_length
        col = agent_idx % grid_length
        group = group_labels_grid[row, col]
        return (group_labels_grid == group).reshape(-1) & neighbours_mask[agent_idx]
        # return (group_labels_grid == group).reshape(-1)

    @jax.jit
    def update_library(
        key,
        agent_idx,
        curr_yield_per_plant,
        libraries,
        recipe_ids,
        foraged_plants,
        foraged_levels,
        roles,
        best_recipe_idxs,
        recipe_ages,
        group_labels_grid,
    ):
        # ROLE KEY: 0 = innovate, 1 = imitate
        innov_key, adopt_key = jax.random.split(key)

        # compute avg yield of current library over n-step history
        plant_batch = foraged_plants[agent_idx]
        level_batch = foraged_levels[agent_idx]

        curr_yield = curr_yield_per_plant.mean()

        def compute_new_avg_yield(new_library, recipe_idx):
            new_plant_scores = evaluate_recipe(
                plant_batch, level_batch, new_library[recipe_idx]
            )
            return jnp.maximum(curr_yield_per_plant, new_plant_scores).mean()

        def do_innovate(_):
            innov_keys = jax.random.split(innov_key, n_innov_attempts)
            candidate_libraries, new_idxs, parent_1_ids, parent_2_ids = jax.vmap(
                lambda k: innovate(
                    k,
                    libraries[agent_idx],
                    recipe_ages[agent_idx],
                    recipe_ids[agent_idx],
                )
            )(innov_keys)

            yields = jax.vmap(compute_new_avg_yield)(candidate_libraries, new_idxs)

            # yields = jax.vmap(
            #     lambda lib: evaluate_library(plant_batch, level_batch, lib)[0]
            # )(candidate_libraries)
            best_innov_idx = yields.argmax()
            return (
                candidate_libraries[best_innov_idx],
                yields[best_innov_idx],
                new_idxs[best_innov_idx],
                EMPTY_RECIPE_ID,
                parent_1_ids[best_innov_idx],
                parent_2_ids[best_innov_idx],
            )

        def do_imitate(_):
            imit_mask = can_imitate(agent_idx, group_labels_grid)
            imitation_library, new_idx, copied_recipe_id = imitate_recipe(
                key,
                libraries,
                recipe_ids,
                imit_mask,
                agent_idx,
                n_agents,
                best_recipe_idxs,
                recipe_ages,
            )
            imitation_yield = compute_new_avg_yield(imitation_library, new_idx)
            # imitation_yield = evaluate_library(
            #     plant_batch, level_batch, imitation_library
            # )[0]
            return (
                imitation_library,
                imitation_yield,
                new_idx,
                copied_recipe_id,
                EMPTY_RECIPE_ID,
                EMPTY_RECIPE_ID,
            )

        # obtain new library (and corresponding yield) based on chosen role
        (
            new_library,
            new_yield,
            new_idx,
            copied_recipe_id,
            parent_1_id,
            parent_2_id,
        ) = jax.lax.cond(
            roles[agent_idx] == ROLE_INNOVATE,
            do_innovate,
            do_imitate,
            operand=None,
        )
        size_delta = get_diff_size(libraries[agent_idx], new_library)

        # adopt new library with probability based on yield improvement
        yield_delta = new_yield - curr_yield
        p_accept = get_acceptance_prob(yield_delta)
        accept = jax.random.bernoulli(adopt_key, p_accept)

        new_library = jnp.where(accept, new_library, libraries[agent_idx])
        yield_delta = jnp.where(accept, yield_delta, 0.0)
        new_ages = jnp.where(
            accept, recipe_ages[agent_idx].at[new_idx].set(0), recipe_ages[agent_idx]
        )

        return (
            new_library,
            yield_delta,
            size_delta,
            new_ages,
            accept,
            new_idx,
            copied_recipe_id,
            parent_1_id,
            parent_2_id,
        )

    @jax.jit
    def compute_role_cost_adjustments(roles, group_labels_grid, group_norm_values):
        group_labels_1d = group_labels_grid.reshape(-1)

        def per_group(group_idx):
            group_mask = group_labels_1d == group_idx
            n_innov = ((roles == ROLE_INNOVATE) & group_mask).sum()
            n_imit = ((roles == ROLE_IMITATE) & group_mask).sum()
            subsidy = (group_norm_values[group_idx] * n_imit) / jnp.maximum(n_innov, 1)
            return group_mask * jnp.where(
                roles == ROLE_INNOVATE, -subsidy, group_norm_values[group_idx]
            )

        adjustments = jax.vmap(per_group)(jnp.arange(max_n_groups))
        return adjustments.sum(axis=0)

    # (keys, agent_idxs, libraries, recipe_ids, plants, levels, roles, best_recipe_idxs, recipe_ages, group_labels_grid)
    # -> updated_libraries, yield_deltas, size_deltas, updated_ages, accepts, update_idxs, copied_recipe_ids, parent_1_ids, parent_2_ids
    vmapped_update_library = jax.vmap(
        update_library,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None, None),
    )

    # (libraries) -> library_entropies
    vmapped_get_library_entropy = jax.vmap(get_library_entropy)

    def body_fn(carry, t):
        (
            key,
            libraries,
            energies,
            yield_emas,
            q_vals,
            agent_ages,
            agent_ids,
            next_agent_id,
            next_recipe_id,
            recipe_ids,
            recipe_ages,
            recipe_parent_1_ids,
            recipe_parent_2_ids,
            recipe_creator_agent_ids,
            recipe_birth_timesteps,
            group_norm_values,
            group_labels_grid,
            group_instance_ids_by_label,
            next_group_instance_id,
            group_parent_instance_ids,
            group_birth_timesteps,
        ) = carry

        # get new keys
        key, death_key, forage_key, policy_key, innov_key, ca_key = jax.random.split(
            key, 6
        )
        forage_keys = jax.random.split(forage_key, n_agents)
        innov_keys = jax.random.split(innov_key, n_agents)

        # determine which agents die this timestep
        p_death_actual = jnp.where(t >= T - final_phase, 0.0, p_death)
        deaths = jax.random.bernoulli(
            death_key, p_death_actual, shape=(n_agents,)
        ).astype(jnp.int32)

        # assign ids to newborn agents
        new_ids = next_agent_id + jnp.cumsum(deaths) - 1
        agent_ids = jnp.where(deaths, new_ids, agent_ids)
        next_agent_id += deaths.sum()

        # reset libraries, energies, etc for newborn agents
        libraries = jnp.where(deaths[:, None, None], initial_library, libraries)
        energies = jnp.where(deaths, 0.0, energies)
        q_vals = jnp.where(deaths[:, None], 1.0, q_vals)
        agent_ages = jnp.where(deaths, 0, agent_ages + 1)
        recipe_ids = jnp.where(deaths[:, None], initial_recipe_ids, recipe_ids)
        recipe_ages = jnp.where(deaths[:, None], 0, recipe_ages + 1)

        # each agent forages a plant based on their current energy level
        foraged_plants, foraged_levels = vmapped_forage(forage_keys, energies)

        # process each agent's foraged batch with their current library
        avg_yields, per_plant_yields, best_recipe_idxs = vmapped_eval_library(
            foraged_plants, foraged_levels, libraries
        )

        # update yield EMAs for each agent
        new_yield_emas = (yield_ema_alpha * avg_yields) + (
            (1 - yield_ema_alpha) * yield_emas
        )

        # Each agent follows an epsilon-greedy policy: with probability epsilon
        # they explore by sampling a random role uniformly, otherwise they
        # exploit the highest-Q role with random tie-breaking.
        explore_key, random_role_key, tie_break_key = jax.random.split(policy_key, 3)
        explore = jax.random.bernoulli(explore_key, EPSILON_GREEDY, shape=(n_agents,))
        random_roles = jax.random.randint(
            random_role_key,
            shape=(n_agents,),
            minval=0,
            maxval=2,
        )
        tie_break_noise = jax.random.uniform(
            tie_break_key,
            shape=q_vals.shape,
            minval=0.0,
            maxval=1e-6,
        )
        greedy_roles = jnp.argmax(q_vals + tie_break_noise, axis=1)
        roles = jnp.where(explore, random_roles, greedy_roles)

        # each agent updates their library based on their chosen role
        (
            new_libraries,
            yield_deltas,
            size_deltas,
            new_recipe_ages,
            accepts,
            update_idxs,
            copied_recipe_ids,
            parent_1_ids,
            parent_2_ids,
        ) = vmapped_update_library(
            innov_keys,
            jnp.arange(n_agents),
            per_plant_yields,
            libraries,
            recipe_ids,
            foraged_plants,
            foraged_levels,
            roles,
            best_recipe_idxs,
            recipe_ages,
            group_labels_grid,
        )

        slot_mask = jnp.arange(MAX_LIBRARY_SIZE)[None, :] == update_idxs[:, None]
        accepted_imitations = accepts & (roles == ROLE_IMITATE)
        accepted_innovations = accepts & (roles == ROLE_INNOVATE)

        recipe_ids = jnp.where(
            accepted_imitations[:, None] & slot_mask,
            copied_recipe_ids[:, None],
            recipe_ids,
        )

        accepted_innovations_int = accepted_innovations.astype(jnp.int32)
        innovation_orders = jnp.cumsum(accepted_innovations_int) - 1
        new_recipe_ids = next_recipe_id + innovation_orders
        recipe_ids = jnp.where(
            accepted_innovations[:, None] & slot_mask,
            new_recipe_ids[:, None],
            recipe_ids,
        )

        lineage_update_ids = jnp.where(accepted_innovations, new_recipe_ids, 0)
        recipe_parent_1_ids = recipe_parent_1_ids.at[lineage_update_ids].set(
            jnp.where(accepted_innovations, parent_1_ids, recipe_parent_1_ids[0])
        )
        recipe_parent_2_ids = recipe_parent_2_ids.at[lineage_update_ids].set(
            jnp.where(accepted_innovations, parent_2_ids, recipe_parent_2_ids[0])
        )
        recipe_creator_agent_ids = recipe_creator_agent_ids.at[lineage_update_ids].set(
            jnp.where(
                accepted_innovations,
                agent_ids,
                recipe_creator_agent_ids[0],
            )
        )
        recipe_birth_timesteps = recipe_birth_timesteps.at[lineage_update_ids].set(
            jnp.where(
                accepted_innovations,
                t,
                recipe_birth_timesteps[0],
            )
        )
        next_recipe_id = next_recipe_id + accepted_innovations_int.sum()

        # compute costs and rewards
        role_costs = jnp.where(roles == ROLE_INNOVATE, innov_cost * size_deltas, 0.0)
        role_costs += compute_role_cost_adjustments(
            roles, group_labels_grid, group_norm_values
        )
        costs = foraging_cost(foraged_levels.mean(axis=1)) + role_costs

        # update each agent's energy
        delta_energies = avg_yields - costs
        energies = jnp.clip(energies + delta_energies, 0, MAX_ENERGY)

        # update q-values based on reward prediction error
        rewards = yield_deltas - role_costs
        rpe = rewards - q_vals[jnp.arange(n_agents), roles]
        new_q_vals = q_vals.at[jnp.arange(n_agents), roles].add(learning_rate * rpe)

        # compute average reward for each role
        total_n_innov, total_n_imit = (roles == ROLE_INNOVATE).sum(), (
            roles == ROLE_IMITATE
        ).sum()
        avg_reward_innov = (rewards * (roles == ROLE_INNOVATE)).sum() / jnp.maximum(
            total_n_innov, 1
        )
        avg_reward_imit = (rewards * (roles == ROLE_IMITATE)).sum() / jnp.maximum(
            total_n_imit, 1
        )
        avg_rewards = jnp.array([avg_reward_innov, avg_reward_imit])

        # maybe run CA to update groups
        should_run_ca = run_ca & (t % run_ca_every == 0) & (t < T - final_phase)
        yield_emas_grid = yield_emas.reshape(grid_length, grid_length)
        (
            group_labels_grid,
            group_norm_values,
            group_instance_ids_by_label,
            next_group_instance_id,
            group_parent_instance_ids,
            group_birth_timesteps,
        ) = jax.lax.cond(
            should_run_ca,
            lambda args: step_ca(*args),
            lambda args: (
                args[2],
                args[3],
                args[5],
                args[6],
                args[7],
                args[8],
            ),
            (
                ca_key,
                t,
                group_labels_grid,
                group_norm_values,
                yield_emas_grid,
                group_instance_ids_by_label,
                next_group_instance_id,
                group_parent_instance_ids,
                group_birth_timesteps,
            ),
        )

        return (
            key,
            new_libraries,
            energies,
            new_yield_emas,
            new_q_vals,
            agent_ages,
            agent_ids,
            next_agent_id,
            next_recipe_id,
            recipe_ids,
            new_recipe_ages,
            recipe_parent_1_ids,
            recipe_parent_2_ids,
            recipe_creator_agent_ids,
            recipe_birth_timesteps,
            group_norm_values,
            group_labels_grid,
            group_instance_ids_by_label,
            next_group_instance_id,
            group_parent_instance_ids,
            group_birth_timesteps,
        ), (
            foraged_levels.mean(axis=-1),
            avg_yields,
            vmapped_get_library_entropy(libraries),
            avg_rewards,
            roles,
            agent_ages,
            group_norm_values,
            group_labels_grid,
            group_instance_ids_by_label,
        )

    libraries = jnp.tile(initial_library[None, ...], (n_agents, 1, 1))
    energies = jnp.zeros(n_agents, dtype=jnp.float32)
    yield_emas = jnp.zeros(n_agents, dtype=jnp.float32)
    q_vals = jnp.ones((n_agents, 2), dtype=jnp.float32)
    agent_ages = jnp.zeros(n_agents, dtype=jnp.int32)
    agent_ids = jnp.arange(n_agents, dtype=jnp.int32)
    next_agent_id = jnp.int32(n_agents)
    recipe_ids = jnp.tile(initial_recipe_ids[None, :], (n_agents, 1))
    next_recipe_id = jnp.int32(NUM_RULES_IN_INITIAL_LIBRARY)
    recipe_ages = jnp.zeros((n_agents, MAX_LIBRARY_SIZE), dtype=jnp.int32)
    recipe_parent_1_ids = jnp.full(max_recipe_ids, EMPTY_RECIPE_ID, dtype=jnp.int32)
    recipe_parent_2_ids = jnp.full(max_recipe_ids, EMPTY_RECIPE_ID, dtype=jnp.int32)
    recipe_creator_agent_ids = jnp.full(
        max_recipe_ids, EMPTY_RECIPE_ID, dtype=jnp.int32
    )
    recipe_birth_timesteps = jnp.full(max_recipe_ids, -1, dtype=jnp.int32)

    group_norm_values = jnp.full(max_n_groups, 0.0)
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
        libraries,
        energies,
        yield_emas,
        q_vals,
        agent_ages,
        agent_ids,
        next_agent_id,
        next_recipe_id,
        recipe_ids,
        recipe_ages,
        recipe_parent_1_ids,
        recipe_parent_2_ids,
        recipe_creator_agent_ids,
        recipe_birth_timesteps,
        group_norm_values,
        group_labels_grid,
        group_instance_ids_by_label,
        next_group_instance_id,
        group_parent_instance_ids,
        group_birth_timesteps,
    )

    carry, metrics = jax.lax.scan(body_fn, carry, jnp.arange(T))

    libraries = carry[1]
    final_agent_ids = carry[6]
    final_next_recipe_id = carry[8]
    final_recipe_ids = carry[9]
    final_recipe_parent_1_ids = carry[11]
    final_recipe_parent_2_ids = carry[12]
    final_recipe_creator_agent_ids = carry[13]
    final_recipe_birth_timesteps = carry[14]
    final_next_group_instance_id = carry[18]
    final_group_parent_instance_ids = carry[19]
    final_group_birth_timesteps = carry[20]

    return (
        *metrics,
        libraries,
        final_recipe_ids,
        final_agent_ids,
        final_recipe_parent_1_ids,
        final_recipe_parent_2_ids,
        final_recipe_creator_agent_ids,
        final_recipe_birth_timesteps,
        final_next_recipe_id,
        final_group_parent_instance_ids,
        final_group_birth_timesteps,
        final_next_group_instance_id,
    )


seeds = [4]
grid_length, T_main, T_extra = 25, int(3e4), 100
T = (
    T_main + T_extra
)  # total timesteps to run (including extra for averaging agent metrics at the end)


all_agent_levels = []
all_agent_yields = []
all_agent_lib_entropies = []
all_pop_role_rewards = []
all_agent_roles = []
all_agent_ages = []
all_group_norm_values = []
all_group_labels_grids = []
all_group_instance_ids_by_label_history = []
all_final_libraries = []
all_final_recipe_ids = []
all_final_agent_ids = []
all_final_next_recipe_ids = []
all_recipe_lineage_arrays = []
all_group_lineage_arrays = []
all_final_next_group_instance_ids = []
for seed in tqdm(seeds):
    key = jax.random.PRNGKey(seed)
    plants = pregenerate_plants(key, num_per_level=500, max_level=MAX_COMPLEXITY_LEVEL)

    # arrays starting with "agent_" have shape (n_fees, T, n_agents, ...)
    # arrays starting with "pop_" have shape (n_fees, T, ...)
    (
        agent_levels,
        agent_yields,
        agent_lib_entropies,
        pop_role_rewards,
        agent_roles,
        agent_ages,
        group_norm_values,
        group_labels_grid,
        group_instance_ids_by_label_history,
        final_libraries,
        final_recipe_ids,
        final_agent_ids,
        final_recipe_parent_1_ids,
        final_recipe_parent_2_ids,
        final_recipe_creator_agent_ids,
        final_recipe_birth_timesteps,
        final_next_recipe_ids,
        final_group_parent_instance_ids,
        final_group_birth_timesteps,
        final_next_group_instance_id,
    ) = jax.block_until_ready(
        run_simulation_loop(key, plants, grid_length, T, final_phase=T_extra)
    )

    all_agent_levels.append(np.asarray(agent_levels))
    all_agent_yields.append(np.asarray(agent_yields))
    all_agent_lib_entropies.append(np.asarray(agent_lib_entropies))
    all_pop_role_rewards.append(np.asarray(pop_role_rewards))
    all_agent_roles.append(np.asarray(agent_roles))
    all_agent_ages.append(np.asarray(agent_ages))
    all_group_norm_values.append(np.asarray(group_norm_values))
    all_group_labels_grids.append(np.asarray(group_labels_grid))
    all_group_instance_ids_by_label_history.append(
        np.asarray(group_instance_ids_by_label_history)
    )
    all_final_libraries.append(np.asarray(final_libraries))
    all_final_recipe_ids.append(np.asarray(final_recipe_ids))
    all_final_agent_ids.append(np.asarray(final_agent_ids))
    all_final_next_recipe_ids.append(np.asarray(final_next_recipe_ids))
    all_recipe_lineage_arrays.append(
        np.stack(
            [
                np.asarray(final_recipe_parent_1_ids),
                np.asarray(final_recipe_parent_2_ids),
                np.asarray(final_recipe_creator_agent_ids),
                np.asarray(final_recipe_birth_timesteps),
            ],
            axis=1,
        )
    )
    all_group_lineage_arrays.append(
        np.stack(
            [
                np.asarray(final_group_parent_instance_ids),
                np.asarray(final_group_birth_timesteps),
            ],
            axis=1,
        )
    )
    all_final_next_group_instance_ids.append(np.asarray(final_next_group_instance_id))

simulation_outputs = {
    "seeds": np.asarray(seeds),
    "T": np.int32(T),
    "T_main": np.int32(T_main),
    "T_extra": np.int32(T_extra),
    "grid_length": np.int32(grid_length),
    "num_rules_in_initial_library": np.int32(NUM_RULES_IN_INITIAL_LIBRARY),
    "empty_recipe_id": np.int32(EMPTY_RECIPE_ID),
    "role_innovate": np.int32(ROLE_INNOVATE),
    "role_imitate": np.int32(ROLE_IMITATE),
    "agent_levels": np.stack(all_agent_levels, axis=0),
    "agent_yields": np.stack(all_agent_yields, axis=0),
    # "agent_lib_entropies": np.stack(all_agent_lib_entropies, axis=0),
    "pop_role_rewards": np.stack(all_pop_role_rewards, axis=0),
    "agent_roles": np.stack(all_agent_roles, axis=0),
    "agent_ages": np.stack(all_agent_ages, axis=0),
    "group_norm_values": np.stack(all_group_norm_values, axis=0),
    "group_labels_grids": np.stack(all_group_labels_grids, axis=0),
    "group_instance_ids_by_label_history": np.stack(
        all_group_instance_ids_by_label_history, axis=0
    ),
    # "final_libraries": np.stack(all_final_libraries, axis=0),
    # "final_recipe_ids": np.stack(all_final_recipe_ids, axis=0),
    # "final_agent_ids": np.stack(all_final_agent_ids, axis=0),
    # "final_next_recipe_ids": np.stack(all_final_next_recipe_ids, axis=0),
    # "recipe_lineage_arrays": np.stack(all_recipe_lineage_arrays, axis=0),
    "group_lineage_arrays": np.stack(all_group_lineage_arrays, axis=0),
    "final_next_group_instance_ids": np.stack(
        all_final_next_group_instance_ids, axis=0
    ),
}
np.savez(f"simulation_outputs_{seeds[0]}-{seeds[-1]}.npz", **simulation_outputs)
