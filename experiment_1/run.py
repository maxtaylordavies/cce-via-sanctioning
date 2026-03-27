from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from tqdm import tqdm

from grammar import (
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
    recipe_lengths = jnp.sum(library != PAD, axis=1)

    def eval_single_plant(plant, level):
        processed = jax.vmap(lambda recipe: apply_recipe(plant, recipe))(library)

        # determine if each processed plant matches the goal plant
        successes = jnp.all(processed == GOAL_PLANT, axis=1)

        # scale successful yields by original plant complexity
        yields = successes.astype(jnp.float32) * plant_value(level)

        # calculate the actual length of each recipe (number of non-PAD rules)
        yields -= rule_cost * recipe_lengths  # penalize longer recipes

        return jnp.maximum(yields, 0.0)  # ensure yields are non-negative

    yields = jax.vmap(eval_single_plant, in_axes=(0, 0))(plants, levels)

    best_recipe = yields.mean(
        axis=0
    ).argmax()  # recipe that achieved the highest average yield across the batch
    best_yields = yields.max(axis=1)  # best yield per plant
    return best_yields.mean(), (best_yields > 0).mean(), best_recipe


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
    p = can_imitate[agent_idx]
    demonstrator_idx = jax.random.choice(key_agent, n_agents, p=p / p.sum())

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


@jax.jit
def compute_population_jaccard(libraries):
    """
    Computes the mean pairwise Jaccard distance for an entire population.

    Args:
        libraries: jnp.ndarray of shape (n_agents, max_library_len, max_recipe_len)

    Returns:
        mean_dist: Scalar float representing the average diversity.
        dist_matrix: The full (N, N) distance matrix.
    """
    n_agents, library_size, _ = libraries.shape

    # Identify non-empty recipes and deduplicate within each library by keeping
    # only the first occurrence of each recipe, matching the previous semantics.
    valid = jnp.any(libraries != PAD, axis=-1)
    self_matches = jnp.all(
        libraries[:, :, None, :] == libraries[:, None, :, :], axis=-1
    )
    first_occurrence = jnp.arange(library_size)[None, :] == jnp.argmax(
        self_matches, axis=2
    )
    unique_mask = valid & first_occurrence
    unique_sizes = jnp.sum(unique_mask, axis=1)

    # Pairwise recipe equality across the full population.
    pair_matches = jnp.all(
        libraries[:, None, :, None, :] == libraries[None, :, None, :, :], axis=-1
    )
    in_other_library = jnp.any(pair_matches & valid[None, :, None, :], axis=3)
    intersections = jnp.sum(unique_mask[:, None, :] & in_other_library, axis=2)

    unions = unique_sizes[:, None] + unique_sizes[None, :] - intersections
    dist_matrix = jnp.where(unions == 0, 0.0, 1.0 - (intersections / unions))

    # Compute mean of the upper triangle (ignoring the diagonal).
    num_pairs = n_agents * (n_agents - 1) / 2
    upper_tri = jnp.triu(dist_matrix, k=1)
    mean_dist = jnp.sum(upper_tri) / num_pairs

    return mean_dist, dist_matrix


@partial(jax.jit, static_argnames=("T", "grid_length"))
def run_simulation_loop(
    key,
    plants,
    grid_length,
    T,
    final_phase=500,
    n_forage=10,
    n_innov_attempts=1,
    innov_cost=0.2,
    imit_fee=0.1,
    imit_dist_threshold=1,
    learning_rate=0.1,
    diversity_eval_every=10,
    p_death=0.001,
):
    n_agents = grid_length**2
    max_recipe_ids = NUM_RULES_IN_INITIAL_LIBRARY + (T * n_agents)
    agent_idxs = jnp.arange(n_agents)
    agent_locs = jnp.stack(
        [
            agent_idxs // grid_length,  # row index
            agent_idxs % grid_length,  # column index
        ]
    ).T
    agent_dists = jnp.abs(agent_locs[:, None] - agent_locs[None, :]).sum(axis=-1)
    can_imitate = ((agent_dists > 0) & (agent_dists <= imit_dist_threshold)).astype(
        jnp.float32
    )

    plants_per_level = plants.shape[1]

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

    # (foraged_plant, level, libraries) -> avg yields, success_rates, best_recipe_idxs
    vmapped_eval_library = jax.vmap(evaluate_library, in_axes=(0, 0, 0))

    @jax.jit
    def update_library(
        key,
        agent_idx,
        libraries,
        recipe_ids,
        foraged_plants,
        foraged_levels,
        roles,
        best_recipe_idxs,
        recipe_ages,
    ):
        # ROLE KEY: 0 = innovate, 1 = imitate

        innov_key, adopt_key = jax.random.split(key)

        # compute avg yield of current library over n-step history
        plant_batch = foraged_plants[agent_idx]
        level_batch = foraged_levels[agent_idx]
        curr_yield = evaluate_library(plant_batch, level_batch, libraries[agent_idx])[0]

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
            yields = jax.vmap(
                lambda lib: evaluate_library(plant_batch, level_batch, lib)[0]
            )(candidate_libraries)
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
            imitation_library, new_idx, copied_recipe_id = imitate_recipe(
                key,
                libraries,
                recipe_ids,
                can_imitate,
                agent_idx,
                n_agents,
                best_recipe_idxs,
                recipe_ages,
            )
            imitation_yield = evaluate_library(
                plant_batch, level_batch, imitation_library
            )[0]
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

    # (keys, agent_idxs, libraries, recipe_ids, plants, levels, roles, best_recipe_idxs, recipe_ages)
    # -> updated_libraries, yield_deltas, size_deltas, updated_ages, accepts, update_idxs, copied_recipe_ids, parent_1_ids, parent_2_ids
    vmapped_update_library = jax.vmap(
        update_library, in_axes=(0, 0, None, None, None, None, None, None, None)
    )

    # (libraries) -> library_entropies
    vmapped_get_library_entropy = jax.vmap(get_library_entropy)

    def body_fn(carry, t):
        (
            key,
            libraries,
            energies,
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
            prev_diversity,
        ) = carry

        # get new keys
        key, death_key, forage_key, policy_key, innov_key = jax.random.split(key, 5)
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
        avg_yields, success_rates, best_recipe_idxs = vmapped_eval_library(
            foraged_plants, foraged_levels, libraries
        )

        # each agent chooses a role based on their q-values
        role_probs = jax.nn.softmax(q_vals / CHOICE_BETA, axis=1)
        roles = jax.random.categorical(policy_key, jnp.log(role_probs), axis=1)

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
            libraries,
            recipe_ids,
            foraged_plants,
            foraged_levels,
            roles,
            best_recipe_idxs,
            recipe_ages,
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

        # compute costs based on foraged plants and role choices
        forage_costs = foraging_cost(foraged_levels.mean(axis=1))
        n_innov, n_imit = (roles == ROLE_INNOVATE).sum(), (roles == ROLE_IMITATE).sum()
        innov_subsidy = imit_fee * n_imit / jnp.maximum(n_innov, 1)
        innov_costs = (innov_cost * size_deltas) - innov_subsidy
        role_costs = jnp.where(roles == ROLE_INNOVATE, innov_costs, imit_fee)
        costs = forage_costs + role_costs

        # update each agent's energy
        delta_energies = avg_yields - costs
        energies = jnp.clip(energies + delta_energies, 0, MAX_ENERGY)

        # update q-values based on reward prediction error
        rewards = yield_deltas - role_costs
        rpe = rewards - q_vals[jnp.arange(n_agents), roles]
        new_q_vals = q_vals.at[jnp.arange(n_agents), roles].add(learning_rate * rpe)

        # compute average reward for each role
        avg_reward_innov = (rewards * (roles == ROLE_INNOVATE)).sum() / jnp.maximum(
            n_innov, 1
        )
        avg_reward_imit = (rewards * (roles == ROLE_IMITATE)).sum() / jnp.maximum(
            n_imit, 1
        )
        avg_rewards = jnp.array([avg_reward_innov, avg_reward_imit])

        # compute population diversity (in terms of libraries)
        should_eval_div = (t % diversity_eval_every) == 0
        pop_diversity = jax.lax.cond(
            should_eval_div,
            lambda libs: compute_population_jaccard(libs)[0],
            lambda _: prev_diversity,
            libraries,
        )

        return (
            key,
            new_libraries,
            energies,
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
            pop_diversity,
        ), (
            foraged_levels.mean(axis=-1),
            avg_yields,
            vmapped_get_library_entropy(libraries),
            avg_rewards,
            pop_diversity,
            roles,
            agent_ages,
        )

    libraries = jnp.tile(initial_library[None, ...], (n_agents, 1, 1))
    energies = jnp.zeros(n_agents, dtype=jnp.float32)
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
    init_diversity, _ = compute_population_jaccard(libraries)
    carry = (
        key,
        libraries,
        energies,
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
        init_diversity,
    )

    carry, metrics = jax.lax.scan(body_fn, carry, jnp.arange(T))

    libraries = carry[1]
    final_agent_ids = carry[5]
    final_next_recipe_id = carry[7]
    final_recipe_ids = carry[8]
    final_recipe_parent_1_ids = carry[10]
    final_recipe_parent_2_ids = carry[11]
    final_recipe_creator_agent_ids = carry[12]
    final_recipe_birth_timesteps = carry[13]
    jaccard_matrix = compute_population_jaccard(libraries)[1]

    return (
        *metrics,
        jaccard_matrix,
        libraries,
        final_recipe_ids,
        final_agent_ids,
        final_recipe_parent_1_ids,
        final_recipe_parent_2_ids,
        final_recipe_creator_agent_ids,
        final_recipe_birth_timesteps,
        final_next_recipe_id,
    )


seeds = list(range(5))
grid_length, T_main, T_extra = 10, int(5e3), 500
T = (
    T_main + T_extra
)  # total timesteps to run (including extra for averaging agent metrics at the end)
fees = 10.0 ** jnp.linspace(-3, 1, 9)


def run_with_fee(key, plants, fee):
    return jax.block_until_ready(
        run_simulation_loop(
            key, plants, grid_length, T, final_phase=T_extra, imit_fee=fee
        )
    )


all_agent_levels = []
all_agent_yields = []
all_agent_lib_entropies = []
all_pop_role_rewards = []
all_pop_diversities = []
all_agent_roles = []
all_agent_ages = []
all_jaccard_matrices = []
all_final_libraries = []
all_final_recipe_ids = []
all_final_agent_ids = []
all_final_next_recipe_ids = []
all_recipe_lineage_arrays = []
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
        pop_diversities,
        agent_roles,
        agent_ages,
        jaccard_matrices,
        final_libraries,
        final_recipe_ids,
        final_agent_ids,
        final_recipe_parent_1_ids,
        final_recipe_parent_2_ids,
        final_recipe_creator_agent_ids,
        final_recipe_birth_timesteps,
        final_next_recipe_ids,
    ) = jax.vmap(run_with_fee, in_axes=(None, None, 0))(key, plants, fees)

    all_agent_levels.append(np.asarray(agent_levels))
    all_agent_yields.append(np.asarray(agent_yields))
    all_agent_lib_entropies.append(np.asarray(agent_lib_entropies))
    all_pop_role_rewards.append(np.asarray(pop_role_rewards))
    all_pop_diversities.append(np.asarray(pop_diversities))
    all_agent_roles.append(np.asarray(agent_roles))
    all_agent_ages.append(np.asarray(agent_ages))
    all_jaccard_matrices.append(jaccard_matrices)
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
            axis=2,
        )
    )

simulation_outputs = {
    "fees": np.asarray(fees),
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
    "agent_lib_entropies": np.stack(all_agent_lib_entropies, axis=0),
    "pop_role_rewards": np.stack(all_pop_role_rewards, axis=0),
    "pop_diversities": np.stack(all_pop_diversities, axis=0),
    "agent_roles": np.stack(all_agent_roles, axis=0),
    "agent_ages": np.stack(all_agent_ages, axis=0),
    "jaccard_matrices": np.stack(all_jaccard_matrices, axis=0),
    "final_libraries": np.stack(all_final_libraries, axis=0),
    "final_recipe_ids": np.stack(all_final_recipe_ids, axis=0),
    "final_agent_ids": np.stack(all_final_agent_ids, axis=0),
    "final_next_recipe_ids": np.stack(all_final_next_recipe_ids, axis=0),
    "recipe_lineage_arrays": np.stack(all_recipe_lineage_arrays, axis=0),
}
np.savez(f"simulation_outputs_{seeds[0]}-{seeds[-1]}.npz", **simulation_outputs)
