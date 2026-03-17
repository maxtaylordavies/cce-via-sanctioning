from functools import partial
import time

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np

from grammar import (
    PAD,
    MAX_LIBRARY_SIZE,
    MAX_RECIPE_LEN,
    MAX_PLANT_LEN,
    MAX_RULE_LEN,
    MAX_COMPLEXITY_LEVEL,
    GOAL_PLANT,
    pregenerate_plants,
    atomic_rules,
    initial_library,
)

MAX_ENERGY = 500
BASE_YIELD = 1.0
BACKGROUND_ENERGY_LOSS = 0.5

ROLE_INNOVATE, ROLE_IMITATE = 0, 1


@partial(jax.jit, static_argnames=["max_level", "n"])
def forage(key, plants, energy, max_level, n):
    level = 1 + jnp.floor((energy / MAX_ENERGY) * max_level)
    level = jnp.clip(level, 1, max_level).astype(jnp.int32)
    plants_out = jax.random.choice(key, plants[level], shape=(n,), axis=0)
    levels = jnp.full((n,), level, dtype=jnp.int32)
    return plants_out, levels


@jax.jit
def update_forage_history(plant_history, level_history, new_plants, new_levels):
    history_len = plant_history.shape[0]
    plant_history = (
        jnp.zeros_like(plant_history)
        .at[: history_len - 1]
        .set(plant_history[1:])
        .at[-1]
        .set(new_plants)
    )
    level_history = (
        jnp.zeros_like(level_history)
        .at[: history_len - 1]
        .set(level_history[1:])
        .at[-1]
        .set(new_levels)
    )
    return plant_history, level_history


@jax.jit
def apply_rule(
    plant,
    target,
    replacement,
):
    """
    Applies one forward rule to a padded plant token array.

    The first occurrence of `target` is replaced by `replacement`.
    If no occurrence is found, `plant` is returned unchanged.

    Args:
        plant: jnp.ndarray[int32] shape (MAX_PLANT_LEN,), PAD-terminated.
        target: jnp.ndarray[int32] shape (MAX_RULE_LEN,), PAD-terminated.
        replacement: jnp.ndarray[int32] shape (max_replacement_len,), PAD-terminated.
        MAX_PLANT_LEN: Static maximum sequence length.
        MAX_RULE_LEN: Static padded length of `target`.
        max_replacement_len: Static padded length of `replacement`.

    Returns:
        jnp.ndarray[int32] shape (MAX_PLANT_LEN,), updated plant.
    """
    positions = jnp.arange(MAX_PLANT_LEN, dtype=jnp.int32)

    plant_len = jnp.sum(plant != PAD)
    target_len = jnp.sum(target != PAD)
    replacement_len = jnp.sum(replacement != PAD)

    # Candidate start indices that can fit target in current (unpadded) plant.
    can_start = positions <= (plant_len - target_len)

    # Check substring equality at each candidate start.
    offsets = jnp.arange(MAX_RULE_LEN, dtype=jnp.int32)
    idx_matrix = positions[:, None] + offsets[None, :]
    safe_idx_matrix = jnp.clip(idx_matrix, 0, MAX_PLANT_LEN - 1)
    plant_windows = plant[safe_idx_matrix]

    active_target_mask = offsets < target_len
    token_match = plant_windows == target[None, :]
    window_match = jnp.all(
        jnp.where(active_target_mask[None, :], token_match, True), axis=1
    )
    matches = can_start & window_match & (target_len > 0)

    has_match = jnp.any(matches)
    first_match = jnp.argmax(matches.astype(jnp.int32))

    delta = replacement_len - target_len

    # Build new sequence by source-index remap and replacement overlay.
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
    """
    Applies a sequence of rules to a plant.

    Args:
        plant: jnp.ndarray[int32] shape (MAX_PLANT_LEN,), PAD-terminated.
        recipe: jnp.ndarray[int32] shape (num_rules, 2, MAX_RULE_LEN), where each row is [target, replacement] stacked and PAD-terminated.
        MAX_PLANT_LEN: Static maximum sequence length.

    Returns:
        jnp.ndarray[int32] shape (MAX_PLANT_LEN,), updated plant after applying all rules in the recipe.
    """

    def body_fn(plant, rule_idx):
        target, replacement = atomic_rules[rule_idx]
        return apply_rule(plant, target, replacement), None

    return jax.lax.scan(body_fn, plant, recipe)[0]


@jax.jit
def evaluate_library(plants, levels, library, rule_cost=0.05):
    """
    Evaluates a library of recipes on a batch of plants.

    Args:
        plants: jnp.ndarray[int32] shape (batch_size, MAX_PLANT_LEN), PAD-terminated.
        levels: jnp.ndarray[int32] shape (batch_size,), complexity levels for each plant.
        library: jnp.ndarray[int32] shape (num_recipes, num_rules, 2, MAX_RULE_LEN), where each recipe is a sequence of [target, replacement] stacked and PAD-terminated.

    Returns:
        average yield across batch
    """

    recipe_lengths = jnp.sum(library != PAD, axis=1)

    def eval_single_plant(plant, level):
        processed = jax.vmap(lambda recipe: apply_recipe(plant, recipe))(library)

        # determine if each processed plant matches the goal plant
        successes = jnp.all(processed == GOAL_PLANT, axis=1)

        # scale successful yields by original plant complexity
        yields = successes.astype(jnp.float32) * BASE_YIELD * level

        # calculate the actual length of each recipe (number of non-PAD rules)
        yields -= rule_cost * recipe_lengths  # penalize longer recipes

        yields = jnp.maximum(yields, 0.0)  # ensure yields are non-negative
        return yields.max()

    yields = jax.vmap(eval_single_plant, in_axes=(0, 0))(plants, levels)
    return yields.mean(), (yields > 0).mean()  # also return success rate


@jax.jit
def get_recipe_length(recipe):
    return jnp.sum(recipe != PAD)


@jax.jit
def get_library_size(library):
    return jax.vmap(get_recipe_length)(library).sum()


@jax.jit
def get_size_diff(library_1, library_2):
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
def innovate(key, library):
    key_op, key_recipe = jax.random.split(key)

    # select which type of operation to perform: add a rule, delete a rule, or combine two recipes
    op_probs = jnp.ones(3) / 3  # probabilities for add, delete, combine
    op = jax.random.choice(key_op, 3, p=op_probs)

    # sample recipe(s) from the library to modify
    # get the number of actual recipes in the library (those that have at least one non-PAD rule)
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
    insert_idx = num_recipes % MAX_LIBRARY_SIZE
    return library.at[insert_idx].set(new_recipe)


@partial(jax.jit, static_argnames=["n_agents"])
def imitate_recipe(key, libraries, agent_idx, n_agents):
    key_agent, key_recipe = jax.random.split(key)

    # select an agent to imitate from (uniform over all other agents)
    p_agent = jnp.ones(n_agents).at[agent_idx].set(0.0)
    p_agent /= p_agent.sum()
    demonstrator_idx = jax.random.choice(key_agent, n_agents, p=p_agent)

    # select a recipe from the demonstrator's library to imitate
    num_recipes = get_num_recipes(libraries[demonstrator_idx])
    p_recipe = jnp.ones(MAX_LIBRARY_SIZE) * (jnp.arange(MAX_LIBRARY_SIZE) < num_recipes)
    p_recipe /= p_recipe.sum()
    recipe_idx = jax.random.choice(key_recipe, MAX_LIBRARY_SIZE, p=p_recipe)

    # determine where to insert the imitated recipe in the imitator's library
    insert_idx = get_num_recipes(libraries[agent_idx]) % MAX_LIBRARY_SIZE

    # insert the imitated recipe and return the updated library
    return (
        libraries[agent_idx].at[insert_idx].set(libraries[demonstrator_idx, recipe_idx])
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


def run_simulation_loop(
    key,
    plants,
    n_agents,
    T,
    n_forage,
    n_innov_attempts=1,
    innov_cost=0.1,
    imit_cost=0.05,
    history_len=10,
    choice_beta=1.0,
    learning_rate=0.01,
    diversity_eval_every=10,
):
    max_level = plants.shape[0] - 1

    # (keys, energies) -> foraged_plants, levels
    vmapped_forage = jax.vmap(
        lambda k, e: forage(k, plants, e, max_level, n_forage), in_axes=(0, 0)
    )

    # (plant_histories, level_histories, foraged_plants, foraged_levels) -> updated_histories
    vmapped_update_history = jax.vmap(update_forage_history, in_axes=(0, 0, 0, 0))

    # (foraged_plants, levels, libraries) -> avg yields, success_rates
    vmapped_eval_library = jax.vmap(evaluate_library, in_axes=(0, 0, 0))

    @jax.jit
    def update_library(
        key,
        agent_idx,
        libraries,
        plant_histories,
        level_histories,
        roles,
        innov_cost,
        imit_cost,
    ):
        # ROLE KEY: 0 = innovate, 1 = imitate

        # compute avg yield of current library over n-step history
        plant_hist = plant_histories[agent_idx].reshape(-1, MAX_PLANT_LEN)
        level_hist = level_histories[agent_idx].reshape(-1)
        curr_yield = evaluate_library(plant_hist, level_hist, libraries[agent_idx])[0]

        def do_innovate(_):
            innov_keys = jax.random.split(key, n_innov_attempts)
            candidate_libraries = jax.vmap(lambda k: innovate(k, libraries[agent_idx]))(
                innov_keys
            )
            yields = jax.vmap(
                lambda lib: evaluate_library(plant_hist, level_hist, lib)[0]
            )(candidate_libraries)
            best_innov_idx = yields.argmax()
            return candidate_libraries[best_innov_idx], yields[best_innov_idx]

        def do_imitate(_):
            imitation_library = imitate_recipe(key, libraries, agent_idx, n_agents)
            imitation_yield = evaluate_library(
                plant_hist, level_hist, imitation_library
            )[0]
            return imitation_library, imitation_yield

        # obtain new library (and corresponding yield) based on chosen role
        new_library, new_yield = jax.lax.cond(
            roles[agent_idx] == ROLE_INNOVATE,
            do_innovate,
            do_imitate,
            operand=None,
        )

        # compute cost of update based on role
        cost = jax.lax.switch(
            roles[agent_idx],
            [
                lambda: innov_cost * get_size_diff(libraries[agent_idx], new_library),
                lambda: imit_cost,
            ],
        )

        # adopt new library if it improves yield
        yield_delta = new_yield - curr_yield
        new_library = jnp.where(yield_delta > 0, new_library, libraries[agent_idx])
        yield_delta = jnp.where(yield_delta > 0, yield_delta, 0.0)

        return new_library, yield_delta, cost

    # (keys, agent_idxs, libraries, plant_histories, level_histories, roles, innov_cost, imit_cost) -> updated_libraries, yield_deltas, costs
    vmapped_update_library = jax.vmap(
        update_library, in_axes=(0, 0, None, None, None, None, None, None)
    )

    # (libraries) -> library_entropies
    vmapped_get_library_entropy = jax.vmap(get_library_entropy)

    def body_fn(carry, t):
        (
            key,
            libraries,
            energies,
            plant_histories,
            level_histories,
            q_vals,
            prev_diversity,
        ) = carry

        # get new keys
        key, forage_key, policy_key, innov_key = jax.random.split(key, 4)
        forage_keys = jax.random.split(forage_key, n_agents)
        innov_keys = jax.random.split(innov_key, n_agents)

        # each agent forages a batch of plants based on their current energy level
        foraged_plants, foraged_levels = vmapped_forage(forage_keys, energies)

        # update each agent's forage history with new batch
        plant_histories, level_histories = vmapped_update_history(
            plant_histories, level_histories, foraged_plants, foraged_levels
        )

        # process each agent's foraged batch with their current library
        avg_yields, _ = vmapped_eval_library(foraged_plants, foraged_levels, libraries)

        # each agent chooses a role based on their q-values
        role_probs = jax.nn.softmax(q_vals / choice_beta, axis=1)
        roles = jax.random.categorical(policy_key, jnp.log(role_probs), axis=1)

        # each agent updates their library based on their chosen role
        new_libraries, yield_deltas, costs = vmapped_update_library(
            innov_keys,
            jnp.arange(n_agents),
            libraries,
            plant_histories,
            level_histories,
            roles,
            innov_cost,
            imit_cost,
        )

        # update each agent's energy
        energies = jnp.clip(
            energies + avg_yields - costs - BACKGROUND_ENERGY_LOSS, 0, MAX_ENERGY
        )

        # update q-values based on reward prediction error
        rewards = yield_deltas - costs
        rpe = rewards - q_vals[jnp.arange(n_agents), roles]
        q_vals = q_vals.at[jnp.arange(n_agents), roles].add(learning_rate * rpe)

        # compute population diversity (in terms of libraries)
        libraries = new_libraries
        should_eval_div = (t % diversity_eval_every) == 0
        pop_diversity = jax.lax.cond(
            should_eval_div,
            lambda libs: compute_population_jaccard(libs)[0],
            lambda _: prev_diversity,
            libraries,
        )

        return (
            key,
            libraries,
            energies,
            plant_histories,
            level_histories,
            q_vals,
            pop_diversity,
        ), (
            energies,
            foraged_levels.mean(axis=1),
            avg_yields,
            vmapped_get_library_entropy(libraries),
            q_vals,
            pop_diversity,
        )

    plant_histories = jnp.zeros(
        (n_agents, history_len, n_forage, MAX_PLANT_LEN), dtype=jnp.int32
    )
    level_histories = jnp.zeros((n_agents, history_len, n_forage), dtype=jnp.int32)
    libraries = jnp.tile(initial_library[None, ...], (n_agents, 1, 1))
    energies = jnp.zeros(n_agents, dtype=jnp.float32)
    q_vals = 10.0 * jnp.ones(
        (n_agents, 2), dtype=jnp.float32
    )  # optimistic initial values for innovate and imitate

    init_diversity, _ = compute_population_jaccard(libraries)
    carry = (
        key,
        libraries,
        energies,
        plant_histories,
        level_histories,
        q_vals,
        init_diversity,
    )

    carry, metrics = jax.lax.scan(body_fn, carry, jnp.arange(T))
    return metrics


key = jax.random.PRNGKey(0)
plants = pregenerate_plants(key, num_per_level=500, max_level=MAX_COMPLEXITY_LEVEL)
n_agents, n_forage, T = 5, 10, 200

start_time = time.time()
(energies, foraged_levels, yields, library_complexities, q_vals, diversities) = (
    jax.block_until_ready(run_simulation_loop(key, plants, n_agents, T, n_forage))
)
elapsed_time = time.time() - start_time
print(f"Simulation completed in {elapsed_time:.2f} seconds.")

# construct dataframe
start_time = time.time()
single_agent_ma = lambda x: pd.Series(x).rolling(window=20, min_periods=1).mean().values
moving_average = np.vectorize(single_agent_ma, signature="(t)->(t)")
yields_ma = moving_average(yields)

sample_ts = np.arange(0, T, 10)
flat_a = np.repeat(np.arange(n_agents), sample_ts.size)
flat_t = np.tile(sample_ts, n_agents)

df = pd.DataFrame(
    {
        "agent": flat_a,
        "t": flat_t,
        "energy": energies[flat_t, flat_a],
        "yield": yields_ma[flat_t, flat_a],
        "foraged_level": foraged_levels[flat_t, flat_a],
        "library_complexity": library_complexities[flat_t, flat_a],
        "diversity": diversities[flat_t],
        "q_innovate": q_vals[flat_t, flat_a, ROLE_INNOVATE],
        "q_imitate": q_vals[flat_t, flat_a, ROLE_IMITATE],
    }
)

p_innov = np.exp(df["q_innovate"] / 1.0)
p_imit = np.exp(df["q_imitate"] / 1.0)
df["p_innovate"] = p_innov / (p_innov + p_imit)
df["p_imitate"] = p_imit / (p_innov + p_imit)

elapsed_time = time.time() - start_time
print(f"Dataframe construction completed in {elapsed_time:.2f} seconds.")

# save df
df.to_csv("simulation_data.csv", index=False)
