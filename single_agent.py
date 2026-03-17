from functools import partial
import time

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

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


def run_simulation_loop(
    key,
    plants,
    T,
    n_forage,
    n_innov_attempts=1,
    history_len=5,
    choice_beta=1.0,
    learning_rate=0.1,
):
    max_level = plants.shape[0] - 1

    @jax.jit
    def update_library(key, library, plant_history, level_history, innov_cost=0.1):
        # compute avg yield of current library over n-step history
        plant_hist = plant_history.reshape(-1, MAX_PLANT_LEN)
        level_hist = level_history.reshape(-1)
        curr_yield = evaluate_library(plant_hist, level_hist, library)[0]

        # generate candidate innovations and evaluate their yields, then pick the best one
        innov_keys = jax.random.split(key, n_innov_attempts)
        candidate_libraries = jax.vmap(lambda k: innovate(k, library))(innov_keys)
        yields = jax.vmap(lambda lib: evaluate_library(plant_hist, level_hist, lib)[0])(
            candidate_libraries
        )
        best_innov_idx = yields.argmax()
        new_library, new_yield = (
            candidate_libraries[best_innov_idx],
            yields[best_innov_idx],
        )

        # compute the cost of the selected innovation
        cost = innov_cost * get_size_diff(new_library, library)

        # accept the innovation if it improves yield
        accept = new_yield > curr_yield
        library = jnp.where(accept, new_library, library)
        delta_yield = jnp.where(accept, new_yield - curr_yield, 0.0)

        return library, delta_yield, cost

    def body_fn(carry, t):
        (key, library, energy, plant_history, level_history, q_vals) = carry

        # get new keys
        key, forage_key, policy_key, innov_key = jax.random.split(key, 4)

        # agent forages a batch of plants based on their current energy level
        foraged_plants, foraged_levels = forage(
            forage_key, plants, energy, max_level, n_forage
        )

        # update agent's forage history with new batch
        plant_history, level_history = update_forage_history(
            plant_history, level_history, foraged_plants, foraged_levels
        )

        # process agent's foraged batch with their current library
        avg_yield, success_rate = evaluate_library(
            foraged_plants, foraged_levels, library
        )

        # agent chooses a role based on their q-values
        role_probs = jax.nn.softmax(q_vals / choice_beta)
        role = jax.random.categorical(policy_key, jnp.log(role_probs))

        # agent maybe updates their library by innovation
        new_library, delta_yield, cost = update_library(
            innov_key, library, plant_history, level_history
        )
        library = jnp.where(role == 1, new_library, library)

        # update q values
        reward = jnp.where(role == 1, delta_yield - cost, 0.0)
        rpe = reward - q_vals[role]
        new_q_vals = q_vals.at[role].add(learning_rate * rpe)

        # update energy
        energy = jnp.clip(
            energy + avg_yield - BACKGROUND_ENERGY_LOSS - cost, 0, MAX_ENERGY
        )

        return (key, library, energy, plant_history, level_history, new_q_vals), (
            energy,
            avg_yield,
            success_rate,
            get_library_size(library),
            get_library_entropy(library),
            foraged_levels.mean(),
            q_vals,
        )

    plant_history = jnp.zeros((history_len, n_forage, MAX_PLANT_LEN), dtype=jnp.int32)
    level_history = jnp.zeros((history_len, n_forage), dtype=jnp.int32)
    library = jnp.copy(initial_library)
    energy = jnp.zeros((), dtype=jnp.float32)
    q_vals = jnp.array([0.0, 10.0])
    carry = (key, library, energy, plant_history, level_history, q_vals)

    carry, metrics = jax.lax.scan(body_fn, carry, jnp.arange(T))
    return metrics


start_time = time.time()

key = jax.random.PRNGKey(0)
T, max_level, num_per_level, n_forage = 2000, 20, 500, 10
(
    energies,
    yields,
    success_rates,
    library_sizes,
    library_complexities,
    avg_levels,
    q_vals,
) = jax.block_until_ready(
    run_simulation_loop(
        key,
        pregenerate_plants(key, num_per_level, max_level),
        T=T,
        n_forage=n_forage,
    )
)

elapsed = time.time() - start_time
print(f"Simulation completed in {elapsed:.2f} seconds")

# convert success rates and avg levels to moving averages for smoother plots
window_size = 10
success_rates = (
    pd.Series(success_rates).rolling(window_size, min_periods=1).mean().values
)
avg_levels = pd.Series(avg_levels).rolling(window_size, min_periods=1).mean().values

# convert q vals to probabilities
print(q_vals.shape)
print(q_vals[:10])
print(q_vals[-1])
role_probs = jax.nn.softmax(q_vals, axis=1)
p_innovate = role_probs[:, 1]

# plot energy and success rate over time on the same graph
metrics_ = {
    "Energy": energies,
    "Success rate": success_rates,
    "Average foraged level": avg_levels,
    "Pr[innovate]": p_innovate,
}
fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
for ax, (metric_name, metric_values) in zip(axs.flatten(), metrics_.items()):
    sns.lineplot(x=np.arange(T), y=metric_values, ax=ax, color="black")
    ax.set_title(metric_name)
    sns.despine(ax=ax, left=True, bottom=True)

energy_ax = axs.flatten()[0]
for level in range(1, max_level + 1):
    energy_threshold = MAX_ENERGY * (level - 1) / max_level
    next_threshold = MAX_ENERGY * level / max_level
    label_y = 0.5 * (energy_threshold + next_threshold)
    energy_ax.axhline(
        energy_threshold,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.25,
        zorder=0,
    )
    energy_ax.text(
        T - 1,
        label_y,
        f"level {level} ",
        ha="right",
        va="center",
        fontsize=7,
        color="black",
        alpha=0.6,
    )

fig.tight_layout()
plt.show()
