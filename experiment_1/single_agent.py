from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from tqdm import tqdm
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

MAX_ENERGY = 1000
ROLE_NOOP, ROLE_INNOVATE = 0, 1
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


@partial(jax.jit, static_argnames=["n"])
def sample_levels(key, energy, n):
    avg_level = (energy / MAX_ENERGY) * MAX_COMPLEXITY_LEVEL
    lo = jnp.floor(avg_level)
    p = jnp.array([1 - (avg_level - lo), avg_level - lo])
    levels = jax.random.choice(key, jnp.array([lo, lo + 1]), p=p, shape=(n,))
    return jnp.clip(levels, 1, MAX_COMPLEXITY_LEVEL).astype(jnp.int32)


@jax.jit
def update_forage_history(plant_history, level_history, new_plants, new_levels):
    history_len = plant_history.shape[0]
    plant_history = (
        jnp.zeros_like(plant_history)
        .at[: history_len - 1]
        .set(plant_history[1:])
        .at[-1]
        .set(new_plants[-1])
    )
    level_history = (
        jnp.zeros_like(level_history)
        .at[: history_len - 1]
        .set(level_history[1:])
        .at[-1]
        .set(new_levels[-1])
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
        yields = successes.astype(jnp.float32) * plant_value(level)

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
    op = choose_innov_op(key_op)

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
    insert_idx = jnp.where(op == 2, num_recipes % MAX_LIBRARY_SIZE, recipe_idxs[0])
    return library.at[insert_idx].set(new_recipe)


def run_simulation_loop(
    key,
    plants,
    T,
    n_forage=20,
    n_innov_attempts=5,
    innov_cost=0.1,
    history_len=5,
    choice_beta=1.0,
    learning_rate=0.05,
    initial_q_innov=0.0,
):
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

    @jax.jit
    def update_library(key, library, plant_hist, level_hist, role):
        # compute avg yield of current library over n-step history
        plant_hist = plant_hist.reshape(-1, MAX_PLANT_LEN)
        level_hist = level_hist.reshape(-1)
        curr_yield = evaluate_library(plant_hist, level_hist, library)[0]

        def do_innovate(_):
            innov_keys = jax.random.split(key, n_innov_attempts)
            candidate_libraries = jax.vmap(lambda k: innovate(k, library))(innov_keys)
            yields = jax.vmap(
                lambda lib: evaluate_library(plant_hist, level_hist, lib)[0]
            )(candidate_libraries)
            best_innov_idx = yields.argmax()
            return candidate_libraries[best_innov_idx], yields[best_innov_idx]

        def do_nothing(_):
            return library, curr_yield

        # obtain new library (and corresponding yield) based on chosen role
        new_library, new_yield = jax.lax.cond(
            role == ROLE_INNOVATE,
            do_innovate,
            do_nothing,
            operand=None,
        )
        size_delta = get_size_diff(library, new_library)

        # adopt new library if it improves yield
        yield_delta = new_yield - curr_yield
        new_library = jnp.where(yield_delta > 0, new_library, library)
        yield_delta = jnp.where(yield_delta > 0, yield_delta, 0.0)

        return new_library, yield_delta, size_delta

    def body_fn(carry, t):
        (key, library, energy, plant_history, level_history, q_vals) = carry

        # get new keys
        key, forage_key, policy_key, innov_key = jax.random.split(key, 4)

        # agent forages a plant based on their current energy level
        foraged_plants, foraged_levels = forage(forage_key, energy)

        # update agent's forage history with new plant and level
        plant_history, level_history = update_forage_history(
            plant_history, level_history, foraged_plants, foraged_levels
        )

        # process agent's foraged batch with their current library
        avg_yield, success_rate = evaluate_library(
            foraged_plants.reshape(-1, MAX_PLANT_LEN),
            foraged_levels.reshape(-1),
            library,
        )

        # agent chooses a role based on their q-values
        role_probs = jax.nn.softmax(q_vals / choice_beta)
        role = jax.random.categorical(policy_key, jnp.log(role_probs))

        # agent updates their library based on their chosen role
        new_library, yield_delta, size_delta = update_library(
            innov_key, library, plant_history, level_history, role
        )

        # compute cost
        forage_cost = foraging_cost(foraged_levels.mean())
        update_cost = innov_cost * size_delta
        cost = forage_cost + update_cost

        # update agent's energy
        delta_energy = avg_yield - cost
        new_energy = jnp.clip(energy + delta_energy, 0, MAX_ENERGY)

        # update q-values based on reward prediction error
        reward = yield_delta - cost
        rpe = reward - q_vals[role]
        new_q_vals = q_vals.at[role].add(learning_rate * rpe)

        return (
            key,
            new_library,
            new_energy,
            plant_history,
            level_history,
            new_q_vals,
        ), (
            energy,
            foraged_levels.mean(),
            avg_yield,
            success_rate,
            reward,
            get_library_size(library),
            get_library_entropy(library),
            q_vals,
        )

    plant_history = jnp.zeros((history_len, n_forage, MAX_PLANT_LEN), dtype=jnp.int32)
    level_history = jnp.zeros((history_len, n_forage), dtype=jnp.int32)
    library = jnp.copy(initial_library)
    energy = jnp.array(0.0)
    q_vals = jnp.array([0.0, initial_q_innov])

    carry = (
        key,
        library,
        energy,
        plant_history,
        level_history,
        q_vals,
    )

    carry, metrics = jax.lax.scan(body_fn, carry, jnp.arange(T))
    return metrics


n_seeds = 5
T = 4000
plants = pregenerate_plants(
    jax.random.PRNGKey(0), num_per_level=500, max_level=MAX_COMPLEXITY_LEVEL
)

dfs = []
for seed in tqdm(range(n_seeds)):
    key = jax.random.PRNGKey(seed)
    (
        energies,
        levels,
        yields,
        success_rates,
        rewards,
        library_sizes,
        library_entropies,
        q_vals,
    ) = jax.block_until_ready(run_simulation_loop(key, plants, T))
    role_probs = jax.nn.softmax(q_vals / 1.0, axis=1)
    df = pd.DataFrame(
        {
            "t": np.arange(T),
            "energy": energies,
            "level": levels,
            "yield": yields,
            "success_rate": success_rates,
            "reward": rewards,
            "library_size": library_sizes,
            "library_entropy": library_entropies,
            "p_innov": role_probs[:, ROLE_INNOVATE],
        }
    )
    df["seed"] = seed
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df.to_csv("single_agent.csv", index=False)

# df = pd.read_csv("single_agent.csv")

# take moving average of yield and success rate
window = 20
df["yield"] = df["yield"].rolling(window=window).mean()
df["success_rate"] = df["success_rate"].rolling(window=window).mean()

# sample df for easier plotting
df = df[df["t"] % window == 0]

palette = sns.color_palette("husl", n_colors=df["seed"].nunique())
titles = {
    "level": "Average level of foraged plants",
    "yield": "Average yield",
    "success_rate": "Success rate",
    "library_size": "Library size (# rules)",
    "library_entropy": "Library entropy",
    "reward": "Reward",
    "p_innov": "Probability of innovating",
}
metrics = list(titles.keys())

fig, axs = plt.subplots(3, 3, figsize=(10, 8), sharex=True)
axs = axs.flatten()
for i, metric in enumerate(metrics):
    sns.lineplot(
        df, x="t", y=metric, ax=axs[i], hue="seed", palette=palette, legend=False
    )
    axs[i].set_title(titles[metric])
    sns.despine(ax=axs[i], left=True, bottom=True)
fig.tight_layout()
plt.show()


fig, axs = plt.subplots(3, 3, figsize=(10, 8), sharex=True)
axs = axs.flatten()
for i, metric in enumerate(metrics):
    sns.lineplot(df, x="t", y=metric, ax=axs[i], color="black")
    axs[i].set_title(titles[metric])
    sns.despine(ax=axs[i], left=True, bottom=True)
fig.tight_layout()
plt.show()
