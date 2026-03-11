from functools import partial
import time

import jax
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style("whitegrid")

# --- 1. Vocabulary ---
PAD, N, A, B, C = 0, 1, 2, 3, 4
VOCAB = [PAD, N, A, B, C]
VOCAB_JNP = jnp.array(VOCAB, dtype=jnp.int32)

# --- 2. The Environment's "Reverse" Rules ---
# Each key is a token, and the value is a list of possible expansions.
REVERSE_RULES = {
    N: [[A, N, B]],  # Wrap nutrient in soft shell and fiber
    A: [[C, C]],  # Harden the soft shell
    B: [[B, B], [C, B]],  # Thicken the fiber, or add hard nodules to it
}

# JIT-safe rule tables derived from REVERSE_RULES.
MAX_RULES_PER_TOKEN = max(len(REVERSE_RULES.get(token, [])) for token in VOCAB)
MAX_EXPANSION_LEN = max(
    [
        len(expansion)
        for expansions in REVERSE_RULES.values()
        for expansion in expansions
    ]
)
MAX_RULE_LEN = MAX_EXPANSION_LEN
MAX_PLANT_LEN = 30
MAX_RECIPE_LEN = 10
MAX_LIBRARY_SIZE = 50
MAX_ENERGY = 200
BASE_YIELD = 1.0
BACKGROUND_ENERGY_LOSS = 0.5

GOAL_PLANT = jnp.zeros(MAX_PLANT_LEN, dtype=jnp.int32).at[0].set(N)

_rule_lengths = []
_rule_tokens = []
for token in VOCAB:
    expansions = REVERSE_RULES.get(token, [])
    lengths = [len(expansion) for expansion in expansions]
    padded_expansions = [
        expansion + [PAD] * (MAX_EXPANSION_LEN - len(expansion))
        for expansion in expansions
    ]

    pad_rules = MAX_RULES_PER_TOKEN - len(expansions)
    lengths += [0] * pad_rules
    padded_expansions += [[PAD] * MAX_EXPANSION_LEN] * pad_rules

    _rule_lengths.append(lengths)
    _rule_tokens.append(padded_expansions)

RULE_LENGTHS = jnp.array(_rule_lengths, dtype=jnp.int32)
RULE_TOKENS = jnp.array(_rule_tokens, dtype=jnp.int32)
HAS_REVERSE_RULES = jnp.array(
    [token in REVERSE_RULES for token in VOCAB], dtype=jnp.float32
)


# construct initial recipe library from atomic forward rules
atomic_rules = [jnp.zeros((2, MAX_RULE_LEN), dtype=jnp.int32)]
initial_library = jnp.zeros((MAX_LIBRARY_SIZE, MAX_RECIPE_LEN), dtype=jnp.int32)
zeros, counter = jnp.zeros(MAX_RULE_LEN, dtype=jnp.int32), 0
for k, vs in REVERSE_RULES.items():
    rule_result = zeros.at[0].set(k)
    for v in vs:
        rule_target = zeros.at[: len(v)].set(v)

        rule = jnp.stack([rule_target, rule_result], axis=0)
        atomic_rules.append(rule)

        initial_library = initial_library.at[counter, 0].set(counter + 1)
        counter += 1
atomic_rules = jnp.stack(atomic_rules, axis=0)


@partial(jax.jit, static_argnames=["complexity_level"])
def generate_plant(key, complexity_level):
    plant = jnp.zeros(MAX_PLANT_LEN, dtype=jnp.int32).at[0].set(N)  # Start with just N
    actual_length = 1  # Track the actual length of the plant sans padding

    def body_fn(carry, _):
        key, plant, actual_length = carry
        key, idx_key, rule_key = jax.random.split(key, 3)

        p_idx = jnp.zeros(MAX_PLANT_LEN, dtype=jnp.float32)
        for i in range(MAX_PLANT_LEN):
            p_idx = p_idx.at[i].set(HAS_REVERSE_RULES[plant[i]])
        stop = jnp.sum(p_idx) == 0
        p_idx = jnp.where(stop, jnp.ones_like(p_idx), p_idx)
        p_idx /= p_idx.sum()

        target_idx = jax.random.choice(idx_key, MAX_PLANT_LEN, p=p_idx)
        target_token = plant[target_idx]

        token_rule_lengths = RULE_LENGTHS[target_token]
        valid_rule_mask = token_rule_lengths > 0
        valid_rule_probs = valid_rule_mask.astype(jnp.float32)
        valid_rule_probs /= valid_rule_probs.sum()

        rule_idx = jax.random.choice(rule_key, MAX_RULES_PER_TOKEN, p=valid_rule_probs)
        expansion = RULE_TOKENS[target_token, rule_idx]
        expansion_len = token_rule_lengths[rule_idx]

        # Apply the expansion (replace one token with a variable-length expansion)
        new_positions = jnp.arange(MAX_PLANT_LEN, dtype=jnp.int32)

        # Prefix positions copy directly from the original plant.
        src_prefix = new_positions

        # Expansion region sources from expansion[] using target-relative offsets.
        src_expansion = new_positions - target_idx

        # Tail shifts right by (expansion_len - 1) after replacing one token.
        src_tail = new_positions - (expansion_len - 1)

        in_prefix = new_positions < target_idx
        in_expansion = (new_positions >= target_idx) & (
            new_positions < (target_idx + expansion_len)
        )

        src_idx = jnp.where(in_prefix, src_prefix, src_tail)
        src_idx = jnp.clip(src_idx, 0, MAX_PLANT_LEN - 1)
        copied = plant[src_idx]

        expansion_idx = jnp.clip(src_expansion, 0, MAX_EXPANSION_LEN - 1)
        expansion_vals = expansion[expansion_idx]
        expansion_vals = jnp.where(in_expansion, expansion_vals, PAD)

        new_plant: jax.Array = jnp.where(in_expansion, expansion_vals, copied)

        actual_length += expansion_len - 1
        stop: jax.Array = stop | (actual_length >= MAX_PLANT_LEN)
        plant: jax.Array = jnp.where(stop, plant, new_plant)

        return (key, plant, actual_length), None

    carry = (key, plant, actual_length)
    carry, _ = jax.lax.scan(body_fn, carry, jnp.arange(complexity_level))
    _, plant, _ = carry
    return plant


def pregenerate_plants(key, num_per_level, max_level):
    plants = jnp.zeros((max_level + 1, num_per_level, MAX_PLANT_LEN), dtype=jnp.int32)
    for cl in range(max_level + 1):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_per_level)
        tmp = jax.vmap(lambda k: generate_plant(k, cl))(keys)
        plants = plants.at[cl].set(tmp)
    return plants


@partial(jax.jit, static_argnames=["max_level", "n"])
def forage(key, plants, energy, max_level, n):
    curr_max_level_idx = jnp.floor((energy / (MAX_ENERGY + 1)) * max_level).astype(
        jnp.int32
    )

    # Uniform over levels [1 .. curr_max_level_idx], zero otherwise.
    # p_level_idx indexes levels 1..max_level (length max_level).
    idx = jnp.arange(max_level, dtype=jnp.int32)
    active = idx <= curr_max_level_idx

    # uniform over active levels
    p_level_idx = active.astype(jnp.float32)
    p_level_idx /= p_level_idx.sum()

    # small bias toward the current max level
    # max active level index in 0-based space:
    bias_bonus = 0.20  # tune this (e.g. 0.05 to 0.20)
    p_level_idx = p_level_idx.at[curr_max_level_idx].add(bias_bonus)

    # renormalize
    p_level_idx = p_level_idx / p_level_idx.sum()

    def single(key):
        key_level, key_plant = jax.random.split(key)
        level = jax.random.choice(key_level, max_level, p=p_level_idx) + 1
        plant = jax.random.choice(key_plant, plants[level], axis=0)
        return plant, level

    plants_out, levels = jax.vmap(single)(jax.random.split(key, n))
    return plants_out, levels


@jax.jit
def update_forage_history(history, new_foraged):
    history_len = history.shape[0]
    history = (
        jnp.zeros_like(history)
        .at[: history_len - 1]
        .set(history[1:])
        .at[-1]
        .set(new_foraged)
    )
    return history


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
def evaluate_library(plants, library, rule_cost=0.01):
    """
    Evaluates a library of recipes on a batch of plants.

    Args:
        plants: jnp.ndarray[int32] shape (batch_size, MAX_PLANT_LEN), PAD-terminated.
        library: jnp.ndarray[int32] shape (num_recipes, num_rules, 2, MAX_RULE_LEN), where each recipe is a sequence of [target, replacement] stacked and PAD-terminated.

    Returns:
        average yield across batch
    """

    def eval_single_plant(plant):
        processed = jax.vmap(lambda recipe: apply_recipe(plant, recipe))(library)

        # determine if each processed plant matches the goal plant
        successes = jnp.all(processed == GOAL_PLANT, axis=1)

        # scale successful yields by original plant complexity (number of non-PAD tokens)
        plant_length = jnp.sum(plant != PAD)
        yields = successes.astype(jnp.float32) * BASE_YIELD * plant_length

        # calculate the actual length of each recipe (number of non-PAD rules)
        recipe_lengths = jnp.sum(library != PAD, axis=1)
        yields -= rule_cost * recipe_lengths  # penalize longer recipes

        yields = jnp.maximum(yields, 0.0)  # ensure yields are non-negative
        return yields.max()

    yields = jax.vmap(eval_single_plant)(plants)
    return yields.mean(), (yields > 0).mean()  # also return success rate


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
def get_library_entropy(library):
    num_rules = atomic_rules.shape[0]
    transition_counts = jnp.zeros((num_rules, num_rules), dtype=jnp.int32)
    for recipe in library:
        for i in range(recipe.shape[0] - 1):
            rule_from = recipe[i]
            rule_to = recipe[i + 1]
            transition_counts = transition_counts.at[rule_from, rule_to].add(1)

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
    op_probs = jnp.array([0.4, 0.4, 0.2])  # probabilities for add, delete, combine
    op = jax.random.choice(key_op, 3, p=op_probs)

    # sample recipe(s) from the library to modify
    # get the number of actual recipes in the library (those that have at least one non-PAD rule)
    num_recipes = jnp.sum(jnp.any(library != PAD, axis=1))
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
    N_AGENTS, L, R = libraries.shape

    def single_pair_jaccard(lib1, lib2):
        # 1. Identify valid recipes (ignore empty padding)
        # A recipe is valid if it contains at least one non-zero token
        valid1 = jnp.any(lib1 != 0, axis=-1)
        valid2 = jnp.any(lib2 != 0, axis=-1)

        # 2. Handle duplicates WITHIN each library (emulating a Set)
        # self_matches[x, y] is True if lib[x] == lib[y]
        self_matches1 = jnp.all(lib1[:, None, :] == lib1[None, :, :], axis=-1)
        self_matches2 = jnp.all(lib2[:, None, :] == lib2[None, :, :], axis=-1)

        # argmax returns the *first* index of the maximum value (True).
        # This acts as a deduplicator: it only flags the first time a recipe appears.
        is_first1 = jnp.arange(L) == jnp.argmax(self_matches1, axis=1)
        is_first2 = jnp.arange(L) == jnp.argmax(self_matches2, axis=1)

        # The actual "Set" sizes for each agent
        set1_mask = valid1 & is_first1
        set2_mask = valid2 & is_first2
        size1 = jnp.sum(set1_mask)
        size2 = jnp.sum(set2_mask)

        # 3. Find intersection BETWEEN libraries
        # matches[x, y] is True if lib1[x] == lib2[y]
        matches = jnp.all(lib1[:, None, :] == lib2[None, :, :], axis=-1)

        # A unique recipe in lib1 is in lib2 if it matches ANY valid recipe in lib2
        in_lib2 = jnp.any(matches & valid2[None, :], axis=1)

        intersection = jnp.sum(set1_mask & in_lib2)
        union = size1 + size2 - intersection

        # 4. Calculate Jaccard Distance (1 - similarity)
        # Using jnp.where avoids division by zero if both libraries are completely empty
        jaccard_dist = jnp.where(union == 0, 0.0, 1.0 - (intersection / union))
        return jaccard_dist

    # 5. Vectorize across the population
    # vmap across lib2, keeping lib1 fixed
    vmap_lib2 = jax.vmap(single_pair_jaccard, in_axes=(None, 0))

    # vmap across lib1 to generate the full N x N matrix
    vmap_all = jax.vmap(vmap_lib2, in_axes=(0, None))

    dist_matrix = vmap_all(libraries, libraries)

    # 6. Compute mean of the upper triangle
    # (We ignore self-comparisons on the diagonal where distance is 0)
    num_pairs = N_AGENTS * (N_AGENTS - 1) / 2
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
    history_len=10,
):
    max_level = plants.shape[0] - 1

    # (keys, energies) -> foraged_plants, levels
    vmapped_forage = jax.vmap(
        lambda k, e: forage(k, plants, e, max_level, n_forage), in_axes=(0, 0)
    )

    # (histories, foraged_plants) -> updated_histories
    vmapped_update_history = jax.vmap(update_forage_history, in_axes=(0, 0))

    # (foraged_plants, libraries) -> avg yields, success_rates
    vmapped_eval_library = jax.vmap(evaluate_library, in_axes=(0, 0))

    @jax.jit
    def maybe_do_innovation(key, library, history):
        keys = jax.random.split(key, n_innov_attempts)

        # compute avg yield of current library over n-step history
        history_flattened = history.reshape(-1, MAX_PLANT_LEN)
        baseline = evaluate_library(history_flattened, library)[0]

        # produce a set of candidate innovations and evaluate them over the same history
        candidate_libraries = jax.vmap(lambda k: innovate(k, library))(keys)
        yield_deltas = jax.vmap(
            lambda lib: evaluate_library(history_flattened, lib)[0] - baseline
        )(candidate_libraries)
        diff_sizes = jax.vmap(lambda lib: get_diff_size(library, lib))(
            candidate_libraries
        )
        returns = yield_deltas - (innov_cost * diff_sizes)

        # accept the best innovation if it has positive expected return
        best_idx = jnp.argmax(returns)
        new_library = candidate_libraries[best_idx]
        innovation_accepted = returns[best_idx] > 0
        library = jnp.where(innovation_accepted, new_library, library)

        return library, returns[best_idx]

    # (keys, libraries, histories) -> updated_libraries, innov_returns
    vmapped_do_innovation = jax.vmap(maybe_do_innovation, in_axes=(0, 0, 0))

    # (libraries) -> library_sizes
    vmapped_get_library_size = jax.vmap(get_library_size)

    # (libraries) -> library_entropies
    vmapped_get_library_entropy = jax.vmap(get_library_entropy)

    def body_fn(carry, t):
        key, libraries, energies, histories = carry

        # get new keys
        key, forage_key, innov_key = jax.random.split(key, 3)
        forage_keys = jax.random.split(forage_key, n_agents)
        innov_keys = jax.random.split(innov_key, n_agents)

        # each agent forages a batch of plants based on their current energy level
        foraged, _ = vmapped_forage(forage_keys, energies)

        # update each agent's forage history with new batch
        histories = vmapped_update_history(histories, foraged)

        # process each agent's foraged batch with their current library, then update their energy based on yield
        avg_yields, success_rates = vmapped_eval_library(foraged, libraries)
        energies = jnp.clip(
            energies + avg_yields - BACKGROUND_ENERGY_LOSS, 0, MAX_ENERGY
        )

        # each agent attempts an innovation and updates their library if successful
        libraries, expected_innov_returns = vmapped_do_innovation(
            innov_keys, libraries, histories
        )

        # compute population diversity (in terms of libraries)
        pop_diversity, _ = compute_population_jaccard(libraries)

        return (key, libraries, energies, histories), (
            energies,
            avg_yields,
            success_rates,
            vmapped_get_library_size(libraries),
            vmapped_get_library_entropy(libraries),
            expected_innov_returns,
            pop_diversity,
        )

    histories = jnp.zeros(
        (n_agents, history_len, n_forage, MAX_PLANT_LEN), dtype=jnp.int32
    )
    # repeat initial library across agents
    libraries = jnp.tile(initial_library[None, ...], (n_agents, 1, 1))
    energies = jnp.zeros(n_agents, dtype=jnp.float32)
    carry = (key, libraries, energies, histories)

    carry, metrics = jax.lax.scan(body_fn, carry, jnp.arange(T))
    (
        energies,
        yields,
        success_rates,
        library_sizes,
        library_complexities,
        innov_returns,
        diversities,
    ) = metrics

    return (
        energies,
        yields,
        success_rates,
        library_sizes,
        library_complexities,
        innov_returns,
        diversities,
    )


key = jax.random.PRNGKey(0)
max_level = 10
plants = pregenerate_plants(key, num_per_level=100, max_level=max_level)
n_agents, n_forage, T = 10, 10, 1000

start_time = time.time()
(
    energies,
    yields,
    success_rates,
    library_sizes,
    library_complexities,
    innov_returns,
    diversities,
) = jax.block_until_ready(run_simulation_loop(key, plants, n_agents, T, n_forage))
elapsed_time = time.time() - start_time
print(f"Simulation completed in {elapsed_time:.2f} seconds.")

num_innovations = jnp.cumsum((innov_returns > 0).astype(jnp.int32), axis=0)
print(num_innovations.shape)

# compute moving averages
window_size = 50


def single_agent_ma(x):
    cumsum = jnp.cumsum(jnp.insert(x, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


moving_average = jax.vmap(single_agent_ma, in_axes=1, out_axes=1)


t_ma = jnp.arange(T - window_size + 1)
yields_ma = moving_average(yields)
success_rates_ma = moving_average(success_rates)
innov_returns_ma = moving_average(innov_returns)

df = []
for m in range(n_agents):
    for t in range(0, T, 10):
        df.append(
            {
                "agent": m,
                "t": t,
                "t_ma": int(t_ma[t]),
                "energy": float(energies[t, m]),
                "yield": float(yields_ma[t, m]),
                "success_rate": float(success_rates_ma[t, m]),
                "library_size": int(library_sizes[t, m]),
                "library_complexity": float(library_complexities[t, m]),
                "innov_return": float(innov_returns_ma[t, m]),
                "num_innovations": int(num_innovations[t, m]),
                "diversity": float(diversities[t]),
            }
        )
df = pd.DataFrame(df)

for hue in ["agent", None]:
    palette = "crest" if hue else None

    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axs = axs.flatten()

    sns.lineplot(
        df, x="t", y="energy", hue=hue, ax=axs[0], legend=False, palette=palette
    )
    axs[0].set_title("Energy")

    sns.lineplot(
        df, x="t_ma", y="yield", hue=hue, ax=axs[1], legend=False, palette=palette
    )
    axs[1].set_title("Average yield")

    sns.lineplot(
        df,
        x="t_ma",
        y="success_rate",
        hue=hue,
        ax=axs[2],
        legend=False,
        palette=palette,
    )
    axs[2].set_title("Success rate")

    sns.lineplot(
        df,
        x="t",
        y="library_complexity",
        hue=hue,
        ax=axs[3],
        legend=False,
        palette=palette,
    )
    axs[3].set_title("Library complexity")

    sns.lineplot(
        df,
        x="t_ma",
        y="innov_return",
        hue=hue,
        ax=axs[4],
        legend=False,
        palette=palette,
    )
    axs[4].set_title("Expected return of innovation")

    sns.lineplot(
        df,
        x="t",
        y="num_innovations",
        hue=hue,
        ax=axs[5],
        legend=False,
        palette=palette,
    )
    axs[5].set_title("Total number of innovations")

    for ax in axs:
        sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    fig.savefig(f"figures/simulation_plots_{'per_agent' if hue else 'pop_average'}.png")

fig, ax = plt.subplots()
sns.lineplot(df, x="t", y="diversity", ax=ax)
ax.set_title("Population diversity (mean pairwise Jaccard distance)")
sns.despine(ax=ax, left=True, bottom=True)
plt.tight_layout()
fig.savefig("figures/simulation_plots_diversity.png")
