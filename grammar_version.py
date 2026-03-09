from functools import partial

import jax
import jax.numpy as jnp
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

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
MAX_ENERGY = 100
BASE_YIELD = 0.5
BACKGROUND_ENERGY_LOSS = 0.2

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
initial_library = jnp.zeros(
    (MAX_LIBRARY_SIZE, MAX_RECIPE_LEN, 2, MAX_RULE_LEN), dtype=jnp.int32
)
zeros, counter = jnp.zeros(MAX_RULE_LEN, dtype=jnp.int32), 0
for k, vs in REVERSE_RULES.items():
    rule_result = zeros.at[0].set(k)
    for v in vs:
        rule_target = zeros.at[: len(v)].set(v)
        initial_library = initial_library.at[counter, 0, 0].set(rule_target)
        initial_library = initial_library.at[counter, 0, 1].set(rule_result)
        counter += 1
NUM_ATOMIC_RULES = counter


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

    def body_fn(plant, rule):
        target, replacement = rule
        return apply_rule(plant, target, replacement), None

    return jax.lax.scan(body_fn, plant, recipe)[0]


@jax.jit
def evaluate_library_on_plant(plant, library, rule_cost=0.1):
    """
    Evaluates a library of recipes on a single plant.

    Args:
        plant: jnp.ndarray[int32] shape (MAX_PLANT_LEN,), PAD-terminated.
        library: jnp.ndarray[int32] shape (num_recipes, num_rules, 2, MAX_RULE_LEN), where each recipe is a sequence of [target, replacement] stacked and PAD-terminated.

    Returns:
        jnp.ndarray[float32] shape (), yield from best recipe
        jnp.ndarray[int32] shape (), index of best recipe
    """
    processed = jax.vmap(lambda recipe: apply_recipe(plant, recipe))(library)

    # determine if each processed plant matches the goal plant
    successes = jnp.all(processed == GOAL_PLANT, axis=1)

    # scale successful yields by original plant complexity (number of non-PAD tokens)
    complexity = jnp.sum(plant != PAD)
    yields = successes.astype(jnp.float32) * BASE_YIELD * complexity

    # calculate the actual length of each recipe (number of non-PAD rules)
    recipe_lengths = jnp.sum(jnp.any(library != PAD, axis=(2, 3)), axis=1)
    yields -= rule_cost * recipe_lengths  # penalize longer recipes

    yields = jnp.maximum(yields, 0.0)  # ensure yields are non-negative
    best_idx = jnp.argmax(yields)
    return yields[best_idx], best_idx


@jax.jit
def innovate_extend(key, library):
    key_recipe, key_rule, key_insert = jax.random.split(key, 3)

    # get the number of actual recipes in the library (those that have at least one non-PAD rule)
    num_recipes = jnp.sum(jnp.any(library != PAD, axis=(1, 2, 3)))
    library_full = num_recipes >= MAX_LIBRARY_SIZE

    # sample a recipe to extend
    recipe_idx = jax.random.randint(key_recipe, (), 0, num_recipes)

    # sample an atomic rule to add
    atomic_rule_idx = jax.random.randint(key_rule, (), 0, NUM_ATOMIC_RULES)
    atomic_rule = initial_library[atomic_rule_idx, 0]

    # find the first empty slot in the recipe
    num_rules = jnp.sum(jnp.any(library[recipe_idx] != PAD, axis=(1, 2)))
    recipe_full = num_rules >= MAX_RECIPE_LEN
    insert_idx = num_rules % MAX_RECIPE_LEN

    # if the recipe isn't full, insert the selected rule
    new_recipe = library[recipe_idx].at[insert_idx].set(atomic_rule)
    new_recipe = jnp.where(recipe_full, library[recipe_idx], new_recipe)

    # if the library is full, select a random index to overwrite with the new recipe
    overwrite_idx = jax.random.randint(key_insert, (), 0, num_recipes)
    insert_idx = jnp.where(library_full, overwrite_idx, num_recipes)
    return library.at[insert_idx].set(new_recipe)


@partial(jax.jit, static_argnames=["max_level"])
def forage(key, plants, energy, max_level):
    key_level, key_plant = jax.random.split(key)

    # energy is between 0 and MAX_ENERGY
    # we want to bias foraging such that:
    #   - at energy=0, p(level=0) = 1
    #   - at energy=MAX_ENERGY, p(level=max_level) = 1
    #   - p(level) should increase smoothly with energy, e.g. linearly interpolating between these extremes
    m = (max_level * energy) / MAX_ENERGY
    k = jnp.floor(m).astype(jnp.int32)
    alpha = m - k
    k = jnp.minimum(k, max_level)
    p_level = jnp.zeros(max_level + 1).at[k].set(1.0 - alpha)
    p_level = jax.lax.cond(
        k < max_level, lambda p: p.at[k + 1].set(alpha), lambda p: p, p_level
    )

    level = jax.random.choice(key_level, max_level + 1, p=p_level)
    plant = jax.random.choice(key_plant, plants[level], axis=0)
    return plant, level


def run_simulation_loop(key, plants, T):
    max_level = plants.shape[0] - 1

    def body_fn(carry, t):
        key, library, energy = carry
        key, forage_key, innovate_key = jax.random.split(key, 3)

        plant, _ = forage(forage_key, plants, energy, max_level)

        best_yield, _ = evaluate_library_on_plant(plant, library)
        energy = jnp.clip(energy + best_yield - BACKGROUND_ENERGY_LOSS, 0, MAX_ENERGY)

        new_library = innovate_extend(innovate_key, library)
        library = jnp.where(best_yield > 0, library, new_library)

        # compute average length of non-empty recipes in the library
        recipe_lengths = jnp.sum(jnp.any(library != PAD, axis=(2, 3)), axis=1)
        num_non_empty = jnp.sum(recipe_lengths > 0)
        avg_length = recipe_lengths.sum() / jnp.maximum(num_non_empty, 1)

        return (key, library, energy), (energy, best_yield, num_non_empty, avg_length)

    carry = (key, initial_library, 0)
    carry, metrics = jax.lax.scan(body_fn, carry, jnp.arange(T))
    energies, yields, num_recipes, avg_recipe_lengths = metrics
    return energies, yields, num_recipes, avg_recipe_lengths


key = jax.random.PRNGKey(0)
max_level = 6
plants = pregenerate_plants(key, num_per_level=100, max_level=max_level)

T = 2000
energies, yields, num_recipes, avg_recipe_lengths = run_simulation_loop(key, plants, T)

# compute moving average of yields for smoother visualization
window_size = 20
cumsum = jnp.cumsum(jnp.insert(yields, 0, 0))
moving_avg_yields = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
axs = axs.flatten()
sns.lineplot(x=jnp.arange(T), y=energies, ax=axs[0])
axs[0].set_title("Energy Over Time")
sns.lineplot(x=jnp.arange(T - window_size + 1), y=moving_avg_yields, ax=axs[1])
axs[1].set_title("Yield per Forage (Moving Average)")
sns.lineplot(x=jnp.arange(T), y=num_recipes, ax=axs[2])
axs[2].set_title("Number of Recipes in Library")
sns.lineplot(x=jnp.arange(T), y=avg_recipe_lengths, ax=axs[3])
axs[3].set_title("Average Recipe Length in Library")

for ax in axs:
    sns.despine(ax=ax, left=True, bottom=True)

plt.tight_layout()
plt.show()
