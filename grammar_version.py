from functools import partial

import jax
import jax.numpy as jnp

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
MAX_PLANT_LEN = 20
MAX_RECIPE_LEN = 10
MAX_LIBRARY_SIZE = 5

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
        recipe = jnp.stack([jnp.stack([rule_target, rule_result], axis=0)], axis=0)
        recipe = jnp.pad(
            recipe, ((0, MAX_RECIPE_LEN - 1), (0, 0), (0, 0)), constant_values=PAD
        )
        initial_library = initial_library.at[counter].set(recipe)
        counter += 1


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
    plants = jnp.zeros((max_level, num_per_level, MAX_PLANT_LEN), dtype=jnp.int32)
    for cl in range(max_level):
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
    yields = successes.astype(jnp.float32) * complexity

    # calculate the actual length of each recipe (number of non-PAD rules)
    recipe_lengths = jnp.sum(jnp.any(library != PAD, axis=(2, 3)), axis=1)
    yields -= rule_cost * recipe_lengths  # penalize longer recipes

    yields = jnp.maximum(yields, 0.0)  # ensure yields are non-negative
    best_idx = jnp.argmax(yields)
    return yields[best_idx], best_idx


key = jax.random.PRNGKey(0)
plant = generate_plant(key, complexity_level=1)
print("Generated plant:", plant)

best_yield, best_recipe_idx = evaluate_library_on_plant(plant, initial_library)
print("Best yield:", best_yield)
print("Best recipe index:", best_recipe_idx)
