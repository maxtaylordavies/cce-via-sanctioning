from functools import partial

import jax
import jax.numpy as jnp

# --- 1. Vocabulary ---
# N = nutrient, H = husk, S = hard shell, T = toxin, P = spike
PAD, N, H, S, T, P = 0, 1, 2, 3, 4, 5
VOCAB = [PAD, N, H, S, T, P]

# --- 2. The Environment's "Reverse" Rules ---
# Each key is a target subsequence, and the value is a list of possible expansions.
REVERSE_RULES = {
    (N,): [[H, N, H]],  # wrap nutrient in husks
    (H,): [[H, H]],  # double the husk
    (H, H): [[S]],  # fuse adjacent husks into a hard shell
    (S,): [[S, S]],  # add a husk or double the hard shell
    (S, S, S): [
        [S, T, S, T],
        [P, S, S, P],
    ],  # add spikes or toxins around/between triple shells
    (T,): [[T, T]],  # double the toxin
    (P,): [[P, P]],  # double the spike
}

# JIT-safe rule tables derived from REVERSE_RULES.
NUM_REVERSE_RULES = sum(len(expansions) for expansions in REVERSE_RULES.values())
MAX_TARGET_LEN = max(len(target) for target in REVERSE_RULES)
MAX_EXPANSION_LEN = max(
    [
        len(expansion)
        for expansions in REVERSE_RULES.values()
        for expansion in expansions
    ]
)
MAX_RULE_LEN = max(MAX_TARGET_LEN, MAX_EXPANSION_LEN)
MAX_PLANT_LEN = 20
MAX_COMPLEXITY_LEVEL = 10
MAX_RECIPE_LEN = MAX_COMPLEXITY_LEVEL + 5
MAX_LIBRARY_SIZE = 50
EMPTY_RECIPE_ID = -1
NUM_RULES_IN_INITIAL_LIBRARY = 3

GOAL_PLANT = jnp.zeros(MAX_PLANT_LEN, dtype=jnp.int32).at[0].set(N)

_reverse_rule_targets = []
_reverse_rule_target_lengths = []
_reverse_rule_expansions = []
_reverse_rule_expansion_lengths = []
for target, expansions in REVERSE_RULES.items():
    padded_target = list(target) + [PAD] * (MAX_TARGET_LEN - len(target))
    for expansion in expansions:
        _reverse_rule_targets.append(padded_target)
        _reverse_rule_target_lengths.append(len(target))
        _reverse_rule_expansions.append(
            expansion + [PAD] * (MAX_EXPANSION_LEN - len(expansion))
        )
        _reverse_rule_expansion_lengths.append(len(expansion))

RULE_TARGETS = jnp.array(_reverse_rule_targets, dtype=jnp.int32)
RULE_TARGET_LENGTHS = jnp.array(_reverse_rule_target_lengths, dtype=jnp.int32)
RULE_EXPANSIONS = jnp.array(_reverse_rule_expansions, dtype=jnp.int32)
RULE_EXPANSION_LENGTHS = jnp.array(_reverse_rule_expansion_lengths, dtype=jnp.int32)


# construct initial recipe library from atomic forward rules
atomic_rules = [jnp.zeros((2, MAX_RULE_LEN), dtype=jnp.int32)]
initial_library = jnp.zeros((MAX_LIBRARY_SIZE, MAX_RECIPE_LEN), dtype=jnp.int32)
initial_recipe_ids = jnp.full(MAX_LIBRARY_SIZE, EMPTY_RECIPE_ID, dtype=jnp.int32)
zeros, counter = jnp.zeros(MAX_RULE_LEN, dtype=jnp.int32), 0
for target, expansions in REVERSE_RULES.items():
    rule_result = zeros.at[: len(target)].set(jnp.array(target, dtype=jnp.int32))
    for expansion in expansions:
        rule_target = zeros.at[: len(expansion)].set(expansion)
        rule = jnp.stack([rule_target, rule_result], axis=0)
        atomic_rules.append(rule)
        if counter < NUM_RULES_IN_INITIAL_LIBRARY:
            initial_library = initial_library.at[counter, 0].set(counter + 1)
            initial_recipe_ids = initial_recipe_ids.at[counter].set(counter)
        counter += 1
atomic_rules = jnp.stack(atomic_rules, axis=0)


@partial(jax.jit, static_argnames=["complexity_level"])
def generate_plant(key, complexity_level):
    plant = jnp.zeros(MAX_PLANT_LEN, dtype=jnp.int32).at[0].set(N)  # Start with just N
    actual_length = 1  # Track the actual length of the plant sans padding
    positions = jnp.arange(MAX_PLANT_LEN, dtype=jnp.int32)
    offsets = jnp.arange(MAX_TARGET_LEN, dtype=jnp.int32)

    def body_fn(carry, _):
        key, plant, actual_length = carry
        key, idx_key, rule_key = jax.random.split(key, 3)

        idx_matrix = positions[:, None] + offsets[None, :]
        safe_idx_matrix = jnp.clip(idx_matrix, 0, MAX_PLANT_LEN - 1)
        plant_windows = plant[safe_idx_matrix]

        active_target_mask = offsets[None, :] < RULE_TARGET_LENGTHS[:, None]
        token_matches = plant_windows[:, None, :] == RULE_TARGETS[None, :, :]
        window_matches = jnp.all(
            jnp.where(active_target_mask[None, :, :], token_matches, True), axis=2
        )
        can_start = positions[:, None] <= (actual_length - RULE_TARGET_LENGTHS[None, :])
        valid_rule_mask = (
            window_matches & can_start & (RULE_TARGET_LENGTHS[None, :] > 0)
        )

        flat_valid_rule_mask = valid_rule_mask.reshape(-1)
        stop = jnp.sum(flat_valid_rule_mask) == 0
        flat_rule_probs = flat_valid_rule_mask.astype(jnp.float32)
        flat_rule_probs = jnp.where(
            stop, jnp.ones_like(flat_rule_probs), flat_rule_probs
        )
        flat_rule_probs /= flat_rule_probs.sum()

        application_idx = jax.random.choice(
            idx_key, MAX_PLANT_LEN * NUM_REVERSE_RULES, p=flat_rule_probs
        )
        target_idx = application_idx // NUM_REVERSE_RULES
        rule_idx = application_idx % NUM_REVERSE_RULES
        expansion = RULE_EXPANSIONS[rule_idx]
        expansion_len = RULE_EXPANSION_LENGTHS[rule_idx]
        target_len = RULE_TARGET_LENGTHS[rule_idx]

        # Apply the expansion (replace one target subsequence with a variable-length expansion)
        new_positions = jnp.arange(MAX_PLANT_LEN, dtype=jnp.int32)

        # Prefix positions copy directly from the original plant.
        src_prefix = new_positions

        # Expansion region sources from expansion[] using target-relative offsets.
        src_expansion = new_positions - target_idx

        # Tail shifts by the change in subsequence length after replacement.
        src_tail = new_positions - (expansion_len - target_len)

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

        new_plant = jnp.where(in_expansion, expansion_vals, copied)

        new_actual_length = actual_length + expansion_len - target_len
        stop = stop | (new_actual_length >= MAX_PLANT_LEN)

        # If stopped, keep old plant/length; else apply update.
        plant_out = jnp.where(stop, plant, new_plant)
        actual_length_out = jnp.where(stop, actual_length, new_actual_length)

        # Emit the plant state for this step.
        return (key, plant_out, actual_length_out), plant_out

    carry = (key, plant, actual_length)
    carry, history = jax.lax.scan(body_fn, carry, jnp.arange(complexity_level))

    # add the initial plant to the start of the history
    history = jnp.concatenate([plant[None, :], history], axis=0)

    _, final_plant, _ = carry
    return final_plant, history


def pregenerate_plants(key, num_per_level, max_level):
    plants = jnp.zeros((max_level + 1, num_per_level, MAX_PLANT_LEN), dtype=jnp.int32)
    for cl in range(max_level + 1):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_per_level)
        tmp, _ = jax.vmap(lambda k: generate_plant(k, cl))(keys)
        plants = plants.at[cl].set(tmp)
    return plants
