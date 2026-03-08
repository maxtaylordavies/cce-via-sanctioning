from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

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


@partial(jax.jit, static_argnames=["complexity_level", "max_length"])
def generate_plant(key, complexity_level, max_length=20):
    plant = jnp.zeros(max_length, dtype=jnp.int32).at[0].set(N)  # Start with just N
    actual_length = 1  # Track the actual length of the plant sans padding

    def body_fn(carry, _):
        key, plant, actual_length = carry
        key, idx_key, rule_key = jax.random.split(key, 3)

        p_idx = jnp.zeros(max_length, dtype=jnp.float32)
        for i in range(max_length):
            p_idx = p_idx.at[i].set(HAS_REVERSE_RULES[plant[i]])
        stop = jnp.sum(p_idx) == 0
        p_idx = jnp.where(stop, jnp.ones_like(p_idx), p_idx)
        p_idx /= p_idx.sum()

        target_idx = jax.random.choice(idx_key, max_length, p=p_idx)
        target_token = plant[target_idx]

        token_rule_lengths = RULE_LENGTHS[target_token]
        valid_rule_mask = token_rule_lengths > 0
        valid_rule_probs = valid_rule_mask.astype(jnp.float32)
        valid_rule_probs /= valid_rule_probs.sum()

        rule_idx = jax.random.choice(rule_key, MAX_RULES_PER_TOKEN, p=valid_rule_probs)
        expansion = RULE_TOKENS[target_token, rule_idx]
        expansion_len = token_rule_lengths[rule_idx]

        # Apply the expansion (replace one token with a variable-length expansion)
        new_positions = jnp.arange(max_length, dtype=jnp.int32)

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
        src_idx = jnp.clip(src_idx, 0, max_length - 1)
        copied = plant[src_idx]

        expansion_idx = jnp.clip(src_expansion, 0, MAX_EXPANSION_LEN - 1)
        expansion_vals = expansion[expansion_idx]
        expansion_vals = jnp.where(in_expansion, expansion_vals, PAD)

        new_plant: jax.Array = jnp.where(in_expansion, expansion_vals, copied)

        actual_length += expansion_len - 1
        stop: jax.Array = stop | (actual_length >= max_length)
        plant: jax.Array = jnp.where(stop, plant, new_plant)

        return (key, plant, actual_length), None

    carry = (key, plant, actual_length)
    carry, _ = jax.lax.scan(body_fn, carry, jnp.arange(complexity_level))
    _, plant, _ = carry
    return plant


key = jax.random.PRNGKey(0)

for cl in range(20):
    plant = generate_plant(key, complexity_level=cl)
    print(f"Level {cl} Plant: {plant}")
