"""Experiment 1: value capture in the binary-trait environment.

This implements Equations 11-12 of the accompanying paper at the level of
individual imitation events.  When an agent copies a trait, ``value_capture_rate``
(lambda in the paper) of the trait's value is transferred to the agent who
originated that copy lineage.  The imitator retains the rest.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

ROLE_INNOVATE, ROLE_IMITATE = 0, 1
NO_CREATOR = -1
L = 100
MAX_TOTAL_L = 5000


@jax.jit
def compute_trait_payoffs(all_traits, trait_value):
    """Return agents' gross material payoffs from their known traits."""
    return all_traits.sum(axis=1) * trait_value


@jax.jit
def innovate(key, traits, trait_creators, agent_idx, curr_L, p_i):
    """Try to discover one trait and record this agent as its originator."""
    idx_key, success_key = jax.random.split(key)
    trait_idx = jax.random.randint(idx_key, (), 0, curr_L)
    success = (traits[trait_idx] == 0) & jax.random.bernoulli(success_key, p_i)

    new_traits = traits.at[trait_idx].set(jnp.where(success, 1, traits[trait_idx]))
    new_trait_creators = trait_creators.at[trait_idx].set(
        jnp.where(
            success,
            jnp.asarray(agent_idx, dtype=trait_creators.dtype),
            trait_creators[trait_idx],
        )
    )
    return new_traits, new_trait_creators, success.astype(jnp.int32)


@jax.jit
def choose_demonstrator(key, can_imitate, agent_idx):
    """Choose uniformly among an agent's available social-learning partners."""
    weights = jnp.where(can_imitate[agent_idx], 1.0, 0.0)

    # This fallback only matters for parameterisations with an empty
    # neighbourhood.  It preserves the previous experiment's behaviour.
    fallback_mask = jnp.arange(can_imitate.shape[0]) != agent_idx
    fallback_weights = jnp.where(fallback_mask, 1.0, 0.0)
    weights = jnp.where(weights.sum() > 0, weights, fallback_weights)
    logits = jnp.where(weights > 0, jnp.log(weights), -jnp.inf)
    return jax.random.categorical(key, logits)


@jax.jit
def imitate(
    key,
    all_traits,
    all_trait_creators,
    can_imitate,
    agent_idx,
    p_c,
):
    """Copy a demonstrator's traits and preserve each trait's provenance."""
    demonstrator_key, copy_key = jax.random.split(key)
    demonstrator_idx = choose_demonstrator(demonstrator_key, can_imitate, agent_idx)
    copied_traits = all_traits[demonstrator_idx] & jax.random.bernoulli(
        copy_key, p_c, shape=all_traits.shape[1:]
    )
    newly_copied = copied_traits & ~all_traits[agent_idx].astype(jnp.bool_)

    new_traits = all_traits[agent_idx] | copied_traits
    new_trait_creators = jnp.where(
        newly_copied,
        all_trait_creators[demonstrator_idx],
        all_trait_creators[agent_idx],
    )
    copied_creator_ids = jnp.where(
        newly_copied,
        all_trait_creators[demonstrator_idx],
        NO_CREATOR,
    )
    return (
        new_traits,
        new_trait_creators,
        newly_copied.sum(),
        copied_creator_ids,
    )


@jax.jit
def invalidate_dead_creators(all_trait_creators, deaths):
    """Expire value-capture claims belonging to agents who have just died."""
    attributed = all_trait_creators != NO_CREATOR
    safe_creator_ids = jnp.where(attributed, all_trait_creators, 0)
    creator_died = deaths[safe_creator_ids]
    return jnp.where(attributed & creator_died, NO_CREATOR, all_trait_creators)


@partial(jax.jit, static_argnames=("n_agents",))
def compute_capture_transfers(
    copied_creator_ids,
    n_copied,
    value_capture_rate,
    trait_value,
    n_agents,
):
    """Compute imitation costs, creator incomes, and unclaimed capture.

    Every copied trait costs the imitator ``value_capture_rate * trait_value``.
    If its creator is still alive, that amount becomes creator income.  If not,
    the cost is unclaimed and leaves the system.
    """
    attributed = copied_creator_ids != NO_CREATOR
    payment_per_copy = value_capture_rate * trait_value
    attributed_payments = jnp.where(attributed, payment_per_copy, 0.0)
    safe_creator_ids = jnp.where(attributed, copied_creator_ids, 0)

    capture_paid = n_copied.astype(jnp.float32) * payment_per_copy
    capture_income = jnp.zeros((n_agents,), dtype=jnp.float32)
    capture_income = capture_income.at[safe_creator_ids.reshape(-1)].add(
        attributed_payments.reshape(-1)
    )
    n_unattributed_copies = n_copied.sum() - attributed.sum()
    unclaimed_capture = n_unattributed_copies.astype(jnp.float32) * payment_per_copy
    return capture_paid, capture_income, unclaimed_capture


@partial(jax.jit, static_argnames=("n_agents",))
def update_traits_and_capture(
    key,
    all_traits,
    all_trait_creators,
    all_roles,
    can_imitate,
    curr_L,
    p_i,
    p_c,
    value_capture_rate,
    trait_value,
    n_agents,
):
    """Apply agents' actions and settle all value-capture transfers."""

    def per_agent(key_, agent_idx):
        def do_innovate(_):
            new_traits, new_creators, success = innovate(
                key_,
                all_traits[agent_idx],
                all_trait_creators[agent_idx],
                agent_idx,
                curr_L,
                p_i,
            )
            return (
                new_traits,
                new_creators,
                success,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.full_like(new_creators, NO_CREATOR),
            )

        def do_imitate(_):
            new_traits, new_creators, n_copied, copied_creator_ids = imitate(
                key_,
                all_traits,
                all_trait_creators,
                can_imitate,
                agent_idx,
                p_c,
            )
            return (
                new_traits,
                new_creators,
                jnp.asarray(0, dtype=jnp.int32),
                n_copied,
                copied_creator_ids,
            )

        return jax.lax.cond(
            all_roles[agent_idx] == ROLE_INNOVATE,
            do_innovate,
            do_imitate,
            operand=None,
        )

    keys = jax.random.split(key, n_agents)
    (
        new_all_traits,
        new_all_trait_creators,
        n_innovated,
        n_imitated,
        copied_creator_ids,
    ) = jax.vmap(per_agent)(keys, jnp.arange(n_agents))
    capture_paid, capture_income, unclaimed_capture = compute_capture_transfers(
        copied_creator_ids,
        n_imitated,
        value_capture_rate,
        trait_value,
        n_agents,
    )
    unattributed_imitations = (
        (copied_creator_ids == NO_CREATOR)
        & (new_all_traits.astype(jnp.bool_) & ~all_traits.astype(jnp.bool_))
        & (all_roles[:, None] == ROLE_IMITATE)
    ).sum(axis=1)
    return (
        new_all_traits,
        new_all_trait_creators,
        n_innovated,
        n_imitated,
        capture_paid,
        capture_income,
        unattributed_imitations,
        unclaimed_capture,
    )


@jax.jit
def update_role_values(
    all_q_vals,
    all_roles,
    direct_rewards,
    capture_income,
    learning_rate,
):
    """Update role values while crediting delayed income to innovation.

    Capture earned while currently innovating is part of that action's observed
    reward.  Capture earned while imitating is a delayed consequence of an older
    innovation, so it updates the innovation value separately instead of making
    imitation look more rewarding.
    """
    agent_idxs = jnp.arange(all_roles.shape[0])
    innovating = all_roles == ROLE_INNOVATE
    chosen_rewards = direct_rewards + jnp.where(innovating, capture_income, 0.0)
    chosen_rpe = chosen_rewards - all_q_vals[agent_idxs, all_roles]
    new_all_q_vals = all_q_vals.at[agent_idxs, all_roles].add(
        learning_rate * chosen_rpe
    )

    delayed_capture = (~innovating) & (capture_income > 0.0)
    capture_rpe = capture_income - new_all_q_vals[:, ROLE_INNOVATE]
    return new_all_q_vals.at[:, ROLE_INNOVATE].add(
        learning_rate * jnp.where(delayed_capture, capture_rpe, 0.0)
    )


def make_neighbours_mask(grid_length, imit_dist_threshold):
    """Compute pairwise Manhattan distances on a toroidal grid."""
    n_agents = grid_length**2
    agent_idxs = jnp.arange(n_agents)
    agent_locs = jnp.stack([agent_idxs // grid_length, agent_idxs % grid_length]).T
    row_diffs = jnp.abs(agent_locs[:, None, 0] - agent_locs[None, :, 0])
    col_diffs = jnp.abs(agent_locs[:, None, 1] - agent_locs[None, :, 1])
    torus_row_dists = jnp.minimum(row_diffs, grid_length - row_diffs)
    torus_col_dists = jnp.minimum(col_diffs, grid_length - col_diffs)
    agent_dists = torus_row_dists + torus_col_dists
    return (agent_dists > 0) & (agent_dists <= imit_dist_threshold)


@partial(jax.jit, static_argnames=("grid_length", "T"))
def run_simulation_loop(
    key,
    grid_length,
    T,
    value_capture_rate,
    p_i=1.0,
    p_c=1.0,
    trait_value=0.2,
    innov_cost=0.1,
    p_d=0.001,
    init_q=1.0,
    choice_beta=0.1,
    learning_rate=0.1,
    imit_dist_threshold=100,
):
    """Run one value-capture simulation and return its time-series metrics."""
    n_agents = grid_length**2
    neighbours_mask = make_neighbours_mask(grid_length, imit_dist_threshold)

    def body_fn(carry, _):
        key, curr_L, all_traits, all_trait_creators, all_q_vals = carry
        key, death_key, role_key, update_key = jax.random.split(key, 4)

        # A death creates a naive replacement in the same grid position.  Claims
        # are invalidated before action selection so payments never go to that
        # replacement merely because it occupies the originator's old slot.
        deaths = jax.random.bernoulli(death_key, p_d, shape=(n_agents,))
        all_traits = jnp.where(deaths[:, None], 0, all_traits)
        all_trait_creators = invalidate_dead_creators(all_trait_creators, deaths)
        all_trait_creators = jnp.where(deaths[:, None], NO_CREATOR, all_trait_creators)
        all_q_vals = jnp.where(deaths[:, None], init_q, all_q_vals)

        role_probs = jax.nn.softmax(all_q_vals / choice_beta, axis=1)
        all_roles = jax.random.categorical(role_key, jnp.log(role_probs), axis=1)

        (
            new_all_traits,
            new_all_trait_creators,
            n_innovated,
            n_imitated,
            capture_paid,
            capture_income,
            unattributed_imitations,
            unclaimed_capture,
        ) = update_traits_and_capture(
            update_key,
            all_traits,
            all_trait_creators,
            all_roles,
            neighbours_mask,
            curr_L,
            p_i,
            p_c,
            value_capture_rate,
            trait_value,
            n_agents,
        )

        gross_benefits = (n_innovated + n_imitated) * trait_value
        direct_rewards = gross_benefits - capture_paid
        direct_rewards -= jnp.where(all_roles == ROLE_INNOVATE, innov_cost, 0.0)
        new_all_q_vals = update_role_values(
            all_q_vals,
            all_roles,
            direct_rewards,
            capture_income,
            learning_rate,
        )

        total_unique_traits_known = (new_all_traits.sum(axis=0) > 0).sum()
        traits_known = new_all_traits.sum(axis=1)
        most_traits_known = traits_known.max()
        mean_traits_known = traits_known.mean()

        mean_prop_known = mean_traits_known / curr_L
        unlock = (mean_prop_known >= 0.9) & (curr_L < MAX_TOTAL_L)
        new_L = jnp.where(unlock, curr_L + L, curr_L)

        return (
            key,
            new_L,
            new_all_traits,
            new_all_trait_creators,
            new_all_q_vals,
        ), (
            mean_traits_known,
            most_traits_known,
            total_unique_traits_known,
            role_probs.mean(axis=0),
            all_roles,
            n_innovated.sum(),
            n_imitated.sum(),
            capture_paid.sum(),
            capture_income.max(),
            unattributed_imitations.sum(),
            unclaimed_capture,
        )

    carry = (
        key,
        L,
        jnp.zeros((n_agents, MAX_TOTAL_L), dtype=jnp.int8),
        jnp.full((n_agents, MAX_TOTAL_L), NO_CREATOR, dtype=jnp.int16),
        jnp.full((n_agents, 2), init_q, dtype=jnp.float32),
    )

    _, metrics = jax.lax.scan(body_fn, carry, xs=None, length=T)
    metrics = list(metrics)
    metrics[5] = jnp.cumsum(metrics[5])
    metrics[6] = jnp.cumsum(metrics[6])
    metrics[7] = jnp.cumsum(metrics[7])
    metrics[9] = jnp.cumsum(metrics[9])
    metrics[10] = jnp.cumsum(metrics[10])
    return metrics


def main():
    seeds = list(range(10))
    grid_length, T = 10, int(2e3)
    value_capture_rates = jnp.linspace(0.0, 1.0, 20)

    def run_with_value_capture_rate(key, value_capture_rate):
        return jax.block_until_ready(
            run_simulation_loop(
                key,
                grid_length,
                T,
                value_capture_rate=value_capture_rate,
            )
        )

    metric_names = (
        "mean_traits",
        "most_traits",
        "total_unique_traits",
        "role_probs",
        "agent_roles",
        "n_innovated",
        "n_imitated",
        "total_value_captured",
        "max_capture_income",
        "n_unattributed_imitations",
        "total_unclaimed_capture",
    )
    all_metrics = {name: [] for name in metric_names}

    for seed in tqdm(seeds):
        key = jax.random.PRNGKey(seed)
        seed_metrics = jax.vmap(run_with_value_capture_rate, in_axes=(None, 0))(
            key, value_capture_rates
        )
        for name, values in zip(metric_names, seed_metrics, strict=True):
            all_metrics[name].append(np.asarray(values))

    simulation_outputs = {
        "value_capture_rates": np.asarray(value_capture_rates),
        "seeds": np.asarray(seeds),
        "T": np.int32(T),
        "grid_length": np.int32(grid_length),
        "initial_trait_space": np.int32(L),
        "max_total_l": np.int32(MAX_TOTAL_L),
        "trait_value": np.float32(0.2),
        "innov_cost": np.float32(0.1),
        "p_i": np.float32(1.0),
        "p_c": np.float32(1.0),
        "p_d": np.float32(0.001),
        "init_q": np.float32(1.0),
        "choice_beta": np.float32(0.1),
        "learning_rate": np.float32(0.1),
        "imit_dist_threshold": np.int32(100),
        "role_innovate": np.int32(ROLE_INNOVATE),
        "role_imitate": np.int32(ROLE_IMITATE),
        "model_variant": np.asarray("value_capture"),
        "learning_rule": np.asarray(
            "chosen_role_plus_delayed_capture_income_to_innovate"
        ),
    }
    simulation_outputs.update(
        {
            name: np.stack(seed_values, axis=0)
            for name, seed_values in all_metrics.items()
        }
    )

    output_path = f"simulation_outputs_{seeds[0]}-{seeds[-1]}.npz"
    np.savez(output_path, **simulation_outputs)


if __name__ == "__main__":
    main()
