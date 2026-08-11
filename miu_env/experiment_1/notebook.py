"""Experiment 1: value capture in the refinement-bandit environment.

Each known arm version records the agent who created that version.  When an
imitator copies a more valuable version of an arm, ``value_capture_rate``
(lambda in the paper) of that arm-level improvement is withheld.  This applies
even when the copied arm does not immediately become the imitator's best arm.
A living creator receives the amount; if the creator has died, it is unclaimed
and leaves the system.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

ROLE_INNOVATE, ROLE_IMITATE = 0, 1
NO_CREATOR = -1

P_MAX = 100
L_MAX = 100


def get_increment(level):
    increment = 0
    for prior_level in range(1, level + 1):
        increment += 0.95 ** (level - prior_level)
    increment *= 0.05 / (1 - (0.95**L_MAX))
    return increment * P_MAX


INCREMENTS = jnp.array([0] + [get_increment(level) for level in range(1, L_MAX + 1)])


def exploit(agent_arm_levels, agent_arm_payoff_estimates, full_rewards):
    arm_idx = jnp.argmax(agent_arm_payoff_estimates)
    arm_level = agent_arm_levels[arm_idx]
    payoff = full_rewards[arm_idx, arm_level]
    return payoff, agent_arm_payoff_estimates.at[arm_idx].set(payoff)


def explore(
    key,
    agent_arm_levels,
    agent_arm_payoff_estimates,
    agent_arm_creators,
    agent_idx,
    full_rewards,
):
    unknown_mask = agent_arm_levels == 0
    n_unknown = unknown_mask.sum()

    def learn_new_arm():
        p_arm = unknown_mask / n_unknown
        arm_idx = jax.random.choice(key, full_rewards.shape[0], p=p_arm)
        payoff = full_rewards[arm_idx, 1]
        return (
            agent_arm_levels.at[arm_idx].set(1),
            agent_arm_payoff_estimates.at[arm_idx].set(payoff),
            agent_arm_creators.at[arm_idx].set(
                jnp.asarray(agent_idx, dtype=agent_arm_creators.dtype)
            ),
            True,
        )

    def do_nothing():
        return (
            agent_arm_levels,
            agent_arm_payoff_estimates,
            agent_arm_creators,
            False,
        )

    return jax.lax.cond(n_unknown > 0, learn_new_arm, do_nothing)


def refine(
    key,
    agent_arm_levels,
    agent_arm_payoff_estimates,
    agent_arm_creators,
    agent_idx,
    full_rewards,
):
    eligible_mask = (agent_arm_levels > 0) & (agent_arm_levels < L_MAX)
    n_eligible = eligible_mask.sum()

    def refine_weighted_arm():
        arm_weights = jnp.where(
            eligible_mask,
            jnp.maximum(agent_arm_payoff_estimates, 0.0),
            0.0,
        )
        total_arm_weight = arm_weights.sum()

        def sample_weighted_arm():
            p_arm = arm_weights / total_arm_weight
            return jax.random.choice(key, full_rewards.shape[0], p=p_arm)

        def sample_uniform_eligible_arm():
            p_arm = eligible_mask / n_eligible
            return jax.random.choice(key, full_rewards.shape[0], p=p_arm)

        arm_idx = jax.lax.cond(
            total_arm_weight > 0,
            sample_weighted_arm,
            sample_uniform_eligible_arm,
        )
        arm_level = agent_arm_levels[arm_idx]
        payoff = full_rewards[arm_idx, arm_level + 1]
        return (
            agent_arm_levels.at[arm_idx].set(arm_level + 1),
            agent_arm_payoff_estimates.at[arm_idx].set(payoff),
            agent_arm_creators.at[arm_idx].set(
                jnp.asarray(agent_idx, dtype=agent_arm_creators.dtype)
            ),
            True,
        )

    def do_nothing():
        return (
            agent_arm_levels,
            agent_arm_payoff_estimates,
            agent_arm_creators,
            False,
        )

    return jax.lax.cond(n_eligible > 0, refine_weighted_arm, do_nothing)


def innovate(
    key,
    agent_arm_levels,
    agent_arm_payoff_estimates,
    agent_arm_creators,
    agent_idx,
    full_rewards,
):
    operation_key, action_key = jax.random.split(key)
    explore_arm = jax.random.bernoulli(operation_key)
    return jax.lax.cond(
        explore_arm,
        lambda: explore(
            action_key,
            agent_arm_levels,
            agent_arm_payoff_estimates,
            agent_arm_creators,
            agent_idx,
            full_rewards,
        ),
        lambda: refine(
            action_key,
            agent_arm_levels,
            agent_arm_payoff_estimates,
            agent_arm_creators,
            agent_idx,
            full_rewards,
        ),
    )


def imitate(
    key,
    all_agent_arm_levels,
    all_agent_arm_payoff_estimates,
    all_agent_arm_creators,
    neighbours_mask,
    agent_idx,
    full_rewards,
):
    neighbour_key, arm_key = jax.random.split(key)
    p_neighbour = neighbours_mask[agent_idx] / neighbours_mask[agent_idx].sum()
    neighbour_idx = jax.random.choice(
        neighbour_key, all_agent_arm_levels.shape[0], p=p_neighbour
    )

    neighbour_arm_levels = all_agent_arm_levels[neighbour_idx]
    neighbour_payoff_estimates = all_agent_arm_payoff_estimates[neighbour_idx]
    arm_weights = jnp.where(
        neighbour_arm_levels > 0,
        jnp.maximum(neighbour_payoff_estimates, 0.0),
        0.0,
    )
    total_arm_weight = arm_weights.sum()

    def sample_weighted_arm():
        p_arm = arm_weights / total_arm_weight
        return jax.random.choice(arm_key, all_agent_arm_levels.shape[1], p=p_arm)

    def sample_uniform_arm():
        return jax.random.randint(arm_key, (), 0, all_agent_arm_levels.shape[1])

    arm_idx = jax.lax.cond(
        total_arm_weight > 0,
        sample_weighted_arm,
        sample_uniform_arm,
    )

    new_level = all_agent_arm_levels[neighbour_idx, arm_idx]
    new_payoff = all_agent_arm_payoff_estimates[neighbour_idx, arm_idx]
    copied_creator = all_agent_arm_creators[neighbour_idx, arm_idx]
    current_level = all_agent_arm_levels[agent_idx, arm_idx]
    accept = new_level > current_level
    copied_value = jnp.maximum(
        full_rewards[arm_idx, new_level] - full_rewards[arm_idx, current_level],
        0.0,
    )
    return jax.lax.cond(
        accept,
        lambda: (
            all_agent_arm_levels[agent_idx].at[arm_idx].set(new_level),
            all_agent_arm_payoff_estimates[agent_idx].at[arm_idx].set(new_payoff),
            all_agent_arm_creators[agent_idx].at[arm_idx].set(copied_creator),
            True,
            copied_creator,
            copied_value,
        ),
        lambda: (
            all_agent_arm_levels[agent_idx],
            all_agent_arm_payoff_estimates[agent_idx],
            all_agent_arm_creators[agent_idx],
            False,
            jnp.asarray(NO_CREATOR, dtype=all_agent_arm_creators.dtype),
            jnp.asarray(0.0, dtype=all_agent_arm_payoff_estimates.dtype),
        ),
    )


@jax.jit
def invalidate_dead_creators(all_agent_arm_creators, deaths):
    attributed = all_agent_arm_creators != NO_CREATOR
    safe_creator_ids = jnp.where(attributed, all_agent_arm_creators, 0)
    creator_died = deaths[safe_creator_ids]
    return jnp.where(
        attributed & creator_died,
        NO_CREATOR,
        all_agent_arm_creators,
    )


@partial(jax.jit, static_argnames=("n_agents",))
def compute_capture_transfers(
    copied_creator_ids,
    imitation_values,
    value_capture_rate,
    n_agents,
):
    """Withhold lambda of every imitation gain and distribute valid claims."""
    capture_paid = value_capture_rate * imitation_values
    attributed = copied_creator_ids != NO_CREATOR
    attributed_payments = jnp.where(attributed, capture_paid, 0.0)
    safe_creator_ids = jnp.where(attributed, copied_creator_ids, 0)

    capture_income = jnp.zeros((n_agents,), dtype=jnp.float32)
    capture_income = capture_income.at[safe_creator_ids].add(attributed_payments)
    unclaimed_capture = jnp.where(attributed, 0.0, capture_paid).sum()
    return capture_paid, capture_income, unclaimed_capture


@jax.jit
def update_arm_knowledge(
    key,
    all_agent_arm_levels,
    all_agent_arm_payoff_estimates,
    all_agent_arm_creators,
    all_roles,
    neighbours_mask,
    full_rewards,
):
    def per_agent(key_, agent_idx):
        def do_innovate(_):
            new_levels, new_payoffs, new_creators, success = innovate(
                key_,
                all_agent_arm_levels[agent_idx],
                all_agent_arm_payoff_estimates[agent_idx],
                all_agent_arm_creators[agent_idx],
                agent_idx,
                full_rewards,
            )
            return (
                new_levels,
                new_payoffs,
                new_creators,
                success.astype(jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(NO_CREATOR, dtype=new_creators.dtype),
                jnp.asarray(0.0, dtype=new_payoffs.dtype),
            )

        def do_imitate(_):
            (
                new_levels,
                new_payoffs,
                new_creators,
                success,
                copied_creator,
                copied_value,
            ) = imitate(
                key_,
                all_agent_arm_levels,
                all_agent_arm_payoff_estimates,
                all_agent_arm_creators,
                neighbours_mask,
                agent_idx,
                full_rewards,
            )
            return (
                new_levels,
                new_payoffs,
                new_creators,
                jnp.asarray(0, dtype=jnp.int32),
                success.astype(jnp.int32),
                copied_creator,
                copied_value,
            )

        return jax.lax.cond(
            all_roles[agent_idx] == ROLE_INNOVATE,
            do_innovate,
            do_imitate,
            operand=None,
        )

    keys = jax.random.split(key, all_agent_arm_levels.shape[0])
    return jax.vmap(per_agent)(keys, jnp.arange(all_agent_arm_levels.shape[0]))


@jax.jit
def update_role_values(
    q_vals,
    roles,
    direct_rewards,
    capture_income,
    learning_rate,
):
    agent_idxs = jnp.arange(roles.shape[0])
    innovating = roles == ROLE_INNOVATE
    chosen_rewards = direct_rewards + jnp.where(innovating, capture_income, 0.0)
    chosen_rpe = chosen_rewards - q_vals[agent_idxs, roles]
    new_q_vals = q_vals.at[agent_idxs, roles].add(learning_rate * chosen_rpe)

    delayed_capture = (~innovating) & (capture_income > 0.0)
    capture_rpe = capture_income - new_q_vals[:, ROLE_INNOVATE]
    return new_q_vals.at[:, ROLE_INNOVATE].add(
        learning_rate * jnp.where(delayed_capture, capture_rpe, 0.0)
    )


@partial(jax.jit, static_argnames=("n_arms",))
def sample_arm_rewards(key, n_arms):
    rewards = jax.random.exponential(key, shape=(n_arms,))
    rewards = jnp.ceil(rewards**2)

    full_rewards = jnp.zeros((n_arms, 1 + L_MAX))
    return full_rewards.at[:, 1:].set(rewards[:, None] + INCREMENTS[:L_MAX][None, :])


def make_neighbours_mask(grid_length, imit_dist_threshold):
    n_agents = grid_length**2
    agent_idxs = jnp.arange(n_agents)
    agent_locs = jnp.stack([agent_idxs // grid_length, agent_idxs % grid_length]).T
    row_diffs = jnp.abs(agent_locs[:, None, 0] - agent_locs[None, :, 0])
    col_diffs = jnp.abs(agent_locs[:, None, 1] - agent_locs[None, :, 1])
    torus_row_dists = jnp.minimum(row_diffs, grid_length - row_diffs)
    torus_col_dists = jnp.minimum(col_diffs, grid_length - col_diffs)
    agent_dists = torus_row_dists + torus_col_dists
    return (agent_dists > 0) & (agent_dists <= imit_dist_threshold)


@partial(jax.jit, static_argnames=("grid_length", "n_arms", "T"))
def run_simulation_loop(
    key,
    grid_length,
    n_arms,
    T,
    value_capture_rate,
    innov_cost=0.1,
    p_death=0.001,
    p_change=0.0,
    choice_beta=0.1,
    imit_dist_threshold=1,
    learning_rate=0.1,
    init_q=1.0,
):
    n_agents = grid_length**2
    neighbours_mask = make_neighbours_mask(grid_length, imit_dist_threshold)
    vmapped_exploit = jax.vmap(exploit, in_axes=(0, 0, None))

    def body_fn(carry, _):
        (
            key,
            full_rewards,
            agent_arm_levels,
            agent_arm_payoff_estimates,
            agent_arm_creators,
            q_vals,
        ) = carry
        key, change_key, death_key, role_key, update_key = jax.random.split(key, 5)

        full_rewards = jax.lax.cond(
            jax.random.bernoulli(change_key, p=p_change),
            lambda: sample_arm_rewards(key, n_arms),
            lambda: full_rewards,
        )

        deaths = jax.random.bernoulli(death_key, p=p_death, shape=(n_agents,))
        agent_arm_levels = jnp.where(deaths[:, None], 0, agent_arm_levels)
        agent_arm_payoff_estimates = jnp.where(
            deaths[:, None], 0.0, agent_arm_payoff_estimates
        )
        agent_arm_creators = invalidate_dead_creators(agent_arm_creators, deaths)
        agent_arm_creators = jnp.where(deaths[:, None], NO_CREATOR, agent_arm_creators)

        curr_payoffs, agent_arm_payoff_estimates = vmapped_exploit(
            agent_arm_levels,
            agent_arm_payoff_estimates,
            full_rewards,
        )
        role_probs = jax.nn.softmax(q_vals / choice_beta, axis=1)
        roles = jax.random.categorical(role_key, jnp.log(role_probs), axis=1)

        (
            new_agent_arm_levels,
            new_agent_arm_payoff_estimates,
            new_agent_arm_creators,
            innovated,
            imitated,
            copied_creator_ids,
            imitation_values,
        ) = update_arm_knowledge(
            update_key,
            agent_arm_levels,
            agent_arm_payoff_estimates,
            agent_arm_creators,
            roles,
            neighbours_mask,
            full_rewards,
        )
        new_payoffs, _ = vmapped_exploit(
            new_agent_arm_levels,
            new_agent_arm_payoff_estimates,
            full_rewards,
        )

        gross_benefits = new_payoffs - curr_payoffs
        capture_paid, capture_income, unclaimed_capture = compute_capture_transfers(
            copied_creator_ids,
            imitation_values,
            value_capture_rate,
            n_agents,
        )

        action_benefits = jnp.where(
            roles == ROLE_IMITATE,
            imitation_values,
            gross_benefits,
        )
        direct_rewards = action_benefits - capture_paid
        direct_rewards -= jnp.where(roles == ROLE_INNOVATE, innov_cost, 0.0)
        new_q_vals = update_role_values(
            q_vals,
            roles,
            direct_rewards,
            capture_income,
            learning_rate,
        )

        mean_payoff = curr_payoffs.mean()
        mean_avg_level = agent_arm_levels.mean()
        mean_max_level = agent_arm_levels.max(axis=1).mean()
        mean_role_probs = role_probs.mean(axis=0)
        unattributed_imitations = (
            (imitated > 0) & (copied_creator_ids == NO_CREATOR)
        ).sum()

        return (
            key,
            full_rewards,
            new_agent_arm_levels,
            new_agent_arm_payoff_estimates,
            new_agent_arm_creators,
            new_q_vals,
        ), (
            mean_payoff,
            mean_avg_level,
            mean_max_level,
            innovated.sum(),
            imitated.sum(),
            mean_role_probs,
            roles,
            capture_paid.sum(),
            capture_income.max(),
            unattributed_imitations,
            unclaimed_capture,
        )

    initial_full_rewards = sample_arm_rewards(key, n_arms)
    carry = (
        key,
        initial_full_rewards,
        jnp.zeros((n_agents, n_arms), dtype=jnp.int32),
        jnp.zeros((n_agents, n_arms), dtype=jnp.float32),
        jnp.full((n_agents, n_arms), NO_CREATOR, dtype=jnp.int16),
        jnp.full((n_agents, 2), init_q, dtype=jnp.float32),
    )

    _, metrics = jax.lax.scan(body_fn, carry, xs=None, length=T)
    metrics = list(metrics)
    metrics[0] /= initial_full_rewards.max()
    metrics[3] = jnp.cumsum(metrics[3])
    metrics[4] = jnp.cumsum(metrics[4])
    metrics[7] = jnp.cumsum(metrics[7])
    metrics[9] = jnp.cumsum(metrics[9])
    metrics[10] = jnp.cumsum(metrics[10])
    return metrics


def main():
    seeds = list(range(5))
    grid_length, n_arms, T = 10, 200, int(1e3)
    value_capture_rates = jnp.linspace(0.0, 1.0, 30)

    def run_with_value_capture_rate(key, value_capture_rate):
        return jax.block_until_ready(
            run_simulation_loop(
                key,
                grid_length,
                n_arms,
                T,
                value_capture_rate=value_capture_rate,
            )
        )

    metric_names = (
        "payoffs",
        "avg_levels",
        "max_levels",
        "n_innov",
        "n_imit",
        "role_probs",
        "agent_roles",
        "total_value_captured",
        "max_capture_income",
        "n_unattributed_imitations",
        "total_unclaimed_capture",
    )
    all_metrics = {name: [] for name in metric_names}

    for seed in tqdm(seeds):
        key = jax.random.PRNGKey(seed)
        seed_metrics = jax.vmap(
            run_with_value_capture_rate,
            in_axes=(None, 0),
        )(key, value_capture_rates)
        for name, values in zip(metric_names, seed_metrics, strict=True):
            all_metrics[name].append(np.asarray(values))

    simulation_outputs = {
        "value_capture_rates": np.asarray(value_capture_rates),
        "seeds": np.asarray(seeds),
        "T": np.int32(T),
        "grid_length": np.int32(grid_length),
        "n_arms": np.int32(n_arms),
        "innov_cost": np.float32(2.0),
        "p_death": np.float32(0.001),
        "p_change": np.float32(0.01),
        "choice_beta": np.float32(0.1),
        "imit_dist_threshold": np.int32(1),
        "learning_rate": np.float32(0.1),
        "init_q": np.float32(1.0),
        "role_innovate": np.int32(ROLE_INNOVATE),
        "role_imitate": np.int32(ROLE_IMITATE),
        "model_variant": np.asarray("value_capture"),
        "creator_attribution": np.asarray("latest_arm_version_creator"),
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

    np.savez(
        f"simulation_outputs_{seeds[0]}-{seeds[-1]}.npz",
        **simulation_outputs,
    )


if __name__ == "__main__":
    main()
