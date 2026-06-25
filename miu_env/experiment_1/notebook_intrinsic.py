from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

ROLE_INNOVATE, ROLE_IMITATE = 0, 1

p_max = 100
l_max = 100


def get_increment(l):
    tmp = 0
    for j in range(1, l + 1):
        tmp += 0.95 ** (l - j)
    tmp *= 0.05 / (1 - (0.95**l_max))
    return tmp * p_max


increments = jnp.array([0] + [get_increment(l) for l in range(1, l_max + 1)])


def exploit(agent_arm_levels, agent_arm_payoff_estimates, full_rewards):
    arm_idx = jnp.argmax(agent_arm_payoff_estimates)
    arm_level = agent_arm_levels[arm_idx]
    payoff = full_rewards[arm_idx, arm_level]
    return payoff, agent_arm_payoff_estimates.at[arm_idx].set(payoff)


def explore(key, agent_arm_levels, agent_arm_payoff_estimates, full_rewards):
    unknown_mask = agent_arm_levels == 0
    n_unknown = unknown_mask.sum()

    def learn_new_arm():
        p_arm = unknown_mask / n_unknown
        arm_idx = jax.random.choice(key, full_rewards.shape[0], p=p_arm)
        payoff = full_rewards[arm_idx, 1]
        return (
            agent_arm_levels.at[arm_idx].set(1),
            agent_arm_payoff_estimates.at[arm_idx].set(payoff),
            True,
        )

    def do_nothing():
        return agent_arm_levels, agent_arm_payoff_estimates, False

    return jax.lax.cond(n_unknown > 0, learn_new_arm, do_nothing)


def refine(key, agent_arm_levels, agent_arm_payoff_estimates, full_rewards):
    eligible_mask = (agent_arm_levels > 0) & (agent_arm_levels < l_max)
    n_eligible = eligible_mask.sum()

    def refine_weighted_arm():
        arm_weights = jnp.where(
            eligible_mask, jnp.maximum(agent_arm_payoff_estimates, 0.0), 0.0
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

        # update the arm level and payoff estimate for the selected arm
        arm_level = agent_arm_levels[arm_idx]
        payoff = full_rewards[arm_idx, arm_level + 1]
        return (
            agent_arm_levels.at[arm_idx].set(arm_level + 1),
            agent_arm_payoff_estimates.at[arm_idx].set(payoff),
            True,
        )

    def do_nothing():
        return agent_arm_levels, agent_arm_payoff_estimates, False

    return jax.lax.cond(n_eligible > 0, refine_weighted_arm, do_nothing)


def innovate(key, agent_arm_levels, agent_arm_payoff_estimates, full_rewards):
    keys = jax.random.split(key, 2)
    op = jax.random.bernoulli(keys[0])
    return jax.lax.cond(
        op,
        lambda: explore(
            keys[1], agent_arm_levels, agent_arm_payoff_estimates, full_rewards
        ),
        lambda: refine(
            keys[1], agent_arm_levels, agent_arm_payoff_estimates, full_rewards
        ),
    )


def imitate(
    key,
    all_agent_arm_levels,
    all_agent_arm_payoff_estimates,
    neighbours_mask,
    agent_idx,
):
    neighbour_key, arm_key = jax.random.split(key)
    p_neighbour = neighbours_mask[agent_idx] / neighbours_mask[agent_idx].sum()
    neighbour_idx = jax.random.choice(
        neighbour_key, all_agent_arm_levels.shape[0], p=p_neighbour
    )

    neighbour_arm_levels = all_agent_arm_levels[neighbour_idx]
    neighbour_payoff_estimates = all_agent_arm_payoff_estimates[neighbour_idx]
    arm_weights = jnp.where(
        neighbour_arm_levels > 0, jnp.maximum(neighbour_payoff_estimates, 0.0), 0.0
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

    # only accept if the new level is higher than the current level for that arm
    new_level = all_agent_arm_levels[neighbour_idx, arm_idx]
    new_payoff = all_agent_arm_payoff_estimates[neighbour_idx, arm_idx]
    current_level = all_agent_arm_levels[agent_idx, arm_idx]
    accept = new_level > current_level
    return jax.lax.cond(
        accept,
        lambda: (
            all_agent_arm_levels[agent_idx].at[arm_idx].set(new_level),
            all_agent_arm_payoff_estimates[agent_idx].at[arm_idx].set(new_payoff),
            True,
        ),
        lambda: (
            all_agent_arm_levels[agent_idx],
            all_agent_arm_payoff_estimates[agent_idx],
            False,
        ),
    )


@jax.jit
def update_arm_knowledge(
    key,
    all_agent_arm_levels,
    all_agent_arm_payoff_estimates,
    all_roles,
    neighbours_mask,
    full_rewards,
):
    def per_agent(key_, agent_idx):
        def do_innovate(_):
            new_levels, new_payoffs, success = innovate(
                key_,
                all_agent_arm_levels[agent_idx],
                all_agent_arm_payoff_estimates[agent_idx],
                full_rewards,
            )
            return (
                new_levels,
                new_payoffs,
                success.astype(int),
                0,
            )  # levels, payoffs, # innovated, # imitated

        def do_imitate(_):
            new_levels, new_payoffs, success = imitate(
                key_,
                all_agent_arm_levels,
                all_agent_arm_payoff_estimates,
                neighbours_mask,
                agent_idx,
            )
            return (
                new_levels,
                new_payoffs,
                0,
                success.astype(int),
            )  # levels, payoffs, # innovated, # imitated

        return jax.lax.cond(
            all_roles[agent_idx] == ROLE_INNOVATE, do_innovate, do_imitate, operand=None
        )

    keys = jax.random.split(key, all_agent_arm_levels.shape[0])
    new_levels, new_payoffs, innovated, imitated = jax.vmap(per_agent)(
        keys, jnp.arange(all_agent_arm_levels.shape[0])
    )
    return new_levels, new_payoffs, innovated, imitated


@partial(jax.jit, static_argnames=("n_arms",))
def sample_arm_rewards(key, n_arms):
    rewards = jax.random.exponential(key, shape=(n_arms,))
    rewards = jnp.ceil(rewards**2)

    full_rewards = jnp.zeros((n_arms, 1 + l_max))
    return full_rewards.at[:, 1:].set(rewards[:, None] + increments[:l_max][None, :])


@partial(jax.jit, static_argnames=("grid_length", "n_arms", "T"))
def run_simulation_loop(
    key,
    grid_length,
    n_arms,
    T,
    prestige_gain,
    innov_cost=2.0,
    p_death=0.001,
    p_change=0.01,
    choice_beta=0.1,
    imit_dist_threshold=1,
    learning_rate=0.1,
    init_q=1.0,
    prestige_decay=0.01,
    prestige_value=1.0,
):
    n_agents = grid_length**2

    # Compute pairwise toroidal distances between agents for imitation.
    agent_idxs = jnp.arange(n_agents)
    agent_locs = jnp.stack(
        [
            agent_idxs // grid_length,  # row index
            agent_idxs % grid_length,  # column index
        ]
    ).T
    row_diffs = jnp.abs(agent_locs[:, None, 0] - agent_locs[None, :, 0])
    col_diffs = jnp.abs(agent_locs[:, None, 1] - agent_locs[None, :, 1])
    torus_row_dists = jnp.minimum(row_diffs, grid_length - row_diffs)
    torus_col_dists = jnp.minimum(col_diffs, grid_length - col_diffs)
    agent_dists = torus_row_dists + torus_col_dists
    neighbours_mask = ((agent_dists > 0) & (agent_dists <= imit_dist_threshold)).astype(
        jnp.bool_
    )

    vmapped_exploit = jax.vmap(exploit, in_axes=(0, 0, None))

    def body_fn(carry, t):
        (
            key,
            full_rewards,
            agent_arm_levels,
            agent_arm_payoff_estimates,
            q_vals,
            prestiges,
        ) = carry

        # get new keys
        key, change_key, death_key, role_key, update_key = jax.random.split(key, 5)

        # maybe reset environment rewards
        full_rewards = jax.lax.cond(
            jax.random.bernoulli(change_key, p=p_change),
            lambda: sample_arm_rewards(key, n_arms),
            lambda: full_rewards,
        )

        # random deaths
        deaths = jax.random.bernoulli(death_key, p=p_death, shape=(n_agents,))
        agent_arm_levels = jnp.where(deaths[:, None], 0, agent_arm_levels)
        agent_arm_payoff_estimates = jnp.where(
            deaths[:, None], 0.0, agent_arm_payoff_estimates
        )
        prestiges = jnp.where(deaths, 0.0, prestiges)
        prestiges = prestiges * (1.0 - prestige_decay)

        # compute payoffs from exploiting current knowledge
        curr_payoffs, agent_arm_payoff_estimates = vmapped_exploit(
            agent_arm_levels, agent_arm_payoff_estimates, full_rewards
        )

        # agents select roles
        role_probs = jax.nn.softmax(q_vals / choice_beta, axis=1)
        roles = jax.random.categorical(role_key, jnp.log(role_probs), axis=1)

        # update knowledge based on roles and compute new prospective payoffs
        new_agent_arm_levels, new_agent_arm_payoff_estimates, innovated, imitated = (
            update_arm_knowledge(
                update_key,
                agent_arm_levels,
                agent_arm_payoff_estimates,
                roles,
                neighbours_mask,
                full_rewards,
            )
        )
        new_payoffs, _ = vmapped_exploit(
            new_agent_arm_levels, new_agent_arm_payoff_estimates, full_rewards
        )
        prestige_changes = prestige_gain * innovated.astype(jnp.float32)
        new_prestiges = prestiges + prestige_changes

        # compute role rewards and costs
        rewards = new_payoffs - curr_payoffs
        rewards -= jnp.where(roles == ROLE_INNOVATE, innov_cost, 0.0)
        rewards += prestige_value * prestige_changes

        # update q-values
        rpe = rewards - q_vals[jnp.arange(n_agents), roles]
        new_all_q_vals = q_vals.at[jnp.arange(n_agents), roles].add(learning_rate * rpe)

        # compute some metrics for logging
        mean_payoff = curr_payoffs.mean()
        mean_avg_level = agent_arm_levels.mean()
        mean_max_level = agent_arm_levels.max(axis=1).mean()
        mean_role_probs = role_probs.mean(axis=0)
        mean_prestige = new_prestiges.mean()
        max_prestige = new_prestiges.max()

        return (
            key,
            full_rewards,
            new_agent_arm_levels,
            new_agent_arm_payoff_estimates,
            new_all_q_vals,
            new_prestiges,
        ), (
            mean_payoff,
            mean_avg_level,
            mean_max_level,
            innovated.sum(),
            imitated.sum(),
            mean_role_probs,
            roles,
            mean_prestige,
            max_prestige,
        )

    full_rewards = sample_arm_rewards(key, n_arms)
    carry = (
        key,
        full_rewards,
        jnp.zeros((n_agents, n_arms), dtype=jnp.int32),  # agent_arm_levels
        jnp.zeros((n_agents, n_arms), dtype=jnp.float32),  # agent_arm_payoff_estimates
        jnp.full((n_agents, 2), init_q, dtype=jnp.float32),  # q_vals
        jnp.zeros((n_agents,), dtype=jnp.float32),  # prestiges
    )

    _, metrics = jax.lax.scan(body_fn, carry, jnp.arange(T))
    metrics = list(metrics)
    metrics[0] /= full_rewards.max()
    metrics[3] = jnp.cumsum(metrics[3])  # cumulative number innovated
    metrics[4] = jnp.cumsum(metrics[4])  # cumulative number imitated
    return metrics


def main():
    seeds = list(range(10))
    grid_length, n_arms, T = 10, 200, int(1e3)
    prestige_gain_vals = 7 * jnp.linspace(0.0, 1.0, 30)
    prestige_decay = 0.01
    prestige_value = 1.0

    def run_with_prestige_gain(key, prestige_gain):
        return jax.block_until_ready(
            run_simulation_loop(
                key,
                grid_length,
                n_arms,
                T,
                prestige_gain=prestige_gain,
                prestige_decay=prestige_decay,
                prestige_value=prestige_value,
            )
        )

    all_prop_payoffs = []
    all_mean_avg_levels = []
    all_mean_max_levels = []
    all_n_innovs = []
    all_n_imits = []
    all_mean_role_probs = []
    all_agent_roles = []
    all_mean_prestige = []
    all_max_prestige = []

    for seed in tqdm(seeds):
        key = jax.random.PRNGKey(seed)
        (
            prop_payoffs,
            mean_avg_levels,
            mean_max_levels,
            n_innovs,
            n_imits,
            mean_role_probs,
            agent_roles,
            mean_prestige,
            max_prestige,
        ) = jax.vmap(run_with_prestige_gain, in_axes=(None, 0))(key, prestige_gain_vals)

        all_prop_payoffs.append(np.asarray(prop_payoffs))
        all_mean_avg_levels.append(np.asarray(mean_avg_levels))
        all_mean_max_levels.append(np.asarray(mean_max_levels))
        all_n_innovs.append(np.asarray(n_innovs))
        all_n_imits.append(np.asarray(n_imits))
        all_mean_role_probs.append(np.asarray(mean_role_probs))
        all_agent_roles.append(np.asarray(agent_roles))
        all_mean_prestige.append(np.asarray(mean_prestige))
        all_max_prestige.append(np.asarray(max_prestige))

    simulation_outputs = {
        "fees": np.asarray(prestige_gain_vals),
        "prestige_gains": np.asarray(prestige_gain_vals),
        "seeds": np.asarray(seeds),
        "T": np.int32(T),
        "grid_length": np.int32(grid_length),
        "n_arms": np.int32(n_arms),
        "prestige_decay": np.float32(prestige_decay),
        "prestige_value": np.float32(prestige_value),
        "role_innovate": np.int32(ROLE_INNOVATE),
        "role_imitate": np.int32(ROLE_IMITATE),
        "payoffs": np.stack(all_prop_payoffs, axis=0),
        "avg_levels": np.stack(all_mean_avg_levels, axis=0),
        "max_levels": np.stack(all_mean_max_levels, axis=0),
        "n_innov": np.stack(all_n_innovs, axis=0),
        "n_imit": np.stack(all_n_imits, axis=0),
        "role_probs": np.stack(all_mean_role_probs, axis=0),
        "agent_roles": np.stack(all_agent_roles, axis=0),
        "mean_prestige": np.stack(all_mean_prestige, axis=0),
        "max_prestige": np.stack(all_max_prestige, axis=0),
    }

    np.savez(f"simulation_outputs_{seeds[0]}-{seeds[-1]}.npz", **simulation_outputs)


if __name__ == "__main__":
    main()
