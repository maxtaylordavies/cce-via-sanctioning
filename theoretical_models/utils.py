from functools import partial

import jax
import jax.numpy as jnp


def time_to_discovery(eta, p_success, N):
    p_fail = 1 - p_success
    p_no_successes = p_fail ** (N * eta)
    return 1 / (1 - p_no_successes + 1e-10)


def time_to_diffusion(eta, phi, N, beta):
    eta_term = 1 / ((beta * (1 - eta)) + 1e-10)
    phi_term = (phi * (N - 1)) / (1 - phi)
    return eta_term * jnp.log(phi_term)


def progression_rate(eta, p_success, phi, N=100, beta=1.0):
    rate = 1 / (
        time_to_discovery(eta, p_success, N) + time_to_diffusion(eta, phi, N, beta)
    )
    return jnp.where(eta == 0, 0.0, rate)


def compute_rate_optimal_eta(p_success, phi):
    etas = jnp.linspace(0.0, 1.0, 1000)
    rates = progression_rate(etas, p_success, phi)
    return etas[jnp.argmax(rates)]


def get_eta_star_fn(
    v_innov_fn, b_imit_fn, p_success_fn, c_innov_fn, c_imit_fn, u_fn=None, beta=0.1
):
    if u_fn is None:
        u_fn = lambda D: 0.0

    def eta_star_fn(D, eta):
        delta_r = (
            b_imit_fn(D)
            - c_imit_fn(D)
            - (p_success_fn(D) * v_innov_fn(D, eta))
            + c_innov_fn(D)
        )
        return 1 / (1 + jnp.exp((delta_r - u_fn(D)) / beta)), delta_r, p_success_fn(D)

    return eta_star_fn


@partial(jax.jit, static_argnames=("delta_functions", "T", "dt"))
def simulate_trajectory(delta_functions, D0, eta0, T, dt, D_max):
    n_steps = int(T / dt)

    dD_dt, deta_dt = delta_functions

    def step_state(state):
        D, eta = state

        dD, deta = dD_dt(D, eta), deta_dt(D, eta)
        D_next, eta_next = D + dt * dD, eta + dt * deta

        # Keep within bounds
        D_next = jnp.clip(D_next, 0.0, D_max)
        eta_next = jnp.clip(eta_next, 0.0, 1.0)

        return jnp.array([D_next, eta_next])

    def scan_step(state, _):
        next_state = step_state(state)
        return next_state, next_state

    init_state = jnp.array([D0, eta0])
    _, traj = jax.lax.scan(scan_step, init_state, xs=None, length=n_steps)

    return jnp.vstack([init_state, traj])
