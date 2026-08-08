from functools import partial

import jax
import jax.numpy as jnp
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    DrawingArea,
    HPacker,
    TextArea,
)


def get_p_success_fn(p0, k):
    def p_success_fn(D):
        return p0 * jnp.exp(-k * D)

    return p_success_fn


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


def get_p_copy(p, eta, m):
    # return 1 - ((1 - p) ** (m * eta))
    return 1 - ((1 - (eta * p)) ** m)


def get_eta_star_fn(v_fn, b_fn, p_fn, c_innov_fn, c_imit_fn, beta, m):
    def delta_r_fn(D, eta):
        p = p_fn(D)
        p_copy = get_p_copy(p, eta, m)
        r_innov = (p * v_fn(D, eta)) - c_innov_fn(D)
        r_imit = (p_copy * b_fn(D)) - c_imit_fn(D)
        return r_imit - r_innov

    def eta_star_fn(D, eta):
        delta_r = delta_r_fn(D, eta)
        return 1 / (1 + jnp.exp(delta_r / beta)), delta_r, p_fn(D)

    return eta_star_fn


def compute_rho_from_w_and_mu(w, mu=0.01):
    return (w * mu) / ((1 - w) * (1 - mu) + 1e-10)


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


def add_title_with_legend(
    fig,
    title,
    legend_symbol,
    legend_vals,
    legend_colors,
    legend_styles=None,
    font_size=14,
):
    title_box = TextArea(title, textprops={"fontsize": font_size})

    if legend_styles is None:
        legend_styles = ["-"] * len(legend_vals)

    entries = []
    for i, (val, color) in enumerate(zip(legend_vals, legend_colors)):
        line_box = DrawingArea(20, 10, 0, 0)
        line_box.add_artist(
            Line2D(
                [0, 20], [5, 5], color=color, linewidth=3, linestyle=legend_styles[i]
            )
        )
        suffix = "]" if i == len(legend_vals) - 1 else ""
        label = TextArea(
            f"${legend_symbol} = {val}$" + suffix, textprops={"fontsize": font_size}
        )
        entries.append(
            HPacker(children=[line_box, label], align="center", pad=0, sep=5)
        )

    open_bracket = TextArea("[", textprops={"fontsize": font_size})
    entries_box = HPacker(children=entries, align="center", pad=0, sep=16)
    legend_row = HPacker(
        children=[open_bracket, entries_box], align="center", pad=0, sep=4
    )
    title_with_legend = HPacker(
        children=[title_box, legend_row], align="center", pad=0, sep=16
    )
    fig.add_artist(
        AnchoredOffsetbox(
            loc="lower left",
            child=title_with_legend,
            bbox_to_anchor=(0.045, 1.02),
            bbox_transform=fig.transFigure,
            frameon=False,
            borderpad=0,
            pad=0,
        )
    )
