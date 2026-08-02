import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "theoretical_models"))

import parameter_space as parameter_space


class AdaptiveCeilingTests(unittest.TestCase):
    @staticmethod
    def _linear_growth(ceiling):
        return lambda D, eta: 1.0 - (D / ceiling)

    @staticmethod
    def _fixed_grid_ceiling(dD_fn):
        fixed_Ds = jnp.linspace(0.0, 100.0, 1000)
        max_growth = jax.vmap(
            lambda D: jnp.max(jax.vmap(lambda eta: dD_fn(D, eta))(parameter_space.etas))
        )(fixed_Ds)
        feasible_indices = jnp.where(max_growth >= 0.0, jnp.arange(fixed_Ds.size), 0)
        return fixed_Ds[jnp.max(feasible_indices)]

    def test_ceiling_above_initial_bound(self):
        dD_fn = self._linear_growth(250.0)
        D_max, eta_c, resolved = parameter_space.find_ceiling_adaptive(dD_fn)
        label, _ = parameter_space.classify_equilibrium(
            dD_fn,
            lambda D, eta: 0.5 - eta,
            D_max,
            eta_c,
            resolved,
        )

        self.assertTrue(bool(resolved))
        self.assertAlmostEqual(float(D_max), 250.0, delta=1e-3)
        self.assertEqual(int(label), 1)

    def test_ceiling_below_initial_bound_matches_fixed_grid(self):
        dD_fn = self._linear_growth(40.0)
        D_max, _, resolved = parameter_space.find_ceiling_adaptive(dD_fn)
        fixed_D_max = self._fixed_grid_ceiling(dD_fn)

        self.assertTrue(bool(resolved))
        self.assertAlmostEqual(float(D_max), float(fixed_D_max), delta=0.11)

    def test_unresolved_ceiling_returns_label_four(self):
        dD_fn = lambda D, eta: jnp.ones_like(D)
        D_max, eta_c, resolved = parameter_space.find_ceiling_adaptive(dD_fn)
        label, _ = parameter_space.classify_equilibrium(
            dD_fn,
            lambda D, eta: 0.5 - eta,
            D_max,
            eta_c,
            resolved,
        )

        self.assertFalse(bool(resolved))
        self.assertTrue(bool(jnp.isnan(D_max)))
        self.assertTrue(bool(jnp.isnan(eta_c)))
        self.assertEqual(int(label), 4)
        zero_ceiling_label, _ = parameter_space.classify_equilibrium(
            dD_fn,
            lambda D, eta: 0.5 - eta,
            0.0,
            0.5,
            True,
        )
        self.assertEqual(int(zero_ceiling_label), 4)

    def test_no_growth_at_origin_is_degenerate(self):
        dD_fn = lambda D, eta: -1.0 - D
        D_max, eta_c, resolved = parameter_space.find_ceiling_adaptive(dD_fn)
        label, _ = parameter_space.classify_equilibrium(
            dD_fn,
            lambda D, eta: 0.5 - eta,
            D_max,
            eta_c,
            resolved,
        )

        self.assertFalse(bool(resolved))
        self.assertTrue(bool(jnp.isnan(D_max)))
        self.assertTrue(bool(jnp.isnan(eta_c)))
        self.assertEqual(int(label), 4)

    def test_jit_vmap_ceiling_search(self):
        @jax.jit
        def find_ceilings(ceilings):
            return jax.vmap(
                lambda ceiling: parameter_space.find_ceiling_adaptive(
                    self._linear_growth(ceiling)
                )
            )(ceilings)

        D_maxes, _, resolved = find_ceilings(jnp.asarray([40.0, 250.0]))
        self.assertTrue(bool(jnp.all(resolved)))
        self.assertTrue(
            bool(jnp.allclose(D_maxes, jnp.asarray([40.0, 250.0]), atol=1e-3))
        )

    def test_population_size_reaches_model_dynamics(self):
        p_success_fn = parameter_space.get_p_success_fn(0.5, 0.02)
        default_dD, default_deta, _ = parameter_space.get_delta_functions(p_success_fn)
        explicit_dD, explicit_deta, _ = parameter_space.get_delta_functions(
            p_success_fn, N=100
        )
        small_dD, small_deta, _ = parameter_space.get_delta_functions(
            p_success_fn, N=10
        )
        D, eta = 20.0, 0.4

        self.assertTrue(bool(jnp.allclose(default_dD(D, eta), explicit_dD(D, eta))))
        self.assertTrue(bool(jnp.allclose(default_deta(D, eta), explicit_deta(D, eta))))
        self.assertFalse(bool(jnp.allclose(default_dD(D, eta), small_dD(D, eta))))
        self.assertFalse(bool(jnp.allclose(default_deta(D, eta), small_deta(D, eta))))

    def test_pi0_reaches_model_dynamics_and_sampling(self):
        p_success_fn = parameter_space.get_p_success_fn(0.5, 0.02)
        default_dD, default_deta, _ = parameter_space.get_delta_functions(p_success_fn)
        explicit_dD, explicit_deta, _ = parameter_space.get_delta_functions(
            p_success_fn, pi0=parameter_space.pi_0
        )
        zero_pi0_dD, zero_pi0_deta, _ = parameter_space.get_delta_functions(
            p_success_fn, pi0=0.0
        )
        D, eta = 20.0, 0.4

        self.assertTrue(bool(jnp.allclose(default_dD(D, eta), explicit_dD(D, eta))))
        self.assertTrue(bool(jnp.allclose(default_deta(D, eta), explicit_deta(D, eta))))
        self.assertTrue(bool(jnp.allclose(default_dD(D, eta), zero_pi0_dD(D, eta))))
        self.assertFalse(
            bool(jnp.allclose(default_deta(D, eta), zero_pi0_deta(D, eta)))
        )

        systems = parameter_space.sample_systems_latin_hypercube(
            jax.random.key(13), n_samples=20
        )
        self.assertEqual(parameter_space.param_ranges["pi_0"], (0.0, 0.5))
        self.assertIn("pi_0", systems)
        self.assertTrue(
            bool(jnp.all((systems["pi_0"] > 0.0) & (systems["pi_0"] < 0.5)))
        )

    def test_global_summary_and_stacked_barplot(self):
        key = jax.random.key(7)
        systems = parameter_space.sample_systems_latin_hypercube(key, n_samples=3)
        proportions, _ = parameter_space.compute_system_outcomes(systems, N=10)
        self.assertAlmostEqual(sum(proportions[label] for label in range(5)), 1.0)
        success_proportions = [
            proportions[f"success_{margin:g}"]
            for margin in parameter_space.success_pct_margins
        ]
        self.assertEqual(success_proportions, sorted(success_proportions))
        self.assertAlmostEqual(success_proportions[-1], proportions[1])

        rows = []
        for N in (10, 100, 1000):
            for run_type, lambda_val in (
                ("baseline", 0.0),
                ("best_fixed", 0.25),
                ("adaptive", -1.0),
            ):
                rows.append(
                    {
                        "N": N,
                        "run_type": run_type,
                        "lambda": lambda_val,
                        "prop_0": 0.1,
                        "prop_1": 0.4,
                        "prop_success_1": 0.1,
                        "prop_success_5": 0.3,
                        "prop_success_10": 0.4,
                        "prop_2": 0.2,
                        "prop_3": 0.3,
                        "prop_4": 0.0,
                    }
                )

        fig = parameter_space.do_stacked_barplot(parameter_space.pd.DataFrame(rows))
        ax = fig.axes[0]
        self.assertEqual(len(ax.patches), 63)
        self.assertIsNone(ax.get_legend())
        first_outcome_bars = ax.patches[:9]
        self.assertAlmostEqual(first_outcome_bars[0].get_width(), 1.0)
        self.assertAlmostEqual(
            first_outcome_bars[1].get_x() - first_outcome_bars[0].get_x() - 1.0,
            0.15,
        )
        self.assertAlmostEqual(
            first_outcome_bars[3].get_x() - first_outcome_bars[2].get_x() - 1.0,
            0.5,
        )
        self.assertAlmostEqual(ax.patches[0].get_height(), 0.1)
        self.assertAlmostEqual(ax.patches[9].get_height(), 0.2)
        self.assertAlmostEqual(ax.patches[18].get_height(), 0.1)
        success_colour_brightness = [
            sum(ax.patches[segment_idx * 9].get_facecolor()[:3])
            for segment_idx in range(3)
        ]
        self.assertEqual(success_colour_brightness, sorted(success_colour_brightness))
        self.assertEqual(
            [tick.get_text() for tick in ax.get_xticklabels()],
            [
                "None\n[$\\lambda=0$]",
                "Best fixed\n[$\\lambda=0.25$]",
                "Adaptive\n[$\\lambda=\\lambda^*(D)$]",
            ]
            * 3,
        )
        self.assertEqual(
            [text.get_text() for text in ax.texts],
            ["Population size $M=10$", "$M=100$", "$M=1000$"],
        )
        self.assertEqual(ax.get_xlabel(), "Value-capture norm")
        self.assertEqual(list(ax.get_yticks()), [0.2, 0.4, 0.6, 0.8])
        self.assertTrue(all(line.get_visible() for line in ax.get_ygridlines()))
        self.assertEqual(len(ax.child_axes), 0)
        parameter_space.plt.close(fig)

    def test_find_optimal_fixed_lambda_for_systems(self):
        def fake_compute_system_outcomes(
            systems, N=parameter_space.N, lambda_fn=None, fixed_lambda=None
        ):
            lambda_val = float(fixed_lambda)
            score = 1.0 - (lambda_val - systems["peak"]) ** 2
            return {"evaluated_at": lambda_val}, jnp.asarray(score)

        with patch.object(
            parameter_space,
            "compute_system_outcomes",
            side_effect=fake_compute_system_outcomes,
        ):
            for peak in (0.0, 0.65, 1.0):
                with self.subTest(peak=peak):
                    optimal_lambda, outcomes, score = (
                        parameter_space.find_optimal_fixed_lambda_for_systems(
                            {"peak": peak}
                        )
                    )
                    self.assertAlmostEqual(optimal_lambda, peak, delta=3e-4)
                    self.assertAlmostEqual(outcomes["evaluated_at"], optimal_lambda)
                    self.assertAlmostEqual(float(score), 1.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
