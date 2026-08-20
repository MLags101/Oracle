"""Joint filter and gain design (milestone M4).

Three properties are worth more than any amount of eyeballing the numbers:

* the margins the tool *reports* are the margins the recommended parameters
  actually produce, recomputed from scratch;
* the answer moves in the direction the user moved the slider;
* a filter change is only ever recommended when it buys something.

The re-solve budget is tested here too. It is a product requirement rather than
an optimization: the sandbox teaches by letting the user move a control and watch
the trade, and a second of latency destroys the connection between the two.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from rotorid.config import load_config
from rotorid.core.analysis.margins import broken_loop, compute_margins, design_grid
from rotorid.core.design.controller import controller_for
from rotorid.core.design.joint import optimize_jointly
from rotorid.core.design.objectives import DesignTargets
from rotorid.core.design.recommend import identify_axis, recommend_from
from tests.synthetic.generators import make_airframe, make_bundle, make_chain, motor_noise

CONFIG = load_config()


def _analysis(**kw):
    chain = kw.pop("chain", make_chain())
    bundle = make_bundle(kw.pop("airframe", make_airframe()), chain, **kw)
    return identify_axis(bundle, "roll", CONFIG), bundle


# --------------------------------------------------------------------------- #
# The margins are real
# --------------------------------------------------------------------------- #


def test_reported_margins_match_the_recommended_parameters() -> None:
    """Recompute the loop from the recommendation alone and check it agrees.

    The design evaluates ``L`` in a vectorized inner loop where a single indexing
    slip would shift every margin quietly and consistently. Rebuilding the loop
    from the published gains and the published chain is the only check that
    cannot be fooled by the same mistake twice.
    """
    analysis, bundle = _analysis()
    rec = recommend_from(analysis, bundle, CONFIG)

    grid = design_grid(0.05, 200.0, 1200)
    L = broken_loop(
        grid,
        controller_for(bundle.stack, rec.gains, rec.filters.chain),
        analysis.airframe,
        delay=analysis.delay,
        op=analysis.operating_point,
    )
    recomputed = compute_margins(grid, L)

    assert recomputed.phase_margin_deg == pytest.approx(rec.margins.phase_margin_deg, abs=2.0)
    assert recomputed.gain_margin_db == pytest.approx(rec.margins.gain_margin_db, abs=1.0)
    assert recomputed.crossover_hz == pytest.approx(rec.margins.crossover_hz, rel=0.05)


def test_the_designed_gains_hold_every_constraint_at_once() -> None:
    analysis, bundle = _analysis()
    rec = recommend_from(analysis, bundle, CONFIG)

    assert rec.margins.phase_margin_deg >= CONFIG.float_("margins", "pm_floor_deg")
    assert rec.margins.gain_margin_db >= CONFIG.float_("margins", "gm_min_db") - 0.1
    assert rec.margins.peak_sensitivity_db <= CONFIG.float_("margins", "ms_max_db") + 0.1


def test_dterm_noise_stays_under_the_ceiling() -> None:
    """The constraint that decides how much D a real vehicle can carry."""
    analysis, bundle = _analysis(noise=None)
    rec = recommend_from(analysis, bundle, CONFIG)

    limit = CONFIG.float_("noise", "dterm_output_rms_limit_pct")
    assert np.isfinite(rec.dterm_noise_rms_pct)
    assert rec.dterm_noise_rms_pct <= limit + 1e-6


# --------------------------------------------------------------------------- #
# Monotonicity
# --------------------------------------------------------------------------- #


def test_conservatism_moves_the_answer_in_one_direction() -> None:
    analysis, bundle = _analysis()
    results = [
        recommend_from(analysis, bundle, CONFIG, conservatism=c) for c in (0.0, 0.25, 0.5, 0.75)
    ]
    crossovers = [r.margins.crossover_hz for r in results]

    assert crossovers == sorted(crossovers, reverse=True), (
        f"crossover must fall as conservatism rises, got {crossovers}"
    )
    assert results[-1].margins.phase_margin_deg >= results[0].margins.phase_margin_deg - 1.0


def test_a_noisier_vehicle_gets_less_derivative_gain() -> None:
    """More gyro noise means less D. If it does not, the ceiling is decorative."""
    chain = make_chain()
    quiet_analysis, quiet_bundle = _analysis(chain=chain, noise=None)
    quiet = recommend_from(quiet_analysis, quiet_bundle, CONFIG)

    t = np.arange(0.0, 90.0, 1.0 / chain.sample_rate_hz)
    loud_analysis, loud_bundle = _analysis(
        chain=chain, noise=motor_noise(t, fundamental_hz=70.0, broadband_rms=0.2)
    )
    loud = recommend_from(loud_analysis, loud_bundle, CONFIG)

    assert loud.gains.kd <= quiet.gains.kd or (
        loud.filters.chain.gyro_lpf_hz is not None
        and quiet.filters.chain.gyro_lpf_hz is not None
        and loud.filters.chain.gyro_lpf_hz <= quiet.filters.chain.gyro_lpf_hz
    ), "a noisier vehicle must end up with either less D or more filtering, or both"


# --------------------------------------------------------------------------- #
# A filter change has to earn its place
# --------------------------------------------------------------------------- #


def test_filters_are_left_alone_when_there_is_no_noise_evidence() -> None:
    analysis, bundle = _analysis()
    joint = optimize_jointly(
        analysis.airframe,
        analysis.chain,
        None,
        CONFIG,
        stack=bundle.stack,
        delay=analysis.delay,
        targets=_targets(),
        f_grid=design_grid(0.1, 200.0, 600),
        op=analysis.operating_point,
        axis="roll",
        track=None,
    )
    assert not joint.filters_changed
    assert "no usable noise measurement" in joint.filters.rationale
    assert joint.passes == 0
    assert np.isnan(joint.dterm_noise_rms)


def test_the_iteration_converges_rather_than_running_to_its_limit() -> None:
    """A fixed point, not a search. Hitting the pass cap means it is not converging."""
    analysis, bundle = _analysis()
    joint = optimize_jointly(
        analysis.airframe,
        analysis.chain,
        analysis.noise,
        CONFIG,
        stack=bundle.stack,
        delay=analysis.delay,
        targets=_targets(),
        f_grid=design_grid(0.1, 200.0, 600),
        op=analysis.operating_point,
        axis="roll",
        track=analysis.track,
    )
    assert 1 <= joint.passes <= 3


def test_every_recommendation_explains_itself() -> None:
    """A number without a reason is a number to obey rather than to trust.

    An empty list of rejected alternatives is a legitimate answer -- sometimes
    nothing was in the running -- but an alternative listed without a reason is
    not.
    """
    analysis, bundle = _analysis()
    rec = recommend_from(analysis, bundle, CONFIG)

    assert rec.filters.rationale.strip()
    assert rec.binding_constraint
    assert "divided out" in rec.rationale, "the deconvolution must be stated, not implied"
    assert all(alternative and why for alternative, why in rec.filters.rejected)


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


def test_the_resolve_holds_the_interactive_budget() -> None:
    """The sandbox re-solves on every slider movement; 300 ms is the requirement."""
    analysis, bundle = _analysis()
    recommend_from(analysis, bundle, CONFIG)  # warm the import and JIT-free caches

    elapsed = []
    for conservatism in (0.2, 0.35, 0.5, 0.65, 0.8):
        start = time.perf_counter()
        recommend_from(analysis, bundle, CONFIG, conservatism=conservatism)
        elapsed.append(time.perf_counter() - start)

    # Best of five. A shared CI runner can stall any single iteration for longer
    # than the budget through no fault of this code; the fastest run is the least
    # noisy estimate of what the work actually costs.
    best = min(elapsed)
    assert best < 0.3, f"re-solve took {best * 1000:.0f} ms, over the 300 ms budget"


def _targets(conservatism: float = 0.5) -> DesignTargets:
    return DesignTargets(
        pm_min_deg=CONFIG.float_("margins", "pm_min_deg"),
        gm_min_db=CONFIG.float_("margins", "gm_min_db"),
        ms_max_db=CONFIG.float_("margins", "ms_max_db"),
        pm_floor_deg=CONFIG.float_("margins", "pm_floor_deg"),
        crossover_frac_of_loop=CONFIG.float_("margins", "crossover_frac_of_loop"),
        conservatism=conservatism,
    )
