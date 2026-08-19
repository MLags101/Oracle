"""Controller models, margins, and the margin-constrained gain search.

The properties asserted here are the ones a wrong answer would violate quietly:
margins that round-trip, a design that respects its constraints rather than
reporting them, and the structural difference between the two stacks' loops.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotorid.core.analysis.margins import (
    broken_loop,
    compute_margins,
    design_grid,
    loop_delay,
)
from rotorid.core.design.controller import controller_for
from rotorid.core.design.objectives import DesignTargets, design_gains
from rotorid.core.types import GainSet
from tests.synthetic.generators import make_airframe, make_chain

F_GRID = design_grid(0.1, 200.0, 900)
DELAY = loop_delay(loop_rate_hz=400.0, actuator_ms=0.1, zoh_loops=0.5, compute_loops=1.0)

TARGETS = DesignTargets(
    pm_min_deg=45.0,
    gm_min_db=6.0,
    ms_max_db=6.0,
    pm_floor_deg=25.0,
    crossover_frac_of_loop=0.2,
)


def _gains(kp: float = 0.1, ki: float = 0.1, kd: float = 0.003) -> GainSet:
    return GainSet(axis="roll", kp=kp, ki=ki, kd=kd, kff=0.0)


def _margins(stack, gains, chain, airframe):
    """Margins for one stack's controller model, on the shared design grid."""
    controller = controller_for(stack, gains, chain)
    return compute_margins(F_GRID, broken_loop(F_GRID, controller, airframe, delay=DELAY))


# --------------------------------------------------------------------------- #
# Loop assembly
# --------------------------------------------------------------------------- #


def test_loop_delay_terms_are_separate_and_sum() -> None:
    d = loop_delay(loop_rate_hz=400.0, actuator_ms=2.0, zoh_loops=0.5, compute_loops=1.0)
    assert d.zoh_s == pytest.approx(0.00125)
    assert d.compute_s == pytest.approx(0.0025)
    assert d.actuator_s == pytest.approx(0.002)
    assert d.total_s == pytest.approx(0.00575)


def test_delay_is_pure_phase() -> None:
    """A transport delay costs phase and nothing else; a magnitude change is a bug."""
    resp = DELAY.response(np.array([1.0, 10.0, 100.0]))
    assert np.allclose(np.abs(resp), 1.0)


def test_margins_round_trip_through_an_analytic_loop() -> None:
    """A pure integrator has 90 degrees of phase margin at its unity crossover."""
    f = design_grid(0.01, 100.0, 2000)
    w = 2.0 * np.pi * f
    L = 2.0 * np.pi * 5.0 / (1j * w)  # crosses unity at 5 Hz
    report = compute_margins(f, L)

    assert report.crossover_hz == pytest.approx(5.0, rel=0.01)
    assert report.phase_margin_deg == pytest.approx(90.0, abs=0.5)
    assert np.isinf(report.gain_margin_db)


def test_margins_reject_a_band_with_no_crossover() -> None:
    f = design_grid(0.1, 10.0, 100)
    with pytest.raises(ValueError, match="does not cross unity"):
        compute_margins(f, np.full(f.shape, 100.0 + 0j))


def test_delay_margin_matches_phase_margin_at_crossover() -> None:
    airframe, chain = make_airframe(), make_chain()
    controller = controller_for("ardupilot", _gains(), chain)
    L = broken_loop(F_GRID, controller, airframe, delay=DELAY)
    report = compute_margins(F_GRID, L)

    expected_ms = 1000.0 * np.radians(report.phase_margin_deg) / (2 * np.pi * report.crossover_hz)
    assert report.delay_margin_ms == pytest.approx(expected_ms, rel=1e-6)


# --------------------------------------------------------------------------- #
# The two stacks are structurally different
# --------------------------------------------------------------------------- #


def test_same_gains_give_different_margins_on_the_two_stacks() -> None:
    """ArduPilot's FLTE sits in the common feedback path; PX4 has no such filter.

    With an error filter configured the two loops cannot have the same margins,
    and a tool that shared one controller model between stacks would report that
    they do.
    """
    airframe = make_airframe()
    chain = make_chain(error_lpf_hz=30.0)
    gains = _gains()

    ap = _margins("ardupilot", gains, chain, airframe)
    px4 = _margins("px4", gains, chain, airframe)

    assert ap.phase_margin_deg < px4.phase_margin_deg, "FLTE costs ArduPilot phase"


def test_stacks_agree_when_no_error_filter_is_configured() -> None:
    """Without FLTE the two feedback paths are the same algebra, so margins match."""
    airframe = make_airframe()
    chain = make_chain(error_lpf_hz=None)
    gains = _gains()

    ap = _margins("ardupilot", gains, chain, airframe)
    px4 = _margins("px4", gains, chain, airframe)
    assert ap.phase_margin_deg == pytest.approx(px4.phase_margin_deg, abs=1e-9)


def test_reference_paths_differ_even_when_margins_agree() -> None:
    """D on error vs D on measurement: identical margins, different commanded step.

    This is the whole reason the two controller models exist, and it is invisible
    in the margin numbers.
    """
    chain = make_chain(error_lpf_hz=None)
    gains = _gains()
    f = np.array([1.0, 5.0, 20.0])

    ap = controller_for("ardupilot", gains, chain).reference_response(f)
    px4 = controller_for("px4", gains, chain).reference_response(f)
    assert not np.allclose(ap, px4)
    assert np.abs(ap[-1]) > np.abs(px4[-1]), "ArduPilot's D term acts on the setpoint too"


def test_target_filter_never_touches_the_margins() -> None:
    """FLTT is reference-path only. If it moves a margin, it is in the wrong place."""
    airframe = make_airframe()
    gains = _gains()
    without = make_chain(target_lpf_hz=None)
    with_fltt = make_chain(target_lpf_hz=10.0)

    a = compute_margins(
        F_GRID,
        broken_loop(F_GRID, controller_for("ardupilot", gains, without), airframe, delay=DELAY),
    )
    b = compute_margins(
        F_GRID,
        broken_loop(F_GRID, controller_for("ardupilot", gains, with_fltt), airframe, delay=DELAY),
    )
    assert a.phase_margin_deg == pytest.approx(b.phase_margin_deg, abs=1e-12)


def test_unknown_stack_raises_rather_than_guessing() -> None:
    with pytest.raises(ValueError, match="unknown stack"):
        controller_for("betaflight", _gains(), make_chain())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Gain design
# --------------------------------------------------------------------------- #


def test_design_meets_every_constraint_it_claims() -> None:
    airframe, chain = make_airframe(), make_chain()
    result = design_gains(
        airframe, chain, stack="ardupilot", delay=DELAY, targets=TARGETS, f_grid=F_GRID
    )

    assert result.margins.phase_margin_deg >= TARGETS.effective_pm_deg() - 0.5
    assert result.margins.gain_margin_db >= TARGETS.gm_min_db - 0.1
    assert result.margins.peak_sensitivity_db <= TARGETS.ms_max_db + 0.1
    assert result.margins.crossover_hz <= result.crossover_ceiling_hz * 1.01
    assert result.margins.disturbance_rejection_bw_hz > 0.0
    assert result.feasible_count > 0


def test_design_margins_are_reproducible_from_the_gains_alone() -> None:
    """Re-deriving the loop from the reported gains must give the reported margins.

    If these disagree, the optimizer is evaluating something other than what it
    exports -- the failure that turns a plausible report into a bad tune.
    """
    airframe, chain = make_airframe(), make_chain()
    result = design_gains(
        airframe, chain, stack="ardupilot", delay=DELAY, targets=TARGETS, f_grid=F_GRID
    )

    controller = controller_for("ardupilot", result.gains, chain)
    recomputed = compute_margins(F_GRID, broken_loop(F_GRID, controller, airframe, delay=DELAY))

    assert recomputed.phase_margin_deg == pytest.approx(result.margins.phase_margin_deg, abs=1e-9)
    assert recomputed.crossover_hz == pytest.approx(result.margins.crossover_hz, rel=1e-9)


def test_conservatism_slider_trades_bandwidth_for_margin() -> None:
    airframe, chain = make_airframe(), make_chain()
    aggressive = design_gains(
        airframe,
        chain,
        stack="ardupilot",
        delay=DELAY,
        targets=DesignTargets(45.0, 6.0, 6.0, 25.0, 0.2, conservatism=0.0),
        f_grid=F_GRID,
    )
    docile = design_gains(
        airframe,
        chain,
        stack="ardupilot",
        delay=DELAY,
        targets=DesignTargets(45.0, 6.0, 6.0, 25.0, 0.2, conservatism=1.0),
        f_grid=F_GRID,
    )

    assert docile.margins.crossover_hz < aggressive.margins.crossover_hz
    assert docile.margins.disturbance_rejection_bw_hz < (
        aggressive.margins.disturbance_rejection_bw_hz
    )
    assert docile.margins.phase_margin_deg >= aggressive.margins.phase_margin_deg


def test_conservatism_can_never_breach_the_hard_floor() -> None:
    """The slider is a preference. The 25 degree floor is not."""
    targets = DesignTargets(30.0, 6.0, 6.0, 25.0, 0.2, conservatism=0.0)
    assert targets.effective_pm_deg() == 25.0
    assert DesignTargets(45.0, 6.0, 6.0, 25.0, 0.2, conservatism=0.0).effective_pm_deg() == 35.0


def test_more_delay_forces_a_lower_crossover() -> None:
    """A slow ESC costs bandwidth, and the tool must say so rather than absorb it."""

    chain = make_chain()
    fast = design_gains(
        make_airframe(tau_ms=10.0),
        chain,
        stack="ardupilot",
        delay=DELAY,
        targets=TARGETS,
        f_grid=F_GRID,
    )
    slow = design_gains(
        make_airframe(tau_ms=45.0),
        chain,
        stack="ardupilot",
        delay=DELAY,
        targets=TARGETS,
        f_grid=F_GRID,
    )
    assert slow.margins.crossover_hz < fast.margins.crossover_hz
    assert slow.crossover_ceiling_hz < fast.crossover_ceiling_hz


def test_heavier_filtering_costs_achievable_bandwidth() -> None:
    """The joint-design premise: filters, not gains, are what limit most vehicles."""
    airframe = make_airframe()
    light = design_gains(
        airframe,
        make_chain(gyro_lpf_hz=120.0, dterm_lpf_hz=60.0),
        stack="ardupilot",
        delay=DELAY,
        targets=TARGETS,
        f_grid=F_GRID,
    )
    heavy = design_gains(
        airframe,
        make_chain(gyro_lpf_hz=20.0, dterm_lpf_hz=10.0),
        stack="ardupilot",
        delay=DELAY,
        targets=TARGETS,
        f_grid=F_GRID,
    )
    assert heavy.margins.disturbance_rejection_bw_hz < light.margins.disturbance_rejection_bw_hz


def test_feedforward_is_the_inverse_of_the_identified_gain() -> None:
    airframe = make_airframe(K=12.0)
    result = design_gains(
        airframe, make_chain(), stack="ardupilot", delay=DELAY, targets=TARGETS, f_grid=F_GRID
    )
    assert result.gains.kff == pytest.approx(1.0 / 12.0)


def test_binding_constraint_is_named() -> None:
    airframe, chain = make_airframe(), make_chain()
    result = design_gains(
        airframe, chain, stack="ardupilot", delay=DELAY, targets=TARGETS, f_grid=F_GRID
    )
    assert result.binding_constraint in {
        "phase_margin",
        "gain_margin",
        "peak_sensitivity",
        "crossover_limit_delay",
        "crossover_limit_loop_rate",
    }


def test_design_refuses_rather_than_relaxing_a_margin() -> None:
    """An impossible target must raise, not quietly produce a worse tune."""
    airframe, chain = make_airframe(), make_chain()
    impossible = DesignTargets(
        pm_min_deg=120.0,
        gm_min_db=40.0,
        ms_max_db=0.1,
        pm_floor_deg=25.0,
        crossover_frac_of_loop=0.2,
        conservatism=1.0,
    )
    with pytest.raises(ValueError, match="not the gains, is the problem"):
        design_gains(
            airframe, chain, stack="ardupilot", delay=DELAY, targets=impossible, f_grid=F_GRID
        )
