"""Oscillation detection, and the constraint it puts on the design (plan 2.3).

The property worth protecting is in the last section. Detection on its own would
be an annotation, and the failure this exists to prevent is not "the tool did not
mention the oscillation" -- it is "the tool looked at an oscillating aircraft, saw
a healthy model, and recommended more gain".

The fixtures are the closed-loop simulator driven towards its stability limit,
either by proportional gain (which brings the crossover up into the delay) or by
derivative gain (which lifts the loop at high frequency until the gyro low-pass's
phase closes the circle). Both are real ways to make a multirotor ring and they
ring at different frequencies, which is the point of testing both.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rotorid.config import load_config
from rotorid.core.analysis.margins import loop_delay
from rotorid.core.analysis.oscillation import detect_oscillation, model_optimism_db
from rotorid.core.design.controller import controller_for
from rotorid.core.design.recommend import identify_axis, recommend_from
from rotorid.core.filters.chain import OperatingPoint
from rotorid.core.guidance.findings import GuidanceContext, collect_findings
from rotorid.core.types import Finding, GainSet
from tests.synthetic.closed_loop import make_closed_loop_bundle
from tests.synthetic.generators import make_airframe, make_chain

CONFIG = load_config()

#: Gains flown by each fixture. The healthy one is ArduPilot's own default shape;
#: the other two were found by walking each gain up until the modelled loop lost
#: its margin -- 0 degrees of phase margin for the proportional case, 13 for the
#: derivative one.
HEALTHY = (0.135, 0.135, 0.0036)
RINGING_P = (0.45, 0.135, 0.0036)
RINGING_D = (0.135, 0.135, 0.02)

DURATION_S = 90.0


def _bundle(gains: tuple[float, float, float]):
    return make_closed_loop_bundle(gains=gains, noise_rms=0.05, duration_s=DURATION_S)


def _codes(findings: tuple[Finding, ...]) -> set[str]:
    return {f.code for f in findings}


def _context(bundle, *, oscillation=None):
    analysis = identify_axis(bundle, "roll", CONFIG)
    if oscillation is not None:
        analysis = replace(analysis, oscillation=oscillation)
    return GuidanceContext(
        bundle=bundle,
        analyses={"roll": analysis},
        recommendations={"roll": recommend_from(analysis, bundle, CONFIG)},
        config=CONFIG,
    )


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def test_a_healthy_tune_is_not_called_oscillating() -> None:
    """The negative case first. A detector that always fires is not a detector."""
    assert detect_oscillation(_bundle(HEALTHY), "roll", CONFIG, gyro_lpf_hz=60.0) is None


@pytest.mark.parametrize(
    ("gains", "expected_hz"), [(RINGING_P, 5.5), (RINGING_D, 8.6)], ids=["p_gain", "d_gain"]
)
def test_a_ringing_loop_is_found_at_the_right_frequency(
    gains: tuple[float, float, float], expected_hz: float
) -> None:
    found = detect_oscillation(_bundle(gains), "roll", CONFIG, gyro_lpf_hz=60.0)
    assert found is not None
    assert found.f_hz == pytest.approx(expected_hz, rel=0.2)
    assert found.duty > 0.5, "a limit cycle is present most of the time, not once"
    assert found.amplitude_frac > 0.5, "it should dominate what the aircraft is doing"


def test_the_discriminator_is_amplification_rather_than_absence_from_the_command() -> None:
    """The obvious test fails exactly where it matters, so this pins the choice.

    An attitude loop feeds a 5 Hz oscillation straight back into the rate
    setpoint, so "is it in the command too?" dismisses a genuine limit cycle at
    the frequency multirotors most often ring at. The ratio of measured power to
    commanded power does not have that problem, because a resonance raises both
    of its terms together.
    """
    ringing = detect_oscillation(_bundle(RINGING_P), "roll", CONFIG, gyro_lpf_hz=60.0)
    assert ringing is not None
    assert ringing.amplification_db > 10.0


def test_the_search_stops_below_the_gyro_low_pass() -> None:
    """Above it the loop has no authority, so a tone there is not a control problem."""
    bundle = _bundle(RINGING_D)
    assert detect_oscillation(bundle, "roll", CONFIG, gyro_lpf_hz=8.0) is None
    assert detect_oscillation(bundle, "roll", CONFIG, gyro_lpf_hz=60.0) is not None


def test_a_transient_is_not_an_oscillation() -> None:
    """Duty is what separates a tune that rings from a moment that did."""
    bundle = _bundle(HEALTHY)
    signal = bundle.signals["rate.roll.measured"]
    t, y = signal.t, signal.y.copy()
    burst = t < t[0] + 0.05 * (t[-1] - t[0])
    y[burst] += 1.5 * np.sin(2.0 * np.pi * 12.0 * t[burst])
    signals = dict(bundle.signals)
    signals["rate.roll.measured"] = replace(signal, y=y)

    assert detect_oscillation(replace(bundle, signals=signals), "roll", CONFIG) is None


# --------------------------------------------------------------------------- #
# Pricing the model's error
# --------------------------------------------------------------------------- #


def test_the_optimism_is_zero_when_the_model_agrees_the_loop_was_at_its_limit() -> None:
    """On this fixture the identification is right, so there is nothing to correct.

    Worth asserting rather than assuming: an optimism figure that came back
    non-zero here would mean the tool was inventing a correction out of a model
    that had nothing wrong with it.
    """
    bundle = _bundle(RINGING_P)
    analysis = identify_axis(bundle, "roll", CONFIG)
    assert analysis.oscillation is not None
    assert analysis.oscillation.model_optimism_db < 2.0


def test_optimism_is_the_gain_margin_the_model_claims_where_the_aircraft_rings() -> None:
    """A loop with plenty of margin at 30 Hz, told the aircraft oscillates there."""
    airframe, chain = make_airframe(), make_chain()
    controller = controller_for(
        "ardupilot",
        GainSet(
            axis="roll",
            kp=0.135,
            ki=0.135,
            kd=0.0036,
            kff=0.0,
            dterm_lpf_hz=chain.dterm_lpf_hz,
            error_lpf_hz=chain.error_lpf_hz,
            target_lpf_hz=chain.target_lpf_hz,
        ),
        chain,
    )
    delay = loop_delay(loop_rate_hz=400.0, actuator_ms=0.1, zoh_loops=0.5, compute_loops=1.0)
    optimism = model_optimism_db(
        30.0, controller, airframe, delay=delay, op=OperatingPoint(motor_hz=(50.0,))
    )
    assert optimism > 10.0, "a healthy loop claims a lot of margin up there, and is wrong"


# --------------------------------------------------------------------------- #
# The constraint -- the part that matters
# --------------------------------------------------------------------------- #


def test_a_measured_oscillation_makes_the_design_back_off() -> None:
    """The property this whole module exists for.

    Same log, same identification, same everything -- except the model is told the
    aircraft was measured oscillating where the model claims 9 dB of margin. The
    recommendation has to come out softer, and has to say that is why.
    """
    from rotorid.core.analysis.oscillation import Oscillation

    bundle = _bundle(HEALTHY)
    analysis = identify_axis(bundle, "roll", CONFIG)
    baseline = recommend_from(analysis, bundle, CONFIG)

    contradicted = replace(
        analysis,
        oscillation=Oscillation(
            axis="roll",
            f_hz=30.0,
            excess_db=14.0,
            duty=0.8,
            amplitude_rad_s=0.3,
            amplitude_frac=0.5,
            amplification_db=16.0,
            model_optimism_db=9.0,
        ),
    )
    held_back = recommend_from(contradicted, bundle, CONFIG)

    assert held_back.gains.kp < baseline.gains.kp
    assert held_back.margins.gain_margin_db > baseline.margins.gain_margin_db
    assert held_back.binding_constraint == "measured_oscillation"


def test_the_holdback_is_capped_rather_than_unbounded() -> None:
    """Past the cap the model is not describing the aircraft, and the answer to
    that is the blocker, not a tune backed off into uselessness."""
    from rotorid.core.analysis.oscillation import Oscillation

    bundle = _bundle(HEALTHY)
    analysis = identify_axis(bundle, "roll", CONFIG)
    absurd = replace(
        analysis,
        oscillation=Oscillation(
            axis="roll",
            f_hz=30.0,
            excess_db=14.0,
            duty=0.8,
            amplitude_rad_s=0.3,
            amplitude_frac=0.5,
            amplification_db=16.0,
            model_optimism_db=60.0,
        ),
    )
    capped = recommend_from(absurd, bundle, CONFIG)
    cap = CONFIG.float_("oscillation", "max_gm_holdback_db")
    base = CONFIG.float_("margins", "gm_min_db")
    # It obeys the capped target, and does not chase the uncapped one into a
    # design space with nothing in it.
    assert capped.margins.gain_margin_db >= base + cap
    assert capped.gains.kp > 0.0


def test_the_conservatism_slider_cannot_trade_the_holdback_away() -> None:
    """The holdback is the size of a demonstrated error, not a preference."""
    from rotorid.core.analysis.oscillation import Oscillation

    bundle = _bundle(HEALTHY)
    analysis = replace(
        identify_axis(bundle, "roll", CONFIG),
        oscillation=Oscillation(
            axis="roll",
            f_hz=30.0,
            excess_db=14.0,
            duty=0.8,
            amplitude_rad_s=0.3,
            amplitude_frac=0.5,
            amplification_db=16.0,
            model_optimism_db=9.0,
        ),
    )
    aggressive = recommend_from(analysis, bundle, CONFIG, conservatism=0.0)
    floor = CONFIG.float_("margins", "gm_min_db") + 9.0
    assert aggressive.margins.gain_margin_db >= floor


# --------------------------------------------------------------------------- #
# The finding
# --------------------------------------------------------------------------- #


def test_an_oscillating_aircraft_blocks_the_export() -> None:
    findings = collect_findings(_context(_bundle(RINGING_D)))
    assert "OSCILLATION_DETECTED" in _codes(findings)
    found = next(f for f in findings if f.code == "OSCILLATION_DETECTED")
    assert found.severity == "blocker"
    assert "halve" in found.action.lower()


def test_a_healthy_flight_produces_no_oscillation_finding() -> None:
    assert "OSCILLATION_DETECTED" not in _codes(collect_findings(_context(_bundle(HEALTHY))))


def test_the_finding_says_whether_the_model_agreed() -> None:
    """Two quite different situations, and the user needs to be told which."""
    from rotorid.core.analysis.oscillation import Oscillation

    agreeing = next(
        f
        for f in collect_findings(_context(_bundle(RINGING_P)))
        if f.code == "OSCILLATION_DETECTED"
    )
    assert "same story" in agreeing.detail

    contradicted = next(
        f
        for f in collect_findings(
            _context(
                _bundle(HEALTHY),
                oscillation=Oscillation(
                    axis="roll",
                    f_hz=30.0,
                    excess_db=14.0,
                    duty=0.8,
                    amplitude_rad_s=0.3,
                    amplitude_frac=0.5,
                    amplification_db=16.0,
                    model_optimism_db=9.0,
                ),
            )
        )
        if f.code == "OSCILLATION_DETECTED"
    )
    assert "wrong by at least that much" in contradicted.detail


# --------------------------------------------------------------------------- #
# Measured D-term noise
# --------------------------------------------------------------------------- #


def _with_dterm(bundle, values: np.ndarray):
    from rotorid.core.io.base import canonical_signal

    signals = dict(bundle.signals)
    t = bundle.signals["rate.roll.measured"].t
    signals["rate.roll.d_term"] = canonical_signal(
        "rate.roll.d_term", t, values, source_msg="PIDR.D"
    )
    return replace(bundle, signals=signals)


def test_a_quiet_d_term_produces_no_finding() -> None:
    bundle = _bundle(HEALTHY)
    t = bundle.signals["rate.roll.measured"].t
    quiet = 0.001 * np.sin(2.0 * np.pi * 40.0 * t)
    assert "DTERM_NOISE_MEASURED" not in _codes(
        collect_findings(_context(_with_dterm(bundle, quiet)))
    )


def test_a_noisy_d_term_is_reported_from_the_log_rather_than_predicted() -> None:
    """Measured, so it stands whatever the noise model says."""
    bundle = _bundle(HEALTHY)
    t = bundle.signals["rate.roll.measured"].t
    loud = 0.15 * np.sin(2.0 * np.pi * 60.0 * t)
    findings = collect_findings(_context(_with_dterm(bundle, loud)))
    assert "DTERM_NOISE_MEASURED" in _codes(findings)
    found = next(f for f in findings if f.code == "DTERM_NOISE_MEASURED")
    assert found.evidence["measured_pct"] > found.evidence["limit_pct"]


def test_control_work_below_the_band_is_not_counted_as_noise() -> None:
    """A large D term at 1 Hz is the derivative doing its job, not heat."""
    bundle = _bundle(HEALTHY)
    t = bundle.signals["rate.roll.measured"].t
    working = 0.15 * np.sin(2.0 * np.pi * 1.0 * t)
    assert "DTERM_NOISE_MEASURED" not in _codes(
        collect_findings(_context(_with_dterm(bundle, working)))
    )


def test_a_log_without_pid_messages_reports_nothing_rather_than_zero() -> None:
    analysis = identify_axis(_bundle(HEALTHY), "roll", CONFIG)
    assert analysis.dterm_measured_pct is None
