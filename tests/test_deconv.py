"""The measured step response, and the check that compares it with the model.

The value of this module is that it is the only thing in the tool that closes the
loop back onto flown data. Everything else asks whether the identification was
*well-conditioned*; this asks whether it was *right*. So the tests are mostly
about the ground truth: the closed-loop simulator knows exactly what step its own
loop would produce, and the deconvolution has to find it without being told.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from rotorid.config import Config, load_config
from rotorid.core.analysis.deconv import measured_step
from rotorid.core.design.recommend import identify_axis, recommend_from
from rotorid.core.guidance.findings import GuidanceContext, collect_findings
from rotorid.core.types import Finding
from tests.synthetic.closed_loop import inner_loop_step, make_closed_loop_bundle

CONFIG = load_config()

#: Long enough for the stack to be a stack rather than a handful of windows.
DURATION_S = 90.0


def _tweaked(**overrides: float) -> Config:
    data = copy.deepcopy(CONFIG.data)
    data["deconv"].update(overrides)
    return Config(data=data, hash=CONFIG.hash, sources=CONFIG.sources)


def _codes(findings: tuple[Finding, ...]) -> set[str]:
    return {f.code for f in findings}


def _context(bundle):
    analysis = identify_axis(bundle, "roll", CONFIG)
    return GuidanceContext(
        bundle=bundle,
        analyses={"roll": analysis},
        recommendations={"roll": recommend_from(analysis, bundle, CONFIG)},
        config=CONFIG,
    )


# --------------------------------------------------------------------------- #
# Against ground truth
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("with_chirp", [True, False])
def test_the_flown_step_is_recovered_from_the_log(with_chirp: bool) -> None:
    """The whole point. The simulator's own inner loop, found without being told.

    Both cases matter and for different reasons: with a chirp this is the best
    evidence a log can carry, and without one it is an ordinary flight where the
    only excitation is the pilot moving the stick.
    """
    bundle = make_closed_loop_bundle(with_chirp=with_chirp, noise_rms=0.05, duration_s=DURATION_S)
    result = measured_step(bundle, "roll", CONFIG)
    assert result is not None

    _, truth = inner_loop_step(
        duration_s=result.t.size / bundle.sample_rate_hz,
        sample_rate_hz=bundle.sample_rate_hz,
    )
    error = float(np.sqrt(np.mean((truth - result.y) ** 2)))
    assert error < 0.10, f"recovered step is {error:.3f} rms away from the loop that made it"


def test_the_recovered_rise_time_is_slow_rather_than_wrong() -> None:
    """The regularizer is a low-pass, so the method reads slow. Documented, and
    pinned here: anything comparing this against a prediction has to allow for a
    bias in this direction and of roughly this size, and would start firing
    spuriously if it ever changed sign."""
    bundle = make_closed_loop_bundle(noise_rms=0.02, duration_s=DURATION_S)
    result = measured_step(bundle, "roll", CONFIG)
    assert result is not None

    t, truth = inner_loop_step(
        duration_s=result.t.size / bundle.sample_rate_hz,
        sample_rate_hz=bundle.sample_rate_hz,
    )
    from rotorid.core.analysis.step import step_metrics

    ratio = result.metrics.rise_time_s / step_metrics(t, truth).rise_time_s
    assert 1.0 < ratio < 1.4


# --------------------------------------------------------------------------- #
# Refusing
# --------------------------------------------------------------------------- #


def test_a_log_without_a_rate_setpoint_is_not_measured() -> None:
    """There is nothing to deconvolve against, and None means exactly that."""
    bundle = make_closed_loop_bundle(duration_s=20.0)
    signals = {k: v for k, v in bundle.signals.items() if k != "rate.roll.setpoint"}
    assert measured_step(replace(bundle, signals=signals), "roll", CONFIG) is None


def test_noise_is_refused_rather_than_smoothed_into_an_answer() -> None:
    """Stacking hundreds of windows produces a smooth curve whatever went in.

    Smoothness is not accuracy, and a smooth wrong step is worse than none: it is
    the plot a user would trust most.
    """
    quiet = make_closed_loop_bundle(noise_rms=0.05, duration_s=DURATION_S)
    loud = make_closed_loop_bundle(noise_rms=1.0, duration_s=DURATION_S)
    assert measured_step(quiet, "roll", CONFIG) is not None
    assert measured_step(loud, "roll", CONFIG) is None


def test_the_gate_is_on_what_the_response_explains_not_on_window_agreement() -> None:
    """Scatter measures precision, and averaging drives precision up regardless.

    With the explained-variance gate lifted, a noise level the tool should refuse
    comes back as a confident answer -- which is what makes the gate the load-
    bearing one rather than a tidy extra.
    """
    loud = make_closed_loop_bundle(noise_rms=0.5, duration_s=DURATION_S)
    assert measured_step(loud, "roll", CONFIG) is None

    ungated = measured_step(loud, "roll", _tweaked(min_explained=0.0))
    assert ungated is not None
    _, truth = inner_loop_step(
        duration_s=ungated.t.size / loud.sample_rate_hz, sample_rate_hz=loud.sample_rate_hz
    )
    assert float(np.sqrt(np.mean((truth - ungated.y) ** 2))) > 0.3


def test_a_stack_of_one_window_is_not_a_measurement() -> None:
    bundle = make_closed_loop_bundle(noise_rms=0.05, duration_s=DURATION_S)
    assert measured_step(bundle, "roll", _tweaked(min_windows=10_000.0)) is None


def test_how_many_windows_were_thrown_away_is_reported() -> None:
    """ "Measured from 4 windows out of 200" has to be visible, not implied."""
    bundle = make_closed_loop_bundle(noise_rms=0.2, duration_s=DURATION_S)
    result = measured_step(bundle, "roll", CONFIG)
    assert result is not None
    assert result.n_rejected > 0
    assert result.n_windows > result.n_rejected


# --------------------------------------------------------------------------- #
# The finding
# --------------------------------------------------------------------------- #


def test_a_model_that_describes_the_aircraft_is_said_to() -> None:
    bundle = make_closed_loop_bundle(with_chirp=True, noise_rms=0.05, duration_s=DURATION_S)
    findings = collect_findings(_context(bundle))
    assert "STEP_RESPONSE_AGREES" in _codes(findings)
    assert "STEP_RESPONSE_DISAGREES" not in _codes(findings)


def test_a_model_of_a_different_aircraft_is_caught() -> None:
    """The check has to be able to fail, or it is decoration.

    The airframe gain is quartered after identification and before the comparison,
    which is exactly the failure the check exists for: a model that is coherent,
    well fitted and about the wrong vehicle.
    """
    bundle = make_closed_loop_bundle(with_chirp=True, noise_rms=0.05, duration_s=DURATION_S)
    analysis = identify_axis(bundle, "roll", CONFIG)
    wrong = replace(
        analysis.airframe,
        params={**analysis.airframe.params, "K": analysis.airframe.params["K"] / 4.0},
    )
    from rotorid.core.design.recommend import _flown_prediction

    misidentified = replace(
        analysis,
        airframe=wrong,
        flown_prediction=_flown_prediction(
            bundle,
            "roll",
            wrong,
            analysis.chain,
            analysis.delay,
            analysis.operating_point,
            analysis.measured,
        ),
    )
    context = GuidanceContext(
        bundle=bundle,
        analyses={"roll": misidentified},
        recommendations={"roll": recommend_from(misidentified, bundle, CONFIG)},
        config=CONFIG,
    )
    codes = _codes(collect_findings(context))
    assert "STEP_RESPONSE_DISAGREES" in codes
    assert "STEP_RESPONSE_AGREES" not in codes


def test_a_log_that_cannot_support_a_measured_step_produces_neither_finding() -> None:
    """Silence, not a verdict. An unmeasurable step is not an agreeing one."""
    bundle = make_closed_loop_bundle(with_chirp=True, noise_rms=0.05, duration_s=DURATION_S)
    analysis = replace(identify_axis(bundle, "roll", CONFIG), measured=None, flown_prediction=None)
    context = GuidanceContext(
        bundle=bundle,
        analyses={"roll": analysis},
        recommendations={"roll": recommend_from(analysis, bundle, CONFIG)},
        config=CONFIG,
    )
    codes = _codes(collect_findings(context))
    assert "STEP_RESPONSE_AGREES" not in codes
    assert "STEP_RESPONSE_DISAGREES" not in codes


# --------------------------------------------------------------------------- #
# Carrying it
# --------------------------------------------------------------------------- #


def test_the_measurement_survives_a_session_round_trip(tmp_path) -> None:
    """The validation screen has to work on a reopened bundle, not only a fresh run."""
    from rotorid.core.export.session import load_session, save_session
    from rotorid.core.pipeline import analyze

    bundle = make_closed_loop_bundle(with_chirp=True, noise_rms=0.05, duration_s=DURATION_S)
    session = analyze(bundle, ("roll",), CONFIG, tool_version="test").session
    assert "roll" in session.measured_steps

    save_session(tmp_path / "flight.rotorid", session)
    reloaded, _ = load_session(tmp_path / "flight.rotorid")

    before = session.measured_steps["roll"]
    after = reloaded.measured_steps["roll"]
    assert np.array_equal(before.y, after.y)
    assert np.array_equal(before.spread, after.spread)
    assert after.n_windows == before.n_windows
    assert after.explained == before.explained


def test_the_report_prints_the_measured_step_beside_the_predicted_one(tmp_path) -> None:
    """The one number on the page that did not come out of the model."""
    from rotorid.core.export.report import write_report
    from rotorid.core.pipeline import analyze

    bundle = make_closed_loop_bundle(with_chirp=True, noise_rms=0.05, duration_s=DURATION_S)
    session = analyze(bundle, ("roll",), CONFIG, tool_version="test").session
    path = write_report(
        tmp_path / "report.html",
        bundle,
        {str(a): r for a, r in session.recommendations.items()},
        config_hash=CONFIG.hash,
        tool_version="test",
        findings=session.findings,
        measured_steps={str(a): m for a, m in session.measured_steps.items()},
    )
    text = path.read_text(encoding="utf-8")
    assert "measured from the log" in text.lower()
    assert "Windows stacked" in text
    # And it says what it is, so nobody reads it as a prediction of the new tune.
    assert "different tunes" in text


def test_a_report_without_a_measurement_says_so_rather_than_omitting_the_row(tmp_path) -> None:
    from rotorid.core.export.report import write_report
    from rotorid.core.pipeline import analyze

    bundle = make_closed_loop_bundle(with_chirp=True, noise_rms=0.05, duration_s=DURATION_S)
    session = analyze(bundle, ("roll",), CONFIG, tool_version="test").session
    path = write_report(
        tmp_path / "report.html",
        bundle,
        {str(a): r for a, r in session.recommendations.items()},
        config_hash=CONFIG.hash,
        tool_version="test",
    )
    assert "could not be measured" in path.read_text(encoding="utf-8")
