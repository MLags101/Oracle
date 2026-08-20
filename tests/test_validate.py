"""Validation mode: before against after, prediction against outcome (M10).

The thing being tested is a distinction rather than an algorithm. Three claims
live in this feature -- the aircraft changed, the aircraft improved, and the tool
was right -- and the whole value of the screen is that they never get mistaken
for each other. So most of what follows checks that the report refuses to make
the third claim when it has no grounds to.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from rotorid import __version__
from rotorid.config import load_config
from rotorid.core.analysis.compare import compare_logs
from rotorid.core.design.recommend import identify_axis, recommend_from
from rotorid.core.export.comparison import write_comparison
from rotorid.core.guidance.validation import validation_findings
from rotorid.core.pipeline import analyze
from rotorid.core.types import LogBundle, Session, StepMetrics
from tests.synthetic.closed_loop import make_closed_loop_bundle
from tests.synthetic.generators import make_airframe, make_bundle, make_chain

CONFIG = load_config()


@pytest.fixture(scope="module")
def before() -> LogBundle:
    """A closed-loop flight, because validation needs the setpoint.

    Everything this feature reads -- the flown step, the tracking error, the
    D-term noise -- is a comparison of what was asked for against what happened,
    so a fixture that logs only the output and the measurement cannot exercise
    any of it.
    """
    return make_closed_loop_bundle(path="before.bin")


@pytest.fixture(scope="module")
def session(before: LogBundle) -> Session:
    return analyze(before, ("roll",), CONFIG, tool_version=__version__).session


def _after(before: LogBundle, session: Session, *, apply_gains: bool = True) -> LogBundle:
    """A second flight of the same aircraft, optionally flying the new gains.

    Built by relabelling the same flight rather than re-simulating one. What the
    comparison reads off an after-log is its measured step, its spectrum and the
    gains in its parameter snapshot -- so a fixture that changes exactly the
    parameter snapshot isolates the thing under test, which is how the report
    treats what it finds rather than whether the aircraft really improved.
    """
    params = dict(before.params)
    if apply_gains:
        gains = session.recommendations["roll"].gains
        params.update(
            {
                "ATC_RAT_RLL_P": gains.kp,
                "ATC_RAT_RLL_I": gains.ki,
                "ATC_RAT_RLL_D": gains.kd,
            }
        )
    return dataclasses.replace(before, path=Path("after.bin"), params=params)


# --------------------------------------------------------------------------- #
# Scope: what a comparison is allowed to claim
# --------------------------------------------------------------------------- #


def test_without_a_session_it_is_an_outcome_comparison_not_a_validation(
    before: LogBundle, session: Session
) -> None:
    """Nothing recorded what was predicted, so nothing can say the tool was right."""
    report = compare_logs(
        before, _after(before, session), CONFIG, tool_version=__version__, axes=("roll",)
    )
    assert not report.has_predictions
    assert report.predicted_from is None
    assert report.axes["roll"].predicted_step is None
    assert report.axes["roll"].prediction_holds is None


def test_with_a_session_the_prediction_is_carried_through(
    before: LogBundle, session: Session
) -> None:
    report = compare_logs(
        before,
        _after(before, session),
        CONFIG,
        tool_version=__version__,
        session=session,
        axes=("roll",),
    )
    assert report.has_predictions
    assert report.predicted_from == "before.bin"
    assert report.axes["roll"].predicted_step is session.recommendations["roll"].predicted_step


def test_the_report_says_which_of_the_two_it_is(
    before: LogBundle, session: Session, tmp_path: Path
) -> None:
    """Stated in the document, before any number in it."""
    outcome = tmp_path / "outcome.html"
    write_comparison(
        outcome,
        compare_logs(
            before, _after(before, session), CONFIG, tool_version=__version__, axes=("roll",)
        ),
    )
    assert "not a validation" in outcome.read_text(encoding="utf-8")

    validated = tmp_path / "validated.html"
    write_comparison(
        validated,
        compare_logs(
            before,
            _after(before, session),
            CONFIG,
            tool_version=__version__,
            session=session,
            axes=("roll",),
        ),
    )
    assert "This is a validation" in validated.read_text(encoding="utf-8")


def test_two_stacks_are_refused_rather_than_compared(before: LogBundle) -> None:
    px4 = make_bundle(make_airframe(), make_chain(), stack="px4")
    with pytest.raises(ValueError, match="different quantities"):
        compare_logs(before, px4, CONFIG, tool_version=__version__)


# --------------------------------------------------------------------------- #
# A prediction is only tested against a flight that flew it
# --------------------------------------------------------------------------- #


def test_gains_that_were_never_loaded_are_not_a_failed_prediction(
    before: LogBundle, session: Session
) -> None:
    """The staged plan loads filters one flight and gains the next.

    An after-log flying the old gains is the expected outcome of following that
    plan, so reading it as a missed prediction would send the user to debug the
    tool over doing exactly what they were told.
    """
    report = compare_logs(
        before,
        _after(before, session, apply_gains=False),
        CONFIG,
        tool_version=__version__,
        session=session,
        axes=("roll",),
    )
    assert report.axes["roll"].applied is False
    codes = {f.code for f in validation_findings(report)}
    assert "TUNE_NOT_APPLIED" in codes
    assert "PREDICTION_MISSED" not in codes
    assert "PREDICTION_CONFIRMED" not in codes


def test_a_flight_that_did_load_them_is_checked(before: LogBundle, session: Session) -> None:
    report = compare_logs(
        before,
        _after(before, session),
        CONFIG,
        tool_version=__version__,
        session=session,
        axes=("roll",),
    )
    assert report.axes["roll"].applied is True
    codes = {f.code for f in validation_findings(report)}
    assert "TUNE_NOT_APPLIED" not in codes
    assert codes & {"PREDICTION_CONFIRMED", "PREDICTION_MISSED"}


def test_a_prediction_far_from_the_flown_step_is_called_out(
    before: LogBundle, session: Session
) -> None:
    """A model wrong by a factor has to say so, because everything rests on it."""
    wrong = dataclasses.replace(
        session,
        recommendations={
            "roll": dataclasses.replace(
                session.recommendations["roll"],
                predicted_step=StepMetrics(
                    rise_time_s=0.004,
                    overshoot_pct=60.0,
                    settling_time_s=0.02,
                    peak_time_s=0.008,
                    steady_state_error=0.0,
                ),
            )
        },
    )
    report = compare_logs(
        before,
        _after(before, wrong),
        CONFIG,
        tool_version=__version__,
        session=wrong,
        axes=("roll",),
    )
    assert report.axes["roll"].prediction_holds is False
    codes = {f.code for f in validation_findings(report)}
    assert "PREDICTION_MISSED" in codes


# --------------------------------------------------------------------------- #
# Filters: the half that normally goes unchecked
# --------------------------------------------------------------------------- #


def test_a_filter_prediction_that_matches_the_flight_is_confirmed(
    before: LogBundle, session: Session
) -> None:
    report = compare_logs(
        before,
        _after(before, session),
        CONFIG,
        tool_version=__version__,
        session=session,
        axes=("roll",),
    )
    comparison = report.axes["roll"]
    if comparison.filter_prediction_error_db is None:
        pytest.skip("this fixture carries no predicted post-filter spectrum")
    codes = {f.code for f in validation_findings(report)}
    assert codes & {"FILTER_PREDICTION_CONFIRMED", "FILTER_PREDICTION_MISSED"}


def test_a_filter_that_under_delivers_reads_as_noisier_not_merely_different(
    before: LogBundle, session: Session
) -> None:
    """The sign matters: too much attenuation and too little are different faults."""
    recommendation = session.recommendations["roll"]
    if recommendation.filters.predicted_psd_post is None:
        pytest.skip("this fixture carries no predicted post-filter spectrum")
    quiet = dataclasses.replace(
        recommendation.filters,
        predicted_psd_post=recommendation.filters.predicted_psd_post * 0.01,
    )
    optimistic = dataclasses.replace(
        session,
        recommendations={"roll": dataclasses.replace(recommendation, filters=quiet)},
    )
    report = compare_logs(
        before,
        _after(before, optimistic),
        CONFIG,
        tool_version=__version__,
        session=optimistic,
        axes=("roll",),
    )
    error = report.axes["roll"].filter_prediction_error_db
    assert error is not None and error > 3.0
    missed = [f for f in validation_findings(report) if f.code == "FILTER_PREDICTION_MISSED"]
    assert missed and "noisier" in missed[0].title


# --------------------------------------------------------------------------- #
# Outcome numbers
# --------------------------------------------------------------------------- #


def test_an_unchanged_aircraft_is_reported_as_unchanged(
    before: LogBundle, session: Session
) -> None:
    """Two flights of the same aircraft must not generate a wall of findings."""
    report = compare_logs(
        before, _after(before, session), CONFIG, tool_version=__version__, axes=("roll",)
    )
    comparison = report.axes["roll"]
    assert comparison.tracking_change == pytest.approx(0.0, abs=1e-9)
    codes = {f.code for f in validation_findings(report)}
    assert not codes & {"TRACKING_IMPROVED", "TRACKING_WORSE"}


def test_tracking_error_is_measured_from_both_logs(before: LogBundle, session: Session) -> None:
    noisier = dataclasses.replace(
        before,
        path=Path("after.bin"),
        signals={
            **before.signals,
            "rate.roll.measured": dataclasses.replace(
                before.signals["rate.roll.measured"],
                y=before.signals["rate.roll.measured"].y * 1.5,
            ),
        },
    )
    report = compare_logs(before, noisier, CONFIG, tool_version=__version__, axes=("roll",))
    comparison = report.axes["roll"]
    assert comparison.before_tracking_rms is not None
    assert comparison.after_tracking_rms is not None
    assert comparison.after_tracking_rms > comparison.before_tracking_rms


def test_an_axis_missing_from_one_log_is_dropped_rather_than_half_reported(
    before: LogBundle, session: Session
) -> None:
    stripped = dataclasses.replace(
        before,
        path=Path("after.bin"),
        signals={k: v for k, v in before.signals.items() if k != "rate.roll.measured"},
    )
    report = compare_logs(before, stripped, CONFIG, tool_version=__version__, axes=("roll",))
    assert "roll" not in report.axes
    assert any("not present in both logs" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# The document
# --------------------------------------------------------------------------- #


def test_the_html_report_is_self_contained(
    before: LogBundle, session: Session, tmp_path: Path
) -> None:
    """One file a user can attach to a forum post -- no assets, no scripts."""
    path = tmp_path / "comparison.html"
    write_comparison(
        path,
        compare_logs(
            before,
            _after(before, session),
            CONFIG,
            tool_version=__version__,
            session=session,
            axes=("roll",),
        ),
    )
    text = path.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "<script" not in text
    assert "src=" not in text
    assert "before.bin" in text and "after.bin" in text
    assert "<svg" in text


def test_the_report_survives_a_pair_with_almost_nothing_in_it(tmp_path: Path) -> None:
    """Two logs that can barely be compared still produce a readable document."""
    bare = make_bundle(make_airframe(), make_chain(), path="bare.bin")
    empty = ValidationReportFixture(bare)
    path = tmp_path / "bare.html"
    write_comparison(path, empty.report)
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")


class ValidationReportFixture:
    """A comparison with no axes in it at all, built without going near a log."""

    def __init__(self, bundle: LogBundle) -> None:
        from rotorid.core.analysis.compare import ValidationReport

        self.report = ValidationReport(
            before=bundle,
            after=bundle,
            axes={},
            tool_version=__version__,
            created_utc=datetime.now(UTC),
        )


def test_a_measured_step_is_drawn_with_its_spread(
    before: LogBundle, session: Session, tmp_path: Path
) -> None:
    """A mean over windows that disagree is not the same measurement as one that agrees."""
    report = compare_logs(
        before, _after(before, session), CONFIG, tool_version=__version__, axes=("roll",)
    )
    if report.axes["roll"].after_step is None:
        pytest.skip("this fixture produced no measurable step")
    path = tmp_path / "steps.html"
    write_comparison(path, report)
    text = path.read_text(encoding="utf-8")
    assert "fill-opacity" in text, "the spread band should be drawn, not just the mean"


def test_identification_is_never_required(before: LogBundle) -> None:
    """Most after-logs have no usable excitation; the point of the flight was to fly.

    Stripping the injected chirp makes the log unidentifiable as a tuning flight,
    and the comparison still has to work on it -- everything it reads comes off
    the signals directly.
    """
    blind = dataclasses.replace(
        before,
        path=Path("after.bin"),
        signals={k: v for k, v in before.signals.items() if not k.startswith("excite.")},
        declared_kind="tuning",
    )
    with pytest.raises(ValueError):
        identify_axis(blind, "roll", CONFIG)

    report = compare_logs(before, blind, CONFIG, tool_version=__version__, axes=("roll",))
    assert "roll" in report.axes
    assert report.axes["roll"].after_tracking_rms is not None


def test_a_recommendation_can_be_re_solved_without_re_identifying(before: LogBundle) -> None:
    """Guards the split validation depends on: measurement and design are separable."""
    analysis = identify_axis(before, "roll", CONFIG)
    first = recommend_from(analysis, before, CONFIG)
    second = recommend_from(analysis, before, CONFIG)
    assert first.gains.kp == pytest.approx(second.gains.kp)
    assert np.isfinite(first.predicted_step.rise_time_s)


def test_a_re_flown_recommendation_is_confirmed_against_the_aircraft(before: LogBundle) -> None:
    """The end-to-end claim M10 exists to make, on a vehicle we control exactly.

    Recommend gains from one flight, fly the *same simulated airframe* again with
    those gains, and the predicted step has to match the one deconvolved from the
    new flight. Nothing about that is guaranteed by construction: the prediction
    comes from a fitted model driven through the controller model, and the
    measurement comes from a regularized deconvolution of a closed-loop
    simulation. They agree only if the identification, the controller model and
    the step recovery are all right at once, which is exactly the claim.

    Every other fixture in this file relabels the before-flight's parameters,
    which tests how the report treats what it finds. This one actually flies it.
    """
    session = analyze(before, ("roll",), CONFIG, tool_version=__version__).session
    gains = session.recommendations["roll"].gains

    reflown = make_closed_loop_bundle(path="after.bin", gains=(gains.kp, gains.ki, gains.kd))
    report = compare_logs(
        before,
        reflown,
        CONFIG,
        tool_version=__version__,
        session=session,
        axes=("roll",),
    )
    comparison = report.axes["roll"]
    assert comparison.applied is True, "the re-flown log has to be flying the recommendation"
    assert comparison.after_step is not None
    assert comparison.prediction_holds is True, (
        f"rise ratio {comparison.rise_ratio}, "
        f"measured overshoot {comparison.after_step.metrics.overshoot_pct:.1f}%, "
        f"predicted {comparison.predicted_step.overshoot_pct:.1f}%"
        if comparison.predicted_step
        else ""
    )
    assert "PREDICTION_CONFIRMED" in {f.code for f in validation_findings(report)}
