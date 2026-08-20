"""Operating-point sensitivity (spec section 5.9).

The claim under test is narrow and worth stating: a vehicle whose gain moves
across the envelope has not been identified badly, it has been identified at one
point. So these tests check that the spread is *measured* rather than absorbed
into the fit residual, that it is attributed only when the log can actually
attribute it, and that it reaches the design as held-back margin rather than as
a note in a report.
"""

from __future__ import annotations

import dataclasses

import pytest

from rotorid.config import load_config
from rotorid.core.design.recommend import identify_axis, recommend_from
from rotorid.core.guidance.findings import GuidanceContext, collect_findings
from rotorid.core.types import LogBundle
from tests.synthetic.generators import (
    make_airframe,
    make_chain,
    make_general_flight_bundle,
)

CONFIG = load_config()

#: What a well-linearized vehicle and a badly mis-set one look like, as a
#: fractional change in K per unit of throttle.
LINEAR = 0.0
MIS_SET = 0.8


def _flight(gain_per_throttle: float, **kwargs: object) -> LogBundle:
    return make_general_flight_bundle(
        make_airframe(), make_chain(), gain_per_throttle=gain_per_throttle, **kwargs
    )


def _findings(bundle: LogBundle) -> dict[str, object]:
    analysis = identify_axis(bundle, "roll", CONFIG)
    rec = recommend_from(analysis, bundle, CONFIG)
    found = collect_findings(
        GuidanceContext(
            bundle=bundle,
            analyses={"roll": analysis},
            recommendations={"roll": rec},
            config=CONFIG,
        )
    )
    return {f.code: f for f in found}


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def test_a_linear_vehicle_shows_almost_no_spread() -> None:
    analysis = identify_axis(_flight(LINEAR), "roll", CONFIG)
    assert analysis.spread is not None
    assert analysis.spread.spread_pct < CONFIG.float_("operating_point", "warn_spread_pct")


def test_a_mis_set_thrust_curve_shows_up_as_a_spread() -> None:
    analysis = identify_axis(_flight(MIS_SET), "roll", CONFIG)
    assert analysis.spread is not None
    assert analysis.spread.spread_pct > 25.0
    assert analysis.airframe.gain_spread_pct == pytest.approx(analysis.spread.spread_pct)


def test_the_spread_needs_more_than_two_operating_points() -> None:
    """Two points define a line through any two points."""
    analysis = identify_axis(_flight(MIS_SET, n_bursts=2), "roll", CONFIG)
    assert analysis.spread is None


def test_a_tuning_flight_is_not_given_a_spread_it_cannot_measure() -> None:
    """A sweep is flown at one throttle. Three fits of one point is not a spread."""
    bundle = dataclasses.replace(_flight(MIS_SET), declared_kind="tuning")
    # Declared as tuning, the same file has no deliberate excitation at all, so
    # the check is on the capability rather than on the numbers.
    assert not bundle.signals.keys() & {"excite.roll"}
    with pytest.raises(ValueError):
        identify_axis(bundle, "roll", CONFIG)


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #


def test_a_spread_that_tracks_throttle_names_the_thrust_curve() -> None:
    """Voltage falls monotonically in the fixture; throttle does not follow it."""
    analysis = identify_axis(_flight(MIS_SET), "roll", CONFIG)
    assert analysis.spread is not None
    assert analysis.spread.attributed_to_throttle
    assert not analysis.spread.attributed_to_voltage


def test_the_finding_says_which_parameter_to_look_at() -> None:
    found = _findings(_flight(MIS_SET))
    assert "THRUST_LINEARIZATION_SUSPECT" in found
    finding = found["THRUST_LINEARIZATION_SUSPECT"]
    assert "MOT_THST_EXPO" in finding.action  # type: ignore[attr-defined]
    assert finding.severity == "warning"  # type: ignore[attr-defined]


def test_a_stable_vehicle_is_told_so_rather_than_left_silent() -> None:
    found = _findings(_flight(LINEAR))
    assert "OPERATING_POINT_STABLE" in found
    assert found["OPERATING_POINT_STABLE"].severity == "good"  # type: ignore[attr-defined]


def test_two_variables_moving_together_is_reported_as_unattributable() -> None:
    """Two strong correlations make a weaker claim than one, not a stronger one.

    With throttle rising monotonically through the flight it tracks the falling
    pack voltage almost exactly, so no log of this shape can say which of the two
    the gain followed -- and the tool must not pick one.
    """
    bundle = _flight(MIS_SET, throttles=(0.30, 0.42, 0.55, 0.68, 0.80))
    analysis = identify_axis(bundle, "roll", CONFIG)
    assert analysis.spread is not None
    assert analysis.spread.attributed_to_throttle
    assert analysis.spread.attributed_to_voltage

    found = _findings(bundle)
    assert "OPERATING_POINT_SPREAD" in found
    assert "THRUST_LINEARIZATION_SUSPECT" not in found
    assert "cannot be read off this log" in found["OPERATING_POINT_SPREAD"].detail  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# What it changes
# --------------------------------------------------------------------------- #


def test_a_moving_gain_buys_the_design_extra_caution() -> None:
    """The only response that follows from the evidence.

    The model is right at one point in the envelope and the aircraft is flown
    across all of it, so the margin that covers the difference has to come from
    somewhere.
    """
    steady = _flight(LINEAR)
    moving = _flight(MIS_SET)
    quiet = recommend_from(identify_axis(steady, "roll", CONFIG), steady, CONFIG)
    lively = recommend_from(identify_axis(moving, "roll", CONFIG), moving, CONFIG)
    assert lively.conservatism > quiet.conservatism


def test_a_severe_spread_caps_the_confidence() -> None:
    bundle = _flight(1.6)
    analysis = identify_axis(bundle, "roll", CONFIG)
    assert analysis.spread is not None
    assert analysis.spread.spread_pct > CONFIG.float_("operating_point", "severe_spread_pct")
    assert recommend_from(analysis, bundle, CONFIG).confidence == "low"
