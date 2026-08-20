"""What a log is declared to be, and what that changes (spec 5.2).

The declaration is the one input the file cannot supply, so the tests here are
mostly about it being *honoured* rather than about it being clever: a tuning
flight is never quietly identified from stick input, a general flight is never
quietly promoted to high confidence, and a disagreement between what the user
said and what the file holds is reported rather than resolved.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rotorid.config import load_config
from rotorid.core.design.recommend import identify_axis, recommend_from
from rotorid.core.io.base import canonical_signal, gate_signal
from rotorid.core.logkind import capabilities, detect_kind, kind_evidence
from rotorid.core.preprocess.segment import propose_segments
from rotorid.core.types import LogBundle
from tests.synthetic.generators import make_airframe, make_bundle, make_chain

CONFIG = load_config()


def _general() -> LogBundle:
    """A flight with energetic single-axis activity and no record of who asked for it.

    All three axes are present -- the two quiet ones are what the ordinary-flight
    segmenter compares the excited one against -- and there is no ``excite.*``,
    which is what an ArduPilot log flown without SYSTEMID looks like.
    """
    return make_bundle(
        make_airframe(), make_chain(), with_motor_noise=True, record_excitation=False
    )


def _tuning() -> LogBundle:
    """The same flight, with the injected chirp recorded alongside it.

    Built from the general fixture rather than separately so the two differ in
    exactly one thing: whether the vehicle wrote down that it was injecting. That
    is the only difference between the two kinds in a real pair of logs too.
    """
    bundle = _general()
    output = bundle.signals["rate.roll.output"]
    return dataclasses.replace(
        bundle,
        signals={
            **bundle.signals,
            "excite.roll": canonical_signal("excite.roll", output.t, output.y, source_msg="SIDD"),
        },
    )


def _with_autotune(bundle: LogBundle) -> LogBundle:
    """The same flight, with the vehicle claiming its autotune ran throughout."""
    grid = next(iter(bundle.signals.values())).t
    return dataclasses.replace(
        bundle,
        signals={
            **bundle.signals,
            "mode.autotune": gate_signal(
                "mode.autotune", grid, [(float(grid[0]), float(grid[-1]))], source_msg="EV.Id"
            ),
        },
    )


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def test_a_recorded_chirp_makes_it_a_tuning_flight() -> None:
    bundle = _tuning()
    assert detect_kind(bundle) == "tuning"
    assert any("chirp" in line for line in kind_evidence(bundle))


def test_without_a_recorded_chirp_it_is_a_general_flight() -> None:
    bundle = _general()
    assert detect_kind(bundle) == "general"
    assert kind_evidence(bundle) == ()


def test_the_sweep_parameters_alone_are_not_evidence_of_a_sweep() -> None:
    """``SID_AXIS`` says what *would* be injected, not that anything was.

    A vehicle flown for an hour with last week's SYSTEMID parameters still set is
    an ordinary flight, and reading the parameter as evidence would refuse it.
    """
    bundle = _general()
    assert bundle.params["SID_AXIS"] == 7.0
    assert detect_kind(bundle) == "general"


def test_the_firmwares_own_autotune_counts_as_deliberate() -> None:
    bundle = _with_autotune(_general())
    assert detect_kind(bundle) == "tuning"
    assert any("autotune" in line for line in kind_evidence(bundle))


# --------------------------------------------------------------------------- #
# The declaration decides which segments are searched for
# --------------------------------------------------------------------------- #


def test_a_tuning_declaration_never_falls_back_to_stick_input() -> None:
    """The whole point of the distinction: no silent downgrade.

    The bundle has plenty of single-axis energy in it -- the general reading of
    the same file finds segments happily -- so an empty result here is the
    declaration being honoured rather than the log being quiet.
    """
    bundle = dataclasses.replace(_general(), declared_kind="tuning")
    assert propose_segments(bundle) == ()
    assert propose_segments(dataclasses.replace(bundle, declared_kind="general")) != ()


def test_a_general_declaration_does_not_use_the_sweep_that_is_there() -> None:
    bundle = dataclasses.replace(_tuning(), declared_kind="general")
    kinds = {s.kind for s in propose_segments(bundle)}
    assert kinds == {"pilot_input"}


def test_an_autotune_run_is_segmented_as_deliberate_excitation() -> None:
    bundle = dataclasses.replace(_with_autotune(_general()), declared_kind="tuning")
    segments = propose_segments(bundle)
    assert segments
    assert {s.kind for s in segments} == {"autotune_twitch"}
    # Worth more than stick input and less than a commanded sweep, and the
    # ordering has to survive into the confidence rating.
    assert all(0.3 < s.confidence < 1.0 for s in segments)


def test_a_px4_autotune_is_labelled_as_px4s_own() -> None:
    px4 = make_bundle(make_airframe(), make_chain(), stack="px4")
    bundle = dataclasses.replace(_with_autotune(px4), declared_kind="tuning")
    assert {s.kind for s in propose_segments(bundle)} == {"px4_autotune"}


def test_refusing_a_tuning_flight_says_which_way_to_fix_it() -> None:
    bundle = dataclasses.replace(_general(), declared_kind="tuning")
    with pytest.raises(ValueError, match="general flight log"):
        identify_axis(bundle, "roll", CONFIG)


# --------------------------------------------------------------------------- #
# The declaration bounds what the answer may claim
# --------------------------------------------------------------------------- #


def test_a_general_flight_cannot_reach_high_confidence() -> None:
    """However well it fits. The fit is over a band the pilot chose."""
    sweep = _tuning()
    tuned = recommend_from(identify_axis(sweep, "roll", CONFIG), sweep, CONFIG)
    assert tuned.confidence == "high"

    ordinary = dataclasses.replace(_general(), declared_kind="general")
    general = recommend_from(identify_axis(ordinary, "roll", CONFIG), ordinary, CONFIG)
    assert general.confidence in ("medium", "low")


def test_a_general_flight_is_designed_no_bolder_than_its_floor() -> None:
    ordinary = dataclasses.replace(_general(), declared_kind="general")
    analysis = identify_axis(ordinary, "roll", CONFIG)
    rec = recommend_from(analysis, ordinary, CONFIG, conservatism=0.0)
    assert rec.conservatism == pytest.approx(capabilities("general").conservatism_floor)


def test_the_floor_is_a_floor_and_not_a_setting() -> None:
    """A user who asks for more caution than the floor still gets it."""
    ordinary = dataclasses.replace(_general(), declared_kind="general")
    analysis = identify_axis(ordinary, "roll", CONFIG)
    rec = recommend_from(analysis, ordinary, CONFIG, conservatism=0.9)
    assert rec.conservatism == pytest.approx(0.9)


def test_a_tuning_flight_is_not_held_back() -> None:
    sweep = _tuning()
    rec = recommend_from(identify_axis(sweep, "roll", CONFIG), sweep, CONFIG, conservatism=0.0)
    assert rec.conservatism == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #


def test_only_a_general_flight_offers_operating_point_sensitivity() -> None:
    """A sweep is flown at one throttle, so the spread cannot be measured from it."""
    assert capabilities("general").allows("operating_point")
    assert not capabilities("tuning").allows("operating_point")


def test_every_kind_states_what_it_costs() -> None:
    for kind in ("general", "tuning"):
        caps = capabilities(kind)
        assert caps.limits
        assert caps.summary


def test_an_unknown_kind_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="unknown log kind"):
        capabilities("autotune")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The gate signal
# --------------------------------------------------------------------------- #


def test_a_gate_is_built_rather_than_interpolated() -> None:
    """A spline through a step rings, and a ringing gate is on when it is not."""
    grid = np.arange(0.0, 10.0, 0.01)
    gate = gate_signal("mode.autotune", grid, [(2.0, 4.0)], source_msg="EV.Id")
    assert set(np.unique(gate.y)) <= {0.0, 1.0}
    assert gate.y[grid < 2.0].max() == 0.0
    assert gate.y[(grid >= 2.0) & (grid <= 4.0)].min() == 1.0
    assert gate.y[grid > 4.0].max() == 0.0


# --------------------------------------------------------------------------- #
# The rating has to discriminate *within* a kind, not only between kinds
# --------------------------------------------------------------------------- #


def test_a_well_flown_general_log_reaches_the_top_of_its_range() -> None:
    """Otherwise the rating carries no information about the log it describes.

    A hand-flown slow-to-fast sweep and one stick waggle are not the same
    evidence, and a rule that rated both `low` because neither was a SYSTEMID
    chirp would be reporting the kind twice and the flight not at all -- the
    ceiling already says the kind.
    """
    from tests.synthetic.generators import make_general_flight_bundle

    bundle = make_general_flight_bundle(make_airframe(), make_chain())
    rec = recommend_from(identify_axis(bundle, "roll", CONFIG), bundle, CONFIG)
    assert rec.confidence == "medium"


def test_a_narrow_band_general_log_is_still_low() -> None:
    """Band width is the thing that decides, and it decides in both directions."""
    from tests.synthetic.generators import make_general_flight_bundle

    bundle = make_general_flight_bundle(
        make_airframe(), make_chain(), f_start_hz=4.5, f_stop_hz=5.5
    )
    rec = recommend_from(identify_axis(bundle, "roll", CONFIG), bundle, CONFIG)
    assert rec.confidence == "low"


def test_the_recommendation_records_which_kind_of_flight_produced_it() -> None:
    """A reloaded session has to be able to say why a good fit is only medium."""
    from tests.synthetic.generators import make_general_flight_bundle

    bundle = make_general_flight_bundle(make_airframe(), make_chain())
    assert recommend_from(identify_axis(bundle, "roll", CONFIG), bundle, CONFIG).log_kind == (
        "general"
    )

    sweep = _tuning()
    assert recommend_from(identify_axis(sweep, "roll", CONFIG), sweep, CONFIG).log_kind == "tuning"


def test_the_why_trace_explains_a_capped_rating() -> None:
    """A trace that argues for `high` next to a `medium` is the worst outcome."""
    from rotorid.core.guidance.explain import explain
    from tests.synthetic.generators import make_general_flight_bundle

    bundle = make_general_flight_bundle(make_airframe(), make_chain())
    rec = recommend_from(identify_axis(bundle, "roll", CONFIG), bundle, CONFIG)
    trace = explain("confidence", rec)
    assert trace is not None
    assert any("caps confidence at medium" in line for line in trace.because)
