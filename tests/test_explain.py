"""The "why this number?" layer (spec section 8).

An explanation is not prose about control theory -- the glossary is that, and it
is tested separately. An explanation is the trace behind one number on the
screen, and the property that makes it worth having is that it contains the
user's own values. So most of what is checked here is exactly that: that the
numbers quoted are the ones the design actually produced, and that an
explanation never appears for a value that was never recommended.
"""

from __future__ import annotations

import re

import pytest

from rotorid.config import load_config
from rotorid.core.design.recommend import analyze_axis
from rotorid.core.guidance.explain import (
    GLOSSARY,
    explain,
    explainable,
    glossary_for,
)
from tests.synthetic.generators import make_airframe, make_bundle, make_chain

CONFIG = load_config()


@pytest.fixture(scope="module")
def rec():
    return analyze_axis(make_bundle(make_airframe(), make_chain()), "roll", CONFIG)


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def test_every_offered_key_actually_explains_something(rec) -> None:
    for key in explainable(rec):
        exp = explain(key, rec)
        assert exp is not None, key
        assert exp.because, f"{key} has no reasoning"
        assert exp.headline
        assert exp.value


def test_an_unknown_key_gets_no_affordance_rather_than_an_empty_one(rec) -> None:
    """A "why?" link that opens onto nothing is worse than no link."""
    assert explain("ATC_RAT_RLL_SMAX", rec) is None
    assert explain("nonsense", rec) is None


def test_a_parameter_is_explained_under_the_name_the_user_sees(rec) -> None:
    """The GUI shows ``ATC_RAT_PIT_D``; the explanation must answer to that."""
    for axis, suffix in (("roll", "RLL"), ("pitch", "PIT"), ("yaw", "YAW")):
        one = analyze_axis(make_bundle(make_airframe(), make_chain(), axis=axis), axis, CONFIG)
        exp = explain(f"ATC_RAT_{suffix}_D", one)
        assert exp is not None
        assert exp.key == "rate_d"
        assert axis in exp.title


# --------------------------------------------------------------------------- #
# The numbers have to be this analysis's numbers
# --------------------------------------------------------------------------- #


def _numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+\.?\d*", text)]


def test_the_gain_explanation_quotes_the_gain_that_was_recommended(rec) -> None:
    exp = explain("ATC_RAT_RLL_P", rec)
    assert exp is not None
    assert exp.value == f"{rec.gains.kp:.4g}"
    body = " ".join(exp.because)
    assert f"{rec.baseline_gains.kp:.4g}" in body, "must say what it was before"
    assert f"{rec.margins.crossover_hz:.2f}" in body


def test_the_derivative_explanation_names_the_noise_it_costs(rec) -> None:
    """D is limited by noise, not by stability, and the explanation must say so."""
    exp = explain("ATC_RAT_RLL_D", rec)
    assert exp is not None
    body = " ".join(exp.because)
    assert f"{rec.dterm_noise_rms_pct:.2f}" in body
    assert f"{rec.filters.phase_cost_deg:.1f}" in body


def test_the_margin_explanation_agrees_with_the_margin_report(rec) -> None:
    exp = explain("phase_margin", rec)
    assert exp is not None
    assert exp.value == f"{rec.margins.phase_margin_deg:.0f} deg"
    assert f"{rec.margins.delay_margin_ms:.0f}" in " ".join(exp.because)


def test_no_explanation_invents_a_number_the_design_does_not_have(rec) -> None:
    """A number in an explanation must be readable off the recommendation.

    This is the guard against explanations drifting away from the design as the
    design changes: the tempting failure mode is a hand-written sentence with a
    plausible constant baked into it.
    """
    allowed = {
        round(v, 4)
        for v in (
            rec.gains.kp,
            rec.gains.ki,
            rec.gains.kd,
            rec.gains.kff,
            rec.baseline_gains.kp,
            rec.baseline_gains.ki,
            rec.baseline_gains.kd,
            rec.baseline_gains.kff,
            rec.margins.phase_margin_deg,
            rec.margins.gain_margin_db,
            rec.margins.crossover_hz,
            rec.margins.delay_margin_ms,
            rec.margins.disturbance_rejection_bw_hz,
            rec.margins.disturbance_rejection_peak_db,
            rec.dterm_noise_rms_pct,
            rec.filters.phase_cost_deg,
            rec.filters.cpu_cost_rel,
            rec.model.coherence_mean,
            rec.model.fit_rms_db,
            rec.model.fit_rms_deg,
            rec.model.valid_band_hz[0],
            rec.model.valid_band_hz[1],
            rec.predicted_step.rise_time_s * 1000.0,
            rec.predicted_step.overshoot_pct,
            rec.latency.airframe_tau_deg + rec.latency.actuator_deg,
        )
    }
    # Structural constants that legitimately appear as prose: the outer-loop
    # separation factor, the margin conventions, the radians-per-turn conversion.
    structural = {0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 20.0, 23.0, 25.0, 45.0, 180.0, 6.283}

    exp = explain("phase_margin", rec)
    assert exp is not None
    for number in _numbers(" ".join(exp.because)):
        near = any(abs(number - a) <= max(0.06, abs(a) * 0.02) for a in allowed)
        assert near or number in structural, (
            f"{number} in the phase-margin explanation is neither a value from this "
            f"recommendation nor a documented constant"
        )


# --------------------------------------------------------------------------- #
# Honesty about what was not recommended
# --------------------------------------------------------------------------- #


def test_notch_keys_are_withheld_when_no_notch_is_recommended(rec) -> None:
    """No notch means no notch bandwidth to explain."""
    if rec.filters.chain.notches:
        pytest.skip("this fixture does recommend a notch")
    assert not [k for k in explainable(rec) if k.startswith("INS_HNTCH_")]


def test_the_filter_explanation_carries_the_alternatives_that_lost(rec) -> None:
    exp = explain("INS_HNTCH_BW", rec)
    if exp is None:
        pytest.skip("no notch recommended for this fixture")
    assert exp.alternatives == rec.filters.rejected


def test_the_binding_constraint_is_named_in_words_not_as_a_code(rec) -> None:
    exp = explain("crossover", rec)
    assert exp is not None
    assert exp.binding == rec.binding_constraint
    assert rec.binding_constraint not in " ".join(exp.because), (
        "the reasoning should say what the constraint is, not repeat its identifier"
    )


# --------------------------------------------------------------------------- #
# Glossary
# --------------------------------------------------------------------------- #


def test_every_glossary_link_resolves(rec) -> None:
    for key in explainable(rec):
        exp = explain(key, rec)
        assert exp is not None
        assert len(glossary_for(exp)) == len(exp.glossary), f"{key} links a missing term"


def test_glossary_cross_references_resolve() -> None:
    for key, entry in GLOSSARY.items():
        for other in entry.see_also:
            assert other in GLOSSARY, f"{key} points at missing term {other}"
            assert other != key


def test_glossary_entries_are_general_and_explanations_are_specific(rec) -> None:
    """The split is the whole design: definitions are shared, traces are not."""
    exp = explain("drb", rec)
    assert exp is not None
    assert GLOSSARY["drb"].short == exp.headline
    assert f"{rec.margins.disturbance_rejection_peak_db:.1f}" in " ".join(exp.because)
