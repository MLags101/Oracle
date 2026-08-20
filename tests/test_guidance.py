"""Findings and the staged flight plan (milestone M5).

Each check gets a fixture that makes its condition true and one that makes it
false. A check that only ever fires is as useless as one that never does, and
both failure modes are invisible without the negative case.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest

from rotorid.config import load_config
from rotorid.core.design.recommend import analyze_axis, identify_axis, recommend_from
from rotorid.core.guidance.findings import GuidanceContext, collect_findings
from rotorid.core.guidance.plan import build_plan
from rotorid.core.io.base import canonical_signal
from rotorid.core.types import Finding
from tests.synthetic.generators import make_airframe, make_bundle, make_chain

CONFIG = load_config()


def _context(**kw):
    bundle = kw.pop("bundle", None) or make_bundle(make_airframe(), make_chain())
    analysis = identify_axis(bundle, "roll", CONFIG)
    rec = recommend_from(analysis, bundle, CONFIG)
    return GuidanceContext(
        bundle=bundle,
        analyses={"roll": analysis},
        recommendations={"roll": rec},
        config=CONFIG,
    )


def _codes(findings: tuple[Finding, ...]) -> set[str]:
    return {f.code for f in findings}


def _with_signal(bundle, key, values, source="RATE"):
    """A copy of a bundle carrying one extra signal on the roll axis."""
    t = bundle.signals["rate.roll.measured"].t
    signals = dict(bundle.signals)
    signals[key] = canonical_signal(key, t, np.asarray(values, dtype=np.float64), source_msg=source)
    return replace(bundle, signals=signals)


# --------------------------------------------------------------------------- #
# The contract every finding has to meet
# --------------------------------------------------------------------------- #


def test_every_finding_carries_an_action_and_evidence_of_its_claim() -> None:
    """A finding without an action is an observation, and nobody can act on it."""
    findings = collect_findings(_context())
    assert findings

    for finding in findings:
        assert finding.title.strip()
        assert finding.detail.strip()
        assert finding.action.strip(), f"{finding.code} has no action"
        assert finding.severity in ("blocker", "warning", "info", "good")


def test_findings_come_back_worst_first() -> None:
    findings = collect_findings(_context())
    order = {"blocker": 0, "warning": 1, "info": 2, "good": 3}
    severities = [order[f.severity] for f in findings]
    assert severities == sorted(severities)


def test_codes_are_unique_per_condition_not_per_message() -> None:
    """Two axes with the same problem share a code; tests and reports rely on it."""
    bundle = make_bundle(make_airframe(), make_chain())
    analysis = identify_axis(bundle, "roll", CONFIG)
    context = _context(bundle=bundle)
    context = GuidanceContext(
        bundle=bundle,
        analyses={"roll": analysis, "pitch": analysis},
        recommendations=context.recommendations,
        config=CONFIG,
    )
    findings = collect_findings(context)
    assert findings


# --------------------------------------------------------------------------- #
# Individual checks, each with its negative case
# --------------------------------------------------------------------------- #


def test_a_log_without_prefilter_gyro_says_exactly_what_to_set() -> None:
    findings = collect_findings(_context())
    finding = next(f for f in findings if f.code == "NO_RAW_IMU_DATA")
    assert "INS_LOG_BAT_OPT = 4" in finding.action
    assert finding.severity == "info", "a missing option is not a fault"


def test_esc_telemetry_present_but_unused_is_flagged() -> None:
    bundle = make_bundle(make_airframe(), make_chain())
    t = bundle.signals["rate.roll.measured"].t
    bundle = _with_signal(bundle, "motor.1.rpm", np.full(t.size, 3000.0), source="ESC")
    assert "ESC_TELEM_AVAILABLE_UNUSED" in _codes(collect_findings(_context(bundle=bundle)))


def test_esc_telemetry_already_in_use_is_not_flagged() -> None:
    bundle = make_bundle(make_airframe(), make_chain())
    t = bundle.signals["rate.roll.measured"].t
    bundle = _with_signal(bundle, "motor.1.rpm", np.full(t.size, 3000.0), source="ESC")
    bundle = replace(bundle, params={**bundle.params, "INS_HNTCH_MODE": 3.0})
    assert "ESC_TELEM_AVAILABLE_UNUSED" not in _codes(collect_findings(_context(bundle=bundle)))


def test_high_cpu_load_blocks_the_expensive_filter_options() -> None:
    bundle = make_bundle(make_airframe(), make_chain())
    t = bundle.signals["rate.roll.measured"].t
    bundle = _with_signal(bundle, "cpu.load", np.full(t.size, 0.92), source="PM")

    findings = collect_findings(_context(bundle=bundle))
    finding = next(f for f in findings if f.code == "CPU_HEADROOM_LOW")
    assert finding.evidence["peak_cpu_load"] == pytest.approx(0.92)
    assert "OPTS" in finding.action


def test_a_quiet_cpu_is_not_flagged() -> None:
    bundle = make_bundle(make_airframe(), make_chain())
    t = bundle.signals["rate.roll.measured"].t
    bundle = _with_signal(bundle, "cpu.load", np.full(t.size, 0.3), source="PM")
    assert "CPU_HEADROOM_LOW" not in _codes(collect_findings(_context(bundle=bundle)))


def test_slew_limiter_activity_is_reported_with_its_duty_cycle() -> None:
    bundle = make_bundle(make_airframe(), make_chain())
    t = bundle.signals["rate.roll.measured"].t
    dmod = np.where(np.arange(t.size) % 4 == 0, 0.6, 1.0)
    bundle = _with_signal(bundle, "rate.roll.dmod", dmod, source="PIDR")

    finding = next(
        f for f in collect_findings(_context(bundle=bundle)) if f.code == "SLEW_LIMITER_ACTIVE"
    )
    assert finding.evidence["duty"] == pytest.approx(0.25, abs=0.02)
    assert finding.evidence["min_dmod"] == pytest.approx(0.6)


def test_an_idle_slew_limiter_is_not_reported() -> None:
    bundle = make_bundle(make_airframe(), make_chain())
    t = bundle.signals["rate.roll.measured"].t
    bundle = _with_signal(bundle, "rate.roll.dmod", np.ones(t.size), source="PIDR")
    assert "SLEW_LIMITER_ACTIVE" not in _codes(collect_findings(_context(bundle=bundle)))


def test_a_saturated_integrator_is_reported_as_a_trim_problem() -> None:
    bundle = make_bundle(make_airframe(), make_chain())
    t = bundle.signals["rate.roll.measured"].t
    bundle = _with_signal(bundle, "rate.roll.i_term", np.full(t.size, 0.5), source="PIDR")
    bundle = replace(bundle, params={**bundle.params, "ATC_RAT_RLL_IMAX": 0.5})

    finding = next(
        f for f in collect_findings(_context(bundle=bundle)) if f.code == "INTEGRATOR_WINDUP"
    )
    assert finding.evidence["duty"] == pytest.approx(1.0)
    assert "trim" in finding.detail or "trim" in finding.action


def test_windup_is_not_claimed_without_an_imax_to_compare_against() -> None:
    """No IMAX in the snapshot means no claim, rather than a guessed one."""
    bundle = make_bundle(make_airframe(), make_chain())
    t = bundle.signals["rate.roll.measured"].t
    bundle = _with_signal(bundle, "rate.roll.i_term", np.full(t.size, 99.0), source="PIDR")
    assert "INTEGRATOR_WINDUP" not in _codes(collect_findings(_context(bundle=bundle)))


def test_a_large_gain_step_is_announced() -> None:
    bundle = make_bundle(make_airframe(), make_chain(), gains=(0.005, 0.005, 0.0001))
    findings = collect_findings(_context(bundle=bundle))
    finding = next(f for f in findings if f.code == "GAINS_FAR_FROM_CURRENT")
    assert "half way" in finding.action


def test_a_structural_peak_becomes_a_mechanical_finding() -> None:
    """The classification has to survive all the way to advice the user can act on."""
    from rotorid.core.analysis.noise import noise_profile
    from tests.synthetic.generators import make_noise_bundle

    chain = make_chain(gyro_lpf_hz=100.0, notch_freq_hz=0.0, harmonics=())
    noise_bundle = make_noise_bundle(chain, hover_hz=50.0)
    profile = noise_profile(noise_bundle, "roll", t_start=0.0, t_end=40.0, chain=chain, op=None)

    context = _context()
    analysis = replace(context.analyses["roll"], noise=profile)
    context = GuidanceContext(
        bundle=context.bundle,
        analyses={"roll": analysis},
        recommendations=context.recommendations,
        config=CONFIG,
    )

    finding = next(f for f in collect_findings(context) if f.code == "STRUCTURAL_RESONANCE")
    assert "not motor noise" in finding.detail
    assert finding.evidence["f_hz"] > 0.0


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


def test_filters_and_gains_are_never_in_the_same_flight() -> None:
    """The rule that makes a bad outcome attributable."""
    rec = analyze_axis(make_bundle(make_airframe(), make_chain()), "roll", CONFIG)
    plan = build_plan({"roll": rec})

    # ATC_RAT_*_FLTD is a filter, not a gain, despite the shared prefix -- so the
    # gain test has to be exact rather than a prefix match.
    is_gain = re.compile(r"ATC_RAT_[A-Z]{3}_(P|I|D|FF)\Z").fullmatch

    for stage in plan.stages:
        touches_filters = any(k.startswith("INS_") or k.endswith("_FLTD") for k in stage.changes)
        touches_gains = any(is_gain(k) for k in stage.changes)
        assert not (touches_filters and touches_gains), (
            f"stage {stage.index} changes filters and gains together"
        )


def test_the_stages_are_ordered_and_numbered_from_one() -> None:
    rec = analyze_axis(make_bundle(make_airframe(), make_chain()), "roll", CONFIG)
    plan = build_plan({"roll": rec})

    assert plan.stages
    assert [s.index for s in plan.stages] == list(range(1, len(plan.stages) + 1))
    titles = [s.title for s in plan.stages]
    if "Filters only" in titles and "Rate loop P and D" in titles:
        assert titles.index("Filters only") < titles.index("Rate loop P and D")


def test_every_stage_says_what_to_check_afterwards() -> None:
    """A flight whose result cannot be checked in the log has not been flown."""
    rec = analyze_axis(make_bundle(make_airframe(), make_chain()), "roll", CONFIG)
    for stage in build_plan({"roll": rec}).stages:
        assert stage.watch_in_flight
        assert stage.check_in_log


def test_unchanged_gains_do_not_produce_a_stage() -> None:
    """A plan with a do-nothing step in it teaches the reader to skim."""
    rec = analyze_axis(make_bundle(make_airframe(), make_chain()), "roll", CONFIG)
    same = replace(rec, gains=rec.baseline_gains, filters=replace(rec.filters, params={}))
    plan = build_plan({"roll": same})

    titles = [s.title for s in plan.stages]
    assert "Filters only" not in titles
    assert "Rate loop P and D" not in titles


def test_the_outer_loop_is_sized_from_the_rate_loop_that_now_exists() -> None:
    rec = analyze_axis(make_bundle(make_airframe(), make_chain()), "roll", CONFIG)
    plan = build_plan({"roll": rec})
    outer = next(s for s in plan.stages if s.title.startswith("Attitude"))

    expected = 2.0 * np.pi * rec.margins.crossover_hz / 4.0
    assert outer.changes["ATC_ANG_RLL_P"] == pytest.approx(expected, rel=0.02)


def test_blocking_findings_are_named_in_the_preamble() -> None:
    rec = analyze_axis(make_bundle(make_airframe(), make_chain()), "roll", CONFIG)
    blocker = Finding(
        severity="blocker",
        code="COHERENCE_NARROW_BAND",
        title="t",
        detail="d",
        action="a",
    )
    plan = build_plan({"roll": rec}, (blocker,))
    assert "COHERENCE_NARROW_BAND" in plan.preamble
    assert "Back up your parameters" in plan.preamble
