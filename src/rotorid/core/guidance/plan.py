"""The staged flight plan (spec section 8.2).

The output of an analysis is not a parameter file. It is an *ordered set of
flights*, because the single most common way a tuning session goes wrong is
changing several things at once and then being unable to attribute the result.

The ladder is fixed:

1. **Filters only.** They change what the loop sees, so every gain designed
   afterwards assumes they are in place.
2. **P and D.** The fast loop, and the pair that produce oscillation if anything
   is wrong. Flown before the integrator so that any oscillation seen is
   unambiguously theirs.
3. **I and feed-forward.** Trim and tracking, which cannot be judged until the
   fast loop is settled.
4. **The outer loop.** Attitude gains, sized from the rate loop that now exists.

Each stage carries what to watch for in the air and what to look for in the log
afterwards, because "it felt fine" is not evidence and the next flight should
close the loop on the last one.
"""

from __future__ import annotations

import numpy as np

from rotorid.core.types import (
    Axis,
    Finding,
    FlightTestPlan,
    FlightTestStage,
    Stack,
    TuneRecommendation,
)

__all__ = ["build_plan"]

#: Inner/outer timescale separation. The attitude loop is sized from the rate
#: loop's crossover rather than from a table, because that is what it has to be
#: slower than.
_OUTER_LOOP_SEPARATION = 4.0

_AP_SUFFIX: dict[Axis, str] = {"roll": "RLL", "pitch": "PIT", "yaw": "YAW"}
_PX4_SUFFIX: dict[Axis, str] = {"roll": "ROLL", "pitch": "PITCH", "yaw": "YAW"}

_ATT_PARAM: dict[Stack, dict[Axis, str]] = {
    "ardupilot": {
        "roll": "ATC_ANG_RLL_P",
        "pitch": "ATC_ANG_PIT_P",
        "yaw": "ATC_ANG_YAW_P",
    },
    "px4": {"roll": "MC_ROLL_P", "pitch": "MC_PITCH_P", "yaw": "MC_YAW_P"},
}


def _rate_prefix(stack: Stack, axis: Axis) -> str:
    if stack == "px4":
        return f"MC_{_PX4_SUFFIX[axis]}RATE_"
    return f"ATC_RAT_{_AP_SUFFIX[axis]}_"


def build_plan(
    recommendations: dict[Axis, TuneRecommendation],
    findings: tuple[Finding, ...] = (),
    params: dict[str, float] | None = None,
) -> FlightTestPlan:
    """Turn recommendations into flights, in the order they should be flown.

    Stages with nothing to change are omitted rather than shown empty: a plan
    with a do-nothing step in it teaches the reader to skim.

    Args:
        findings: Used to attribute each stage to the observations that motivated
            it, so a user can see why they are being asked to fly again.
        params: The flown parameter snapshot. Needed only on PX4, to undo the
            ``K`` scaling: gains are effective everywhere inside this tool, but
            ``MC_*RATE_P`` is the standard-form number the firmware multiplies by
            ``K``, so writing an effective gain into it would inflate the tune by
            exactly the factor the user chose.
    """
    blockers = tuple(f.code for f in findings if f.severity == "blocker")
    stages: list[FlightTestStage] = []

    filter_changes = _filter_changes(recommendations)
    if filter_changes:
        stages.append(_filter_stage(len(stages) + 1, filter_changes, findings))

    pd_changes = _gain_changes(recommendations, ("P", "D"), params)
    if pd_changes:
        stages.append(_pd_stage(len(stages) + 1, pd_changes, recommendations, findings))

    i_changes = _gain_changes(recommendations, ("I", "FF"), params)
    if i_changes:
        stages.append(_i_stage(len(stages) + 1, i_changes, findings))

    outer = _outer_loop_changes(recommendations)
    if outer:
        stages.append(_outer_stage(len(stages) + 1, outer))

    return FlightTestPlan(stages=tuple(stages), preamble=_preamble(blockers))


def _preamble(blockers: tuple[str, ...]) -> str:
    base = (
        "Back up your parameters before changing anything. Fly one stage per flight, "
        "at altitude, with room to recover, and download the log after each one -- a "
        "stage you cannot check in the log afterwards has not really been flown."
    )
    if not blockers:
        return base
    return (
        base
        + " This plan is built on an analysis with unresolved blocking findings ("
        + ", ".join(blockers)
        + "). Resolve those first; flying this as it stands means testing a "
        "recommendation the tool has already said it cannot stand behind."
    )


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def _filter_stage(
    index: int, changes: dict[str, float], findings: tuple[Finding, ...]
) -> FlightTestStage:
    return FlightTestStage(
        index=index,
        title="Filters only",
        changes=changes,
        watch_in_flight=(
            "Motor temperature after landing -- warmer than usual means noise is still "
            "reaching them",
            "Any new low-frequency wallow, which would mean a filter is costing more phase "
            "than expected",
        ),
        check_in_log=(
            "The gyro spectrum: the motor peaks the notch was aimed at should be gone, and "
            "the floor between them should be roughly unchanged",
            "PID D-term amplitude, which should fall without the gains having changed",
            "Scheduler load (PM.Load), to confirm the new filters fit",
        ),
        motivating_findings=_codes(
            findings, ("STRUCTURAL_RESONANCE", "ESC_TELEM_AVAILABLE_UNUSED", "DTERM_NOISE_HIGH")
        ),
    )


def _pd_stage(
    index: int,
    changes: dict[str, float],
    recommendations: dict[Axis, TuneRecommendation],
    findings: tuple[Finding, ...],
) -> FlightTestStage:
    worst = min(
        (r.margins.phase_margin_deg for r in recommendations.values()), default=float("nan")
    )
    return FlightTestStage(
        index=index,
        title="Rate loop P and D",
        changes=changes,
        watch_in_flight=(
            "Sharp stick inputs and a quick stop: oscillation that continues after the stick "
            "centres is too much D; a slow overshoot is too much P",
            f"Designed phase margin is {worst:.0f} degrees, so the response should look "
            f"damped rather than crisp-and-ringing",
        ),
        check_in_log=(
            "Rate tracking: RATE.RDes against RATE.R on a sharp input",
            "The D term for high-frequency content that was not there before",
            "PIDR.Dmod -- if the slew limiter engaged, the gains are above what the airframe "
            "can carry and the flown tune was not the configured one",
        ),
        motivating_findings=_codes(findings, ("SLEW_LIMITER_ACTIVE", "GAINS_FAR_FROM_CURRENT")),
    )


def _i_stage(
    index: int, changes: dict[str, float], findings: tuple[Finding, ...]
) -> FlightTestStage:
    return FlightTestStage(
        index=index,
        title="Rate loop I and feed-forward",
        changes=changes,
        watch_in_flight=(
            "Held attitude in wind: a persistent lean means the integrator is not carrying "
            "the trim",
            "A slow wallow after aggressive manoeuvres, which is an integrator that wound up "
            "and is unwinding",
        ),
        check_in_log=(
            "The I term should settle to a steady non-zero value in hover, not drift",
            "The I term should not sit at IMAX",
        ),
        motivating_findings=_codes(findings, ("INTEGRATOR_WINDUP",)),
    )


def _outer_stage(index: int, changes: dict[str, float]) -> FlightTestStage:
    return FlightTestStage(
        index=index,
        title="Attitude (outer) loop",
        changes=changes,
        watch_in_flight=(
            "Attitude overshoot on a step input, and any bounce as the vehicle arrives at "
            "the commanded angle",
        ),
        check_in_log=(
            "ATT.DesRoll against ATT.Roll: the outer loop should arrive without overshoot",
        ),
    )


# --------------------------------------------------------------------------- #
# What changes at each rung
# --------------------------------------------------------------------------- #


def _filter_changes(recommendations: dict[Axis, TuneRecommendation]) -> dict[str, float]:
    changes: dict[str, float] = {}
    for rec in recommendations.values():
        changes.update(rec.filters.params)
    return changes


def _gain_changes(
    recommendations: dict[Axis, TuneRecommendation],
    terms: tuple[str, ...],
    params: dict[str, float] | None = None,
) -> dict[str, float]:
    """Parameter changes for the named gain terms, omitting anything unchanged."""
    changes: dict[str, float] = {}
    for axis, rec in recommendations.items():
        stack = rec.filters.stack
        prefix = _rate_prefix(stack, axis)
        scale = _standard_form_scale(stack, prefix, params)
        values = {
            "P": (rec.gains.kp, rec.baseline_gains.kp),
            "I": (rec.gains.ki, rec.baseline_gains.ki),
            "D": (rec.gains.kd, rec.baseline_gains.kd),
            "FF": (rec.gains.kff, rec.baseline_gains.kff),
        }
        for term in terms:
            new, old = values[term]
            if not np.isclose(new, old, rtol=1e-3, atol=1e-9):
                # Five significant figures, not five decimal places: a D gain of
                # 0.0036 and a P gain of 0.14 need the same relative precision,
                # and decimal rounding gives the small one far less of it.
                written = float(new) if term == "FF" else float(new) / scale
                changes[f"{prefix}{term}"] = float(f"{written:.5g}")
    return changes


def _standard_form_scale(stack: Stack, prefix: str, params: dict[str, float] | None) -> float:
    """The ``K`` the firmware will multiply the written gain by.

    1.0 on ArduPilot, which has no such factor, and on PX4 when the snapshot does
    not record one. Note that ``FF`` is exempt: PX4 scales P, I and D by ``K`` and
    leaves the feed-forward alone.
    """
    if stack != "px4" or params is None:
        return 1.0
    k = float(params.get(f"{prefix}K", 1.0))
    return k if k > 0.0 else 1.0


def _outer_loop_changes(recommendations: dict[Axis, TuneRecommendation]) -> dict[str, float]:
    """Attitude P sized from the rate loop that now exists.

    The attitude loop is a pure gain on an inner loop that behaves like a first
    order lag, so its bandwidth has to sit a factor below the rate crossover for
    the two not to interact. That factor, not a table of typical values, is what
    sets the number.
    """
    changes: dict[str, float] = {}
    for axis, rec in recommendations.items():
        crossover = rec.margins.crossover_hz
        if crossover <= 0.0:
            continue
        # ATC_ANG_*_P and MC_*_P are both rate-per-angle gains in 1/s: the
        # outer-loop bandwidth expressed directly, in rad/s.
        name = _ATT_PARAM[rec.filters.stack][axis]
        changes[name] = round(2.0 * np.pi * crossover / _OUTER_LOOP_SEPARATION, 2)
    return changes


def _codes(findings: tuple[Finding, ...], wanted: tuple[str, ...]) -> tuple[str, ...]:
    """Which of the interesting findings are actually present."""
    present = {f.code for f in findings}
    return tuple(code for code in wanted if code in present)
