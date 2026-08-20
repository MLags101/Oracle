"""Vibration and accelerometer clipping (plan phase 2.1).

The precondition check. Everything else in the tool assumes the gyro trace is the
aircraft moving; these tests are about noticing when it is the frame shaking
instead, and about not claiming a clean frame when the log said nothing at all.
"""

from __future__ import annotations

import glob
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rotorid.config import load_config
from rotorid.core.analysis.vibration import vibration_summary
from rotorid.core.design.recommend import identify_axis, recommend_from
from rotorid.core.export.params import ExportBlockedError, write_param_files
from rotorid.core.guidance.findings import GuidanceContext, collect_findings
from rotorid.core.guidance.plan import build_plan
from rotorid.core.io.ardupilot import read_ardupilot
from rotorid.core.io.base import canonical_signal, signal_units
from rotorid.core.types import Finding
from tests.synthetic.generators import make_airframe, make_bundle, make_chain

CONFIG = load_config()

_LOGS = sorted(glob.glob("logs/*.bin"))


def _bundle_with(**series: np.ndarray):
    """A standard synthetic bundle carrying extra IMU signals.

    Keyword names use underscores because that is all Python allows; they map to
    the dotted canonical keys, so ``imu_1_vibe_z`` is ``imu.1.vibe.z``.
    """
    bundle = make_bundle(make_airframe(), make_chain())
    t = bundle.signals["rate.roll.measured"].t
    signals = dict(bundle.signals)
    for key, values in series.items():
        name = key.replace("_", ".")
        signals[name] = canonical_signal(
            name, t, np.asarray(values, dtype=np.float64), source_msg="VIBE"
        )
    return replace(bundle, signals=signals)


def _flat(bundle, level: float) -> np.ndarray:
    return np.full(bundle.signals["rate.roll.measured"].t.shape, level)


def _context(bundle):
    analysis = identify_axis(bundle, "roll", CONFIG)
    return GuidanceContext(
        bundle=bundle,
        analyses={"roll": analysis},
        recommendations={"roll": recommend_from(analysis, bundle, CONFIG)},
        config=CONFIG,
    )


def _codes(findings: tuple[Finding, ...]) -> set[str]:
    return {f.code for f in findings}


# --------------------------------------------------------------------------- #
# Canonical keys
# --------------------------------------------------------------------------- #


def test_the_indexed_key_resolver_handles_a_four_part_key() -> None:
    """``imu.1.vibe.z`` has its index in the middle and a two-part tail.

    The previous resolver special-cased ``motor.{n}.{field}`` by counting parts,
    so it would have rejected this key and every reader that produced it.
    """
    assert signal_units("imu.1.vibe.z") == "m/s^2"
    assert signal_units("imu.0.clip") == "count"
    assert signal_units("motor.3.rpm") == "rev/min"
    with pytest.raises(KeyError):
        signal_units("imu.1.vibe.w")


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _LOGS, reason="no logs/*.bin to read")
def test_vibe_reaches_canonical_keys_from_a_real_log() -> None:
    """VIBE was requested and decoded before this change, and then dropped."""
    bundle = read_ardupilot(Path(_LOGS[0]))
    vibe = {k for k in bundle.signals if k.startswith("imu.") and ".vibe." in k}
    assert vibe, "the reader decoded VIBE and threw it away"

    for key in vibe:
        assert bundle.signals[key].units == "m/s^2"
        # VIBE is on the medium-rate schedule and the analysis grid is not. The
        # native rate has to survive the resampler, or every rate check downstream
        # reads the grid rate instead and believes it.
        native = bundle.signals[key].native_rate_hz
        assert native is not None
        assert native < 100.0

    assert any(k.startswith("imu.") and k.endswith(".clip") for k in bundle.signals)


@pytest.mark.skipif(not _LOGS, reason="no logs/*.bin to read")
def test_a_real_log_reports_a_physical_vibration_level() -> None:
    summary = vibration_summary(read_ardupilot(Path(_LOGS[0])))
    assert summary.measured
    assert 0.0 <= summary.level_m_s2 < 15.0
    assert not summary.clipped


# --------------------------------------------------------------------------- #
# Summarizing
# --------------------------------------------------------------------------- #


def test_a_log_without_vibration_data_says_so_rather_than_reporting_zero() -> None:
    """The distinction the whole check rests on: clean versus unknown."""
    summary = vibration_summary(make_bundle(make_airframe(), make_chain()))
    assert not summary.measured
    assert summary.level_m_s2 == 0.0


def test_the_worst_imu_and_axis_are_named_not_averaged() -> None:
    """One shaking sensor corrupts what that sensor feeds, whatever the others say."""
    bundle = make_bundle(make_airframe(), make_chain())
    bundle = _bundle_with(
        imu_0_vibe_x=_flat(bundle, 2.0),
        imu_0_vibe_y=_flat(bundle, 2.0),
        imu_0_vibe_z=_flat(bundle, 2.0),
        imu_1_vibe_z=_flat(bundle, 40.0),
    )
    summary = vibration_summary(bundle)
    assert summary.worst_imu == 1
    assert summary.worst_component == "z"
    assert summary.level_m_s2 == pytest.approx(40.0, rel=1e-6)
    assert summary.per_imu_m_s2[0] == pytest.approx(2.0, rel=1e-6)


def test_windows_decide_the_verdict() -> None:
    """A frame that shook on the ground and flew smoothly is not a shaking frame.

    And the converse, which matters more: a frame that only shook during the
    sweep must not have that averaged away by a long calm hover either.
    """
    bundle = make_bundle(make_airframe(), make_chain())
    t = bundle.signals["rate.roll.measured"].t
    level = np.where(t < t[0] + 2.0, 60.0, 1.0)
    bundle = _bundle_with(imu_0_vibe_z=level)

    early = vibration_summary(bundle, ((t[0], t[0] + 1.5),))
    late = vibration_summary(bundle, ((t[0] + 5.0, t[-1]),))
    assert early.level_m_s2 > 50.0
    assert late.level_m_s2 < 2.0


def test_windows_outside_the_log_raise_rather_than_report_a_clean_aircraft() -> None:
    bundle = make_bundle(make_airframe(), make_chain())
    bundle = _bundle_with(imu_0_vibe_z=_flat(bundle, 50.0))
    with pytest.raises(ValueError, match="select no samples"):
        vibration_summary(bundle, ((1.0e6, 2.0e6),))


def test_a_single_spike_does_not_condemn_the_flight() -> None:
    """The level is a high percentile, not the maximum.

    A 10 Hz message with one bad sample, or the resampler's spline overshooting a
    step, must not be able to block an export on its own.
    """
    bundle = make_bundle(make_airframe(), make_chain())
    level = _flat(bundle, 1.0)
    level[level.size // 2] = 500.0
    assert vibration_summary(_bundle_with(imu_0_vibe_z=level)).level_m_s2 < 15.0


# --------------------------------------------------------------------------- #
# Clipping
# --------------------------------------------------------------------------- #


def test_a_clip_counter_that_never_moves_is_not_clipping() -> None:
    """The counter is cumulative. A large constant means it clipped on some past flight."""
    bundle = make_bundle(make_airframe(), make_chain())
    summary = vibration_summary(_bundle_with(imu_0_clip=_flat(bundle, 4000.0)))
    assert summary.clip_measured
    assert not summary.clipped
    assert summary.clip_count == 0


def test_a_counter_that_rises_is_clipping_and_names_the_imu() -> None:
    bundle = make_bundle(make_airframe(), make_chain())
    t = bundle.signals["rate.roll.measured"].t
    steps = np.where(t < t[t.size // 2], 0.0, 7.0)
    summary = vibration_summary(_bundle_with(imu_0_clip=_flat(bundle, 0.0), imu_2_clip=steps))
    assert summary.clipping_imus == (2,)
    assert summary.clip_count == 7


def test_resampling_ringing_around_a_step_does_not_invent_clipping() -> None:
    """A cubic through a step rings on both sides of it.

    The counters arrive here already splined onto the analysis grid, so a check
    written as "did any sample differ from the first" would fire on a log whose
    counter never actually moved. Sub-count wobble is not clipping; a real step,
    which is at least one whole count, still clears the tolerance easily.
    """
    bundle = make_bundle(make_airframe(), make_chain())
    size = bundle.signals["rate.roll.measured"].t.size
    ripple = 0.2 * np.sin(np.linspace(0.0, 40.0, size))
    assert not vibration_summary(_bundle_with(imu_1_clip=ripple)).clipped
    assert vibration_summary(_bundle_with(imu_1_clip=ripple + np.linspace(0.0, 1.0, size))).clipped


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


def test_a_quiet_frame_is_said_to_be_quiet() -> None:
    """A user who only ever sees problems learns nothing about a healthy log."""
    bundle = make_bundle(make_airframe(), make_chain())
    findings = collect_findings(_context(_bundle_with(imu_0_vibe_z=_flat(bundle, 3.0))))
    assert "VIBRATION_LOW" in _codes(findings)
    assert next(f for f in findings if f.code == "VIBRATION_LOW").severity == "good"


def test_a_log_with_no_vibration_message_is_told_it_is_unmeasured() -> None:
    findings = collect_findings(_context(make_bundle(make_airframe(), make_chain())))
    codes = _codes(findings)
    assert "VIBRATION_NOT_LOGGED" in codes
    assert "VIBRATION_LOW" not in codes
    assert "VIBRATION_HIGH" not in codes


def test_moderate_vibration_warns_and_severe_vibration_blocks() -> None:
    bundle = make_bundle(make_airframe(), make_chain())
    warn = collect_findings(_context(_bundle_with(imu_0_vibe_z=_flat(bundle, 20.0))))
    severe = collect_findings(_context(_bundle_with(imu_0_vibe_z=_flat(bundle, 45.0))))

    assert next(f for f in warn if f.code == "VIBRATION_HIGH").severity == "warning"
    assert next(f for f in severe if f.code == "VIBRATION_HIGH").severity == "blocker"


def test_a_frame_that_is_calm_on_average_and_violent_sometimes_still_warns() -> None:
    """Sustained level and peak are two facts, and the second is not the first.

    A frame with a loose arm is quiet until something excites it. Judging it on
    the level it spends most of its time at would pass exactly that aircraft.
    """
    bundle = make_bundle(make_airframe(), make_chain())
    t = bundle.signals["rate.roll.measured"].t
    # Violent over 3% of the flight and calm over the other 97%, spread through
    # the whole record so it lands inside whichever windows the identification
    # picked rather than depending on where those happened to fall.
    level = np.where(((t - t[0]) * 4.0) % 1.0 < 0.03, 25.0, 4.0)
    findings = collect_findings(_context(_bundle_with(imu_0_vibe_z=level)))

    high = next(f for f in findings if f.code == "VIBRATION_HIGH")
    assert high.severity == "warning"
    assert "reaches" in high.title
    assert high.evidence["level_m_s2"] < 15.0 < high.evidence["peak_m_s2"]


def test_one_bad_sample_is_not_an_excursion() -> None:
    """The same property as the summary's, asserted where the user would see it."""
    bundle = make_bundle(make_airframe(), make_chain())
    level = _flat(bundle, 4.0)
    level[level.size // 2] = 500.0
    codes = _codes(collect_findings(_context(_bundle_with(imu_0_vibe_z=level))))
    assert "VIBRATION_LOW" in codes
    assert "VIBRATION_HIGH" not in codes


def test_the_vibration_finding_says_no_gain_fixes_it() -> None:
    """The one conclusion a tuning tool must not let the user reach."""
    bundle = make_bundle(make_airframe(), make_chain())
    finding = next(
        f
        for f in collect_findings(_context(_bundle_with(imu_0_vibe_z=_flat(bundle, 45.0))))
        if f.code == "VIBRATION_HIGH"
    )
    assert "mechanical" in finding.detail.lower()
    assert "balance" in finding.action.lower()


def test_clipping_blocks_regardless_of_how_low_the_level_reads() -> None:
    """Clipping is categorical. A saturated sample is an absent measurement."""
    bundle = make_bundle(make_airframe(), make_chain())
    t = bundle.signals["rate.roll.measured"].t
    findings = collect_findings(
        _context(
            _bundle_with(
                imu_0_vibe_z=_flat(bundle, 1.0),
                imu_0_clip=np.where(t < t[t.size // 3], 0.0, 3.0),
            )
        )
    )
    codes = _codes(findings)
    assert "ACCEL_CLIPPING" in codes
    # The level really is low, and the tool reports both facts rather than
    # picking whichever one it likes better.
    assert "VIBRATION_LOW" in codes
    assert next(f for f in findings if f.code == "ACCEL_CLIPPING").severity == "blocker"


def test_clipping_stops_the_export_until_it_is_acknowledged(tmp_path: Path) -> None:
    bundle = make_bundle(make_airframe(), make_chain())
    t = bundle.signals["rate.roll.measured"].t
    context = _context(_bundle_with(imu_0_clip=np.where(t < t[t.size // 3], 0.0, 3.0)))
    findings = collect_findings(context)
    plan = build_plan(context.recommendations)

    def _export(**kw):
        return write_param_files(
            tmp_path,
            plan,
            log_name="flight.bin",
            tool_version="test",
            config_hash="abcd1234",
            findings=findings,
            **kw,
        )

    with pytest.raises(ExportBlockedError, match="ACCEL_CLIPPING"):
        _export()
    assert not list(tmp_path.iterdir()), "a blocked export must leave nothing behind"

    written = _export(
        acknowledgements={f.code: "accepted for test" for f in findings if f.severity == "blocker"}
    )
    assert written
    # The acknowledgement has to reach the file, so that whoever flies these gains
    # can see what was waved through to produce them.
    assert "ACCEL_CLIPPING" in written[0].read_text(encoding="utf-8")
