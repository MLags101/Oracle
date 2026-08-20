"""PX4 support (milestone M9).

Two things are being checked, and they are different in kind.

The **reader** is checked against real bytes: a uLog written by
``tests/synthetic/ulog.py`` and handed to the reader as a file, because the
reader's job is interpretation -- which topic supplies which canonical signal,
which of them are post-filter -- and none of that is exercised by constructing a
bundle in Python.

The **model** is checked for the places PX4 genuinely differs from ArduPilot. Any
of these silently taken from the other stack produces numbers that look entirely
reasonable and are wrong, which is the failure mode this whole file exists for:
the ``K`` factor, D on the measurement rather than on the error, a 2-pole D-term
filter rather than a 1-pole, notches with no attenuation setting, and no
throttle-derived notch tracking at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rotorid.config import load_config
from rotorid.core.filters.chain import ardupilot_chain, px4_chain
from rotorid.core.io.px4 import read_px4
from rotorid.core.preprocess.params import chain_from_bundle, gains_from_bundle
from tests.synthetic.ulog import write_px4_log

CONFIG = load_config()


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return write_px4_log(tmp_path / "flight.ulg")


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def test_a_ulog_reads_into_the_same_canonical_keys_as_a_bin(log: Path) -> None:
    """Everything above the IO layer is stack-agnostic only if this holds."""
    bundle = read_px4(log)
    assert bundle.stack == "px4"
    for axis in ("roll", "pitch", "yaw"):
        assert f"rate.{axis}.measured" in bundle.signals
        assert f"rate.{axis}.output" in bundle.signals
        assert f"att.{axis}.measured" in bundle.signals
    assert bundle.signals["rate.roll.measured"].units == "rad/s"
    assert bundle.signals["rate.roll.output"].units == "normalized"


def test_the_logged_gyro_is_marked_post_filter(log: Path) -> None:
    """The provenance trap, on this stack too.

    ``vehicle_angular_velocity`` comes out of the same chain the controller reads.
    If this flag were wrong the deconvolution would either count the filters twice
    or not at all, and the recommendation would be timid or unstable accordingly.
    """
    bundle = read_px4(log)
    assert bundle.signals["rate.roll.measured"].filtered is True
    assert bundle.signals["rate.roll.measured"].source_msg == "vehicle_angular_velocity"


def test_the_older_actuator_topic_is_accepted(tmp_path: Path) -> None:
    """PX4 renamed the rate-controller output; both names are in the wild."""
    path = write_px4_log(tmp_path / "old.ulg", torque_topic="actuator_controls_0")
    bundle = read_px4(path)
    assert bundle.signals["rate.roll.output"].source_msg == "actuator_controls_0"


def test_attitude_is_converted_from_the_quaternion_px4_actually_logs(log: Path) -> None:
    bundle = read_px4(log)
    roll = bundle.signals["att.roll.measured"]
    assert roll.units == "rad"
    assert np.max(np.abs(roll.y)) < 0.5, "0.05 rad of commanded roll, not 0.05 of anything else"


def test_esc_telemetry_lands_under_the_key_the_notch_designer_looks_for(log: Path) -> None:
    bundle = read_px4(log)
    assert "motor.0.rpm" in bundle.signals
    assert bundle.signals["motor.0.rpm"].units == "rev/min"


def test_a_log_without_raw_gyro_says_the_spectrum_will_be_reconstructed(log: Path) -> None:
    """The reconstruction is blind inside a deep notch, so its use is announced."""
    bundle = read_px4(log)
    assert any("sensor_gyro_fifo" in w for w in bundle.warnings)


def test_a_log_without_angular_velocity_is_refused_with_the_fix(tmp_path: Path) -> None:
    from tests.synthetic.ulog import ULogWriter

    path = tmp_path / "empty.ulg"
    writer = ULogWriter()
    writer.parameters({"IMU_GYRO_RATEMAX": 400.0})
    writer.topic(
        "cpuload",
        [("float", "load")],
        {"timestamp": np.arange(0.0, 1.0, 0.1), "load": np.full(10, 0.3)},
    )
    writer.write(path)

    with pytest.raises(ValueError, match="SDLOG_PROFILE"):
        read_px4(path)


def test_the_index_pass_names_the_topics_without_reading_them_all(log: Path) -> None:
    from rotorid.core.io.px4 import PX4Reader

    counts = PX4Reader(log).index()
    assert counts["vehicle_angular_velocity"] > 1000
    assert "esc_status" in counts


# --------------------------------------------------------------------------- #
# Gains: the K factor
# --------------------------------------------------------------------------- #


def test_the_k_factor_is_resolved_at_the_io_boundary(tmp_path: Path) -> None:
    """PX4 stores standard form. Every gain downstream must already be effective."""
    path = write_px4_log(
        tmp_path / "k.ulg",
        params={
            "IMU_GYRO_RATEMAX": 400.0,
            "MC_ROLLRATE_K": 2.0,
            "MC_ROLLRATE_P": 0.15,
            "MC_ROLLRATE_I": 0.2,
            "MC_ROLLRATE_D": 0.003,
        },
    )
    gains = gains_from_bundle(read_px4(path), "roll")
    assert gains.kp == pytest.approx(0.30)
    assert gains.ki == pytest.approx(0.40)
    assert gains.kd == pytest.approx(0.006)


# --------------------------------------------------------------------------- #
# The filter chain
# --------------------------------------------------------------------------- #


def _chain(**params: float):
    base = {"IMU_GYRO_CUTOFF": 80.0, "IMU_DGYRO_CUTOFF": 50.0}
    return px4_chain({**base, **params}, "roll", gyro_sample_rate_hz=1000.0, loop_rate_hz=1000.0)


def test_the_chain_is_built_from_px4_parameter_names(log: Path) -> None:
    chain = chain_from_bundle(read_px4(log), "roll")
    assert chain.stack == "px4"
    assert chain.gyro_lpf_hz == pytest.approx(80.0)
    assert chain.dterm_lpf_hz == pytest.approx(50.0)
    assert chain.error_lpf_hz is None, "PX4's rate controller has no error low-pass"
    assert chain.target_lpf_hz is None


def test_a_px4_notch_is_a_true_null_with_no_depth_setting() -> None:
    """PX4's NotchFilter has bandwidth and nothing else. Depth is not a knob."""
    chain = _chain(IMU_GYRO_NF0_FRQ=120.0, IMU_GYRO_NF0_BW=20.0)
    at_centre = float(np.abs(chain.sensor_response(np.array([120.0])))[0])
    assert at_centre < 1e-6
    assert "att" not in chain.describe(), "there is no attenuation parameter to report"


def test_the_notch_minus_3db_points_sit_at_the_configured_bandwidth() -> None:
    """``BW`` means the -3 dB width, and the discrete filter has to deliver it.

    Checked at 8 kHz rather than at the 1 kHz the other tests use, because the
    bilinear transform warps the skirt as the centre approaches Nyquist: the same
    200 Hz / 40 Hz notch measures -1.9 dB at its stated edges when run at 1 kHz.
    That is the firmware's behaviour, not an error here -- but it means "BW is the
    -3 dB width" is a statement about the design, and only about the realised
    filter when the centre is well below Nyquist.
    """
    fast = px4_chain(
        {"IMU_GYRO_NF0_FRQ": 200.0, "IMU_GYRO_NF0_BW": 40.0},
        "roll",
        gyro_sample_rate_hz=8000.0,
        loop_rate_hz=8000.0,
    )
    edges = 20.0 * np.log10(np.abs(fast.sensor_response(np.array([180.0, 220.0]))))
    assert np.allclose(edges, -3.0, atol=0.5), edges


def test_the_dynamic_notch_reads_its_harmonic_count_and_floor() -> None:
    chain = _chain(
        IMU_GYRO_DNF_EN=1.0, IMU_GYRO_DNF_MIN=40.0, IMU_GYRO_DNF_BW=15.0, IMU_GYRO_DNF_HMC=3.0
    )
    assert len(chain.notches) == 1
    notch = chain.notches[0]
    assert notch.harmonics == (1, 2, 3)
    assert notch.freq_hz == pytest.approx(40.0)
    assert notch.flavor == "px4"


def test_px4_notches_never_become_composite_however_the_bits_are_set() -> None:
    """``OPTS`` is an ArduPilot parameter. Reading it here would triple the phase."""
    from dataclasses import replace

    from rotorid.core.filters.harmonic import NotchOption

    chain = _chain(IMU_GYRO_NF0_FRQ=120.0, IMU_GYRO_NF0_BW=20.0)
    notch = replace(chain.notches[0], opts=NotchOption.TRIPLE_NOTCH)
    assert notch.composite_notches == 1
    assert len(notch.stages(120.0)) == 1


def test_the_dterm_filter_is_two_pole_on_px4_and_one_pole_on_ardupilot() -> None:
    """Same cutoff, different phase. Modelling one with the other misprices D."""
    f = np.array([30.0])
    px4 = _chain()
    ap = ardupilot_chain(
        {"INS_GYRO_FILTER": 80.0, "ATC_RAT_RLL_FLTD": 50.0},
        "roll",
        gyro_sample_rate_hz=1000.0,
        loop_rate_hz=1000.0,
    )
    px4_lag = -float(np.degrees(np.angle(px4.dterm_lpf_response(f)))[0])
    ap_lag = -float(np.degrees(np.angle(ap.dterm_lpf_response(f)))[0])
    assert px4_lag > 1.5 * ap_lag, (
        f"a 2-pole filter should cost roughly twice the phase of a 1-pole "
        f"({px4_lag:.1f} deg against {ap_lag:.1f} deg)"
    )


# --------------------------------------------------------------------------- #
# Designing filters for PX4
# --------------------------------------------------------------------------- #


def _noise_fixture(stack_chain):
    from rotorid.core.analysis.noise import motor_track, noise_profile
    from tests.synthetic.generators import make_chain, make_noise_bundle

    del stack_chain
    chain = make_chain(gyro_lpf_hz=100.0, notch_freq_hz=0.0, harmonics=())
    bundle = make_noise_bundle(chain, hover_hz=50.0)
    profile = noise_profile(bundle, "roll", t_start=0.0, t_end=40.0, chain=chain, op=None)
    return profile, chain, motor_track(bundle, 0.0, 40.0)


def test_px4_filter_parameters_use_px4_names() -> None:
    from dataclasses import replace

    from rotorid.core.design.filters import recommend_filters
    from rotorid.core.filters.chain import OperatingPoint

    profile, chain, track = _noise_fixture(None)
    rec = recommend_filters(
        profile,
        replace(chain, stack="px4"),
        CONFIG,
        track=track,
        op=OperatingPoint(motor_hz=(50.0,)),
        crossover_hz=4.0,
        kd=0.003,
    )
    assert rec.stack == "px4"
    assert set(rec.params) <= {
        "IMU_GYRO_CUTOFF",
        "IMU_DGYRO_CUTOFF",
        "IMU_GYRO_DNF_EN",
        "IMU_GYRO_DNF_MIN",
        "IMU_GYRO_DNF_BW",
        "IMU_GYRO_DNF_HMC",
    }
    assert rec.params.get("IMU_GYRO_DNF_EN") == 1.0, "ESC RPM is bit 0"


def test_px4_refuses_to_pretend_it_can_track_from_throttle() -> None:
    """ArduPilot degrades to a throttle model. PX4 has no such mode at all.

    So the honest answer on a PX4 log with neither ESC RPM nor the onboard FFT is
    a refusal with the two parameters that would fix it -- not a static notch
    pinned to whatever frequency this particular hover happened to sit at, which
    would be in the wrong place at every other throttle setting while looking
    exactly like a working recommendation.
    """
    from rotorid.core.analysis.noise import MotorTrack
    from rotorid.core.design.filters import choose_notch_source

    throttle_only = MotorTrack(
        t=np.linspace(0.0, 10.0, 100),
        f_hz=np.full(100, 0.6),
        source="throttle_model",
    )
    with pytest.raises(ValueError, match="IMU_GYRO_FFT_EN"):
        choose_notch_source(
            track=throttle_only,
            hover_thrust=0.35,
            fft_available=False,
            fundamental_hz=90.0,
            stack="px4",
        )

    # ArduPilot, given exactly the same evidence, does have somewhere to go.
    ardupilot = choose_notch_source(
        track=throttle_only,
        hover_thrust=0.35,
        fft_available=False,
        fundamental_hz=90.0,
        stack="ardupilot",
    )
    assert ardupilot.mode == 1


def test_px4_uses_the_onboard_fft_when_there_is_no_esc_telemetry() -> None:
    from rotorid.core.analysis.noise import MotorTrack
    from rotorid.core.design.filters import choose_notch_source

    source = choose_notch_source(
        track=MotorTrack(
            t=np.linspace(0.0, 10.0, 100),
            f_hz=np.full(100, 0.6),
            source="throttle_model",
        ),
        hover_thrust=0.35,
        fft_available=True,
        fundamental_hz=90.0,
        stack="px4",
    )
    assert source.mode == 2, "IMU_GYRO_DNF_EN bit 1"
    assert "IMU_GYRO_DNF_EN bit 1" in source.rationale


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_a_px4_log_runs_the_whole_pipeline_and_recovers_the_airframe() -> None:
    """The same airframe, the same numbers, through the other firmware's names."""
    from rotorid.core.pipeline import analyze
    from tests.synthetic.generators import make_airframe, make_bundle, make_chain

    airframe = make_airframe()
    bundle = make_bundle(
        airframe, make_chain(), stack="px4", with_motor_noise=True, path="flight.ulg"
    )
    result = analyze(bundle, ("roll",), CONFIG, tool_version="test")

    assert not result.failures, result.failures
    rec = result.session.recommendations["roll"]
    assert rec.model.params["K"] == pytest.approx(airframe.params["K"], rel=0.10)
    assert rec.model.params["wn"] == pytest.approx(airframe.params["wn"], rel=0.10)
    # Same 10% as the other two. A PX4 log has no injected-signal message, so the
    # sweep's start and stop frequencies are unknown and the band is bounded by
    # where the excitation is measured to have had power rather than by where the
    # firmware was told to put it. That trims the fade-in at the bottom of the
    # sweep, which is exactly the part a delay estimate leans on hardest.
    assert rec.model.params["tau"] == pytest.approx(airframe.params["tau"], rel=0.10)


def test_a_px4_plan_is_written_in_px4_parameter_names() -> None:
    """An ArduPilot name in a PX4 export is a change that would simply not apply."""
    from rotorid.core.pipeline import analyze
    from tests.synthetic.generators import make_airframe, make_bundle, make_chain

    bundle = make_bundle(
        make_airframe(), make_chain(), stack="px4", with_motor_noise=True, path="flight.ulg"
    )
    plan = analyze(bundle, ("roll",), CONFIG, tool_version="test").session.next_steps
    assert plan is not None

    names = {name for stage in plan.stages for name in stage.changes}
    assert names
    assert not any(n.startswith(("ATC_", "INS_")) for n in names), names
    assert any(n.startswith("MC_ROLLRATE_") for n in names)


def test_the_k_factor_is_undone_on_the_way_back_out() -> None:
    """Gains are effective inside the tool and standard-form in the file.

    PX4 multiplies ``MC_*RATE_P`` by ``MC_*RATE_K``, so writing an effective gain
    straight into it would inflate the whole tune by exactly the factor the user
    chose -- and it would look like an opinion about tuning rather than a units
    bug.
    """
    from rotorid.core.guidance.plan import build_plan
    from rotorid.core.pipeline import analyze
    from tests.synthetic.generators import make_airframe, make_bundle, make_chain

    bundle = make_bundle(
        make_airframe(), make_chain(), stack="px4", with_motor_noise=True, path="flight.ulg"
    )
    result = analyze(bundle, ("roll",), CONFIG, tool_version="test")
    rec = result.session.recommendations["roll"]

    at_k1 = build_plan({"roll": rec}, (), {**bundle.params, "MC_ROLLRATE_K": 1.0})
    at_k2 = build_plan({"roll": rec}, (), {**bundle.params, "MC_ROLLRATE_K": 2.0})

    def gain(plan, name):
        return next(s.changes[name] for s in plan.stages if name in s.changes)

    assert gain(at_k1, "MC_ROLLRATE_P") == pytest.approx(
        2.0 * gain(at_k2, "MC_ROLLRATE_P"), rel=1e-3
    )
    assert gain(at_k1, "MC_ROLLRATE_P") == pytest.approx(rec.gains.kp, rel=1e-3)
    assert gain(at_k1, "MC_ROLLRATE_FF") == pytest.approx(gain(at_k2, "MC_ROLLRATE_FF")), (
        "PX4 scales P, I and D by K and leaves the feed-forward alone"
    )


def test_findings_point_at_the_document_for_the_stack_in_hand() -> None:
    """An ArduPilot parameter recipe in a PX4 finding is advice that cannot be taken."""
    from rotorid.core.pipeline import analyze
    from tests.synthetic.generators import make_airframe, make_bundle, make_chain

    bundle = make_bundle(
        make_airframe(), make_chain(), stack="px4", with_motor_noise=True, path="flight.ulg"
    )
    findings = analyze(bundle, ("roll",), CONFIG, tool_version="test").session.findings
    assert findings
    for finding in findings:
        if finding.doc_link is not None:
            assert finding.doc_link == "docs/logging-setup-px4.md", finding.code
        for ardupilot_only in ("INS_LOG_BAT", "LOG_BITMASK", "SID_", "ATC_RAT_"):
            assert ardupilot_only not in finding.action, f"{finding.code}: {finding.action}"


# --------------------------------------------------------------------------- #
# PX4's own autotune, ingested and compared (M9)
# --------------------------------------------------------------------------- #


@pytest.fixture
def autotuned(tmp_path: Path) -> Path:
    return write_px4_log(tmp_path / "autotune.ulg", duration_s=20.0, with_autotune=True)


def test_an_autotune_run_becomes_a_stack_agnostic_gate(autotuned: Path) -> None:
    """ArduPilot says it in the event log, PX4 in a status topic, both land here."""
    bundle = read_px4(autotuned)
    gate = bundle.signals["mode.autotune"]
    assert set(np.unique(gate.y)) <= {0.0, 1.0}
    assert gate.y.any()
    assert not gate.y.all(), "the fixture is idle at both ends and the gate should say so"


def test_a_log_with_an_autotune_in_it_reads_as_a_tuning_flight(autotuned: Path) -> None:
    assert read_px4(autotuned).kind == "tuning"


def test_px4s_own_gains_are_ingested_rather_than_ignored(autotuned: Path) -> None:
    """A second opinion about the same aircraft is the only external check there is."""
    vendor = read_px4(autotuned).vendor_tunes
    assert len(vendor) == 1
    assert vendor[0].source == "px4_autotune"
    assert vendor[0].fitness == pytest.approx(0.02, rel=1e-3)
    assert len(vendor[0].coefficients) == 5


def test_the_k_factor_is_resolved_for_the_autotune_too(autotuned: Path) -> None:
    """The same conversion the parameter reader does, for the same reason.

    PX4 publishes the autotune result in standard form -- an overall ``kc`` with
    the proportional term at unity -- so the effective gains are ``kc``,
    ``kc * ki`` and ``kc * kd``. A number that arrived in a different
    parameterization and was not converted is a silent factor error in every
    comparison downstream.
    """
    vendor = read_px4(autotuned).vendor_tunes[0]
    kc, ki, kd = 0.15, 0.2, 0.003
    assert vendor.gains.kp == pytest.approx(kc, rel=1e-4)
    assert vendor.gains.ki == pytest.approx(kc * ki, rel=1e-4)
    assert vendor.gains.kd == pytest.approx(kc * kd, rel=1e-4)


def test_the_converged_value_is_taken_not_the_average(tmp_path: Path) -> None:
    """An autotune's intermediate values are a search, not an answer."""
    log = write_px4_log(
        tmp_path / "converged.ulg",
        duration_s=20.0,
        with_autotune=True,
        autotune_gains=(0.4, 0.1, 0.01),
    )
    assert read_px4(log).vendor_tunes[0].gains.kp == pytest.approx(0.4, rel=1e-4)


def test_the_axis_comes_from_which_one_moved_not_from_the_state_enum(autotuned: Path) -> None:
    """The enum has been renumbered between releases; the aircraft has not.

    The fixture excites roll hardest, so roll is the answer whatever number the
    state field happens to carry.
    """
    assert read_px4(autotuned).vendor_tunes[0].axis == "roll"


def test_a_log_without_an_autotune_carries_no_vendor_opinion(log: Path) -> None:
    """Absent, not zero. A tuner that did not run has not agreed with anything."""
    bundle = read_px4(log)
    assert bundle.vendor_tunes == ()
    assert "mode.autotune" not in bundle.signals


def test_a_disagreement_between_the_two_tuners_is_reported(autotuned: Path) -> None:
    """One of them describes a vehicle that does not exist, and it matters which."""
    from rotorid.core.guidance.findings import check_vendor_tune

    bundle = read_px4(autotuned)
    vendor = bundle.vendor_tunes[0]
    ours = _recommendation_scaled_from(vendor, factor=4.0)
    findings = check_vendor_tune(_context(bundle, {vendor.axis: ours}))
    assert [f.code for f in findings] == ["VENDOR_TUNE_DISAGREES"]
    assert findings[0].severity == "warning"


def test_agreement_between_the_two_tuners_is_reported_too(autotuned: Path) -> None:
    """Two independent estimates landing together is evidence neither can produce alone."""
    from rotorid.core.guidance.findings import check_vendor_tune

    bundle = read_px4(autotuned)
    vendor = bundle.vendor_tunes[0]
    ours = _recommendation_scaled_from(vendor, factor=1.1)
    findings = check_vendor_tune(_context(bundle, {vendor.axis: ours}))
    assert [f.code for f in findings] == ["VENDOR_TUNE_AGREES"]
    assert findings[0].severity == "good"


def _recommendation_scaled_from(vendor, *, factor: float):
    """A recommendation that is the vendor's answer scaled by a known factor.

    Built rather than analysed, and scaled from the vendor's own numbers rather
    than invented: what is under test is the ratio the check computes, so the
    fixture has to control that ratio exactly. Every term is scaled together,
    because a fixture that moved P and D by different factors would be testing
    the check's choice of worst term rather than its threshold.
    """
    from unittest.mock import Mock

    from rotorid.core.types import GainSet

    rec = Mock()
    rec.gains = GainSet(
        axis=vendor.axis,
        kp=vendor.gains.kp * factor,
        ki=vendor.gains.ki * factor,
        kd=vendor.gains.kd * factor,
        kff=0.0,
    )
    return rec


def _context(bundle, recommendations):
    from rotorid.core.guidance.findings import GuidanceContext

    return GuidanceContext(
        bundle=bundle, analyses={}, recommendations=recommendations, config=load_config()
    )
