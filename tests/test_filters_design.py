"""Notch and low-pass recommendation (milestone M3, second half).

What is under test is the *reasoning*, not the arithmetic: that a tracking notch
is only proposed where something tracks, that the phase budget is genuinely
binding rather than decorative, and that the parameter names and values are the
ones a user could paste into Mission Planner without translation.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotorid.config import load_config
from rotorid.core.analysis.noise import MotorTrack, motor_track, noise_profile
from rotorid.core.design.filters import GYRO_LPF_LADDER, choose_notch_source, recommend_filters
from rotorid.core.filters.chain import OperatingPoint
from tests.synthetic.generators import make_chain, make_noise_bundle

CONFIG = load_config()
HOVER_HZ = 50.0
STRUCTURAL_HZ = 118.0


def _profile(**kw):
    """A noise profile plus everything ``recommend_filters`` needs alongside it."""
    chain = kw.pop("chain", make_chain(gyro_lpf_hz=100.0, notch_freq_hz=0.0, harmonics=()))
    bundle = make_noise_bundle(chain, hover_hz=HOVER_HZ, **kw)
    profile = noise_profile(bundle, "roll", t_start=0.0, t_end=40.0, chain=chain, op=None)
    return profile, chain, motor_track(bundle, 0.0, 40.0), bundle


def _recommend(**kw):
    profile, chain, track, bundle = _profile(**kw.pop("fixture", {}))
    return recommend_filters(
        profile,
        chain,
        CONFIG,
        track=track,
        op=OperatingPoint(motor_hz=(HOVER_HZ,)),
        crossover_hz=kw.pop("crossover_hz", 4.0),
        kd=kw.pop("kd", 0.0036),
        hover_thrust=bundle.param("MOT_THST_HOVER"),
        **kw,
    )


# --------------------------------------------------------------------------- #
# Source selection
# --------------------------------------------------------------------------- #


def test_measured_rpm_wins_over_the_throttle_model() -> None:
    source = choose_notch_source(
        track=MotorTrack(np.zeros(1), np.array([50.0]), "esc_telemetry"),
        hover_thrust=0.35,
        fft_available=True,
        fundamental_hz=50.0,
    )
    assert source.mode == 3
    assert source.ref == 1.0
    assert any("throttle" in alternative for alternative, _ in source.rejected)


def test_throttle_mode_carries_hover_thrust_as_its_reference() -> None:
    """``REF`` is a *thrust* reference, not a frequency. Mixing them up is silent."""
    source = choose_notch_source(
        track=MotorTrack(np.zeros(0), np.zeros(0), "none"),
        hover_thrust=0.42,
        fft_available=False,
        fundamental_hz=80.0,
    )
    assert source.mode == 1
    assert source.ref == pytest.approx(0.42)
    assert source.freq_hz == pytest.approx(80.0)


def test_static_is_the_last_resort_and_says_what_it_costs() -> None:
    source = choose_notch_source(
        track=MotorTrack(np.zeros(0), np.zeros(0), "none"),
        hover_thrust=None,
        fft_available=False,
        fundamental_hz=80.0,
    )
    assert source.mode == 0
    assert "wrong place" in source.rationale


def test_a_notch_without_a_frequency_is_refused() -> None:
    with pytest.raises(ValueError, match="no motor fundamental"):
        choose_notch_source(
            track=MotorTrack(np.zeros(0), np.zeros(0), "none"),
            hover_thrust=0.35,
            fft_available=False,
            fundamental_hz=0.0,
        )


# --------------------------------------------------------------------------- #
# The recommendation
# --------------------------------------------------------------------------- #


def test_notch_is_centred_on_the_measured_fundamental_and_tracks_it() -> None:
    rec = _recommend()
    params = rec.params

    assert params["INS_HNTCH_ENABLE"] == 1.0
    assert params["INS_HNTCH_MODE"] == 3.0, "ESC telemetry is in this log"
    # A tracking mode sets FREQ to the lowest frequency worth following, which
    # sits below the hover fundamental by FM_RAT.
    assert params["INS_HNTCH_FREQ"] < HOVER_HZ
    assert params["INS_HNTCH_FREQ"] > 0.5 * HOVER_HZ
    assert 0.0 < params["INS_HNTCH_FM_RAT"] <= 1.0


def test_harmonics_are_included_only_where_a_tracking_peak_sits() -> None:
    """The fixture has three motor harmonics and one frame resonance.

    The frame resonance must not turn into a harmonic: it is not one, and a notch
    that tracked it would move away from it the moment the throttle changed.
    """
    rec = _recommend()
    mask = int(rec.params["INS_HNTCH_HMNCS"])
    included = [n + 1 for n in range(16) if mask & (1 << n)]

    assert included == [1, 2, 3]
    assert rec.chain.notches[0].harmonics == (1, 2, 3)


def test_a_structural_peak_is_called_out_as_a_mechanical_problem() -> None:
    rec = _recommend()
    assert f"{STRUCTURAL_HZ:.0f} Hz" in rec.rationale
    assert "structural resonance" in rec.rationale
    assert "fix the mechanics" in rec.rationale


def test_no_notch_is_proposed_when_nothing_tracks() -> None:
    """A log that cannot prove anything tracks must not get a tracking notch."""
    rec = _recommend(fixture={"with_esc_telemetry": False})
    assert "INS_HNTCH_ENABLE" not in rec.params
    assert not rec.chain.notches
    assert any("tracking notch" in why for _, why in rec.rejected)


def test_the_phase_budget_is_binding() -> None:
    """Pushed close enough to the noise, the design must give ground.

    A crossover just below the motor fundamental cannot carry the same notch stack
    as one a decade below it. If the recommendation is identical either way, the
    budget is not being enforced.
    """
    relaxed = _recommend(crossover_hz=2.0)
    tight = _recommend(crossover_hz=25.0)

    assert tight.phase_cost_deg >= relaxed.phase_cost_deg
    tight_harmonics = len(tight.chain.notches[0].harmonics) if tight.chain.notches else 0
    relaxed_harmonics = len(relaxed.chain.notches[0].harmonics) if relaxed.chain.notches else 0
    tight_bw = tight.params.get("INS_HNTCH_BW", 0.0)
    relaxed_bw = relaxed.params.get("INS_HNTCH_BW", 0.0)
    assert tight_harmonics < relaxed_harmonics or tight_bw < relaxed_bw, (
        "a crossover close to the noise must buy fewer harmonics or a narrower notch"
    )


def test_the_recommended_chain_actually_attenuates_the_peaks() -> None:
    rec = _recommend()
    tracked = [p for p in rec.attenuation_at_peaks_db.items()]
    assert tracked

    fundamental_att = min(
        db for f, db in rec.attenuation_at_peaks_db.items() if abs(f - HOVER_HZ) < 0.2 * HOVER_HZ
    )
    assert fundamental_att < -10.0, "the notch must actually be on the peak it was designed for"


def test_the_predicted_spectrum_is_the_pre_filter_one_put_through_the_new_chain() -> None:
    rec = _recommend()
    profile, _chain, _track, _ = _profile()
    assert rec.predicted_psd_post is not None
    assert profile.psd_pre is not None

    op = OperatingPoint(motor_hz=(HOVER_HZ,))
    expected = profile.psd_pre * np.abs(rec.chain.sensor_response(profile.f_hz, op)) ** 2
    assert np.allclose(rec.predicted_psd_post, expected, rtol=1e-6)
    assert rec.predicted_psd_post.sum() < profile.psd_pre.sum(), (
        "a chain that does not reduce total noise is not worth its phase"
    )


# --------------------------------------------------------------------------- #
# Low-pass selection
# --------------------------------------------------------------------------- #


def test_the_gyro_cutoff_is_the_highest_that_holds_the_noise_limit() -> None:
    """Filtering is phase lag. The right cutoff is the least that does the job."""
    quiet = _recommend(kd=0.0005)
    noisy = _recommend(kd=0.02)

    assert quiet.chain.gyro_lpf_hz is not None
    assert noisy.chain.gyro_lpf_hz is not None
    assert quiet.chain.gyro_lpf_hz >= noisy.chain.gyro_lpf_hz
    assert quiet.chain.gyro_lpf_hz in GYRO_LPF_LADDER


def test_the_dterm_cutoff_stays_below_the_gyro_cutoff() -> None:
    rec = _recommend()
    assert rec.chain.dterm_lpf_hz is not None
    assert rec.chain.gyro_lpf_hz is not None
    assert rec.chain.dterm_lpf_hz <= 0.75 * rec.chain.gyro_lpf_hz


def test_yaw_gets_a_lower_dterm_cutoff_than_roll() -> None:
    """Yaw carries less D and more noise; the community prior is FLTD = gyro/4."""
    profile_roll, chain, track, _bundle = _profile()
    profile_yaw = noise_profile(
        make_noise_bundle(chain, axis="yaw", hover_hz=HOVER_HZ),
        "yaw",
        t_start=0.0,
        t_end=40.0,
        chain=chain,
        op=None,
    )
    common = {
        "track": track,
        "op": OperatingPoint(motor_hz=(HOVER_HZ,)),
        "crossover_hz": 4.0,
        "kd": 0.0036,
    }
    roll = recommend_filters(profile_roll, chain, CONFIG, **common)
    yaw = recommend_filters(profile_yaw, chain, CONFIG, **common)

    assert yaw.chain.dterm_lpf_hz is not None
    assert roll.chain.dterm_lpf_hz is not None
    assert yaw.chain.dterm_lpf_hz < roll.chain.dterm_lpf_hz


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


def test_the_baseline_chain_is_carried_through_for_the_diff() -> None:
    rec = _recommend()
    assert rec.baseline_chain is not rec.chain
    assert rec.baseline_chain.gyro_lpf_hz == 100.0


def test_a_px4_chain_is_designed_in_px4s_own_names() -> None:
    """Same evidence, different firmware: the parameters must not be ArduPilot's."""
    from dataclasses import replace

    profile, chain, track, _ = _profile()
    rec = recommend_filters(
        profile,
        replace(chain, stack="px4"),
        CONFIG,
        track=track,
        op=OperatingPoint(motor_hz=(HOVER_HZ,)),
        crossover_hz=4.0,
        kd=0.0036,
    )
    assert rec.stack == "px4"
    assert not any(name.startswith(("INS_", "ATC_")) for name in rec.params)
    assert "IMU_GYRO_CUTOFF" in rec.params
    assert "INS_HNTCH_ATT" not in rec.params, "PX4 has no notch attenuation parameter"


def test_designing_without_a_prefilter_spectrum_is_refused() -> None:
    """Designing against a post-filter spectrum would count the flown chain twice."""
    from dataclasses import replace

    profile, chain, track, _ = _profile()
    with pytest.raises(ValueError, match="pre-filter spectrum"):
        recommend_filters(
            replace(profile, psd_pre=None),
            chain,
            CONFIG,
            track=track,
            op=OperatingPoint(motor_hz=(HOVER_HZ,)),
            crossover_hz=4.0,
            kd=0.0036,
        )
