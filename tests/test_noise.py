"""Noise characterization and peak classification (milestone M3).

The acceptance criterion for M3 is that a synthetic flight whose noise sources
are known produces the right *classification*, not merely the right frequencies.
Finding a peak at 60 Hz is easy; deciding whether it wants a tracking notch, a
static notch or a spanner is the part that has to be tested, because on a real
log nobody knows the answer and a wrong classification is silent.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotorid.core.analysis.noise import (
    dterm_noise_rms,
    find_spectral_peaks,
    motor_track,
    noise_profile,
    prefilter_psd,
)
from tests.synthetic.generators import make_chain, make_noise_bundle

HOVER_HZ = 50.0
STRUCTURAL_HZ = 118.0

#: Peaks are measured over the steadiest stretch of the flight, which sits at an
#: extreme of the throttle oscillation rather than at its mean -- so the measured
#: frequency is a few percent off the nominal hover value by construction.
TOLERANCE = 0.15


def _bundle(**kw):
    chain = kw.pop("chain", make_chain(gyro_lpf_hz=100.0, notch_freq_hz=0.0, harmonics=()))
    return make_noise_bundle(chain, hover_hz=HOVER_HZ, **kw), chain


# --------------------------------------------------------------------------- #
# Motor tracking
# --------------------------------------------------------------------------- #


def test_motor_track_reads_esc_telemetry_in_hz() -> None:
    bundle, _ = _bundle()
    track = motor_track(bundle, 0.0, 40.0)

    assert track.source == "esc_telemetry"
    assert track.is_measured
    assert track.mean_hz() == pytest.approx(HOVER_HZ, rel=0.05)
    assert len(track.per_motor_hz) == 4


def test_motor_track_falls_back_to_nothing_without_rpm() -> None:
    """No RPM and no motor outputs means no track -- not a fabricated one."""
    bundle, _ = _bundle(with_esc_telemetry=False)
    track = motor_track(bundle, 0.0, 40.0)

    assert track.source == "none"
    assert track.mean_hz() == 0.0
    assert track.span_fraction == 0.0


def test_a_steady_hover_is_reported_as_unusable_for_correlation() -> None:
    """Correlation against a constant proves nothing, and must not be trusted.

    This is the case that would otherwise let a structural resonance be labelled
    as motor noise: at constant RPM, everything correlates with everything.
    """
    bundle, _ = _bundle(sweep_fraction=0.0)
    assert motor_track(bundle, 0.0, 40.0).span_fraction < 0.05


# --------------------------------------------------------------------------- #
# Peak finding
# --------------------------------------------------------------------------- #


def test_peaks_are_found_at_the_motor_harmonics() -> None:
    bundle, chain = _bundle()
    profile = noise_profile(bundle, "roll", t_start=0.0, t_end=40.0, chain=chain, op=None)
    found = sorted(p.f_hz for p in profile.peaks)

    for expected in (HOVER_HZ, 2 * HOVER_HZ, 3 * HOVER_HZ, STRUCTURAL_HZ):
        assert any(abs(f - expected) < TOLERANCE * expected for f in found), (
            f"no peak near {expected} Hz in {found}"
        )


def test_peak_finder_ignores_the_rate_response_at_the_bottom_of_the_band() -> None:
    """The airframe's own response is signal. A notch there would be a disaster."""
    f = np.geomspace(0.1, 200.0, 2000)
    psd = 1.0 / (1.0 + (f / 2.5) ** 4)  # a rate response, no noise peaks at all
    assert find_spectral_peaks(f, psd, prominence_db=6.0) == ()


# --------------------------------------------------------------------------- #
# Classification -- the M3 acceptance criterion
# --------------------------------------------------------------------------- #


def test_motor_harmonics_are_classified_as_tracking_and_the_frame_peak_is_not() -> None:
    bundle, chain = _bundle()
    profile = noise_profile(bundle, "roll", t_start=0.0, t_end=40.0, chain=chain, op=None)

    tracking = [p for p in profile.peaks if p.tracks_rpm]
    structural = [p for p in profile.peaks if p.kind == "structural"]

    assert tracking, "the motor tones sweep with RPM and must be seen to"
    assert any(abs(p.f_hz - HOVER_HZ) < TOLERANCE * HOVER_HZ for p in tracking)
    assert any(abs(p.f_hz - STRUCTURAL_HZ) < TOLERANCE * STRUCTURAL_HZ for p in structural), (
        f"the fixed {STRUCTURAL_HZ:.0f} Hz tone is a frame resonance, not something a "
        f"tracking notch can chase"
    )
    assert all(not p.tracks_rpm for p in structural)


def test_the_fundamental_is_labelled_as_the_fundamental() -> None:
    bundle, chain = _bundle()
    profile = noise_profile(bundle, "roll", t_start=0.0, t_end=40.0, chain=chain, op=None)

    fundamentals = [p for p in profile.peaks if p.kind == "motor_fundamental"]
    assert len(fundamentals) == 1
    assert fundamentals[0].harmonic_index == 1
    assert fundamentals[0].f_hz == pytest.approx(HOVER_HZ, rel=TOLERANCE)

    seconds = [p for p in profile.peaks if p.harmonic_index == 2]
    assert seconds and seconds[0].kind == "motor_harmonic"


def test_a_flight_with_no_rpm_evidence_does_not_claim_tracking() -> None:
    """Without RPM data, "tracks the motors" is a claim the log cannot support."""
    bundle, chain = _bundle(with_esc_telemetry=False)
    profile = noise_profile(bundle, "roll", t_start=0.0, t_end=40.0, chain=chain, op=None)

    assert profile.peaks
    assert all(not p.tracks_rpm for p in profile.peaks)


# --------------------------------------------------------------------------- #
# Dividing the flown chain out
# --------------------------------------------------------------------------- #


def test_prefilter_psd_undoes_the_chain_it_is_given() -> None:
    chain = make_chain(gyro_lpf_hz=60.0, notch_freq_hz=0.0, harmonics=())
    f = np.geomspace(1.0, 200.0, 400)
    true_pre = np.full_like(f, 1e-4)
    post = true_pre * np.abs(chain.sensor_response(f)) ** 2

    recovered = prefilter_psd(f, post, chain, op=None, floor_db=-20.0)
    # Only where the chain is within the clamp: past -20 dB the division is
    # deliberately floored, and the next test is what pins that behaviour.
    unclamped = np.abs(chain.sensor_response(f)) >= 10.0 ** (-20.0 / 20.0)
    assert unclamped.any()
    assert np.allclose(recovered[unclamped], true_pre[unclamped], rtol=1e-9)
    assert (recovered[~unclamped] < true_pre[~unclamped]).all(), (
        "clamping must under-estimate the pre-filter spectrum, never over-estimate it"
    )


def test_prefilter_psd_stays_bounded_inside_a_deep_notch() -> None:
    """Inside a notch the post-filter spectrum knows nothing; it must not explode."""
    chain = make_chain(gyro_lpf_hz=100.0, notch_freq_hz=90.0, notch_att_db=40.0, harmonics=(1,))
    f = np.geomspace(1.0, 200.0, 800)
    post = np.full_like(f, 1e-6)

    recovered = prefilter_psd(f, post, chain, op=None, floor_db=-20.0)
    assert np.isfinite(recovered).all()
    assert recovered.max() <= 1e-6 * 10.0 ** (20.0 / 10.0) * (1.0 + 1e-9)


def test_the_profile_finds_peaks_the_flown_notch_had_already_removed() -> None:
    """A working notch hides its own evidence, and the tool must not be fooled.

    If peaks were found on the post-filter spectrum, a chain that is doing its job
    would look like a vehicle with no noise -- and the recommendation would be to
    remove the notch that is the only reason the trace is clean.
    """
    chain = make_chain(gyro_lpf_hz=100.0, notch_freq_hz=HOVER_HZ, notch_bw_hz=30.0, harmonics=(1,))
    bundle = make_noise_bundle(chain, hover_hz=HOVER_HZ, structural_hz=None)

    profile = noise_profile(
        bundle,
        "roll",
        t_start=0.0,
        t_end=40.0,
        chain=chain,
        op=None,
    )
    assert any(abs(p.f_hz - HOVER_HZ) < TOLERANCE * HOVER_HZ for p in profile.peaks), (
        "the notched-out fundamental is still in the pre-filter spectrum and must be found"
    )


# --------------------------------------------------------------------------- #
# D-term noise
# --------------------------------------------------------------------------- #


def test_dterm_noise_scales_with_kd_and_with_cutoff() -> None:
    f = np.geomspace(1.0, 200.0, 500)
    psd = np.full_like(f, 1e-6)
    loose = make_chain(gyro_lpf_hz=120.0, notch_freq_hz=0.0, harmonics=(), dterm_lpf_hz=60.0)
    tight = make_chain(gyro_lpf_hz=40.0, notch_freq_hz=0.0, harmonics=(), dterm_lpf_hz=20.0)

    assert dterm_noise_rms(f, psd, loose, kd=0.008) > dterm_noise_rms(f, psd, loose, kd=0.004)
    assert dterm_noise_rms(f, psd, tight, kd=0.004) < dterm_noise_rms(f, psd, loose, kd=0.004)


def test_dterm_noise_is_zero_without_derivative_gain() -> None:
    f = np.geomspace(1.0, 200.0, 200)
    chain = make_chain()
    assert dterm_noise_rms(f, np.full_like(f, 1e-6), chain, kd=0.0) == 0.0
