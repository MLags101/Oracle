"""Parity tests for the firmware-exact filter primitives.

These assert the properties the ArduPilot formulas are *supposed* to deliver, so a
transcription error shows up here rather than as a mysteriously conservative tune.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotorid.core.filters.biquad import (
    biquad_response,
    cascade_response,
    lpf2p_biquad,
    notch_A_Q,
    notch_biquad,
    onepole_alpha,
    onepole_response,
    phase_lag_deg,
)


def _db(x: np.ndarray | complex) -> np.ndarray:
    return 20.0 * np.log10(np.abs(x))


# --------------------------------------------------------------------------- #
# Notch
# --------------------------------------------------------------------------- #

NOTCH_CASES = [
    # (center_hz, bandwidth_hz, attenuation_db, sample_rate_hz)
    (80.0, 40.0, 40.0, 1000.0),
    (80.0, 40.0, 40.0, 4000.0),
    (160.0, 40.0, 30.0, 4000.0),
    (60.0, 15.0, 20.0, 2000.0),
    (240.0, 120.0, 40.0, 8000.0),
]


@pytest.mark.parametrize(("f0", "bw", "att", "fs"), NOTCH_CASES)
def test_notch_depth_equals_attenuation(f0: float, bw: float, att: float, fs: float) -> None:
    """Magnitude at the centre must be exactly -ATT dB.

    This falls out of ``A = 10^(-att/40)`` appearing squared in b0/b2, and is the
    single most load-bearing property of the transcription.
    """
    A, Q = notch_A_Q(f0, bw, att)
    stage = notch_biquad(f0, A, Q, fs)
    assert _db(stage.response(np.array([f0]))[0]) == pytest.approx(-att, abs=1e-6)


@pytest.mark.parametrize(("f0", "bw", "att", "fs"), NOTCH_CASES)
def test_notch_half_power_points(f0: float, bw: float, att: float, fs: float) -> None:
    """The -3 dB points sit at f0 -/+ BW/2.

    Discrete time makes this slightly asymmetric, and the asymmetry grows with
    f0/fs -- which is exactly why the tool never uses an analog approximation.
    The lower edge is the tight one; the upper edge is allowed more slack.
    """
    A, Q = notch_A_Q(f0, bw, att)
    stage = notch_biquad(f0, A, Q, fs)
    lower = _db(stage.response(np.array([f0 - bw / 2.0]))[0])
    upper = _db(stage.response(np.array([f0 + bw / 2.0]))[0])
    assert lower == pytest.approx(-3.0, abs=0.25)
    assert upper == pytest.approx(-3.0, abs=1.5)
    assert upper < lower, "discrete notch is deeper on the high side"


def test_notch_disabled_when_bandwidth_exceeds_twice_centre() -> None:
    """f0 <= BW/2 makes the octave expression undefined; firmware disables the notch."""
    _, q = notch_A_Q(center_hz=20.0, bandwidth_hz=40.0, attenuation_db=40.0)
    assert q == 0.0

    a, q = notch_A_Q(center_hz=20.0, bandwidth_hz=60.0, attenuation_db=40.0)
    stage = notch_biquad(20.0, a, q, 4000.0)
    f = np.array([1.0, 20.0, 100.0])
    assert np.allclose(np.abs(stage.response(f)), 1.0)
    assert np.allclose(phase_lag_deg(stage.response(f)), 0.0)


def test_notch_above_nyquist_is_passthrough() -> None:
    """A notch centre at or above Nyquist cannot be realized and must not alias."""
    a, q = notch_A_Q(500.0, 250.0, 40.0)
    stage = notch_biquad(500.0, a, q, 1000.0)
    assert np.allclose(np.abs(stage.response(np.array([10.0, 100.0]))), 1.0)


def test_notch_is_constant_q_across_harmonics() -> None:
    """Harmonics reuse the fundamental's A and Q, so bandwidth scales with centre.

    ``HarmonicNotchFilter::init`` computes A and Q once and only multiplies the
    centre frequency, which makes the stack constant-Q. Check that harmonic n has
    an absolute -3 dB width of about n * BW.
    """
    f0, bw, att, fs = 80.0, 40.0, 40.0, 8000.0
    A, Q = notch_A_Q(f0, bw, att)
    for n in (1, 2, 3):
        stage = notch_biquad(f0 * n, A, Q, fs)
        lower = _db(stage.response(np.array([f0 * n - (bw * n) / 2.0]))[0])
        assert lower == pytest.approx(-3.0, abs=0.3)


def test_notch_phase_cost_matches_documented_example() -> None:
    """The worked example in the spec (3 harmonics at 80/160/240 Hz, fs = 4 kHz).

    2:1 FREQ:BW costs about 7.6 / 15.7 / 24.6 degrees at 10 / 20 / 30 Hz; the 4:1
    ratio used with per-motor tracking costs about 3.5 / 7.2 / 11.4. Those numbers
    are quoted to users, so pin them.
    """
    fs = 4000.0
    f = np.array([10.0, 20.0, 30.0])

    A2, Q2 = notch_A_Q(80.0, 40.0, 40.0)
    wide = cascade_response([notch_biquad(80.0 * n, A2, Q2, fs) for n in (1, 2, 3)], f)
    assert phase_lag_deg(wide) == pytest.approx([7.62, 15.65, 24.63], abs=0.1)

    A4, Q4 = notch_A_Q(80.0, 20.0, 40.0)
    narrow = cascade_response([notch_biquad(80.0 * n, A4, Q4, fs) for n in (1, 2, 3)], f)
    assert phase_lag_deg(narrow) == pytest.approx([3.50, 7.22, 11.43], abs=0.1)

    assert np.all(phase_lag_deg(narrow) < phase_lag_deg(wide))


# --------------------------------------------------------------------------- #
# Low pass
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cutoff", [10.0, 20.0, 42.0, 80.0, 120.0])
def test_lpf2p_is_minus_3db_at_cutoff(cutoff: float) -> None:
    """A 2-pole Butterworth is -3 dB at its cutoff."""
    stage = lpf2p_biquad(cutoff, 2000.0)
    assert _db(stage.response(np.array([cutoff]))[0]) == pytest.approx(-3.01, abs=0.05)


def test_lpf2p_dc_gain_is_unity() -> None:
    stage = lpf2p_biquad(42.0, 2000.0)
    assert np.abs(stage.response(np.array([0.0]))[0]) == pytest.approx(1.0, abs=1e-9)


def test_lpf2p_rolls_off_at_40db_per_decade() -> None:
    """Two poles means 40 dB/decade well above the corner."""
    stage = lpf2p_biquad(20.0, 8000.0)
    high = _db(stage.response(np.array([200.0, 400.0])))
    assert (high[0] - high[1]) == pytest.approx(12.04, abs=0.5)


def test_lpf2p_cutoff_clamped_to_40_percent_of_sample_rate() -> None:
    """Firmware clamps the cutoff; an unclamped design would be unstable."""
    clamped = lpf2p_biquad(900.0, 1000.0)
    explicit = lpf2p_biquad(400.0, 1000.0)
    assert clamped.b == pytest.approx(explicit.b)
    assert clamped.a == pytest.approx(explicit.a)


def test_lpf2p_non_positive_cutoff_is_passthrough() -> None:
    stage = lpf2p_biquad(0.0, 1000.0)
    assert np.allclose(np.abs(stage.response(np.array([1.0, 50.0, 400.0]))), 1.0)


# --------------------------------------------------------------------------- #
# One pole (FLTT / FLTE / FLTD)
# --------------------------------------------------------------------------- #


def test_onepole_alpha_matches_ac_pid_formula() -> None:
    dt = 1.0 / 400.0
    fc = 20.0
    expected = dt / (dt + 1.0 / (2.0 * np.pi * fc))
    assert onepole_alpha(fc, dt) == pytest.approx(expected)


def test_onepole_disabled_when_cutoff_non_positive() -> None:
    assert onepole_alpha(0.0, 1.0 / 400.0) == 1.0
    resp = onepole_response(1.0, np.array([1.0, 50.0]), 400.0)
    assert np.allclose(np.abs(resp), 1.0)


def test_onepole_dc_gain_is_unity() -> None:
    alpha = onepole_alpha(20.0, 1.0 / 400.0)
    assert np.abs(onepole_response(alpha, np.array([0.0]), 400.0)[0]) == pytest.approx(1.0)


def test_onepole_phase_cost_is_large_at_typical_fltd() -> None:
    """A 20 Hz FLTD at 400 Hz loop rate is often the biggest single lag term.

    Roughly 26 degrees at 10 Hz and 40 at 20 Hz -- worth more phase than a whole
    3-harmonic notch stack, which is why the tool optimizes it rather than
    defaulting it.
    """
    fs = 400.0
    alpha = onepole_alpha(20.0, 1.0 / fs)
    lag = phase_lag_deg(onepole_response(alpha, np.array([10.0, 20.0]), fs))
    assert lag == pytest.approx([25.6, 40.4], abs=0.5)


# --------------------------------------------------------------------------- #
# Cascade
# --------------------------------------------------------------------------- #


def test_cascade_of_nothing_is_unity() -> None:
    assert np.allclose(cascade_response([], np.array([1.0, 10.0])), 1.0)


def test_cascade_multiplies_stages() -> None:
    f = np.array([5.0, 25.0, 60.0])
    a, b = lpf2p_biquad(30.0, 2000.0), lpf2p_biquad(60.0, 2000.0)
    assert np.allclose(cascade_response([a, b], f), a.response(f) * b.response(f))


def test_cascade_rejects_mixed_sample_rates() -> None:
    """Mixing rates silently produces a wrong phase, so it must raise."""
    with pytest.raises(ValueError, match="mixes sample rates"):
        cascade_response([lpf2p_biquad(30.0, 2000.0), lpf2p_biquad(30.0, 400.0)], np.array([10.0]))


def test_biquad_response_matches_scipy() -> None:
    """Cross-check the response evaluation against scipy's own freqz."""
    from scipy.signal import freqz

    fs = 4000.0
    A, Q = notch_A_Q(80.0, 40.0, 40.0)
    stage = notch_biquad(80.0, A, Q, fs)
    f = np.linspace(1.0, 1500.0, 257)
    _, h_scipy = freqz(stage.b, stage.a, worN=f, fs=fs)
    ours = biquad_response(stage.b, stage.a, f, fs)
    assert np.allclose(ours, h_scipy, rtol=1e-10, atol=1e-12)
