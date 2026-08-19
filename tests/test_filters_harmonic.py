"""Tests for the ArduPilot harmonic notch stack.

Each test pins a behaviour transcribed from ``HarmonicNotchFilter.cpp``. Several of
these are exactly the details a plausible-looking reimplementation gets wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotorid.core.filters.biquad import cascade_response, phase_lag_deg
from rotorid.core.filters.harmonic import (
    HarmonicNotch,
    NotchOption,
    composite_count,
    harmonics_from_bitmask,
)


def _db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(x))


def _notch(**kwargs: object) -> HarmonicNotch:
    base: dict[str, object] = {
        "freq_hz": 80.0,
        "bandwidth_hz": 40.0,
        "attenuation_db": 40.0,
        "harmonics": (1,),
        "sample_rate_hz": 4000.0,
    }
    base.update(kwargs)
    return HarmonicNotch(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Bitmask decoding
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("hmncs", "expected"),
    [(1, (1,)), (3, (1, 2)), (7, (1, 2, 3)), (5, (1, 3)), (0, ()), (0x8000, (16,))],
)
def test_harmonics_bitmask(hmncs: int, expected: tuple[int, ...]) -> None:
    assert harmonics_from_bitmask(hmncs) == expected


def test_composite_count_prefers_triple_over_double() -> None:
    """Upstream says pick one; triple is the preferred option, so it wins."""
    assert composite_count(0) == 1
    assert composite_count(NotchOption.DOUBLE_NOTCH) == 2
    assert composite_count(NotchOption.TRIPLE_NOTCH) == 3
    assert composite_count(NotchOption.DOUBLE_NOTCH | NotchOption.TRIPLE_NOTCH) == 3


# --------------------------------------------------------------------------- #
# Stack construction
# --------------------------------------------------------------------------- #


def test_single_notch_has_expected_depth() -> None:
    notch = _notch()
    stages = notch.stages(80.0)
    assert len(stages) == 1
    assert _db(cascade_response(stages, np.array([80.0])))[0] == pytest.approx(-40.0, abs=1e-6)


def test_three_harmonics_produce_three_stages_at_multiples() -> None:
    notch = _notch(harmonics=(1, 2, 3))
    stages = notch.stages(80.0)
    assert len(stages) == 3
    f = np.array([80.0, 160.0, 240.0])
    depths = _db(cascade_response(stages, f))
    assert np.all(depths < -30.0)


def test_double_notch_has_no_centre_notch() -> None:
    """Two composite notches sit at 1 -/+ spread, with nothing at the centre.

    A reimplementation that puts a notch at the centre plus two skirts would give
    a deeper, narrower response than the firmware and mispredict both attenuation
    and phase.
    """
    notch = _notch(opts=NotchOption.DOUBLE_NOTCH)
    stages = notch.stages(80.0)
    assert len(stages) == 2
    spread = notch.bandwidth_hz / (32.0 * notch.freq_hz)
    at_centre = _db(cascade_response(stages, np.array([80.0])))[0]
    at_lower = _db(cascade_response(stages, np.array([80.0 * (1.0 - spread)])))[0]
    assert at_lower < at_centre, "the sub-notches, not the centre, are the deep points"


def test_triple_notch_has_centre_plus_two_skirts() -> None:
    notch = _notch(opts=NotchOption.TRIPLE_NOTCH)
    assert len(notch.stages(80.0)) == 3


def test_composite_divides_bandwidth_for_shaping() -> None:
    """Each sub-notch is designed with BW/N, so the composite matches a single notch.

    The composite exists to widen the *stopband* without deepening it, so the
    combined depth should stay in the same neighbourhood as the single notch
    rather than tripling in dB.
    """
    single = _db(cascade_response(_notch().stages(80.0), np.array([80.0])))[0]
    triple = _db(
        cascade_response(_notch(opts=NotchOption.TRIPLE_NOTCH).stages(80.0), np.array([80.0]))
    )[0]
    assert triple < single  # deeper, but not by a factor of three
    assert abs(triple) < 3.0 * abs(single)


def test_multi_source_makes_one_stack_per_motor() -> None:
    notch = _notch(harmonics=(1, 2), opts=NotchOption.MULTI_SOURCE)
    assert notch.per_motor
    stages = notch.stages([80.0, 85.0, 90.0, 95.0])
    assert len(stages) == 8  # 4 motors x 2 harmonics


# --------------------------------------------------------------------------- #
# Frequency limits
# --------------------------------------------------------------------------- #


def test_harmonic_above_nyquist_cutoff_is_dropped() -> None:
    """Notches at or above 0.48 * fs are disabled by the firmware."""
    notch = _notch(harmonics=(1, 2, 3), sample_rate_hz=1000.0, freq_hz=200.0, bandwidth_hz=100.0)
    stages = notch.stages(200.0)
    # 200 and 400 Hz survive; 600 Hz is above 0.48 * 1000.
    assert len(stages) == 2


def test_notch_disabled_below_quarter_of_minimum_frequency() -> None:
    notch = _notch(freq_min_ratio=0.7)
    minimum = notch.minimum_freq_hz  # 56 Hz
    assert notch.stages(minimum * 0.2) == []
    assert notch.stages(minimum * 0.5) != []


def test_attenuation_fades_out_below_minimum_frequency() -> None:
    """Attenuation is interpolated toward unity below the minimum, not switched off.

    Checked at the notch's own centre, which is clamped to the minimum frequency,
    so the depth there is what actually changes.
    """
    notch = _notch(freq_min_ratio=0.7)
    minimum = notch.minimum_freq_hz
    at_min = _db(cascade_response(notch.stages(minimum), np.array([minimum])))[0]
    part_way = _db(cascade_response(notch.stages(minimum * 0.5), np.array([minimum])))[0]
    assert at_min == pytest.approx(-40.0, abs=1e-6)
    assert -40.0 < part_way < -1.0, "faded, but not yet gone"


def test_centre_is_floored_at_minimum_frequency() -> None:
    notch = _notch(freq_min_ratio=0.7)
    assert notch.tracked_center_hz(motor_hz=10.0) == [pytest.approx(56.0)]


def test_treat_low_as_min_scales_minimum_with_harmonic() -> None:
    """With the option set, harmonics keep their spacing instead of collapsing."""
    notch = _notch(harmonics=(1, 2), freq_min_ratio=0.7, opts=NotchOption.TREAT_LOW_AS_MIN)
    assert notch.treat_low_as_min
    assert notch.stages(20.0)  # not disabled -- clamped to the per-harmonic minimum


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #


def test_throttle_tracking_uses_square_root_of_thrust_ratio() -> None:
    """f = FREQ * sqrt(throttle / REF). REF is a thrust reference, not a frequency."""
    notch = _notch(freq_hz=80.0, freq_min_ratio=0.0)
    at_hover = notch.tracked_center_hz(throttle=0.35, ref=0.35)[0]
    assert at_hover == pytest.approx(80.0)
    # Four times the thrust doubles the frequency.
    assert notch.tracked_center_hz(throttle=1.4, ref=0.35)[0] == pytest.approx(160.0)


def test_throttle_tracking_needs_a_reference() -> None:
    with pytest.raises(ValueError, match="throttle and a positive ref"):
        _notch().tracked_center_hz(throttle=0.4, ref=0.0)


def test_measured_frequency_takes_priority_over_throttle_model() -> None:
    notch = _notch(freq_min_ratio=0.0)
    assert notch.tracked_center_hz(motor_hz=123.0, throttle=0.4, ref=0.35) == [123.0]


# --------------------------------------------------------------------------- #
# Phase cost -- the number the optimizer trades against attenuation
# --------------------------------------------------------------------------- #


def test_narrower_bandwidth_costs_less_phase_at_crossover() -> None:
    """The 4:1 ratio recommended with per-motor tracking roughly halves the lag."""
    f = np.array([20.0])
    wide = phase_lag_deg(cascade_response(_notch(harmonics=(1, 2, 3)).stages(80.0), f))[0]
    narrow = phase_lag_deg(
        cascade_response(_notch(harmonics=(1, 2, 3), bandwidth_hz=20.0).stages(80.0), f)
    )[0]
    assert narrow < wide
    assert narrow == pytest.approx(wide / 2.0, rel=0.15)


def test_more_harmonics_cost_more_phase() -> None:
    f = np.array([20.0])
    lags = [
        phase_lag_deg(cascade_response(_notch(harmonics=h).stages(80.0), f))[0]
        for h in [(1,), (1, 2), (1, 2, 3)]
    ]
    assert lags[0] < lags[1] < lags[2]
