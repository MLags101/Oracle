"""Defects a real tuning-grade log exposed, each pinned by a synthetic case.

Every test here corresponds to something that was wrong and that no synthetic
fixture could have caught, because the fixtures were built out of the same
assumptions the code was. They are written against constructed data anyway, so
they run without the log and keep running once it is gone.

The log that found them: half an hour of ArduPilot 4.7, ``RATE`` at the loop rate,
raw gyro at 1.6 kHz, the onboard FFT running, and no ESC telemetry.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotorid.config import load_config
from rotorid.core.analysis.noise import motor_track
from rotorid.core.analysis.spectra import choose_shared_nperseg, lowest_resolvable_hz
from rotorid.core.io.base import canonical_signal, gate_signal
from rotorid.core.preprocess.resample import resample_to_grid, uniform_grid
from rotorid.core.preprocess.segment import airborne_windows, propose_segments
from tests.synthetic.generators import make_airframe, make_chain, make_general_flight_bundle

CONFIG = load_config()


# --------------------------------------------------------------------------- #
# Units: a signal that lies about itself is worse than one that is missing
# --------------------------------------------------------------------------- #


def test_a_heading_is_converted_rather_than_passed_through() -> None:
    """``degheading`` is degrees, and ATT.Yaw is logged in it.

    Missing this put yaw attitude into the bundle at 57 times its true value
    under a key whose contract says radians -- and the only trace was one warning
    among thirty. Every consumer downstream believes the label.
    """
    from rotorid.core.io.ardupilot import _UNIT_SCALE

    assert _UNIT_SCALE["degheading"] == pytest.approx(np.pi / 180.0)
    assert _UNIT_SCALE["deg"] == _UNIT_SCALE["degheading"]


def test_pwm_microseconds_are_left_alone_deliberately() -> None:
    """``motor.{n}.output`` is defined as the PWM the vehicle wrote."""
    from rotorid.core.io.ardupilot import _UNIT_SCALE

    assert _UNIT_SCALE["us"] == 1.0


def test_an_unknown_unit_drops_the_signal_instead_of_mislabelling_it() -> None:
    """A missing signal is reportable. A mislabelled one is not, by anything."""
    from rotorid.core.io.ardupilot import ArduPilotReader

    reader = ArduPilotReader.__new__(ArduPilotReader)
    reader._warnings = []
    assert reader._scale("furlongs", "RATE", "R") is None
    assert reader._scale("deg/s", "RATE", "R") == pytest.approx(np.pi / 180.0)
    assert any("dropped rather than guessed at" in w for w in reader._warnings)


# --------------------------------------------------------------------------- #
# One window length per axis, because the spectra are summed
# --------------------------------------------------------------------------- #


def test_every_segment_lands_on_one_grid() -> None:
    """Estimates on different grids cannot be summed, and summing them is the plan.

    Choosing the window per segment is invisible on a fixture whose segments are
    all the same length and fatal on a flight whose stick inputs are not.
    """
    sizes = [4000, 5200, 6400, 4096]
    nperseg, _, usable = choose_shared_nperseg(sizes, 800.0, f_lowest_hz=None, min_averages=5)
    assert usable, "some segment has to be usable"
    assert all(sizes[i] >= nperseg for i in usable)
    assert nperseg & (nperseg - 1) == 0, "Welch wants a power of two"


def test_a_declared_sweep_that_cannot_be_resolved_is_an_error() -> None:
    """The user asked for 0.05 Hz and the record cannot show it. That is actionable."""
    with pytest.raises(ValueError, match="cannot resolve"):
        choose_shared_nperseg([4000], 800.0, f_lowest_hz=0.05, min_averages=5)


def test_ordinary_flight_takes_the_band_its_windows_can_give() -> None:
    """Nobody declared anything, so refusing for missing 0.5 Hz would be inventing a test.

    Short windows simply start the band higher, and the narrower result reaches
    the confidence rating on its own.
    """
    nperseg, lowest, usable = choose_shared_nperseg(
        [2400, 2600], 800.0, f_lowest_hz=None, min_averages=5
    )
    assert usable
    assert lowest > 0.5, "these windows are too short to reach the pilot-input default"
    assert lowest == pytest.approx(lowest_resolvable_hz(nperseg, 800.0))


def test_resolution_still_wins_over_averaging() -> None:
    """A long record must not be chopped up to buy averages it was not asked for.

    An FRF averaged three times is noisy but usable; one that cannot see the
    lowest excited frequency is useless, and that trade is settled the same way
    for one record and for many.
    """
    nperseg, lowest, _ = choose_shared_nperseg([72000], 400.0, f_lowest_hz=0.2, min_averages=5)
    assert lowest == 0.2
    assert lowest_resolvable_hz(nperseg, 400.0) <= 0.2


# --------------------------------------------------------------------------- #
# Ground data is a different plant
# --------------------------------------------------------------------------- #


def _grounded(bundle, flying: list[tuple[float, float]]):
    """The same flight, with the vehicle declaring when it was off the ground."""
    import dataclasses

    grid = bundle.signals["rate.roll.output"].t
    return dataclasses.replace(
        bundle,
        signals={
            **bundle.signals,
            "mode.flying": gate_signal("mode.flying", grid, flying, source_msg="EV.Id"),
        },
    )


def test_excitation_on_the_ground_is_not_identified_from() -> None:
    """A vehicle on its legs is not free to rotate, so the rate loop sees a
    constraint rather than an inertia -- and nothing downstream can tell."""
    bundle = make_general_flight_bundle(make_airframe(), make_chain())
    assert propose_segments(bundle), "the fixture has usable excitation to begin with"

    never_flew = _grounded(bundle, [])
    assert airborne_windows(never_flew) == []
    assert propose_segments(never_flew) == ()


def test_the_airborne_window_is_honoured_rather_than_the_whole_record() -> None:
    bundle = make_general_flight_bundle(make_airframe(), make_chain())
    grid = bundle.signals["rate.roll.output"].t
    half = float(grid[0] + 0.5 * (grid[-1] - grid[0]))
    airborne = _grounded(bundle, [(float(grid[0]), half)])

    segments = propose_segments(airborne)
    assert segments, "the first half of the flight still has excitation in it"
    assert all(s.t_end <= half + 1e-6 for s in segments)


def test_a_log_that_never_says_is_searched_whole() -> None:
    """Absent landing state and "it never flew" are different, and read differently."""
    bundle = make_general_flight_bundle(make_airframe(), make_chain())
    assert airborne_windows(bundle) is None
    assert propose_segments(bundle)


# --------------------------------------------------------------------------- #
# Resampling a signal that is faster, or full of holes
# --------------------------------------------------------------------------- #


def test_a_fast_signal_is_filtered_before_it_is_decimated() -> None:
    """Otherwise its top half folds down as a mirror image of itself.

    A 500 Hz line splined onto an 800 Hz grid reappears at 300 Hz, at a plausible
    frequency, in the band the notch designer works in, and nothing afterwards
    can tell it from a frame resonance.
    """
    fs = 1600.0
    t = np.arange(0.0, 8.0, 1.0 / fs)
    y = np.sin(2.0 * np.pi * 500.0 * t)
    signal = canonical_signal("gyro.roll.prefilter", t, y, source_msg="GYR.GyrX", filtered=False)

    out = resample_to_grid(signal, uniform_grid(0.0, 7.9, 800.0))

    # Nothing above the grid's Nyquist survives, so a tone that was entirely above
    # it leaves almost nothing behind. Splined directly it would arrive at full
    # amplitude, sitting at 300 Hz and looking exactly like a frame resonance.
    survived = float(np.sqrt(np.mean(np.square(out.y))))
    assert survived < 0.02, f"the 500 Hz tone came through at {survived:.3f} RMS"
    assert out.native_rate_hz == pytest.approx(800.0), (
        "after decimation the signal cannot describe anything past the grid's Nyquist"
    )


def test_a_spline_is_not_trusted_across_a_dropout() -> None:
    """A cubic through a hole is fitting a polynomial to four distant points.

    Raw gyro with dropouts came back with excursions of forty thousand radians
    per second, which is not a value anything downstream is equipped to
    disbelieve.
    """
    rng = np.random.default_rng(0)
    fs = 1600.0
    t = np.arange(0.0, 20.0, 1.0 / fs)
    y = 0.05 * np.sin(2.0 * np.pi * 90.0 * t) + 0.02 * rng.standard_normal(t.size)
    hole = (t > 8.0) & (t < 9.5)
    signal = canonical_signal(
        "gyro.roll.prefilter", t[~hole], y[~hole], source_msg="GYR.GyrX", filtered=False
    )

    out = resample_to_grid(signal, uniform_grid(0.0, 19.9, 800.0))
    assert np.abs(out.y).max() < 2.0 * np.abs(y).max()


# --------------------------------------------------------------------------- #
# The onboard FFT, on a vehicle with no ESC telemetry
# --------------------------------------------------------------------------- #


def test_the_onboard_fft_stands_in_for_esc_telemetry() -> None:
    """Without it every line in the spectrum has to be called structural.

    A tracking notch cannot be recommended for a peak that nothing has shown to
    move, so a vehicle with no RPM source gets a static notch or nothing at all.
    """
    import dataclasses

    bundle = make_general_flight_bundle(make_airframe(), make_chain())
    grid = bundle.signals["rate.roll.output"].t
    peak = 90.0 + 10.0 * np.sin(2.0 * np.pi * 0.05 * grid)
    with_fft = dataclasses.replace(
        bundle,
        signals={
            **bundle.signals,
            "fft.roll.peak_hz": canonical_signal(
                "fft.roll.peak_hz", grid, peak, source_msg="FTN2.PkX"
            ),
        },
    )

    assert motor_track(bundle, float(grid[0]), float(grid[-1])).source != "onboard_fft"
    track = motor_track(with_fft, float(grid[0]), float(grid[-1]))
    assert track.source == "onboard_fft"
    assert 80.0 < float(np.median(track.f_hz)) < 100.0


def test_a_lock_that_never_happened_is_not_a_very_slow_motor() -> None:
    """Zeros mean the FFT had nothing, and averaging them in drags the notch down."""
    import dataclasses

    bundle = make_general_flight_bundle(make_airframe(), make_chain())
    grid = bundle.signals["rate.roll.output"].t
    silent = dataclasses.replace(
        bundle,
        signals={
            **bundle.signals,
            "fft.roll.peak_hz": canonical_signal(
                "fft.roll.peak_hz", grid, np.zeros_like(grid), source_msg="FTN2.PkX"
            ),
        },
    )
    assert motor_track(silent, float(grid[0]), float(grid[-1])).source != "onboard_fft"
