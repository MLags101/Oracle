"""The gap between the rate a vehicle runs at and the rate it writes to the card.

This is the one failure mode in the whole tool that every other check is blind
to. ``SCHED_LOOP_RATE`` says 400 Hz, the gyro says 2 kHz, the analysis grid is
built at 800 Hz -- and none of those is a claim about how often ``RATE`` reached
the SD card. On ArduPilot that is a separate decision: with ``LOG_BITMASK``
bit 0 (ATTITUDE_FAST) clear, the rate and attitude messages go out on the 10 Hz
medium-rate schedule instead, and nothing else in the file says so.

What makes it dangerous rather than merely limiting is that it does not look
like a problem. The resampler splines 10 Hz onto the grid; coherence between the
two splined signals stays high all the way to 200 Hz, because they were smoothed
by the same interpolator; and a confident airframe model comes back fitted to
the shape of a cubic. So the guard has to be an explicit one, at the boundary
where the log's own timestamps are still visible.

Every property below was found by running the tool on real ArduCopter 4.7 logs.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotorid.config import load_config
from rotorid.core.analysis.noise import noise_profile
from rotorid.core.design.recommend import evidence_ceiling_hz, identify_axis
from rotorid.core.guidance.findings import GuidanceContext, collect_findings
from rotorid.core.io.base import canonical_signal, native_rate_hz
from rotorid.core.preprocess.resample import resample_to_grid, uniform_grid
from rotorid.core.types import LogBundle
from tests.synthetic.generators import make_airframe, make_bundle, make_chain

# --------------------------------------------------------------------------- #
# The measurement itself
# --------------------------------------------------------------------------- #


def test_native_rate_is_the_median_not_the_mean() -> None:
    """A handful of dropped writes must not be mistaken for a slower schedule."""
    t = np.arange(0.0, 10.0, 0.0025)
    gapped = np.delete(t, slice(1000, 1400))  # one long stall mid-record
    assert native_rate_hz(gapped) == pytest.approx(400.0, rel=1e-6)


def test_native_rate_survives_being_put_on_a_faster_grid() -> None:
    """The whole point: on the grid, every signal reports the grid rate.

    If the native rate were recomputed downstream instead of carried, a 10 Hz
    message would report 800 Hz the moment it was resampled, and the number that
    says how much of it is real would be gone.
    """
    t = np.arange(0.0, 20.0, 0.1)
    slow = canonical_signal(
        "rate.roll.measured", t, np.sin(2.0 * np.pi * 0.3 * t), source_msg="RATE.R"
    )
    assert slow.native_rate_hz == pytest.approx(10.0)

    on_grid = resample_to_grid(slow, uniform_grid(0.0, 20.0, 800.0))
    assert on_grid.rate_hz == pytest.approx(800.0)
    assert on_grid.native_rate_hz == pytest.approx(10.0)
    assert on_grid.native_nyquist_hz == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# What it protects
# --------------------------------------------------------------------------- #


def _starve(bundle: LogBundle, rate_hz: float) -> LogBundle:
    """The same bundle, claiming its rate messages were logged this slowly.

    The samples are left alone deliberately. What is being tested is that the
    stated provenance is *believed* -- a guard that only works when the data is
    also visibly degraded would not catch the real case, where interpolated data
    looks perfectly well behaved.
    """
    from dataclasses import replace

    signals = {
        key: (
            replace(signal, native_rate_hz=rate_hz)
            if key.startswith("rate.") or key.startswith("excite.")
            else signal
        )
        for key, signal in bundle.signals.items()
    }
    return replace(bundle, signals=signals)


def test_the_identification_band_stops_where_the_evidence_does() -> None:
    """A sweep that ran to 30 Hz is still only identified to the log's Nyquist."""
    config = load_config()
    bundle = make_bundle(make_airframe(), make_chain())
    full = identify_axis(bundle, "roll", config)

    starved = identify_axis(_starve(bundle, 16.0), "roll", config)
    assert evidence_ceiling_hz(_starve(bundle, 16.0), "rate.roll.measured") == pytest.approx(8.0)
    assert starved.deconvolved.valid_band_hz[1] <= 8.0
    assert full.deconvolved.valid_band_hz[1] > starved.deconvolved.valid_band_hz[1]


def test_a_log_with_no_band_left_refuses_rather_than_fitting_a_spline() -> None:
    bundle = _starve(make_bundle(make_airframe(), make_chain()), 0.4)
    with pytest.raises(ValueError, match="no band left to identify over"):
        identify_axis(bundle, "roll", load_config())


def test_noise_is_not_characterized_above_where_the_log_can_see() -> None:
    """Interpolating a slow message manufactures a forest of evenly spaced lines.

    On a real 10 Hz log this produced twenty-odd STRUCTURAL_RESONANCE findings
    at 245, 255, 275, 295 Hz and up -- every one of them an artefact of the
    spline, every one of them advising the user to go looking for a loose bolt.
    Below a ceiling that could contain a motor fundamental at all, the honest
    answer is no noise profile.
    """
    bundle = make_bundle(make_airframe(), make_chain(), with_motor_noise=True)
    with pytest.raises(ValueError, match="no noise spectrum here"):
        noise_profile(bundle, "roll", t_start=0.0, t_end=10.0, evidence_ceiling_hz=5.0)


def test_a_generous_ceiling_leaves_the_noise_profile_alone() -> None:
    bundle = make_bundle(make_airframe(), make_chain(), with_motor_noise=True)
    kwargs = {"t_start": 0.0, "t_end": bundle.signals["rate.roll.measured"].duration_s}
    unbounded = noise_profile(bundle, "roll", **kwargs)  # type: ignore[arg-type]
    capped = noise_profile(bundle, "roll", evidence_ceiling_hz=1e6, **kwargs)  # type: ignore[arg-type]
    assert capped.f_hz.size == unbounded.f_hz.size
    assert len(capped.peaks) == len(unbounded.peaks)


# --------------------------------------------------------------------------- #
# What it says
# --------------------------------------------------------------------------- #


def _findings(bundle: LogBundle) -> tuple[str, ...]:
    context = GuidanceContext(bundle=bundle, analyses={}, recommendations={}, config=load_config())
    return tuple(f.code for f in collect_findings(context))


def test_a_starved_log_is_named_as_such_even_when_nothing_could_be_analysed() -> None:
    """The case the user most needs it, and the one it is easiest to lose.

    A log logged at 10 Hz fails on all three axes at once, so there are no
    recommendations to hang per-axis findings off. Before this, the whole
    findings pass was skipped in that case and the user got three copies of "no
    usable excitation" -- true, unhelpful, and not the thing to change.
    """
    starved = _starve(make_bundle(make_airframe(), make_chain()), 10.0)
    assert "LOG_RATE_TOO_LOW" in _findings(starved)


def test_it_is_one_finding_and_not_one_per_axis() -> None:
    """Three axes share one logging schedule the user set once."""
    starved = _starve(make_bundle(make_airframe(), make_chain()), 10.0)
    codes = _findings(starved)
    assert codes.count("LOG_RATE_TOO_LOW") == 1


def test_a_properly_logged_flight_is_not_accused_of_this() -> None:
    assert "LOG_RATE_TOO_LOW" not in _findings(make_bundle(make_airframe(), make_chain()))


def test_the_action_names_the_parameter_that_fixes_it() -> None:
    starved = _starve(make_bundle(make_airframe(), make_chain()), 10.0)
    context = GuidanceContext(bundle=starved, analyses={}, recommendations={}, config=load_config())
    finding = next(f for f in collect_findings(context) if f.code == "LOG_RATE_TOO_LOW")
    assert "LOG_BITMASK" in finding.action
    assert "ATTITUDE_FAST" in finding.action
    assert finding.severity == "blocker"


def test_the_yardstick_is_not_the_crossover_the_starved_log_produced() -> None:
    """Otherwise the check exonerates exactly the logs it exists to catch.

    A starved log fits a sluggish airframe, which designs a low crossover, which
    makes the starved log look like it had plenty of bandwidth to spare. The
    comparison has to be against the crossover the vehicle's loop rate would
    have allowed, which is a property of the aircraft rather than of the fit.
    """
    starved = _starve(make_bundle(make_airframe(), make_chain()), 10.0)
    analysis = identify_axis(starved, "roll", load_config())
    from rotorid.core.design.recommend import recommend_from

    rec = recommend_from(analysis, starved, load_config())
    assert rec.margins.crossover_hz < 5.0, "the starved fit really is this slow"

    context = GuidanceContext(
        bundle=starved,
        analyses={"roll": analysis},
        recommendations={"roll": rec},
        config=load_config(),
    )
    assert any(f.code == "LOG_RATE_TOO_LOW" for f in collect_findings(context))
