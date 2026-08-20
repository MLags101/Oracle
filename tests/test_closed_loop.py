"""Identifying an aircraft from data it flew under its own control.

Every log RotorID reads is closed-loop data. The mixer command is the
controller's output, so it contains the gyro noise fed back through the
controller, and the ordinary ``Puy/Puu`` estimate of the plant is biased by
exactly that -- towards ``-1/C``, an estimate of the inverse controller wearing
the coherence of a good measurement.

The remedy is an instrument: an exogenous signal against which both the plant
input and the response are measured, giving ``Pry/Pru``. It is unbiased for any
``r`` uncorrelated with the noise, whatever the controller is and wherever ``r``
enters the loop, which is why the same estimator serves an injected chirp and a
pilot's stick.

None of this can be tested against :mod:`tests.synthetic.generators`, whose
bundles have no feedback in them at all and whose "injected chirp" is literally
the same array as the mixer command. Everything here runs on
:mod:`tests.synthetic.closed_loop` instead.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rotorid.config import load_config
from rotorid.core.analysis.instrument import choose_instrument, windowed_signals
from rotorid.core.design.recommend import identify_axis
from rotorid.core.preprocess.segment import propose_segments
from rotorid.core.types import LogBundle
from tests.synthetic.closed_loop import make_closed_loop_bundle

CONFIG = load_config()

#: The airframe every fixture here flies, plus the loop delay the tool cannot
#: separate from it. ``tau`` comes out as the airframe's 18 ms plus the 3.85 ms
#: of ZOH, compute and actuator lag, because the deconvolution removes the filter
#: chain and nothing else -- which is the documented contract.
TRUTH = {"K": 12.0, "wn": 2.0 * np.pi * 2.5, "zeta": 1.0, "tau": 0.018 + 0.00385}


def _blind(bundle: LogBundle) -> LogBundle:
    """The same flight with every exogenous signal removed.

    Forces the fall-back to the direct estimator, which is how the biased answer
    is obtained for comparison without reaching into the estimator itself.
    """
    return dataclasses.replace(
        bundle,
        signals={
            k: v
            for k, v in bundle.signals.items()
            if not k.endswith(".setpoint") and not k.startswith("excite.")
        },
    )


def _error(params: dict[str, float], key: str) -> float:
    """Fractional error against the known truth."""
    return abs(float(params[key]) - TRUTH[key]) / TRUTH[key]


# --------------------------------------------------------------------------- #
# The estimator recovers the aircraft
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("injection", ["rate", "mixer"])
def test_a_chirp_flown_in_the_loop_recovers_the_airframe(injection: str) -> None:
    """The case that was broken, at both injection points.

    Before the instrument-variable estimator, this path computed
    ``S_ey / S_ee`` -- the transfer from the injected waveform to the gyro, which
    with the waveform inside the loop is the *closed-loop* response and not the
    plant at all.
    """
    bundle = make_closed_loop_bundle(injection=injection)
    analysis = identify_axis(bundle, "roll", CONFIG)

    assert analysis.effective.estimator == "instrument_variable"
    assert analysis.effective.instrument == "excite.roll"
    params = analysis.airframe.params
    assert _error(params, "K") < 0.10
    assert _error(params, "wn") < 0.05
    assert _error(params, "zeta") < 0.10
    assert _error(params, "tau") < 0.15


def test_both_injection_points_identify_the_same_aircraft() -> None:
    """Where the waveform enters must not change what the aircraft is."""
    at_rate = identify_axis(make_closed_loop_bundle(injection="rate"), "roll", CONFIG)
    at_mixer = identify_axis(make_closed_loop_bundle(injection="mixer"), "roll", CONFIG)

    for key in ("K", "wn", "zeta"):
        a, b = float(at_rate.airframe.params[key]), float(at_mixer.airframe.params[key])
        assert abs(a - b) / max(a, b) < 0.05, key


def test_the_plant_input_assembly_is_load_bearing() -> None:
    """Mislabel where the chirp entered and the answer is nonsense.

    For ``SID_AXIS`` 10-12 the waveform is added after the rate controller, so
    ``RATE.ROut`` is only half the plant input and the other half is ``SIDD.Targ``.
    This test exists because that is an assumption about firmware rather than
    something the log states, and an assumption that could be wrong without
    anything noticing is not one worth making.
    """
    bundle = make_closed_loop_bundle(injection="mixer")
    honest = identify_axis(bundle, "roll", CONFIG)
    mislabelled = identify_axis(
        dataclasses.replace(bundle, params={**bundle.params, "SID_AXIS": 7.0}), "roll", CONFIG
    )

    assert _error(honest.airframe.params, "K") < 0.10
    assert _error(mislabelled.airframe.params, "K") > 1.0, (
        "reassembling the plant input wrongly must be visible, not absorbed by the fit"
    )


# --------------------------------------------------------------------------- #
# The direct estimator, and why it is not used
# --------------------------------------------------------------------------- #


def test_the_direct_estimator_is_badly_wrong_on_an_ordinary_flight() -> None:
    """The headline. Same flight, same data, two estimators, one right.

    On a pilot-flown log with a realistic gyro noise floor, ``Puy/Puu`` returns
    an airframe gain several times the truth -- and does it while reporting high
    coherence out to the top of the band, because the noise it is fitting is
    genuinely correlated between the two signals. A tool that used it would
    recommend gains several times wrong and say it was sure.
    """
    bundle = make_closed_loop_bundle(with_chirp=False, noise_rms=0.2)

    unbiased = identify_axis(bundle, "roll", CONFIG)
    biased = identify_axis(_blind(bundle), "roll", CONFIG)

    assert unbiased.effective.estimator == "instrument_variable"
    assert biased.effective.estimator == "direct_h1"
    assert biased.effective.instrument is None

    assert _error(unbiased.airframe.params, "K") < 0.25
    assert _error(biased.airframe.params, "K") > 1.0


def test_the_two_estimators_agree_when_the_chirp_drowns_the_noise() -> None:
    """Bias is a function of how much of the command is noise, not of feedback.

    With a strong injected sweep there is little of the controller's noise
    reaction left in the mixer command, so the direct estimate is nearly right and
    the reported disagreement is nearly zero. That is what makes the number a
    useful diagnostic rather than a constant.
    """
    analysis = identify_axis(make_closed_loop_bundle(noise_rms=0.02), "roll", CONFIG)
    assert abs(analysis.effective.bias_db) < 1.0
    assert abs(analysis.effective.bias_deg) < 5.0


def test_the_reported_bias_grows_with_the_noise_it_measures() -> None:
    quiet = identify_axis(make_closed_loop_bundle(with_chirp=False, noise_rms=0.05), "roll", CONFIG)
    loud = identify_axis(make_closed_loop_bundle(with_chirp=False, noise_rms=1.0), "roll", CONFIG)
    assert abs(loud.effective.bias_db) > abs(quiet.effective.bias_db)


# --------------------------------------------------------------------------- #
# What a pilot's stick can and cannot buy
# --------------------------------------------------------------------------- #


def test_a_pilot_stick_alone_identifies_the_aircraft() -> None:
    """No chirp, no SYSTEMID, nothing but a hand on the sticks."""
    bundle = make_closed_loop_bundle(with_chirp=False, noise_rms=0.1)
    analysis = identify_axis(bundle, "roll", CONFIG)

    assert analysis.effective.instrument == "att.roll.setpoint"
    assert _error(analysis.airframe.params, "K") < 0.25
    assert _error(analysis.airframe.params, "wn") < 0.30


def test_the_band_stops_where_the_stick_stopped() -> None:
    """The honest limit of a general log, and it has to be visible in the band.

    A pilot puts almost no energy above a couple of Hz, so there is nothing up
    there to identify from. Reporting a band that runs to the Nyquist would be
    claiming evidence the flight never produced.
    """
    analysis = identify_axis(
        make_closed_loop_bundle(with_chirp=False, noise_rms=0.1), "roll", CONFIG
    )
    top = analysis.airframe.valid_band_hz[1]
    assert top < 25.0, f"a stick-flown log cannot identify to {top:.0f} Hz"

    swept = identify_axis(make_closed_loop_bundle(noise_rms=0.1), "roll", CONFIG)
    assert swept.airframe.valid_band_hz[1] > top, "a sweep must buy more band than a stick"


def test_more_noise_narrows_the_band_rather_than_corrupting_the_fit() -> None:
    """Degrading gracefully means losing band, not gaining a confident error."""
    quiet = identify_axis(make_closed_loop_bundle(with_chirp=False, noise_rms=0.1), "roll", CONFIG)
    loud = identify_axis(make_closed_loop_bundle(with_chirp=False, noise_rms=0.5), "roll", CONFIG)
    assert loud.airframe.valid_band_hz[1] < quiet.airframe.valid_band_hz[1]


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #


def test_the_ladder_prefers_the_chirp_then_the_stick_then_the_rate_target() -> None:
    swept = make_closed_loop_bundle()
    assert choose_instrument(swept, "roll") == ("excite.roll", "injected_chirp")

    flown = make_closed_loop_bundle(with_chirp=False)
    assert choose_instrument(flown, "roll") == ("att.roll.setpoint", "attitude_setpoint")

    no_attitude = dataclasses.replace(
        flown, signals={k: v for k, v in flown.signals.items() if k != "att.roll.setpoint"}
    )
    assert choose_instrument(no_attitude, "roll") == ("rate.roll.setpoint", "rate_setpoint")

    assert choose_instrument(_blind(flown), "roll") == (None, "none")


def test_a_stick_that_never_moved_is_not_an_instrument() -> None:
    """Otherwise the estimate becomes one small number divided by another."""
    bundle = make_closed_loop_bundle(with_chirp=False)
    segment = next(s for s in propose_segments(bundle) if s.axis == "roll")

    still = dataclasses.replace(
        bundle,
        signals={
            **bundle.signals,
            "att.roll.setpoint": dataclasses.replace(
                bundle.signals["att.roll.setpoint"],
                y=np.zeros_like(bundle.signals["att.roll.setpoint"].y),
            ),
        },
    )
    cut = windowed_signals(
        still, "roll", segment, instrument_key="att.roll.setpoint", rung="attitude_setpoint"
    )
    assert cut.instrument is None
    assert cut.rung == "none"


def test_the_plant_input_is_required_and_saying_so_beats_guessing() -> None:
    bundle = make_closed_loop_bundle()
    segment = next(s for s in propose_segments(bundle) if s.axis == "roll")
    without = dataclasses.replace(
        bundle, signals={k: v for k, v in bundle.signals.items() if k != "rate.roll.output"}
    )
    with pytest.raises(ValueError, match="what drove the aircraft is unknown"):
        windowed_signals(
            without, "roll", segment, instrument_key="excite.roll", rung="injected_chirp"
        )


# --------------------------------------------------------------------------- #
# What the user is told
# --------------------------------------------------------------------------- #


def _findings(bundle: LogBundle) -> dict[str, object]:
    from rotorid import __version__
    from rotorid.core.pipeline import analyze

    result = analyze(bundle, ("roll",), CONFIG, tool_version=__version__)
    return {f.code: f for f in result.session.findings}


def test_a_log_with_no_independent_signal_is_blocked() -> None:
    """Not a weaker measurement of the aircraft -- a measurement of something else."""
    bundle = _blind(make_closed_loop_bundle(with_chirp=False, noise_rms=0.2))
    finding = _findings(bundle)["ESTIMATOR_BIASED"]

    assert finding.severity == "blocker"  # type: ignore[attr-defined]
    assert "LOG_BITMASK" in finding.action  # type: ignore[attr-defined]


def test_an_instrumented_log_is_not_accused_of_bias() -> None:
    bundle = make_closed_loop_bundle(with_chirp=False, noise_rms=0.2)
    assert "ESTIMATOR_BIASED" not in _findings(bundle)


def test_a_large_disagreement_is_reported_without_blocking() -> None:
    """The right answer was used; the size of the gap is still worth seeing."""
    bundle = make_closed_loop_bundle(with_chirp=False, noise_rms=1.0)
    codes = _findings(bundle)
    assert "ESTIMATOR_BIASED" not in codes
    finding = codes.get("ESTIMATOR_BIAS_LARGE")
    assert finding is not None, "a flight this noisy must say the naive reading differs"
    assert finding.severity == "warning"  # type: ignore[attr-defined]


def test_the_model_carries_how_the_loop_was_removed() -> None:
    """The report and the screen both read this off the model, not off the plant."""
    instrumented = identify_axis(make_closed_loop_bundle(), "roll", CONFIG).airframe
    assert instrumented.estimator == "instrument_variable"
    assert instrumented.instrument == "excite.roll"

    blind = identify_axis(_blind(make_closed_loop_bundle()), "roll", CONFIG).airframe
    assert blind.estimator == "direct_h1"
    assert blind.instrument is None
