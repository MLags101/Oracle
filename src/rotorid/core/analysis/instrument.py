"""Choosing what to identify against, and assembling the plant input.

Flight data is closed-loop data. The mixer command is the controller's own
output, so it contains the gyro noise fed back through the controller, and the
ordinary ``Puy/Puu`` estimate of the plant is biased by exactly that -- towards
``-1/C``, an estimate of the inverse controller wearing the coherence of a good
measurement. The remedy is an *instrument*: an exogenous signal, uncorrelated
with the gyro noise, against which both the plant input and the response are
measured. See :class:`~rotorid.core.analysis.spectra.InstrumentedEstimate` for
why the ratio of those two is the plant.

This module answers the two questions that has to be answered per log:

* **Which signal is the instrument?** A ladder, best first, because logs differ
  in what they were asked to record and a general flight has no injected chirp.
* **What exactly is the plant input?** Not always the logged mixer command.
  ArduPilot's SYSTEMID adds its waveform *after* the rate controller for
  ``SID_AXIS`` 10-12, so on those flights the aircraft was driven by the sum and
  the logged ``RATE.ROut`` is only half of it.

Both answers are recorded on the result rather than inferred later. A tool that
cannot say which signal it identified against cannot say how much to trust the
answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rotorid.core.types import Axis, ExcitationSegment, FloatArray, LogBundle

__all__ = ["Instrumented", "Rung", "choose_instrument", "windowed_signals"]

#: Which rung of the ladder an instrument came from, best first. ``"none"`` means
#: the log offered nothing exogenous and the estimate will be the biased one.
Rung = Literal["injected_chirp", "attitude_setpoint", "rate_setpoint", "none"]

#: Plain-language name per rung, for findings and for the screen.
RUNG_NAMES: dict[Rung, str] = {
    "injected_chirp": "the injected SYSTEMID chirp",
    "attitude_setpoint": "the pilot's commanded lean angle",
    "rate_setpoint": "the rate setpoint",
    "none": "nothing exogenous",
}

#: A signal whose RMS over the window is below this fraction of the response's
#: cannot instrument anything: it did not move, so nothing can be attributed to
#: it, and the ratio it would produce is two small numbers dividing each other.
_MIN_INSTRUMENT_RMS_RATIO = 1e-3


@dataclass(frozen=True, slots=True)
class Instrumented:
    """One segment's three signals, windowed and ready to estimate from.

    Attributes:
        instrument: The exogenous signal, or ``None`` when the ladder ran out and
            the caller must fall back to the biased direct estimate.
        plant_input: What actually drove the aircraft -- see
            :func:`windowed_signals` for when that is not the logged command.
        summed_injection: ``True`` when the chirp had to be added to the logged
            mixer command to reconstruct the plant input. Worth surfacing: it is
            an assumption about firmware, and the estimator-agreement check is
            what tests it.
    """

    instrument: FloatArray | None
    plant_input: FloatArray
    response: FloatArray
    instrument_key: str | None
    input_key: str
    output_key: str
    rung: Rung
    summed_injection: bool = False


def choose_instrument(bundle: LogBundle, axis: Axis) -> tuple[str | None, Rung]:
    """Pick the best exogenous signal this log offers for one axis.

    The ladder, and why each rung sits where it does:

    1. ``excite.{axis}`` -- the injected chirp. Exogenous by construction: the
       firmware generated it, nothing in the aircraft influenced it.
    2. ``att.{axis}.setpoint`` -- the pilot's commanded lean angle. In Stabilize
       and AltHold this is a pure function of stick position, so it is exogenous
       to the rate loop. **This is the rung that makes an ordinary flight
       identifiable at all.**
    3. ``rate.{axis}.setpoint`` -- weaker, and only used when the attitude
       setpoint is missing. In Stabilize it is ``ATC_ANG_*_P`` times the attitude
       error, so gyro noise reaches it the long way round through the outer loop.
       It removes most of the bias rather than all of it.
    4. Nothing. The caller falls back to the direct estimate and says so.

    Returns:
        ``(canonical key, rung)``. The key is ``None`` only for ``"none"``.
    """
    ladder: tuple[tuple[str, Rung], ...] = (
        (f"excite.{axis}", "injected_chirp"),
        (f"att.{axis}.setpoint", "attitude_setpoint"),
        (f"rate.{axis}.setpoint", "rate_setpoint"),
    )
    for key, rung in ladder:
        if key in bundle.signals:
            return key, rung
    return None, "none"


def windowed_signals(
    bundle: LogBundle,
    axis: Axis,
    segment: ExcitationSegment,
    *,
    instrument_key: str | None,
    rung: Rung,
) -> Instrumented:
    """Cut one segment out of the log and assemble the plant input.

    The plant input is the logged rate-controller output, *except* when the
    SYSTEMID waveform was injected downstream of that controller. ArduPilot's
    ``SID_AXIS`` 10-12 add the sample to the actuator command after the rate
    controller has run, and ArduPilot's own documentation states the plant input
    is "the sum of the sweep and the rate controller output". So on those flights
    the aircraft was driven by ``RATE.ROut + SIDD.Targ`` and identifying against
    ``RATE.ROut`` alone would be identifying against half the input.

    For ``SID_AXIS`` 7-9 the waveform is added to the rate *target*, upstream of
    the controller, so its effect is already inside ``RATE.ROut`` and the logged
    command is the whole plant input.

    An instrument that did not move over this window is discarded and the rung
    demoted -- a stick held still cannot instrument anything, and pretending
    otherwise turns the estimate into one small number divided by another.

    Raises:
        ValueError: if the rate measurement or the mixer command is missing. Both
            are required now: the mixer command is the plant input, and there is
            no substitute for it.
    """
    output_key = f"rate.{axis}.measured"
    input_key = f"rate.{axis}.output"
    if output_key not in bundle.signals:
        raise ValueError(f"{output_key} is not in the log; nothing to identify against")
    if input_key not in bundle.signals:
        raise ValueError(
            f"{input_key} is not in the log, so what drove the aircraft is unknown. "
            "The rate-controller output is the plant input; without it the response "
            "cannot be attributed to anything."
        )

    y_sig = bundle.signal(output_key)
    u_sig = bundle.signal(input_key)
    window = (y_sig.t >= segment.t_start) & (y_sig.t <= segment.t_end)
    response = np.asarray(y_sig.y[window], dtype=np.float64)
    plant_input = np.asarray(u_sig.y[window], dtype=np.float64)

    excite_key = f"excite.{axis}"
    summed = segment.injection_point == "mixer" and excite_key in bundle.signals
    if summed:
        plant_input = plant_input + np.asarray(
            bundle.signal(excite_key).y[window], dtype=np.float64
        )

    instrument: FloatArray | None = None
    if instrument_key is not None and instrument_key in bundle.signals:
        candidate = np.asarray(bundle.signal(instrument_key).y[window], dtype=np.float64)
        if _moved(candidate, response):
            instrument = candidate
        else:
            instrument_key, rung = None, "none"
    elif instrument_key is not None:
        instrument_key, rung = None, "none"

    return Instrumented(
        instrument=instrument,
        plant_input=plant_input,
        response=response,
        instrument_key=instrument_key,
        input_key=input_key,
        output_key=output_key,
        rung=rung,
        summed_injection=summed,
    )


def _moved(candidate: FloatArray, response: FloatArray) -> bool:
    """Whether a candidate instrument carries enough signal to be one."""
    if candidate.size == 0:
        return False
    scale = float(np.std(response))
    if scale <= 0.0:
        return False
    return float(np.std(candidate)) > _MIN_INSTRUMENT_RMS_RATIO * scale
