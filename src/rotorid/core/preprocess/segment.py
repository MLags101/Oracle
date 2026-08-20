"""Finding the parts of a flight worth identifying from (spec section 5.2).

The tool is chirp-first. A deliberate frequency sweep gives a clean, wide-band,
single-axis excitation with a known schedule, and everything downstream is much
better conditioned because of it. Ordinary flight is supported as a fallback, but
never silently: a segment carries the confidence it earned, and a weak one caps
the confidence of the recommendation built on it.

Which of the two is searched for is the log's declared kind
(:mod:`rotorid.core.logkind`) rather than whichever happens to be found. Silently
falling back from a sweep to stick inputs is how a user ends up reading a number
that came from evidence they would have rejected: the label on the screen still
says the flight they flew, and the model underneath came from somewhere else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from scipy.signal import butter, sosfiltfilt

from rotorid.core.types import (
    AXES,
    Axis,
    BoolArray,
    ExcitationSegment,
    FloatArray,
    LogBundle,
    LogKind,
    SegmentKind,
)

__all__ = ["propose_segments"]

#: Confidence by excitation kind. A commanded sweep is worth several times what
#: an energetic bit of stick input is, and the difference has to survive into the
#: final recommendation rather than being averaged away.
_CONFIDENCE = {
    "systemid_chirp": 1.0,
    "px4_autotune": 0.8,
    "autotune_twitch": 0.5,
    "pilot_input": 0.3,
}

#: An excitation is only usable if the *other* two axes are comparatively quiet;
#: otherwise the single-axis assumption behind the identification is violated.
_CROSS_AXIS_QUIET_RATIO = 0.4

#: Fallback detection: high-pass corner, and what counts as "excited".
#:
#: The threshold is a fraction of the axis's own *peak* rather than a multiple of
#: its median, and that choice matters more than it looks. A deliberate slow-to-
#: fast sweep -- exactly the thing worth identifying from -- has a nearly flat
#: envelope, so it never rises to several times its own median and a
#: median-relative rule cannot see it at all. It can only find bursts, which is
#: the one kind of excitation that identifies badly.
_ENERGY_HIGHPASS_HZ = 0.3
_EXCITED_FRACTION_OF_PEAK = 0.3

#: Below this peak envelope, in normalized mixer output, there is not enough
#: signal to identify anything and the "excitation" is the controller reacting to
#: air. 1% of full output is already a very small stick input.
_ENERGY_MIN_AMPLITUDE = 0.01

#: Shorter than this and there is no low-frequency information in the window.
_MIN_SEGMENT_S = 5.0


def propose_segments(
    bundle: LogBundle, kind: LogKind | None = None
) -> tuple[ExcitationSegment, ...]:
    """Auto-propose identification windows, best evidence first.

    Args:
        kind: Which class of excitation to look for. Defaults to the log's own
            :attr:`~rotorid.core.types.LogBundle.kind`. A tuning flight is
            identified from deliberate excitation only; a general flight from
            ordinary stick activity only. Neither falls back to the other,
            because the fallback is invisible in every downstream number.

    Returns:
        Segments in descending confidence order. Empty if nothing of the
        requested class is present -- which is a finding for the caller to
        report, not something to paper over with a shorter window.
    """
    if (kind if kind is not None else bundle.kind) == "tuning":
        return _deliberate_segments(bundle)
    return _energy_segments(bundle)


def _deliberate_segments(bundle: LogBundle) -> tuple[ExcitationSegment, ...]:
    """Excitation somebody asked for: an injected sweep, or an autotune run."""
    chirps = _systemid_segments(bundle)
    if chirps:
        return chirps
    return _autotune_segments(bundle)


def _systemid_segments(bundle: LogBundle) -> tuple[ExcitationSegment, ...]:
    """Windows where a SYSTEMID chirp was actually being injected.

    Bounded by the injected signal itself rather than by the configured times.
    ``SID_T_REC`` says what was asked for; the recorded chirp says what happened,
    and a sweep aborted by a mode change or a low-battery failsafe differs.
    """
    out: list[ExcitationSegment] = []
    for axis in AXES:
        key = f"excite.{axis}"
        if key not in bundle.signals:
            continue
        signal = bundle.signals[key]
        active = np.abs(signal.y) > 1e-6
        for start, end in _runs(active, signal.t):
            if end - start < _MIN_SEGMENT_S:
                continue
            out.append(
                ExcitationSegment(
                    axis=axis,
                    t_start=start,
                    t_end=end,
                    kind="systemid_chirp",
                    amplitude_estimate=float(
                        np.max(np.abs(signal.y[(signal.t >= start) & (signal.t <= end)]))
                    ),
                    confidence=_CONFIDENCE["systemid_chirp"],
                    injection_point=_injection_point(bundle),
                    f_start_hz=bundle.param("SID_F_START_HZ"),
                    f_stop_hz=bundle.param("SID_F_STOP_HZ"),
                )
            )
    return tuple(out)


def _injection_point(bundle: LogBundle) -> str | None:
    """Where the chirp entered the loop, decoded from ``SID_AXIS``.

    It matters which: injecting at the mixer measures the plant directly, while
    injecting at the rate-controller input measures it through the controller, so
    the two need different reference signals.
    """
    from rotorid.core.io.ardupilot import SID_AXIS_MAP

    code = bundle.param("SID_AXIS")
    if code is None:
        return None
    mapped = SID_AXIS_MAP.get(int(code))
    return mapped[1] if mapped else None


def _autotune_segments(bundle: LogBundle) -> tuple[ExcitationSegment, ...]:
    """Twitches from the firmware's own autotune, if it ran.

    The window comes from the vehicle (``mode.autotune``); the axis comes from the
    data. Both stacks say *that* autotune was running far more reliably than they
    say which axis it was working on -- ArduPilot's per-axis progress is prose in
    the message log, PX4's is an internal state enum that has been renumbered --
    so the axis is decided by which one was actually being moved, which is a fact
    about the flight rather than about the firmware version.

    Worth less than a chirp and more than a stick input: an autotune twitch is a
    deliberate, single-axis, repeatable excitation, but it is a step rather than a
    sweep, so it excites a band nobody chose.
    """
    gate = bundle.signals.get("mode.autotune")
    if gate is None or not gate.y.size:
        return ()
    windows = _runs(gate.y > 0.5, gate.t)
    if not windows:
        return ()
    kind: SegmentKind = "px4_autotune" if bundle.stack == "px4" else "autotune_twitch"
    out = [
        replace(segment, kind=kind, confidence=_CONFIDENCE[kind])
        for segment in _energy_segments(bundle, windows=windows)
    ]
    return tuple(out)


def _energy_segments(
    bundle: LogBundle, windows: Sequence[tuple[float, float]] | None = None
) -> tuple[ExcitationSegment, ...]:
    """Fallback: stretches of ordinary flight with strong single-axis activity.

    Always low confidence. Pilot input is narrow-band, correlated across axes, and
    mixed with the controller's own response to disturbances, so an airframe
    identified this way is a rough estimate wearing the same clothes as a good
    one. The confidence value is what keeps them distinguishable downstream.
    """
    rate = bundle.sample_rate_hz
    envelopes: dict[Axis, FloatArray] = {}
    for axis in AXES:
        key = f"rate.{axis}.output"
        if key not in bundle.signals:
            return ()
        envelopes[axis] = _envelope(bundle.signals[key].y, rate)

    t = bundle.signals[f"rate.{AXES[0]}.output"].t
    # Confining the search also confines what "peak" means, which is the point:
    # the threshold is a fraction of the largest excitation *in the window*, so a
    # gentle autotune twitch is not measured against a violent stick input that
    # happened somewhere else in the flight.
    inside = _mask_for(t, windows)
    out: list[ExcitationSegment] = []
    for axis in AXES:
        others = np.maximum.reduce([envelopes[a] for a in AXES if a != axis])
        peak = float(np.max(envelopes[axis][inside])) if inside.any() else 0.0
        if peak < _ENERGY_MIN_AMPLITUDE:
            continue
        threshold = _EXCITED_FRACTION_OF_PEAK * peak
        excited = (
            inside
            & (envelopes[axis] > threshold)
            & (others < envelopes[axis] * _CROSS_AXIS_QUIET_RATIO)
        )
        for start, end in _runs(excited, t):
            if end - start < _MIN_SEGMENT_S:
                continue
            window = (t >= start) & (t <= end)
            out.append(
                ExcitationSegment(
                    axis=axis,
                    t_start=start,
                    t_end=end,
                    kind="pilot_input",
                    amplitude_estimate=float(np.max(np.abs(envelopes[axis][window]))),
                    confidence=_CONFIDENCE["pilot_input"],
                )
            )
    return tuple(sorted(out, key=lambda s: -s.duration_s))


def _mask_for(t: FloatArray, windows: Sequence[tuple[float, float]] | None) -> BoolArray:
    """Samples inside any of ``windows``; everything, when there are none."""
    if windows is None:
        return np.ones(t.shape, dtype=np.bool_)
    mask = np.zeros(t.shape, dtype=np.bool_)
    for start, end in windows:
        mask |= (t >= start) & (t <= end)
    return mask


def _envelope(y: FloatArray, sample_rate_hz: float) -> FloatArray:
    """Smoothed magnitude of the high-passed signal.

    High-passing first is what separates excitation from trim: a vehicle held in a
    steady banked turn has a large mean output and no information in it at all.
    """
    nyquist = 0.5 * sample_rate_hz
    corner = min(_ENERGY_HIGHPASS_HZ / nyquist, 0.99)
    sos = butter(2, corner, btype="highpass", output="sos")
    filtered = np.abs(sosfiltfilt(sos, y))
    window = max(1, round(sample_rate_hz))
    kernel = np.ones(window) / window
    return np.asarray(np.convolve(filtered, kernel, mode="same"), dtype=np.float64)


def _runs(mask: BoolArray, t: FloatArray) -> list[tuple[float, float]]:
    """Contiguous ``True`` runs of ``mask``, as ``(t_start, t_end)`` pairs."""
    if not mask.any():
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.nonzero(edges == 1)[0] + 1)
    ends = list(np.nonzero(edges == -1)[0])
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size - 1)
    return [(float(t[a]), float(t[b])) for a, b in zip(starts, ends, strict=True)]
