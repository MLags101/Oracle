"""The step response the aircraft actually flew (spec section 5.3).

Everything else in the tool predicts. A model is fitted, a loop is closed around
it on paper, and a step is computed through it. That prediction is only as good as
the model, and the one number a pilot judges a tune by -- how it answers a stick
input -- is exactly where a wrong model is least obvious, because a wrong model
still produces a plausible-looking step.

This measures it instead, from the log, with no model in the path. The method is
Wiener deconvolution of the rate setpoint against the measured rate, which is the
technique the blackbox tools (PID-Analyzer, PIDtoolbox) use to pull a step
response out of arbitrary flight data:

.. math::

    H(f) = \\frac{Y(f)\\,\\overline{R(f)}}{|R(f)|^2 + \\lambda}

The regularizer :math:`\\lambda` is what makes it usable on flight data. A plain
division by ``R`` explodes wherever the pilot happened not to excite the aircraft,
which on an ordinary flight is most frequencies; adding :math:`\\lambda` trades a
little bias for not amplifying those bins by a factor of ten thousand.

The geometry -- a window of about 1.5 s, a response of about 0.5 s, a stride of
about 0.2 s, windows stacked and weighted by how much the stick moved in them --
is PID-Analyzer's, adopted rather than rederived. It works because a multirotor
rate loop settles in well under half a second, so a 1.5 s window contains the
whole response several times over and the stride gives many overlapping looks at
it.

**The regularizer biases rise time slow.** :math:`\\lambda` is a low-pass on the
recovered response, so even on clean data the rise time comes back around 15%
longer than the loop's actual one, and the overshoot a point or two shallower.
That is a property of the method, not of the aircraft, and anything comparing this
against a predicted step has to allow for it rather than report it as a
disagreement.

Three things this cannot do, all of which it says rather than hides:

* **It assumes the loop is linear and time-invariant over the window.** A window
  containing a saturating motor, or a mode change, or a gain change, describes
  something that is not one system. Windows whose recovered step does not settle
  near unity are rejected on exactly that basis.
* **It cannot see what the stick did not excite.** A window where the pilot held
  still deconvolves noise against noise and returns noise, so windows are gated
  on how far the setpoint actually moved.
* **It measures the closed loop, not the aircraft.** The result is the response
  of the tune that was flown. It is the right thing to compare a *predicted* step
  against, and the wrong thing to identify an airframe from.
"""

from __future__ import annotations

import numpy as np
from scipy.signal.windows import tukey

from rotorid.config import Config
from rotorid.core.analysis.step import step_metrics
from rotorid.core.types import Axis, FloatArray, LogBundle, MeasuredStep

__all__ = ["MeasuredStep", "measured_step"]

#: Taper fraction of the Tukey window applied to each slice. Windowing both
#: signals is not exactly right -- ``w * (h conv r)`` is not ``h conv (w * r)`` --
#: but the error lives in the first and last taper, and the response being
#: recovered is short compared with the window, so it stays out of the part that
#: is kept. A rectangular window would instead put a discontinuity at both ends of
#: every slice and smear it across the whole spectrum.
_TAPER = 0.25


def measured_step(
    bundle: LogBundle,
    axis: Axis,
    config: Config,
    *,
    windows: tuple[tuple[float, float], ...] | None = None,
) -> MeasuredStep | None:
    """Recover the flown step response on one axis.

    Args:
        windows: Restrict to these ``(t_start, t_end)`` spans, normally the
            identification segments. ``None`` uses the whole record, which is
            usually what you want here -- unlike identification, this benefits
            from every stick input in the flight, not only the ones that made a
            good frequency response.

    Returns:
        The stacked step, or ``None`` if the log lacks the signals or no window
        survived. ``None`` means "not measured", never "measured as flat".
    """
    setpoint_key = f"rate.{axis}.setpoint"
    measured_key = f"rate.{axis}.measured"
    if setpoint_key not in bundle.signals or measured_key not in bundle.signals:
        return None

    fs = bundle.sample_rate_hz
    r_all = bundle.signals[setpoint_key].y
    y_all = bundle.signals[measured_key].y
    t_all = bundle.signals[setpoint_key].t

    n_window = round(config.float_("deconv", "window_s") * fs)
    n_response = round(config.float_("deconv", "response_s") * fs)
    n_stride = max(1, round(config.float_("deconv", "stride_s") * fs))
    if n_response >= n_window or r_all.size < n_window:
        return None

    reference_swing = _swing(r_all)
    if reference_swing <= 0.0:
        return None
    min_swing = reference_swing * config.float_("deconv", "min_window_swing_frac")
    regularization = config.float_("deconv", "regularization")
    tolerance = config.float_("deconv", "max_steady_state_error")
    min_explained = config.float_("deconv", "min_explained")

    taper = tukey(n_window, _TAPER)
    steps: list[FloatArray] = []
    weights: list[float] = []
    explained: list[float] = []
    rejected = 0

    for start in range(0, r_all.size - n_window + 1, n_stride):
        stop = start + n_window
        if not _inside(t_all[start], t_all[stop - 1], windows):
            continue

        swing = _swing(r_all[start:stop])
        if swing < min_swing:
            rejected += 1
            continue

        step, fit = _one_window(
            r_all[start:stop] * taper, y_all[start:stop] * taper, regularization
        )
        step = step[:n_response]
        if fit < min_explained:
            # The recovered response does not account for what the aircraft did.
            # This is the gate that matters, and it is the only one that can see a
            # *bias*: scatter between windows measures precision, and averaging
            # more windows drives scatter down whether or not the answer is
            # moving towards the truth. How much of the measured rate the response
            # explains does not improve by averaging.
            rejected += 1
            continue
        if abs(_settled(step) - 1.0) > tolerance:
            # The recovered response does not end up where a step response has to
            # end up. Something in this window was not one linear time-invariant
            # system -- a limiter, a mode change, or simply too little excitation
            # for the regularizer to leave alone.
            rejected += 1
            continue

        steps.append(step)
        weights.append(swing)
        explained.append(fit)

    if len(steps) < round(config.float_("deconv", "min_windows")):
        return None

    stack = np.stack(steps, axis=0)
    weight = np.asarray(weights, dtype=np.float64)
    mean = np.asarray(np.average(stack, axis=0, weights=weight), dtype=np.float64)
    spread = np.asarray(np.std(stack, axis=0), dtype=np.float64)
    t = np.arange(mean.size, dtype=np.float64) / fs

    return MeasuredStep(
        t=t,
        y=mean,
        spread=spread,
        n_windows=len(steps),
        n_rejected=rejected,
        metrics=step_metrics(t, mean),
        scatter=float(np.mean(spread)) / max(abs(_settled(mean)), 1e-9),
        explained=float(np.median(explained)),
    )


def _one_window(r: FloatArray, y: FloatArray, regularization: float) -> tuple[FloatArray, float]:
    """Wiener-deconvolve one window, integrate to a step, and score the fit.

    Zero-padded to twice the window, because a DFT convolution is circular and
    without the pad the tail of the response wraps round onto its own beginning --
    which looks exactly like an aircraft that answers before it is asked.

    Returns:
        ``(step, explained)``, where ``explained`` is the fraction of the measured
        rate's variance that the recovered response accounts for. On the
        closed-loop simulator this tracks the error against the known truth almost
        exactly, and does so identically whether the excitation was a chirp or a
        pilot's stick -- which is what makes it usable as a threshold at all.
    """
    n = r.size * 2
    R = np.fft.rfft(r, n=n)
    Y = np.fft.rfft(y, n=n)

    power = np.abs(R) ** 2
    # Scaled off the mean rather than the peak: on flight data the peak sits at
    # DC or at whatever the pilot's favourite frequency was, and scaling by it
    # would set the floor by one bin's worth of accident.
    floor = regularization * float(np.mean(power))
    H = Y * np.conj(R) / (power + floor)
    impulse = np.fft.irfft(H, n=n)

    # What this response says the aircraft should have done, against what it did.
    # Compared over the original window rather than the padded one: the pad is
    # zeros on both sides and would flatter the fit by the ratio of the lengths.
    modelled = np.fft.irfft(R * H, n=n)[: r.size]
    variance = float(np.var(y))
    fit = 1.0 - float(np.var(y - modelled)) / variance if variance > 0.0 else 0.0

    # A step is the running sum of an impulse response. Not scaled by dt: the
    # DFT pair already carries the sample interval on both sides, so the sum is
    # dimensionless gain -- 1.0 for a loop that ends up tracking its command.
    return np.asarray(np.cumsum(impulse[: r.size]), dtype=np.float64), fit


def _settled(step: FloatArray) -> float:
    """Where the recovered step ended up, over its last tenth."""
    tail = step[max(1, int(0.9 * step.size)) :]
    return float(np.mean(tail)) if tail.size else 0.0


def _swing(values: FloatArray) -> float:
    """Peak-to-peak, which is what "did the stick move" means.

    Not the RMS: a setpoint held at a large constant offset during a turn has a
    large RMS and carries no information about the response at all.
    """
    return float(np.max(values) - np.min(values)) if values.size else 0.0


def _inside(t_start: float, t_end: float, windows: tuple[tuple[float, float], ...] | None) -> bool:
    """Whether a slice falls entirely within one of the requested spans."""
    if windows is None:
        return True
    return any(start <= t_start and t_end <= stop for start, stop in windows)
