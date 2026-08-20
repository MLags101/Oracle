"""Predicted closed-loop step response (spec section 5.8).

The step is computed through the **reference** path, which is not the same as the
loop used for margins. On ArduPilot the target filter and the feed-forward act
only here; on PX4 the derivative term does not act here at all. Predicting the
step from the feedback loop -- the obvious shortcut -- gets both stacks wrong in
opposite directions, and gets them wrong in the one plot users judge a tune by.

Evaluation is by FFT rather than by a state-space simulation, so the exact
``exp(-tau*s)`` delay and the exact discrete filter responses carry through
without a rational approximation anywhere.
"""

from __future__ import annotations

import numpy as np

from rotorid.core.analysis.margins import LoopDelay, plant_path
from rotorid.core.analysis.model_eval import airframe_response
from rotorid.core.design.controller import RateController
from rotorid.core.filters.chain import OperatingPoint
from rotorid.core.types import AirframeModel, FloatArray, StepMetrics

__all__ = ["step_metrics", "step_response"]

#: Settling band, as a fraction of the final value. 2% is the usual convention.
_SETTLING_BAND = 0.02


def step_response(
    controller: RateController,
    airframe: AirframeModel,
    *,
    delay: LoopDelay,
    op: OperatingPoint | None = None,
    duration_s: float = 2.0,
    sample_rate_hz: float = 1000.0,
) -> tuple[FloatArray, FloatArray]:
    """Unit rate-step response of the closed loop.

    ``T(s) = C_ref(s) * G(s) / (1 + C_fb(s) * G(s))`` where ``G`` is the sensor
    path, airframe and transport delay together.

    Returns:
        ``(t, y)`` in seconds and rad/s, for a 1 rad/s commanded step.
    """
    n_keep = round(duration_s * sample_rate_hz)
    t = np.arange(n_keep, dtype=np.float64) / sample_rate_hz

    # Evaluated over four times the span that is kept. An inverse FFT is a
    # *circular* inverse, so whatever of the impulse response has not decayed by
    # the end of the record reappears at the start of it -- which reads as an
    # aircraft that begins responding before it is commanded. Padding pushes the
    # wrap-around out past the part anyone looks at.
    n = 4 * n_keep
    f = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    # The integrator and any 1/s term are infinite at f = 0. Evaluate just below
    # the first real bin and set the DC value explicitly afterwards, rather than
    # letting an inf propagate through the division and come back as a NaN.
    f_eval = f.copy()
    f_eval[0] = f[1] * 1e-3

    # The forward path from the mixer command, without the sensor filters: the
    # gyro filters are in the feedback path, not between the controller and the
    # airframe, so the commanded response is not shaped by them.
    forward = airframe_response(airframe, f_eval) * delay.response(f_eval)
    L = controller.feedback_response(f_eval) * plant_path(
        f_eval, controller, airframe, delay=delay, op=op
    )
    T = controller.reference_response(f_eval) * forward / (1.0 + L)

    if controller.gains.ki > 0.0:
        # An integrator has infinite gain at DC, so the closed loop tracks a
        # constant command exactly. The finite-arithmetic evaluation cannot say
        # that -- 1/s is simply dropped at f = 0 -- and leaving it would invent a
        # steady-state error that the real controller does not have.
        T[0] = 1.0

    # The step is the running sum of the impulse response. Not a multiplication
    # by the transform of a constant: the DFT of a constant sequence is a single
    # spike at DC, so that produces T(0) at every sample and nothing else -- a
    # flat line, for any controller and any aircraft.
    impulse = np.fft.irfft(T, n=n)
    y = np.cumsum(impulse)[:n_keep]
    return t, np.asarray(y, dtype=np.float64)


def step_metrics(t: FloatArray, y: FloatArray) -> StepMetrics:
    """Rise, overshoot, settling and steady-state error of a step response.

    The final value is taken from the last tenth of the record rather than from
    the DC gain of the model, so a response that has not settled reports a
    settling time it genuinely failed to achieve instead of one computed against
    a value it never reached.
    """
    tail = y[int(0.9 * y.size) :]
    final = float(np.mean(tail))
    if abs(final) < 1e-9:
        return StepMetrics(
            rise_time_s=float("nan"),
            overshoot_pct=float("nan"),
            settling_time_s=float("nan"),
            peak_time_s=float("nan"),
            steady_state_error=1.0,
        )

    normalized = y / final
    rise = _crossing(t, normalized, 0.9) - _crossing(t, normalized, 0.1)
    peak_index = int(np.argmax(np.abs(normalized)))
    overshoot = max(0.0, float(normalized[peak_index]) - 1.0) * 100.0

    outside = np.nonzero(np.abs(normalized - 1.0) > _SETTLING_BAND)[0]
    settling = float(t[outside[-1]]) if outside.size and outside[-1] + 1 < t.size else 0.0

    return StepMetrics(
        rise_time_s=rise,
        overshoot_pct=overshoot,
        settling_time_s=settling,
        peak_time_s=float(t[peak_index]),
        steady_state_error=1.0 - final,
    )


def _crossing(t: FloatArray, y: FloatArray, level: float) -> float:
    """First time ``y`` reaches ``level``, interpolated, or NaN if it never does."""
    reached = np.nonzero(y >= level)[0]
    if reached.size == 0:
        return float("nan")
    i = int(reached[0])
    if i == 0:
        return float(t[0])
    y0, y1 = y[i - 1], y[i]
    if y1 == y0:
        return float(t[i])
    return float(t[i - 1] + (level - y0) * (t[i] - t[i - 1]) / (y1 - y0))
