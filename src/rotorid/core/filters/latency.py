"""Where the phase went: an itemized latency budget.

The single most instructive number the tool produces is not a gain -- it is "you
have 62 degrees of lag at 20 Hz, and 40 of them are your D-term filter". This
module builds that breakdown.

Two rules keep it honest:

* **Nothing is counted twice.** The filter terms come from the modeled chain; the
  airframe term is the residual delay left in the identified model *after* that
  chain was divided out (spec section 0, rule 6). If you find yourself adding the
  gyro filter delay to ``tau``, something upstream already went wrong.
* **The budget is a diagnostic, not the loop phase.** The D-term filter sits in the
  derivative branch only, so the arithmetic sum of every item is not the phase of
  ``L(jw)``. Margins always come from evaluating the full loop; the budget explains
  the result rather than computing it. :attr:`LatencyBudget.common_path_deg` gives
  the subset that is genuinely in series with everything.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from rotorid.core.filters.biquad import lpf2p_biquad, phase_lag_deg
from rotorid.core.filters.chain import FilterChain, OperatingPoint
from rotorid.core.types import LatencyBudget

__all__ = ["actuator_latency_ms", "build_budget", "delay_phase_deg"]

#: ArduPilot ``MOT_PWM_TYPE`` values, mapped to the latency classes in
#: ``rotorid.toml`` under ``[design.actuator_latency_ms]``.
_AP_PWM_TYPE_CLASS: Final[dict[int, str]] = {
    0: "pwm",  # Normal PWM
    1: "oneshot",
    2: "oneshot",  # OneShot125
    3: "pwm",  # Brushed
    4: "dshot",  # DShot150
    5: "dshot",  # DShot300
    6: "dshot",  # DShot600
    7: "dshot",  # DShot1200
}


def delay_phase_deg(delay_s: float, f_hz: float) -> float:
    """Phase lag in degrees contributed by a pure transport delay.

    ``exp(-s*T)`` costs ``360 * f * T`` degrees -- linear in frequency, which is why
    delay dominates the budget at high crossover and is invisible at low.
    """
    return 360.0 * f_hz * delay_s


def actuator_latency_ms(params: dict[str, float], table: dict[str, float]) -> float:
    """Transport latency of the ESC/motor command path, in milliseconds.

    This covers command transport only. Motor spin-up dynamics are identified as
    part of the airframe model, so counting them here as well would double up.

    Args:
        params: Vehicle parameter snapshot. ``MOT_PWM_TYPE`` selects the class.
        table: The ``[design.actuator_latency_ms]`` mapping from config.

    Returns:
        Latency in milliseconds, falling back to the ``unknown`` entry.
    """
    raw = params.get("MOT_PWM_TYPE")
    key = _AP_PWM_TYPE_CLASS.get(int(raw), "unknown") if raw is not None else "unknown"
    return float(table.get(key, table.get("unknown", 0.0)))


def build_budget(
    f_hz: float,
    *,
    chain: FilterChain,
    airframe_tau_s: float,
    actuator_ms: float,
    zoh_loops: float,
    compute_loops: float,
    op: OperatingPoint | None = None,
) -> LatencyBudget:
    """Assemble the phase-lag breakdown at one frequency.

    Args:
        f_hz: Frequency to evaluate at -- normally the design crossover.
        chain: The filter chain, current or candidate.
        airframe_tau_s: Residual delay from the identified airframe model, with the
            filter chain already divided out.
        actuator_ms: ESC/motor transport latency, from :func:`actuator_latency_ms`.
        zoh_loops: Zero-order-hold delay in loop periods (0.5 is exact for a ZOH).
        compute_loops: Controller compute delay in loop periods.
        op: Operating point, for tracked notch centres.

    Returns:
        The itemized budget.

    Raises:
        ValueError: if ``f_hz`` is not positive.
    """
    if f_hz <= 0.0:
        raise ValueError(f"latency budget needs a positive frequency, got {f_hz}")

    freqs = np.array([f_hz], dtype=np.float64)
    loop_period = 1.0 / chain.loop_rate_hz

    gyro_lpf_deg = 0.0
    if chain.gyro_lpf_hz:
        stage = lpf2p_biquad(chain.gyro_lpf_hz, chain.sample_rate_hz)
        gyro_lpf_deg = float(phase_lag_deg(stage.response(freqs))[0])

    # Notch contribution is the whole sensor path minus the gyro low-pass, so the
    # two terms partition the sensor response exactly rather than overlapping.
    sensor_deg = float(chain.phase_deg(freqs, op)[0])
    notches_deg = sensor_deg - gyro_lpf_deg

    return LatencyBudget(
        at_hz=f_hz,
        gyro_lpf_deg=gyro_lpf_deg,
        notches_deg=notches_deg,
        dterm_lpf_deg=float(phase_lag_deg(chain.dterm_lpf_response(freqs))[0]) + 0.0,
        error_lpf_deg=float(phase_lag_deg(chain.error_lpf_response(freqs))[0]) + 0.0,
        zoh_deg=delay_phase_deg(zoh_loops * loop_period, f_hz),
        compute_deg=delay_phase_deg(compute_loops * loop_period, f_hz),
        actuator_deg=delay_phase_deg(actuator_ms / 1000.0, f_hz),
        airframe_tau_deg=delay_phase_deg(airframe_tau_s, f_hz),
    )
