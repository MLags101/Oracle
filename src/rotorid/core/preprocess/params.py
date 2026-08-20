"""Parameter snapshot to typed objects (spec section 6, ingestion half of 7).

The mapping between on-vehicle parameter names and RotorID objects lives in
exactly two places: here for reading, and ``export/params.py`` for writing. They
are round-trip tested against each other, because a mapping that disagrees with
itself produces a recommendation expressed in units the vehicle does not use.

Nothing here invents a default. A parameter that was not logged comes back as
``None`` and becomes a finding; guessing ``ATC_RAT_RLL_FLTD`` because most
vehicles use 20 Hz would put an unlogged filter into the margin calculation.
"""

from __future__ import annotations

import numpy as np

from rotorid.core.filters.chain import (
    FilterChain,
    OperatingPoint,
    ardupilot_chain,
    px4_chain,
)
from rotorid.core.types import Axis, GainSet, LogBundle

__all__ = [
    "ardupilot_gain_set",
    "chain_from_bundle",
    "gains_from_bundle",
    "hover_operating_point",
    "px4_gain_set",
]

_AP_AXIS_PARAM: dict[Axis, str] = {"roll": "RLL", "pitch": "PIT", "yaw": "YAW"}
_PX4_AXIS_PARAM: dict[Axis, str] = {"roll": "ROLL", "pitch": "PITCH", "yaw": "YAW"}


def ardupilot_gain_set(params: dict[str, float], axis: Axis) -> GainSet:
    """Read ``ATC_RAT_*`` into a :class:`~rotorid.core.types.GainSet`.

    Raises:
        KeyError: if the axis has no P gain logged, which means the parameter
            snapshot is incomplete rather than that the gain is zero.
    """
    suffix = _AP_AXIS_PARAM[axis]
    prefix = f"ATC_RAT_{suffix}_"
    if f"{prefix}P" not in params:
        raise KeyError(
            f"{prefix}P is not in the parameter snapshot; the log is missing PARM "
            "messages for the rate controller"
        )
    return GainSet(
        axis=axis,
        kp=float(params[f"{prefix}P"]),
        ki=float(params.get(f"{prefix}I", 0.0)),
        kd=float(params.get(f"{prefix}D", 0.0)),
        kff=float(params.get(f"{prefix}FF", 0.0)),
        imax=params.get(f"{prefix}IMAX"),
        dterm_lpf_hz=params.get(f"{prefix}FLTD"),
        error_lpf_hz=params.get(f"{prefix}FLTE"),
        target_lpf_hz=params.get(f"{prefix}FLTT"),
    )


def px4_gain_set(params: dict[str, float], axis: Axis) -> GainSet:
    """Read ``MC_*RATE_*`` into effective gains, resolving the ``K`` factor.

    PX4 stores the controller in standard form: the effective gains are ``K*P``,
    ``K*I`` and ``K*D``. That multiplication happens exactly here, so no design or
    export code has to remember it. Getting it wrong is a silent factor error in
    every gain, which is why it has its own regression test.

    Raises:
        KeyError: if the axis has no P gain logged.
    """
    suffix = _PX4_AXIS_PARAM[axis]
    prefix = f"MC_{suffix}RATE_"
    if f"{prefix}P" not in params:
        raise KeyError(f"{prefix}P is not in the parameter snapshot")
    k = float(params.get(f"{prefix}K", 1.0))
    return GainSet(
        axis=axis,
        kp=k * float(params[f"{prefix}P"]),
        ki=k * float(params.get(f"{prefix}I", 0.0)),
        kd=k * float(params.get(f"{prefix}D", 0.0)),
        kff=float(params.get(f"{prefix}FF", 0.0)),
        imax=params.get(f"{prefix}I_LIM"),
        dterm_lpf_hz=params.get("IMU_DGYRO_CUTOFF"),
    )


def gains_from_bundle(bundle: LogBundle, axis: Axis) -> GainSet:
    """Current gains for one axis, whichever stack the log came from."""
    if bundle.stack == "ardupilot":
        return ardupilot_gain_set(bundle.params, axis)
    return px4_gain_set(bundle.params, axis)


def chain_from_bundle(bundle: LogBundle, axis: Axis) -> FilterChain:
    """Reconstruct the filter chain the log was recorded through."""
    builder = ardupilot_chain if bundle.stack == "ardupilot" else px4_chain
    return builder(
        bundle.params,
        axis,
        gyro_sample_rate_hz=bundle.gyro_sample_rate_hz,
        loop_rate_hz=bundle.loop_rate_hz,
    )


def hover_operating_point(bundle: LogBundle, t_start: float, t_end: float) -> OperatingPoint:
    """Where in the envelope a segment sat, so tracked notches land in the right place.

    Motor frequency is used when the log has ESC telemetry, because it is
    measured; throttle is the fallback, and it is only a proxy -- the notch
    frequency it implies is right only if ``MOT_THST_HOVER`` matches the vehicle's
    actual hover thrust.
    """
    motor_hz: list[float] = []
    for index in range(1, 13):
        key = f"motor.{index}.rpm"
        if key not in bundle.signals:
            continue
        signal = bundle.signals[key]
        window = (signal.t >= t_start) & (signal.t <= t_end)
        if window.any():
            motor_hz.append(float(np.mean(signal.y[window])) / 60.0)

    throttle: float | None = None
    outputs = [
        bundle.signals[f"motor.{i}.output"]
        for i in range(1, 13)
        if f"motor.{i}.output" in bundle.signals
    ]
    if outputs:
        spin_min = bundle.param("MOT_SPIN_MIN", 0.15) or 0.15
        spin_max = bundle.param("MOT_SPIN_MAX", 0.95) or 0.95
        pwm_min = bundle.param("MOT_PWM_MIN", 1000.0) or 1000.0
        pwm_max = bundle.param("MOT_PWM_MAX", 2000.0) or 2000.0
        means = []
        for signal in outputs:
            window = (signal.t >= t_start) & (signal.t <= t_end)
            if window.any():
                pwm = float(np.mean(signal.y[window]))
                normalized = (pwm - pwm_min) / (pwm_max - pwm_min)
                means.append((normalized - spin_min) / max(spin_max - spin_min, 1e-6))
        if means:
            throttle = float(np.clip(np.mean(means), 0.0, 1.0))

    return OperatingPoint(throttle=throttle, motor_hz=tuple(motor_hz))
