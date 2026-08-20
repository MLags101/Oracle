"""The vehicle's filter chain, assembled from parameters and evaluated exactly.

This is the object that both halves of the tool depend on:

* **Identification** divides :meth:`FilterChain.sensor_response` out of the measured
  effective plant to recover the airframe (spec section 5.3, step 4).
* **Design** multiplies a *candidate* chain back into the broken loop (spec 5.7).

Those are the only two sites allowed to do either, and they must use the same
model, or filter phase gets counted twice or not at all.

**Where each filter lives.** Not every filter in the chain sits in the same place,
and putting them in one lump is the classic way to get the loop wrong:

===================  ==========================  ===========================
Filter               Path                        Affects
===================  ==========================  ===========================
gyro LPF, notches    sensor (common feedback)    margins and step
``FLTE`` error LPF   common feedback             margins and step
``FLTD`` D-term LPF  derivative branch only      margins and step
``FLTT`` target LPF  reference only              step only, never margins
===================  ==========================  ===========================

So :meth:`sensor_response` deliberately excludes the PID-local filters; the
controller model owns those and applies them to the branches they belong to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rotorid.core.filters.biquad import (
    BiquadCoeffs,
    cascade_response,
    lpf2p_biquad,
    onepole_alpha,
    onepole_response,
    phase_lag_deg,
    px4_lpf2p_biquad,
)
from rotorid.core.filters.harmonic import HarmonicNotch, NotchOption, harmonics_from_bitmask

__all__ = ["FilterChain", "OperatingPoint", "ardupilot_chain", "px4_chain"]

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

Stack = Literal["ardupilot", "px4"]
Axis = Literal["roll", "pitch", "yaw"]

_AP_AXIS_PARAM = {"roll": "RLL", "pitch": "PIT", "yaw": "YAW"}

#: Relative CPU cost weights, used only to rank candidate configurations against a
#: board's measured headroom. Not an absolute prediction.
_COST_PER_BIQUAD = 1.0
_COST_LOOP_RATE_UPDATE = 4.0
_COST_ALL_IMUS = 2.0


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """Where in the flight envelope the chain is being evaluated.

    Notch centres move with throttle or motor speed, so a chain has no single
    frequency response -- it has one per operating point. Identification uses the
    operating point of the segment being analysed.

    Attributes:
        throttle: Normalized throttle, for throttle-tracked notches.
        motor_hz: Measured motor frequencies. One value, or one per motor when the
            notch is configured for multi-source tracking.
    """

    throttle: float | None = None
    motor_hz: tuple[float, ...] = ()

    @property
    def has_measured_frequency(self) -> bool:
        """Whether a measured motor frequency is available."""
        return len(self.motor_hz) > 0


@dataclass(frozen=True, slots=True)
class FilterChain:
    """Everything between the gyro and the controller output, as configured.

    Attributes:
        sample_rate_hz: Rate the gyro-side filters run at. Notch and gyro LPF
            coefficients depend on it, so it must be the *sensor* rate, not the
            analysis grid rate.
        loop_rate_hz: Rate the PID-local filters run at.
        notch_ref: ``INS_HNTCH_REF``. Thrust reference for throttle tracking.
    """

    stack: Stack
    sample_rate_hz: float
    loop_rate_hz: float
    gyro_lpf_hz: float | None = None
    notches: tuple[HarmonicNotch, ...] = ()
    notch_ref: float | None = None
    dterm_lpf_hz: float | None = None
    error_lpf_hz: float | None = None
    target_lpf_hz: float | None = None
    all_imus: bool = False

    # ----------------------------------------------------------------- #
    # Sensor path -- the part that goes into the loop transfer function
    # ----------------------------------------------------------------- #

    def sensor_response(self, f_hz: FloatArray, op: OperatingPoint | None = None) -> ComplexArray:
        """Complex response of the gyro LPF and every notch, at one operating point.

        This is what identification divides out and design multiplies in. It does
        **not** include ``FLTE``/``FLTD``/``FLTT`` -- see the module docstring.

        Args:
            f_hz: Frequencies in Hz.
            op: Operating point. Required if any notch tracks; ignored for static ones.

        Returns:
            Complex response, same shape as ``f_hz``.
        """
        f = np.asarray(f_hz, dtype=np.float64)
        stages = self.sensor_stages(op)
        return cascade_response(stages, f) if stages else np.ones_like(f, dtype=np.complex128)

    def sensor_stages(self, op: OperatingPoint | None = None) -> list[BiquadCoeffs]:
        """The sensor-path biquads themselves, in firmware order.

        Same stages :meth:`sensor_response` evaluates. Exposed so that time-domain
        code -- the firmware-parity check against a pre/post-filter log, and the
        synthetic generators -- filters samples through the *identical* objects
        rather than a second transcription of the same maths.
        """
        stages: list[BiquadCoeffs] = []
        if self.gyro_lpf_hz:
            lpf = px4_lpf2p_biquad if self.stack == "px4" else lpf2p_biquad
            stages.append(lpf(self.gyro_lpf_hz, self.sample_rate_hz))
        for notch in self.notches:
            stages.extend(notch.stages(self._notch_centers(notch, op)))
        return stages

    def _notch_centers(self, notch: HarmonicNotch, op: OperatingPoint | None) -> list[float]:
        """Resolve one notch's tracked centre(s) at the operating point."""
        if op is None:
            return [notch.freq_hz]
        if op.has_measured_frequency:
            centers = list(op.motor_hz) if notch.per_motor else [op.motor_hz[0]]
            return notch.tracked_center_hz(motor_hz=centers)
        if op.throttle is not None and self.notch_ref:
            return notch.tracked_center_hz(throttle=op.throttle, ref=self.notch_ref)
        return [notch.freq_hz]

    # ----------------------------------------------------------------- #
    # PID-local filters, kept separate so the controller can place them
    # ----------------------------------------------------------------- #

    def error_lpf_response(self, f_hz: FloatArray) -> ComplexArray:
        """``FLTE``. In the common feedback path, so it does move the margins."""
        return self._onepole(self.error_lpf_hz, f_hz)

    def dterm_lpf_response(self, f_hz: FloatArray) -> ComplexArray:
        """``FLTD`` / ``IMU_DGYRO_CUTOFF``. Derivative branch only.

        The two stacks do not filter this branch the same way, and the difference
        is not cosmetic. ArduPilot's ``FLTD`` is a 1-pole IIR running at the loop
        rate; PX4's ``IMU_DGYRO_CUTOFF`` is a 2-pole Butterworth running at the
        gyro rate, on the angular *acceleration* estimate. At the same cutoff the
        2-pole costs roughly twice the phase, so modelling one with the other
        would misprice the D term on every PX4 aircraft.
        """
        if self.stack == "px4":
            if not self.dterm_lpf_hz:
                return np.ones_like(np.asarray(f_hz, dtype=np.float64), dtype=np.complex128)
            return cascade_response(
                [px4_lpf2p_biquad(self.dterm_lpf_hz, self.sample_rate_hz)],
                np.asarray(f_hz, dtype=np.float64),
            )
        return self._onepole(self.dterm_lpf_hz, f_hz)

    def target_lpf_response(self, f_hz: FloatArray) -> ComplexArray:
        """``FLTT``. Reference path only -- shapes the step, never the margins."""
        return self._onepole(self.target_lpf_hz, f_hz)

    def _onepole(self, cutoff_hz: float | None, f_hz: FloatArray) -> ComplexArray:
        f = np.asarray(f_hz, dtype=np.float64)
        if not cutoff_hz:
            return np.ones_like(f, dtype=np.complex128)
        alpha = onepole_alpha(cutoff_hz, 1.0 / self.loop_rate_hz)
        return onepole_response(alpha, f, self.loop_rate_hz)

    # ----------------------------------------------------------------- #
    # Reporting
    # ----------------------------------------------------------------- #

    def phase_deg(self, f_hz: FloatArray, op: OperatingPoint | None = None) -> FloatArray:
        """Sensor-path phase lag in degrees, positive for lag."""
        return phase_lag_deg(self.sensor_response(f_hz, op))

    def group_delay_ms(self, f_hz: FloatArray, op: OperatingPoint | None = None) -> FloatArray:
        """Sensor-path group delay in milliseconds.

        Computed as ``-d(phase)/d(omega)`` by finite difference, so it needs at
        least two frequency points.

        Raises:
            ValueError: if fewer than two frequencies are given.
        """
        f = np.asarray(f_hz, dtype=np.float64)
        if f.size < 2:
            raise ValueError("group delay needs at least two frequencies")
        phase_rad = np.radians(self.phase_deg(f, op))
        omega = 2.0 * np.pi * f
        return np.asarray(np.gradient(phase_rad, omega) * 1000.0, dtype=np.float64)

    def n_biquads(self, op: OperatingPoint | None = None) -> int:
        """How many biquad stages the firmware would be running."""
        count = 1 if self.gyro_lpf_hz else 0
        for notch in self.notches:
            count += len(notch.stages(self._notch_centers(notch, op)))
        return count

    def cpu_cost(self, op: OperatingPoint | None = None) -> float:
        """Relative CPU cost, for ranking candidates against board headroom."""
        cost = _COST_PER_BIQUAD * self.n_biquads(op)
        for notch in self.notches:
            if notch.opts & NotchOption.LOOP_RATE_UPDATE:
                cost += _COST_LOOP_RATE_UPDATE
        if self.all_imus:
            cost *= _COST_ALL_IMUS
        return cost

    def describe(self) -> str:
        """One-line human summary, for rationale text and reports."""
        parts = []
        if self.gyro_lpf_hz:
            parts.append(f"gyro LPF {self.gyro_lpf_hz:g} Hz")
        for notch in self.notches:
            harmonics = "+".join(str(h) for h in notch.harmonics)
            # PX4 notches have no attenuation setting, so printing one would
            # invite a user to go looking for a parameter that does not exist.
            depth = "" if notch.flavor == "px4" else f"att {notch.attenuation_db:g} dB "
            parts.append(
                f"notch {notch.freq_hz:g} Hz BW {notch.bandwidth_hz:g} "
                f"{depth}harmonics {harmonics}" + (" per-motor" if notch.per_motor else "")
            )
        if self.dterm_lpf_hz:
            parts.append(f"D LPF {self.dterm_lpf_hz:g} Hz")
        if self.error_lpf_hz:
            parts.append(f"error LPF {self.error_lpf_hz:g} Hz")
        return "; ".join(parts) if parts else "no filtering"


def ardupilot_chain(
    params: dict[str, float],
    axis: Axis,
    *,
    gyro_sample_rate_hz: float,
    loop_rate_hz: float,
) -> FilterChain:
    """Build a :class:`FilterChain` from an ArduPilot parameter snapshot.

    Reads ``INS_GYRO_FILTER``, both harmonic notch banks (``INS_HNTCH_*`` and
    ``INS_HNTC2_*``) and the per-axis ``ATC_RAT_*_FLT{D,E,T}`` cutoffs. Absent
    parameters are treated as "not configured", never as a default guess -- a
    missing parameter is a data-quality finding, not something to invent.

    Args:
        params: Parameter name to value, as logged.
        axis: Which rate loop's PID filters to read.
        gyro_sample_rate_hz: Rate the gyro filters run at.
        loop_rate_hz: Rate the PID filters run at.

    Returns:
        The reconstructed chain.
    """
    suffix = _AP_AXIS_PARAM[axis]

    notches: list[HarmonicNotch] = []
    ref: float | None = None
    all_imus = False
    for prefix in ("INS_HNTCH", "INS_HNTC2"):
        if not params.get(f"{prefix}_ENABLE", 0.0):
            continue
        harmonics = harmonics_from_bitmask(int(params.get(f"{prefix}_HMNCS", 1)))
        if not harmonics:
            continue
        opts = int(params.get(f"{prefix}_OPTS", 0))
        notches.append(
            HarmonicNotch(
                freq_hz=float(params.get(f"{prefix}_FREQ", 0.0)),
                bandwidth_hz=float(params.get(f"{prefix}_BW", 0.0)),
                attenuation_db=float(params.get(f"{prefix}_ATT", 0.0)),
                harmonics=harmonics,
                sample_rate_hz=gyro_sample_rate_hz,
                freq_min_ratio=float(params.get(f"{prefix}_FM_RAT", 1.0)),
                opts=opts,
            )
        )
        if ref is None:
            ref = float(params.get(f"{prefix}_REF", 0.0)) or None
        all_imus = all_imus or bool(opts & NotchOption.ALL_IMUS)

    return FilterChain(
        stack="ardupilot",
        sample_rate_hz=gyro_sample_rate_hz,
        loop_rate_hz=loop_rate_hz,
        gyro_lpf_hz=params.get("INS_GYRO_FILTER") or None,
        notches=tuple(notches),
        notch_ref=ref,
        dterm_lpf_hz=params.get(f"ATC_RAT_{suffix}_FLTD") or None,
        error_lpf_hz=params.get(f"ATC_RAT_{suffix}_FLTE") or None,
        target_lpf_hz=params.get(f"ATC_RAT_{suffix}_FLTT") or None,
        all_imus=all_imus,
    )


#: ``IMU_GYRO_DNF_EN`` bits.
class DynamicNotchSource:
    """Bit values of ``IMU_GYRO_DNF_EN``."""

    ESC_RPM = 1 << 0
    FFT = 1 << 1


def px4_chain(
    params: dict[str, float],
    axis: Axis,
    *,
    gyro_sample_rate_hz: float,
    loop_rate_hz: float,
) -> FilterChain:
    """Build a :class:`FilterChain` from a PX4 parameter snapshot.

    Reads ``IMU_GYRO_CUTOFF``, the two static notches ``IMU_GYRO_NF0_*`` and
    ``IMU_GYRO_NF1_*``, the dynamic notch ``IMU_GYRO_DNF_*``, and
    ``IMU_DGYRO_CUTOFF``.

    Three things differ from the ArduPilot reader in ways that change the answer
    rather than only the parameter names:

    * PX4's notches are true nulls with no attenuation setting, so the depth
      column is not a design variable on this stack (see
      :func:`~rotorid.core.filters.biquad.px4_notch_A_Q`).
    * ``IMU_GYRO_DNF_MIN`` is an absolute frequency floor, expressed here as a
      ratio of the fundamental so one notch type serves both stacks.
    * There is no error or target low-pass. PX4's rate controller has neither, and
      inventing them would put phase in the feedback path that the aircraft does
      not have.

    Args:
        axis: Accepted for symmetry with :func:`ardupilot_chain`. PX4's filters
            are per-IMU rather than per-axis, so every axis gets the same chain --
            which is itself worth knowing when comparing the two stacks.
    """
    del axis  # PX4 filters are per-IMU, not per-axis.

    notches: list[HarmonicNotch] = []
    for prefix in ("IMU_GYRO_NF0", "IMU_GYRO_NF1"):
        freq = float(params.get(f"{prefix}_FRQ", 0.0))
        bandwidth = float(params.get(f"{prefix}_BW", 0.0))
        if freq <= 0.0 or bandwidth <= 0.0:
            continue
        notches.append(
            HarmonicNotch(
                freq_hz=freq,
                bandwidth_hz=bandwidth,
                attenuation_db=0.0,
                harmonics=(1,),
                sample_rate_hz=gyro_sample_rate_hz,
                flavor="px4",
            )
        )

    enabled = int(params.get("IMU_GYRO_DNF_EN", 0))
    if enabled:
        harmonics = tuple(range(1, int(params.get("IMU_GYRO_DNF_HMC", 3)) + 1))
        minimum = float(params.get("IMU_GYRO_DNF_MIN", 0.0))
        # The dynamic notch has no configured centre: it follows the motors. The
        # minimum is the only frequency the parameters name, so it stands in as
        # the reference the operating point scales from.
        notches.append(
            HarmonicNotch(
                freq_hz=minimum,
                bandwidth_hz=float(params.get("IMU_GYRO_DNF_BW", 0.0)),
                attenuation_db=0.0,
                harmonics=harmonics or (1,),
                sample_rate_hz=gyro_sample_rate_hz,
                freq_min_ratio=1.0,
                flavor="px4",
            )
        )

    return FilterChain(
        stack="px4",
        sample_rate_hz=gyro_sample_rate_hz,
        loop_rate_hz=loop_rate_hz,
        gyro_lpf_hz=params.get("IMU_GYRO_CUTOFF") or None,
        notches=tuple(notches),
        notch_ref=None,
        dterm_lpf_hz=params.get("IMU_DGYRO_CUTOFF") or None,
        error_lpf_hz=None,
        target_lpf_hz=None,
    )
