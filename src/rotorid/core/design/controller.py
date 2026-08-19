"""Stack-specific rate controllers (spec section 5.8).

The two stacks are not the same controller with different parameter names, and
the difference is not cosmetic:

* ArduPilot takes the derivative of the **filtered error**, and puts ``FLTE`` in
  the common feedback path -- so ``FLTE`` moves the stability margins.
* PX4 takes the derivative of the **measurement** (angular acceleration, filtered
  at ``IMU_DGYRO_CUTOFF``) and has no error filter at all.

Both give the same margins for a given set of gains only if the filters are
absent. With realistic filters they diverge, and a tool that modeled one shape
for both would hand PX4 users gains designed against the wrong loop.

Reference-path terms -- ArduPilot's ``FLTT`` and the feed-forward -- deliberately
do **not** appear in :meth:`feedback_response`. They shape the commanded step and
have no effect on stability, and folding them into the loop is a standard way to
get a margin calculation quietly wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rotorid.core.filters.chain import FilterChain
from rotorid.core.types import ComplexArray, FloatArray, GainSet, Stack

__all__ = ["ArduPilotRatePID", "PX4RateController", "RateController", "controller_for"]


def _jw(f_hz: FloatArray) -> ComplexArray:
    return np.asarray(2j * np.pi * np.asarray(f_hz, dtype=np.float64), dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class RateController:
    """Common surface. Subclasses differ only in where the derivative is taken."""

    gains: GainSet
    chain: FilterChain

    @property
    def stack(self) -> Stack:
        """Which firmware this controller models."""
        raise NotImplementedError

    def feedback_response(self, f_hz: FloatArray) -> ComplexArray:
        """``C_fb(jw)`` -- everything in the feedback path, for margins."""
        raise NotImplementedError

    def reference_response(self, f_hz: FloatArray) -> ComplexArray:
        """``C_ref(jw)`` -- what acts on the setpoint, for the predicted step."""
        raise NotImplementedError

    def _pi(self, f_hz: FloatArray) -> ComplexArray:
        s = _jw(f_hz)
        with np.errstate(divide="ignore", invalid="ignore"):
            integral = np.where(np.abs(s) > 0.0, self.gains.ki / s, 0.0)
        return np.asarray(self.gains.kp + integral, dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class ArduPilotRatePID(RateController):
    """``AC_PID::update_all()``.

    ``C_fb(s) = [Kp + Ki/s + Kd*s*L_FLTD(s)] * L_FLTE(s)``

    The error filter multiplies the *whole* controller, derivative included,
    because in the firmware it is applied to the error before any term sees it.
    """

    @property
    def stack(self) -> Stack:
        """Always ``"ardupilot"``."""
        return "ardupilot"

    def feedback_response(self, f_hz: FloatArray) -> ComplexArray:
        s = _jw(f_hz)
        derivative = self.gains.kd * s * self.chain.dterm_lpf_response(f_hz)
        return np.asarray(
            (self._pi(f_hz) + derivative) * self.chain.error_lpf_response(f_hz),
            dtype=np.complex128,
        )

    def reference_response(self, f_hz: FloatArray) -> ComplexArray:
        """Target path: ``FLTT`` shapes the setpoint, then FF bypasses the PID.

        ``C_ref(s) = L_FLTT(s) * [C_fb(s) + Kff]``. The feed-forward is the reason
        an ArduPilot step can look quick without the margins improving at all --
        which is exactly the confusion the tool exists to clear up.
        """
        return np.asarray(
            self.chain.target_lpf_response(f_hz) * (self.feedback_response(f_hz) + self.gains.kff),
            dtype=np.complex128,
        )


@dataclass(frozen=True, slots=True)
class PX4RateController(RateController):
    """``RateControl::update()``.

    ``C_fb(s) = Kp + Ki/s + Kd*s*L_DGYRO(s)``, with no error filter.

    The gains here are already **effective** gains. PX4 stores ``MC_*RATE_P`` and
    a separate ``MC_*RATE_K`` multiplier, and the effective proportional gain is
    the product; that conversion happens once, at the IO boundary, so nothing in
    the design layer has to remember it.
    """

    @property
    def stack(self) -> Stack:
        """Always ``"px4"``."""
        return "px4"

    def feedback_response(self, f_hz: FloatArray) -> ComplexArray:
        s = _jw(f_hz)
        derivative = self.gains.kd * s * self.chain.dterm_lpf_response(f_hz)
        return np.asarray(self._pi(f_hz) + derivative, dtype=np.complex128)

    def reference_response(self, f_hz: FloatArray) -> ComplexArray:
        """Setpoint path: P and I act on the error, D does not, FF is direct.

        Because D is taken on the measurement, the derivative term contributes
        nothing to the reference path -- so for identical gains PX4 shows less
        step overshoot than ArduPilot while having exactly the same margins. That
        asymmetry is asserted in the tests.
        """
        return np.asarray(self._pi(f_hz) + self.gains.kff, dtype=np.complex128)


def controller_for(stack: Stack, gains: GainSet, chain: FilterChain) -> RateController:
    """Build the controller model matching a stack.

    Raises:
        ValueError: on an unknown stack, rather than defaulting to one of them --
            silently modeling PX4 as ArduPilot is a wrong-answer bug, not a
            degraded one.
    """
    if stack == "ardupilot":
        return ArduPilotRatePID(gains=gains, chain=chain)
    if stack == "px4":
        return PX4RateController(gains=gains, chain=chain)
    raise ValueError(f"unknown stack {stack!r}")
