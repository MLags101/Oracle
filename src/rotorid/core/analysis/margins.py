"""Broken-loop assembly and stability margins (spec sections 5.7 and 5.8).

:func:`broken_loop` is the **only** place a filter chain is multiplied back into
the loop, the mirror of the single divide in
:func:`rotorid.core.analysis.sysid.deconvolve`. Two sites, no more: that pairing
is what keeps filter phase from being counted twice or lost.

Margins are read off the assembled complex response rather than from a rational
model, because the delay is exact here and a Pade approximation of 20 ms of lag
is worth several degrees at crossover -- more than the difference between a good
tune and a marginal one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rotorid.core.analysis.model_eval import airframe_response
from rotorid.core.design.controller import RateController
from rotorid.core.filters.chain import OperatingPoint
from rotorid.core.types import AirframeModel, ComplexArray, FloatArray, MarginReport

__all__ = [
    "LoopDelay",
    "broken_loop",
    "compute_margins",
    "design_grid",
    "loop_delay",
]

#: The disturbance-rejection bandwidth is where the rejection function reaches
#: this level. Standard rotorcraft handling-qualities definition.
_DRB_LEVEL_DB = -3.0


@dataclass(frozen=True, slots=True)
class LoopDelay:
    """Transport lag in the loop that is *not* already inside the airframe model.

    ``tau`` from the identification is deliberately excluded: it was fitted after
    the filter chain was divided out, so it already carries actuator and motor
    lag. Adding these terms to it as well would double-count (spec 5.4).
    """

    zoh_s: float
    compute_s: float
    actuator_s: float

    @property
    def total_s(self) -> float:
        """Total additional transport delay in seconds."""
        return self.zoh_s + self.compute_s + self.actuator_s

    def response(self, f_hz: FloatArray) -> ComplexArray:
        """``exp(-j*w*T)`` on the given grid."""
        f = np.asarray(f_hz, dtype=np.float64)
        return np.asarray(np.exp(-2j * np.pi * f * self.total_s), dtype=np.complex128)


def loop_delay(
    *,
    loop_rate_hz: float,
    actuator_ms: float,
    zoh_loops: float,
    compute_loops: float,
) -> LoopDelay:
    """Assemble the loop delay from config-driven terms.

    Args:
        loop_rate_hz: ``SCHED_LOOP_RATE`` / ``IMU_GYRO_RATEMAX``.
        actuator_ms: From the ESC-protocol table in ``filters/latency.py``.
        zoh_loops: ``[design].zoh_delay_loops``. Half a period, exactly, for a ZOH.
        compute_loops: ``[design].compute_delay_loops``.
    """
    period = 1.0 / loop_rate_hz
    return LoopDelay(
        zoh_s=zoh_loops * period,
        compute_s=compute_loops * period,
        actuator_s=actuator_ms / 1000.0,
    )


def design_grid(f_min_hz: float, f_max_hz: float, points: int = 600) -> FloatArray:
    """Log-spaced frequency grid for design work.

    Fixed once per session and reused: precomputing the airframe and each
    candidate chain on this grid is what keeps the interactive re-solve inside its
    budget, since only the controller changes as sliders move.
    """
    if not 0.0 < f_min_hz < f_max_hz:
        raise ValueError(f"invalid design band {f_min_hz}..{f_max_hz} Hz")
    return np.asarray(np.geomspace(f_min_hz, f_max_hz, points), dtype=np.float64)


def broken_loop(
    f_hz: FloatArray,
    controller: RateController,
    airframe: AirframeModel,
    *,
    delay: LoopDelay,
    op: OperatingPoint | None = None,
    plant_response: ComplexArray | None = None,
) -> ComplexArray:
    """``L(jw) = C_fb(jw) * F_sensor(jw) * G_air(jw) * D_loop(jw)``.

    Args:
        plant_response: Precomputed ``F * G_air * D_loop`` on the same grid. Pass
            it when sweeping many gain candidates -- that product does not change
            with the gains, and recomputing it per candidate is most of the cost.

    Note:
        ``F_sensor`` is the sensor path only. ``FLTE`` and ``FLTD`` live in the
        controller, applied to the branches they actually sit in.
    """
    if plant_response is None:
        plant_response = plant_path(f_hz, controller, airframe, delay=delay, op=op)
    return np.asarray(controller.feedback_response(f_hz) * plant_response, dtype=np.complex128)


def plant_path(
    f_hz: FloatArray,
    controller: RateController,
    airframe: AirframeModel,
    *,
    delay: LoopDelay,
    op: OperatingPoint | None = None,
) -> ComplexArray:
    """Everything in the loop that the gains do not change: ``F * G_air * D``."""
    return np.asarray(
        controller.chain.sensor_response(f_hz, op)
        * airframe_response(airframe, f_hz)
        * delay.response(f_hz),
        dtype=np.complex128,
    )


def _interp_crossing(x: FloatArray, y: FloatArray, level: float, index: int) -> float:
    """Linear interpolation, in log-x, of the crossing between ``index`` and +1."""
    x0, x1 = np.log(x[index]), np.log(x[index + 1])
    y0, y1 = y[index], y[index + 1]
    if y1 == y0:
        return float(x[index])
    return float(np.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0)))


def _downward_crossings(f: FloatArray, values: FloatArray, level: float) -> list[float]:
    """Every frequency where ``values`` falls through ``level``, ascending."""
    above = values >= level
    edges = np.nonzero(above[:-1] & ~above[1:])[0]
    return [_interp_crossing(f, values, level, int(i)) for i in edges]


def _first_downward_crossing(f: FloatArray, values: FloatArray, level: float) -> float | None:
    """Lowest frequency where ``values`` falls through ``level``."""
    crossings = _downward_crossings(f, values, level)
    return crossings[0] if crossings else None


def _first_upward_crossing(f: FloatArray, values: FloatArray, level: float) -> float | None:
    """Lowest frequency at which ``values`` has reached ``level``.

    Deliberately *not* "the first rising edge". If the series is already at or
    above the level in the first bin, the answer is the bottom of the band, not
    whatever later edge happens to be the first clean crossing -- reading it the
    other way reports a wide disturbance-rejection bandwidth for a loop that
    rejects nothing anywhere.
    """
    reached = np.nonzero(values >= level)[0]
    if reached.size == 0:
        return None
    i = int(reached[0])
    if i == 0:
        return float(f[0])
    return _interp_crossing(f, values, level, i - 1)


def compute_margins(f_hz: FloatArray, L: ComplexArray) -> MarginReport:
    """Read gain, phase, delay and sensitivity margins off the broken loop.

    A loop with a lightly damped airframe can cross unity gain more than once:
    ``|L|`` falls through 0 dB, rises back through it around the resonance, and
    falls again. Every one of those crossings is a place the loop could be
    destabilized, so the reported phase margin is the **worst** of them, and the
    reported crossover is the frequency where that worst margin occurs. Likewise
    the gain margin is the smallest over every phase crossing of -180 degrees.

    Reporting the *first* crossing instead -- which this did until it was caught
    by a design that changed by 65% under a 4e-15 perturbation of the measured
    response -- is both optimistic and numerically unstable. When ``|L|`` grazes
    0 dB, two extra crossings appear and disappear together as the tangency is
    crossed, so "the first crossing" jumps discontinuously between branches while
    the worst margin varies smoothly: at the moment of tangency the new pair
    carries the phase of the tangency point, which is a value the report already
    had. Stability under perturbation is not a nicety here. A recommendation that
    moves because the tenth significant figure of the input moved is not a
    recommendation.

    ``disturbance_rejection_peak_db`` and ``peak_sensitivity_db`` are the same
    quantity, ``||S||inf``, under the standard definitions. Both names are
    reported because the rotorcraft handling-qualities literature and the control
    literature each use their own, and users arrive from both.

    Raises:
        ValueError: if ``|L|`` never crosses unity in the supplied band, which
            means the design grid is too narrow rather than that the loop is
            stable at every frequency.
    """
    f = np.asarray(f_hz, dtype=np.float64)
    mag_db = 20.0 * np.log10(np.abs(L))
    phase_deg = np.degrees(np.unwrap(np.angle(L)))

    crossings = _downward_crossings(f, mag_db, 0.0)
    if not crossings:
        raise ValueError(
            "|L| does not cross unity anywhere in the design band; "
            f"it runs {mag_db[0]:.1f} to {mag_db[-1]:.1f} dB over {f[0]:g}-{f[-1]:g} Hz"
        )

    # Wrap into (-180, 180]. The unwrapped phase can be several turns down by the
    # time a delay-heavy loop reaches crossover, and the margin is the distance to
    # the *nearest* -180 crossing, not to the first one.
    log_f = np.log(f)
    margins = [
        (180.0 + float(np.interp(np.log(wc), log_f, phase_deg)) + 180.0) % 360.0 - 180.0
        for wc in crossings
    ]
    # Two different questions with two different answers. The loop's *bandwidth*
    # is the highest frequency it still passes unity at; its *margin* is the
    # smallest room to spare at any of them. Quoting the pair from one crossing
    # would understate the bandwidth or overstate the margin.
    crossover_hz = crossings[-1]
    phase_margin_deg = min(margins)

    # Gain margin: the least room to spare over every phase crossing of -180.
    gm_crossings = _downward_crossings(f, phase_deg, -180.0)
    gain_margin_db = (
        min(-float(np.interp(np.log(hz), log_f, mag_db)) for hz in gm_crossings)
        if gm_crossings
        else float("inf")
    )

    S_db = -20.0 * np.log10(np.abs(1.0 + L))
    peak_sensitivity_db = float(np.max(S_db))
    drb_hz = _first_upward_crossing(f, S_db, _DRB_LEVEL_DB)

    # Delay margin is per crossing and the binding one is the smallest: the
    # amount of extra lag the loop tolerates is set by whichever crossing runs
    # out of phase first, which is not necessarily the one with the worst margin
    # in degrees -- a small margin at a low frequency buys more time than a
    # larger one high up.
    delay_margin_ms = min(
        1000.0 * np.radians(max(pm, 0.0)) / (2.0 * np.pi * wc)
        for pm, wc in zip(margins, crossings, strict=True)
    )

    return MarginReport(
        gain_margin_db=gain_margin_db,
        phase_margin_deg=phase_margin_deg,
        crossover_hz=crossover_hz,
        delay_margin_ms=delay_margin_ms,
        peak_sensitivity_db=peak_sensitivity_db,
        disturbance_rejection_bw_hz=drb_hz if drb_hz is not None else 0.0,
        disturbance_rejection_peak_db=peak_sensitivity_db,
    )
