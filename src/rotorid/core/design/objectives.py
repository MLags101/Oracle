"""Margin-constrained rate-gain design (spec section 5.7, inner loop).

The search is parameterized as ``(crossover, Ki/Kp, Kd/Kp)`` rather than as raw
gains. That is not a style preference: raw ``(Kp, Ki, Kd)`` are strongly
correlated -- scaling all three together only slides the loop up and down without
changing its shape -- so a search over them spends most of its effort exploring
directions that do nothing.

Splitting the shape from the level also makes the whole design nearly free to
evaluate. With ``Ki/Kp`` and ``Kd/Kp`` fixed, the loop is *linear in* ``Kp``, so
``Kp`` only slides ``|L|`` up and down and leaves the phase untouched. The phase
margin available at any candidate crossover is therefore known before ``Kp`` is
chosen at all, and the whole crossover sweep collapses into one array operation.
That is what holds the interactive re-solve inside its budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rotorid.core.analysis.margins import LoopDelay, compute_margins, plant_path
from rotorid.core.design.controller import controller_for
from rotorid.core.filters.chain import FilterChain, OperatingPoint
from rotorid.core.types import (
    AirframeModel,
    Axis,
    ComplexArray,
    FloatArray,
    GainSet,
    MarginReport,
    Stack,
)

__all__ = ["DesignResult", "DesignTargets", "design_gains"]

_DRB_LEVEL_DB = -3.0

#: Ki/Kp and Kd/Kp search grids. Ki/Kp is a rate in 1/s -- the integrator corner
#: -- and Kd/Kp is a time in s. The ranges bracket every published multirotor
#: tune by a wide margin; the resolution is what a user could distinguish in
#: flight.
_KI_OVER_KP = np.geomspace(0.5, 40.0, 14)
_KD_OVER_KP = np.geomspace(0.002, 0.10, 18)

#: Crossover must stay below this fraction of the delay-imposed limit
#: ``1/(2*pi*tau)``. Above it, no gain set has usable margins.
_CROSSOVER_DELAY_FRACTION = 0.25

#: How many candidate crossovers to try per gain shape. The design grid is far
#: finer than this because the *margins* are read off it, but trying every one of
#: its points as a crossover costs an n_crossovers x n_frequencies sensitivity
#: evaluation per shape for a resolution no user could fly the difference of.
#: Sixty-four log-spaced candidates put neighbouring crossovers a few percent
#: apart across the whole usable range, which is finer than the difference a
#: pilot could feel.
_MAX_CROSSOVER_CANDIDATES = 64


@dataclass(frozen=True, slots=True)
class DesignTargets:
    """Everything the optimizer is allowed to trade, and the two things it is not.

    ``pm_floor_deg`` is a hard safety limit rather than a preference: flight-test
    work finds phase margins of 20-23 degrees produce PIO tendency, so no slider
    position may design below it.
    """

    pm_min_deg: float
    gm_min_db: float
    ms_max_db: float
    pm_floor_deg: float
    crossover_frac_of_loop: float
    conservatism: float = 0.5
    #: Extra gain margin required because the aircraft was measured oscillating at
    #: a frequency this model says has margin left. Not a preference and not tied
    #: to the slider: it is the size of a demonstrated error in the model, so the
    #: conservatism control cannot trade it away.
    gm_holdback_db: float = 0.0

    def effective_pm_deg(self) -> float:
        """Phase-margin target after the conservatism slider, never below the floor.

        0 = aggressive (10 degrees below the nominal target), 1 = docile (15
        above). Clamped at the floor in both directions, so the slider can never
        be used to talk the designer into an unsafe margin.
        """
        span = -10.0 + 25.0 * float(np.clip(self.conservatism, 0.0, 1.0))
        return max(self.pm_min_deg + span, self.pm_floor_deg)

    def effective_gm_db(self) -> float:
        """Gain-margin target after any measured-oscillation holdback."""
        return self.gm_min_db + max(self.gm_holdback_db, 0.0)

    def crossover_scale(self) -> float:
        """How far the conservatism slider backs the crossover ceiling off.

        The slider has to move both halves of the trade. Phase margin alone is not
        enough: on many vehicles the binding constraint is peak sensitivity, not
        phase margin, and a slider wired only to the phase-margin target then does
        nothing at all -- the user drags it and the answer never changes.
        """
        return 1.0 - 0.4 * float(np.clip(self.conservatism, 0.0, 1.0))

    def max_crossover_hz(self, *, tau_s: float, loop_rate_hz: float) -> tuple[float, str]:
        """Crossover ceiling, and which of the two limits set it.

        Returns:
            ``(f_hz, reason)`` where reason is ``"crossover_limit_delay"`` or
            ``"crossover_limit_loop_rate"``.
        """
        by_delay = _CROSSOVER_DELAY_FRACTION / (2.0 * np.pi * tau_s) if tau_s > 0.0 else np.inf
        by_loop = self.crossover_frac_of_loop * loop_rate_hz / 2.0
        scale = self.crossover_scale()
        if by_delay <= by_loop:
            return float(by_delay * scale), "crossover_limit_delay"
        return float(by_loop * scale), "crossover_limit_loop_rate"


@dataclass(frozen=True, slots=True)
class DesignResult:
    """The chosen gains, what they achieve, and why they are not better.

    ``binding_constraint`` is the single most useful line the tool produces: it
    turns "here are your gains" into "here is what is stopping them being higher",
    which is the difference between a number to trust and a number to obey.

    Attributes:
        designed_crossover_hz: The crossover the search aimed at -- the frequency
            ``Kp`` was chosen to put unity gain at, and the one bounded by
            ``crossover_ceiling_hz``. It is not always where the finished loop
            crosses: a loop with a strong D term can dip through 0 dB, be lifted
            back over it, and cross again higher up, and
            ``margins.crossover_hz`` reports that highest crossing because that
            is the bandwidth the aircraft actually has. The two are separate
            facts and conflating them makes the ceiling look violated when it
            was obeyed.
    """

    gains: GainSet
    margins: MarginReport
    binding_constraint: str
    crossover_ceiling_hz: float
    feasible_count: int
    designed_crossover_hz: float = 0.0
    rejected: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def _unit_controller_shape(
    f_hz: FloatArray, chain: FilterChain, stack: Stack, ki_ratio: float, kd_ratio: float
) -> ComplexArray:
    """``C_fb(jw) / Kp`` for one shape -- the part that ``Kp`` does not scale."""
    gains = GainSet(axis="roll", kp=1.0, ki=ki_ratio, kd=kd_ratio, kff=0.0)
    return controller_for(stack, gains, chain).feedback_response(f_hz)


def _first_upward_crossing_rows(f: FloatArray, values: FloatArray, level: float) -> FloatArray:
    """Per-row lowest frequency where ``values`` rises through ``level``.

    ``values`` is ``(n_candidates, n_frequencies)``. Rows that never cross return
    0.0, which reads downstream as "no disturbance-rejection bandwidth". Rows that
    are already above the level in the first bin return the bottom of the band:
    such a loop rejects nothing anywhere, however clean it looks higher up.

    Written without a Python loop because the design calls it once per gain shape,
    a few hundred times per re-solve, inside the interactive budget.
    """
    reached = values >= level
    any_reached = reached.any(axis=1)
    right = np.argmax(reached, axis=1)
    left = np.maximum(right - 1, 0)
    rows = np.arange(values.shape[0])

    y0 = values[rows, left]
    y1 = values[rows, right]
    x0 = np.log(f[left])
    x1 = np.log(f[right])

    with np.errstate(divide="ignore", invalid="ignore"):
        interpolated = np.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    out = np.where(y1 != y0, interpolated, f[right])
    out = np.where(right == 0, f[0], out)
    return np.asarray(np.where(any_reached, out, 0.0), dtype=np.float64)


def design_gains(
    airframe: AirframeModel,
    chain: FilterChain,
    *,
    stack: Stack,
    delay: LoopDelay,
    targets: DesignTargets,
    f_grid: FloatArray,
    op: OperatingPoint | None = None,
    axis: Axis | None = None,
) -> DesignResult:
    """Choose the gains with the widest disturbance-rejection bandwidth that hold.

    Constraints, all simultaneous: phase margin, gain margin, peak sensitivity,
    and a crossover ceiling set by whichever of transport delay or loop rate binds
    first. The objective is disturbance-rejection bandwidth, not crossover --
    crossover is a means, and optimizing it directly rewards designs that are fast
    at rejecting nothing.

    Raises:
        ValueError: if no gain set in the search space satisfies the constraints.
            Refusing is the required behaviour: quietly relaxing a margin and
            presenting the result as a tune is the failure mode this tool exists
            to prevent.
    """
    axis = axis or airframe.axis
    tau_s = float(airframe.params.get("tau", 0.0))
    ceiling_hz, ceiling_reason = targets.max_crossover_hz(
        tau_s=tau_s, loop_rate_hz=chain.loop_rate_hz
    )
    pm_target = targets.effective_pm_deg()

    plant = plant_path(
        f_grid,
        controller_for(stack, _unit_gains(axis), chain),
        airframe,
        delay=delay,
        op=op,
    )

    wc_choices = np.nonzero((f_grid > f_grid[0]) & (f_grid <= ceiling_hz))[0]
    if wc_choices.size > _MAX_CROSSOVER_CANDIDATES:
        # The grid is log-spaced, so evenly spaced indices are evenly spaced in
        # log frequency -- which is how crossover resolution should be measured.
        picks = np.linspace(0, wc_choices.size - 1, _MAX_CROSSOVER_CANDIDATES)
        wc_choices = wc_choices[np.unique(np.round(picks).astype(int))]
    if wc_choices.size == 0:
        raise ValueError(
            f"crossover ceiling of {ceiling_hz:.2f} Hz sits below the design grid; "
            f"limited by {ceiling_reason}"
        )

    best: tuple[float, GainSet, MarginReport, float] | None = None
    feasible_count = 0
    blockers: list[tuple[float, str]] = []

    for ki_ratio in _KI_OVER_KP:
        for kd_ratio in _KD_OVER_KP:
            L1 = _unit_controller_shape(f_grid, chain, stack, ki_ratio, kd_ratio) * plant
            mag1_db = 20.0 * np.log10(np.abs(L1))
            phase_deg = np.degrees(np.unwrap(np.angle(L1)))

            pm = (360.0 + phase_deg[wc_choices]) % 360.0 - 180.0
            kp_db = -mag1_db[wc_choices]
            kp = 10.0 ** (kp_db / 20.0)

            # Gain margin, from the first phase crossing of -180 degrees.
            gm_db = _gain_margin_row(f_grid, mag1_db, phase_deg, kp_db)

            # Phase and gain margin are one-dimensional and cheap. Sensitivity is
            # not: it needs the whole loop at every candidate crossover, which is
            # the single most expensive array in the design. So the cheap
            # constraints are applied first and only the survivors pay for it.
            passes_margins = (pm >= pm_target) & (gm_db >= targets.effective_gm_db())
            if not passes_margins.any():
                ms_probe = np.full(pm.shape, -np.inf)
                blockers.append(_worst_blocker(pm, gm_db, ms_probe, targets, pm_target))
                continue

            rows = np.nonzero(passes_margins)[0]
            L = kp[rows, None] * L1[None, :]
            S_db = -20.0 * np.log10(np.abs(1.0 + L))
            ms_db = np.max(S_db, axis=1)

            ok = ms_db <= targets.ms_max_db
            if not ok.any():
                blockers.append(_worst_blocker(pm[rows], gm_db[rows], ms_db, targets, pm_target))
                continue

            drb = _first_upward_crossing_rows(f_grid, S_db, _DRB_LEVEL_DB)
            drb = np.where(ok, drb, 0.0)
            feasible_count += int(ok.sum())
            winner = int(np.argmax(drb))
            if drb[winner] <= 0.0:
                continue

            candidate_kp = float(kp[rows[winner]])
            gains = GainSet(
                axis=axis,
                kp=candidate_kp,
                ki=candidate_kp * float(ki_ratio),
                kd=candidate_kp * float(kd_ratio),
                kff=_feedforward_gain(airframe),
                dterm_lpf_hz=chain.dterm_lpf_hz,
                error_lpf_hz=chain.error_lpf_hz,
                target_lpf_hz=chain.target_lpf_hz,
            )
            if best is None or drb[winner] > best[0]:
                # The vectorized screen above checks the margin at the crossover
                # this shape was *aimed* at. A loop that grazes 0 dB elsewhere has
                # further crossings the screen never looked at, and the margin
                # that matters is the worst of them. So the winner is confirmed
                # against the same full evaluation the report will publish --
                # otherwise the tool could recommend a tune whose own margin
                # table shows it failing the constraint it was designed under.
                report = compute_margins(f_grid, candidate_kp * L1)
                if (
                    report.phase_margin_deg < pm_target
                    or report.gain_margin_db < targets.effective_gm_db()
                    or report.peak_sensitivity_db > targets.ms_max_db
                ):
                    continue
                best = (float(drb[winner]), gains, report, float(f_grid[wc_choices[rows[winner]]]))

    if best is None:
        raise ValueError(
            "no gain set in the search space meets the margin constraints "
            f"(PM >= {pm_target:.0f} deg, GM >= {targets.effective_gm_db():.0f} dB, "
            f"Ms <= {targets.ms_max_db:.0f} dB). The airframe or its filter chain, "
            "not the gains, is the problem."
        )

    _, gains, report, designed_hz = best
    binding = _binding_constraint(
        report, targets, pm_target, ceiling_hz, ceiling_reason, designed_hz
    )
    return DesignResult(
        gains=gains,
        margins=report,
        binding_constraint=binding,
        crossover_ceiling_hz=ceiling_hz,
        feasible_count=feasible_count,
        designed_crossover_hz=designed_hz,
        rejected=_summarize_blockers(blockers),
    )


def _summarize_blockers(blockers: list[tuple[float, str]]) -> tuple[tuple[str, str], ...]:
    """Why the shapes that were not chosen lost, one line per distinct reason."""
    worst: dict[str, float] = {}
    for shortfall, name in blockers:
        if shortfall > worst.get(name, -np.inf):
            worst[name] = shortfall
    return tuple(
        (name, f"missed by {shortfall:.1f} at its best crossover")
        for name, shortfall in sorted(worst.items(), key=lambda kv: -kv[1])
    )


def _unit_gains(axis: Axis) -> GainSet:
    return GainSet(axis=axis, kp=1.0, ki=0.0, kd=0.0, kff=0.0)


def _feedforward_gain(airframe: AirframeModel) -> float:
    """Feed-forward that produces the commanded rate open-loop.

    Steady state gives ``rate = K * output``, so ``Kff = 1/K`` asks the mixer for
    exactly the output the vehicle needs and leaves the feedback path to correct
    the error rather than to produce the response. Returns 0 if ``K`` is
    degenerate -- a feed-forward built on a bad gain estimate is worse than none.
    """
    K = float(airframe.params.get("K", 0.0))
    return 1.0 / K if K > 1e-6 else 0.0


def _gain_margin_row(
    f_grid: FloatArray, mag1_db: FloatArray, phase_deg: FloatArray, kp_db: FloatArray
) -> FloatArray:
    """Gain margin for every ``Kp`` candidate at once.

    The phase crossing does not move with ``Kp``, so it is found once and every
    candidate's margin is that point's magnitude shifted by its own ``Kp``.
    """
    idx = np.nonzero((phase_deg[:-1] >= -180.0) & (phase_deg[1:] < -180.0))[0]
    if idx.size == 0:
        return np.full(kp_db.shape, np.inf)
    i = int(idx[0])
    frac = (-180.0 - phase_deg[i]) / (phase_deg[i + 1] - phase_deg[i])
    mag_at_180 = mag1_db[i] + frac * (mag1_db[i + 1] - mag1_db[i])
    return np.asarray(-(mag_at_180 + kp_db), dtype=np.float64)


def _worst_blocker(
    pm: FloatArray,
    gm_db: FloatArray,
    ms_db: FloatArray,
    targets: DesignTargets,
    pm_target: float,
) -> tuple[float, str]:
    """Which constraint rejected this shape, and by how much at its best point."""
    best = int(np.argmax(pm))
    shortfalls = {
        "phase_margin": pm_target - float(pm[best]),
        "gain_margin": targets.effective_gm_db() - float(gm_db[best]),
        "peak_sensitivity": float(ms_db[best]) - targets.ms_max_db,
    }
    name = max(shortfalls, key=lambda k: shortfalls[k])
    return shortfalls[name], name


def _gain_margin_name(targets: DesignTargets) -> str:
    """What to call the gain-margin constraint, given why it is where it is.

    A user told "gain margin" is what stopped the tune will go looking for a
    number in the config. When the binding number is a holdback measured off their
    own aircraft's oscillation, that is a different fact and needs a different
    name, or the explanation points at the wrong thing.
    """
    return "measured_oscillation" if targets.gm_holdback_db > 0.0 else "gain_margin"


def _binding_constraint(
    report: MarginReport,
    targets: DesignTargets,
    pm_target: float,
    ceiling_hz: float,
    ceiling_reason: str,
    designed_crossover_hz: float,
) -> str:
    """Which constraint the winning design is sitting on.

    Reported as the constraint with the least slack, expressed as a fraction of
    its own tolerance so that degrees and decibels can be compared honestly.
    """
    slack = {
        "phase_margin": (report.phase_margin_deg - pm_target) / max(pm_target, 1.0),
        _gain_margin_name(targets): (report.gain_margin_db - targets.effective_gm_db())
        / max(targets.effective_gm_db(), 1.0),
        "peak_sensitivity": (targets.ms_max_db - report.peak_sensitivity_db)
        / max(targets.ms_max_db, 1.0),
        ceiling_reason: (ceiling_hz - designed_crossover_hz) / ceiling_hz,
    }
    return min(slack, key=lambda k: slack[k])
