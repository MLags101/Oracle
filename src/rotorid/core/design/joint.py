"""Filters and gains, designed against one shared budget (spec section 5.7).

Designing these separately is the standard way to end up with a vehicle that is
worse than it started. The filter choice depends on the crossover, because that
is where its phase lag is spent; the crossover depends on the filter choice,
because the lag is what limits it; and the D-term noise ceiling depends on the
derivative gain, which is not known until the gains are designed. Three circular
dependencies, each of which is silent if you break it by assuming.

So the two are solved together, as a fixed-point iteration:

    design gains -> crossover -> design filters -> design gains -> ...

which converges in two or three passes because each step moves the crossover by
less than the last. The alternative -- enumerating every filter configuration
against a full gain search -- costs the same answer several hundred times over
and would not fit the interactive re-solve budget.

The result is always compared against leaving the filters alone. A filter change
has to *earn* its place by widening the disturbance-rejection bandwidth; if it
does not, the recommendation is to change nothing, and the report says why.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from rotorid.config import Config
from rotorid.core.analysis.margins import LoopDelay
from rotorid.core.analysis.noise import MotorTrack, dterm_noise_rms
from rotorid.core.design.filters import describe_chain, recommend_filters
from rotorid.core.design.objectives import DesignResult, DesignTargets, design_gains
from rotorid.core.filters.chain import FilterChain, OperatingPoint
from rotorid.core.types import (
    AirframeModel,
    Axis,
    FilterRecommendation,
    FloatArray,
    NoiseProfile,
    Stack,
)

__all__ = ["JointResult", "optimize_jointly", "unchanged_filters"]

#: The iteration is a contraction, not a search: each pass moves the crossover by
#: less than the last. Four is generous -- convergence is normally reached in two.
_MAX_PASSES = 4

#: Fractional crossover change below which another pass would not change any
#: recommended parameter value.
_CONVERGED = 0.02

#: A filter change must widen disturbance-rejection bandwidth by at least this
#: fraction to be worth asking the user to fly again.
_MIN_DRB_GAIN = 0.02

#: Frequency the flown chain's phase cost is quoted at when no design crossover
#: is involved. Reported for comparison only; margins never come from it.
_PHASE_PROBE_HZ = 5.0


@dataclass(frozen=True, slots=True)
class JointResult:
    """The jointly designed filters and gains, and the evidence behind the choice."""

    design: DesignResult
    filters: FilterRecommendation
    dterm_noise_rms: float
    passes: int
    rejected: tuple[tuple[str, str], ...] = ()

    @property
    def filters_changed(self) -> bool:
        """Whether the recommendation actually asks for a filter change."""
        return self.filters.chain is not self.filters.baseline_chain


def optimize_jointly(
    airframe: AirframeModel,
    baseline_chain: FilterChain,
    noise: NoiseProfile | None,
    config: Config,
    *,
    stack: Stack,
    delay: LoopDelay,
    targets: DesignTargets,
    f_grid: FloatArray,
    op: OperatingPoint,
    axis: Axis,
    track: MotorTrack | None = None,
    hover_thrust: float | None = None,
    fft_available: bool = False,
    per_motor_capable: bool = False,
    chain_override: FilterChain | None = None,
) -> JointResult:
    """Design filters and gains together for one axis.

    Args:
        noise: The measured noise profile. ``None`` -- or a profile with no
            pre-filter spectrum -- means there is no evidence to design filters
            from, and the flown chain is kept and labelled as such rather than
            guessed at.
        chain_override: A chain chosen by the user in the sandbox. When given, no
            filter design happens at all: the gains are designed against exactly
            what was asked for, and the result is reported by the same measures
            as a designed chain. Overriding is allowed to produce a worse answer
            -- that is what makes it a sandbox rather than a set of presets --
            but it is never allowed to produce an unmeasured one.

    Returns:
        The better of the jointly designed chain and the flown one, by
        disturbance-rejection bandwidth.

    Raises:
        ValueError: if no gain set meets the margin constraints even with the
            flown filter chain. Propagated from :func:`design_gains`, because at
            that point the problem is the airframe, not the tuning.
    """
    baseline = design_gains(
        airframe,
        baseline_chain,
        stack=stack,
        delay=delay,
        targets=targets,
        f_grid=f_grid,
        op=op,
        axis=axis,
    )
    rejected: list[tuple[str, str]] = []

    if chain_override is not None:
        forced = design_gains(
            airframe,
            chain_override,
            stack=stack,
            delay=delay,
            targets=targets,
            f_grid=f_grid,
            op=op,
            axis=axis,
        )
        return JointResult(
            design=forced,
            filters=describe_chain(
                chain_override,
                baseline_chain,
                axis=axis,
                stack=stack,
                noise=noise,
                op=op,
                crossover_hz=forced.margins.crossover_hz,
            ),
            dterm_noise_rms=_noise_rms(noise, chain_override, forced, op),
            passes=0,
        )

    if noise is None or noise.psd_pre is None or track is None:
        return JointResult(
            design=baseline,
            filters=unchanged_filters(
                baseline_chain,
                stack,
                "Filters left as flown: this log has no usable noise measurement to "
                "design them from. Hover for 20-30 seconds with ESC telemetry logged "
                "and the filter recommendation becomes available.",
                noise=noise,
                op=op,
                rejected=(("filter redesign", "no noise evidence in this log"),),
            ),
            dterm_noise_rms=_noise_rms(noise, baseline_chain, baseline, op),
            passes=0,
            rejected=(("filter redesign", "no noise evidence in this log"),),
        )

    noise_limit = config.float_("noise", "dterm_output_rms_limit_pct") / 100.0

    best: tuple[DesignResult, FilterRecommendation, float] | None = None
    crossover = baseline.margins.crossover_hz
    kd = baseline.gains.kd
    passes = 0
    evaluated: set[str] = {baseline_chain.describe()}

    for _ in range(_MAX_PASSES):
        passes += 1
        candidate_filters = recommend_filters(
            noise,
            baseline_chain,
            config,
            track=track,
            op=op,
            crossover_hz=crossover,
            kd=kd,
            hover_thrust=hover_thrust,
            fft_available=fft_available,
            per_motor_capable=per_motor_capable,
        )
        if candidate_filters.chain.describe() in evaluated:
            # The iteration has landed back on a chain already designed against.
            # Another pass would reproduce the same numbers at the same cost.
            break
        evaluated.add(candidate_filters.chain.describe())

        try:
            candidate = design_gains(
                airframe,
                candidate_filters.chain,
                stack=stack,
                delay=delay,
                targets=targets,
                f_grid=f_grid,
                op=op,
                axis=axis,
            )
        except ValueError as exc:
            rejected.append((candidate_filters.chain.describe(), str(exc)))
            break

        rms = _noise_rms(noise, candidate_filters.chain, candidate, op)
        if rms > noise_limit:
            # The designed D came out higher than the seed the filters were chosen
            # against, so the chain that looked quiet enough no longer is. Feeding
            # the real kd back in is the whole reason this loop exists.
            rejected.append(
                (
                    candidate_filters.chain.describe(),
                    f"D-term output noise {rms * 100.0:.1f}% exceeds the "
                    f"{noise_limit * 100.0:.0f}% limit at the designed D gain",
                )
            )
        elif best is None or candidate.margins.disturbance_rejection_bw_hz > (
            best[0].margins.disturbance_rejection_bw_hz
        ):
            best = (candidate, candidate_filters, rms)

        moved = abs(candidate.margins.crossover_hz - crossover) / max(crossover, 1e-9)
        crossover = candidate.margins.crossover_hz
        kd = candidate.gains.kd
        if moved < _CONVERGED:
            break

    if best is None:
        return JointResult(
            design=baseline,
            filters=unchanged_filters(
                baseline_chain,
                stack,
                "Filters left as flown: no candidate chain both held the noise limit "
                "and left the loop designable. The rejected alternatives and their "
                "reasons are listed below.",
                noise=noise,
                op=op,
                rejected=tuple(rejected),
            ),
            dterm_noise_rms=_noise_rms(noise, baseline_chain, baseline, op),
            passes=passes,
            rejected=tuple(rejected),
        )

    candidate, candidate_filters, rms = best
    baseline_drb = baseline.margins.disturbance_rejection_bw_hz
    candidate_drb = candidate.margins.disturbance_rejection_bw_hz
    if candidate_drb <= baseline_drb * (1.0 + _MIN_DRB_GAIN):
        rejected.append(
            (
                candidate_filters.chain.describe(),
                f"disturbance-rejection bandwidth {candidate_drb:.2f} Hz is no better "
                f"than the {baseline_drb:.2f} Hz the flown filters already give",
            )
        )
        return JointResult(
            design=baseline,
            filters=unchanged_filters(
                baseline_chain,
                stack,
                "Filters left as flown deliberately: the alternatives were designed "
                "and none of them bought enough bandwidth to be worth a flight. The "
                "filters this vehicle already has are the right ones.",
                noise=noise,
                op=op,
                rejected=tuple(rejected),
            ),
            dterm_noise_rms=_noise_rms(noise, baseline_chain, baseline, op),
            passes=passes,
            rejected=tuple(rejected),
        )

    return JointResult(
        design=candidate,
        filters=replace(candidate_filters, rejected=candidate_filters.rejected + tuple(rejected)),
        dterm_noise_rms=rms,
        passes=passes,
        rejected=tuple(rejected),
    )


def unchanged_filters(
    chain: FilterChain,
    stack: Stack,
    rationale: str,
    *,
    noise: NoiseProfile | None = None,
    op: OperatingPoint | None = None,
    rejected: tuple[tuple[str, str], ...] = (),
) -> FilterRecommendation:
    """The flown chain, presented as a decision rather than as a gap.

    Returning the identical object for both ``chain`` and ``baseline_chain`` is
    the signal downstream code checks to know that nothing is being asked for.
    The measured spectrum is carried anyway: "we looked and the filters you have
    are right" is a far more useful thing to show than a blank panel, and it is
    only believable with the picture attached.
    """
    spectrum: dict[str, object] = {}
    if noise is not None and noise.psd_pre is not None:
        spectrum = {
            "psd_f_hz": noise.f_hz,
            "psd_pre": noise.psd_pre,
            "predicted_psd_post": noise.psd_pre
            * np.abs(chain.sensor_response(noise.f_hz, op)) ** 2,
        }
    return FilterRecommendation(
        stack=stack,
        chain=chain,
        baseline_chain=chain,
        params={},
        phase_cost_deg=float(chain.phase_deg(np.array([_PHASE_PROBE_HZ]), op)[0]),
        cpu_cost_rel=chain.cpu_cost(),
        rationale=rationale,
        rejected=rejected,
        **spectrum,  # type: ignore[arg-type]
    )


def _noise_rms(
    noise: NoiseProfile | None,
    chain: FilterChain,
    design: DesignResult,
    op: OperatingPoint,
) -> float:
    """D-term output RMS for one design, or NaN where there is no spectrum to use."""
    if noise is None or noise.psd_pre is None:
        return float("nan")
    return dterm_noise_rms(noise.f_hz, noise.psd_pre, chain, kd=design.gains.kd, op=op)
