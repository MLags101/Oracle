"""How much the airframe moves under the tune (spec section 5.9).

A rate loop is designed against one number for the airframe gain `K`, and on a
real vehicle that number is not one number. It moves with throttle, because
thrust is not linear in motor command; with battery voltage, because a sagging
pack turns the same command into less thrust; with payload, and with how chewed
up the props are. A tune designed to a 45-degree phase margin at one operating
point and flown across a 30% gain spread does not have a 45-degree phase margin.

The measurement here is deliberately relative. Identifying a separate airframe
per segment would be the direct approach and is the wrong one: each segment is
short, so each fit is poor, and the spread would then be mostly fit noise
wearing the name of a physical effect. Instead the *shape* identified from the
whole flight is held fixed and only its gain is allowed to move, which is the
one parameter a change of operating point actually moves and the one a short
segment can pin down.

This is the analysis a general flight offers and a sweep cannot: a two-minute
SYSTEMID sweep is flown at one throttle, on one battery state, so there is
nothing to compare. An ordinary flight visits the envelope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rotorid.core.analysis.sysid import DeconvolvedPlant
from rotorid.core.types import AirframeModel, ExcitationSegment, FloatArray

__all__ = ["GainSample", "OperatingPointSpread", "gain_spread"]

#: Least number of usable segments before a spread means anything. Two points
#: define a line through any two points; a correlation drawn from them is not
#: evidence of anything and would be quoted as though it were.
_MIN_SAMPLES = 3

#: Bins below this coherence are ignored when measuring a segment's gain. The
#: segment already passed the identification's own gate as a whole; this is the
#: second, per-bin one, because a segment can be usable overall and still have
#: nothing to say at the bottom of the band.
_MIN_COHERENCE = 0.6

#: Correlation strength above which a spread is attributed to a variable rather
#: than reported as unexplained. Pearson r on a handful of points is a blunt
#: instrument, so the bar is set where the relationship would be visible by eye.
_ATTRIBUTION_R = 0.7


@dataclass(frozen=True, slots=True)
class GainSample:
    """The airframe gain measured over one segment, and where that segment sat.

    Attributes:
        gain_ratio: This segment's gain as a multiple of the flight-wide fit. 1.0
            means the segment agrees with the model everybody else produced.
        throttle: Mean normalized throttle over the segment, or ``None`` when the
            log carries no motor outputs.
        voltage: Mean pack voltage over the segment, or ``None`` without ``BAT``.
        n_bins: Coherent bins the ratio was measured over. Carried because a
            ratio from four bins and one from four hundred are different claims.
    """

    segment: ExcitationSegment
    gain_ratio: float
    throttle: float | None
    voltage: float | None
    n_bins: int


@dataclass(frozen=True, slots=True)
class OperatingPointSpread:
    """How far the airframe gain moved across the flight, and with what.

    Attributes:
        spread_pct: Peak-to-peak spread as a percentage of the mean. The number
            that goes on :attr:`~rotorid.core.types.AirframeModel.gain_spread_pct`
            and into how much margin the design holds back.
        throttle_r: Pearson correlation of gain against throttle, or ``None`` if
            throttle was not logged. Signed: positive means the vehicle gets
            *more* responsive as it is pushed, which is what an under-compensated
            thrust curve looks like.
        voltage_r: The same against pack voltage.
        samples: One per segment, so the report can draw the scatter rather than
            asking the reader to trust a single percentage.
    """

    spread_pct: float
    throttle_r: float | None
    voltage_r: float | None
    samples: tuple[GainSample, ...]

    @property
    def attributed_to_throttle(self) -> bool:
        """Whether the spread tracks throttle strongly enough to name it."""
        return self.throttle_r is not None and abs(self.throttle_r) >= _ATTRIBUTION_R

    @property
    def attributed_to_voltage(self) -> bool:
        """Whether the spread tracks pack voltage strongly enough to name it."""
        return self.voltage_r is not None and abs(self.voltage_r) >= _ATTRIBUTION_R

    def describe(self) -> str:
        """One sentence naming the spread and what it moved with."""
        if self.attributed_to_throttle and self.attributed_to_voltage:
            with_what = "with both throttle and pack voltage"
        elif self.attributed_to_throttle:
            with_what = "with throttle"
        elif self.attributed_to_voltage:
            with_what = "with pack voltage"
        else:
            with_what = "with nothing this log measured"
        return (
            f"airframe gain moved {self.spread_pct:.0f}% across "
            f"{len(self.samples)} operating points, {with_what}"
        )


def gain_spread(
    model: AirframeModel,
    per_segment: dict[ExcitationSegment, DeconvolvedPlant],
    throttle: dict[ExcitationSegment, float] | None = None,
    voltage: dict[ExcitationSegment, float] | None = None,
) -> OperatingPointSpread | None:
    """Measure how the airframe gain moved between segments.

    Args:
        model: The flight-wide fit. Its shape is held fixed; only the scale in
            front of it is allowed to differ per segment.
        per_segment: One deconvolved plant per segment, on the same terms as the
            one ``model`` was fitted to -- filters divided out at *that* segment's
            operating point, and the loop divided out. Anything else and the
            "gain spread" would include the tracked notches moving with throttle,
            which is a real effect and a different one.

    Returns:
        The spread, or ``None`` when fewer than three segments produced a usable
        ratio. Refusing is the point: a spread quoted from two segments would
        be indistinguishable in the report from one measured across a flight.
    """
    samples: list[GainSample] = []
    for segment, plant in per_segment.items():
        ratio, n_bins = _gain_ratio(model, plant)
        if ratio is None:
            continue
        samples.append(
            GainSample(
                segment=segment,
                gain_ratio=ratio,
                throttle=(throttle or {}).get(segment),
                voltage=(voltage or {}).get(segment),
                n_bins=n_bins,
            )
        )

    if len(samples) < _MIN_SAMPLES:
        return None

    ratios = np.array([s.gain_ratio for s in samples], dtype=np.float64)
    mean = float(np.mean(ratios))
    if mean <= 0.0:
        return None
    spread_pct = 100.0 * float(np.ptp(ratios)) / mean

    return OperatingPointSpread(
        spread_pct=spread_pct,
        throttle_r=_correlate(ratios, [s.throttle for s in samples]),
        voltage_r=_correlate(ratios, [s.voltage for s in samples]),
        samples=tuple(sorted(samples, key=lambda s: s.segment.t_start)),
    )


def _gain_ratio(model: AirframeModel, plant: DeconvolvedPlant) -> tuple[float | None, int]:
    """This segment's gain as a multiple of the model's, over the valid band.

    The *median* of the per-bin magnitude ratios, not a least-squares scale. A
    single resonant bin where the segment happened to be excited and the model is
    a smooth curve would drag a least-squares answer a long way; the median
    ignores it, and the quantity being estimated -- a scalar multiplying an
    otherwise-agreed shape -- is exactly the kind that a median estimates well.
    """
    band = model.valid_band_hz
    inside = (
        plant.valid_mask
        & (plant.f_hz >= band[0])
        & (plant.f_hz <= band[1])
        & (plant.coherence >= _MIN_COHERENCE)
    )
    if not inside.any():
        return None, 0

    measured = np.abs(plant.G[inside])
    modelled = np.abs(model.response(plant.f_hz[inside]))
    usable = (modelled > 0.0) & np.isfinite(measured) & (measured > 0.0)
    if not usable.any():
        return None, 0
    return float(np.median(measured[usable] / modelled[usable])), int(np.count_nonzero(usable))


def _correlate(ratios: FloatArray, against: list[float | None]) -> float | None:
    """Pearson r, or ``None`` when the variable was not logged or never moved.

    A variable that held still across the flight is not uncorrelated with the
    gain -- it is silent about it, and the two have to read differently. A
    coefficient of zero would be read as "throttle does not affect this
    aircraft", which is a claim this log cannot make.
    """
    if any(value is None for value in against):
        return None
    values = np.array(against, dtype=np.float64)
    if float(np.std(values)) <= 0.0 or float(np.std(ratios)) <= 0.0:
        return None
    return float(np.corrcoef(values, ratios)[0, 1])
