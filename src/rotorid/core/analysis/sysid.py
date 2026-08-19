"""Effective plant to airframe model (spec section 5.3 steps 3-5, and 5.4).

This module owns the single most dangerous operation in the tool: dividing the
vehicle's filter chain out of the measured response. Get it wrong in one
direction and every recommended gain is too timid; wrong in the other and the
aircraft oscillates. So it happens exactly once, here, in :func:`deconvolve`, and
the result is labelled with how it was obtained.

The corresponding multiply -- putting a *candidate* chain back into the loop --
lives in the design package. Two sites, no more.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from rotorid.core.analysis.model_eval import airframe_response
from rotorid.core.filters.chain import FilterChain, OperatingPoint
from rotorid.core.types import (
    AirframeModel,
    Axis,
    BoolArray,
    ComplexArray,
    EffectivePlant,
    FloatArray,
)

__all__ = [
    "DeconvolvedPlant",
    "FilterModelCheck",
    "check_filter_model",
    "deconvolve",
    "fit_airframe",
    "fit_weights",
]

#: Phase error is real information but noisier than magnitude, and a fit that
#: chases it will trade away gain accuracy. Half weight, per spec 5.4.
_PHASE_WEIGHT = 0.5


@dataclass(frozen=True, slots=True)
class DeconvolvedPlant:
    """``G_air`` on the FRF grid, with the bins the division ruined marked invalid."""

    axis: Axis
    f_hz: FloatArray
    G: ComplexArray
    coherence: FloatArray
    valid_mask: BoolArray
    filter_deconvolution: str
    n_bins_lost_to_notches: int

    @property
    def valid_band_hz(self) -> tuple[float, float]:
        """Lowest and highest usable frequency.

        Raises:
            ValueError: if nothing survived.
        """
        if not self.valid_mask.any():
            raise ValueError("no frequency bin survived deconvolution")
        f = self.f_hz[self.valid_mask]
        return float(f[0]), float(f[-1])


def deconvolve(
    plant: EffectivePlant,
    chain: FilterChain | None,
    *,
    op: OperatingPoint | None = None,
    floor_db: float,
) -> DeconvolvedPlant:
    """Divide the modeled filter chain out of the measured effective plant.

    ``G_air(jw) = EffectivePlant(jw) / F_current(jw)``.

    Inside a deep notch ``|F|`` is tiny and the division amplifies whatever noise
    is there into an enormous, entirely fictitious resonance. Those bins are
    marked invalid rather than clamped to something plausible-looking: a fit is
    allowed to have no information at a frequency, but it must never be given
    fabricated information.

    Args:
        plant: The measurement. Must have ``filters_included`` set truthfully --
            a raw-gyro measurement passed here would be filtered twice.
        chain: The chain the log was recorded through, or ``None`` if the
            measurement is already pre-filter.
        op: Operating point for tracked notch centres.
        floor_db: ``[filters].deconv_floor_db``. Bins where ``|F|`` falls this far
            below unity are dropped.

    Raises:
        ValueError: if asked to divide filters out of a signal that never had any.
    """
    frf = plant.frf
    if chain is None or not plant.filters_included:
        if chain is not None and not plant.filters_included:
            raise ValueError(
                "refusing to divide a filter chain out of a measurement marked "
                "filters_included=False; that would count the chain twice with the "
                "wrong sign (spec 5.3)"
            )
        return DeconvolvedPlant(
            axis=plant.axis,
            f_hz=frf.f_hz,
            G=frf.H,
            coherence=frf.coherence,
            valid_mask=frf.valid_mask,
            filter_deconvolution="raw_gyro" if plant.source == "raw_gyro" else "none",
            n_bins_lost_to_notches=0,
        )

    F = chain.sensor_response(frf.f_hz, op)
    floor = 10.0 ** (floor_db / 20.0)
    usable: BoolArray = _chain_usable_across_bin(chain, frf.f_hz, op, floor)

    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(usable, frf.H / F, 0.0)

    valid = frf.valid_mask & usable & np.isfinite(G)
    return DeconvolvedPlant(
        axis=plant.axis,
        f_hz=frf.f_hz,
        G=np.asarray(G, dtype=np.complex128),
        coherence=frf.coherence,
        valid_mask=np.asarray(valid, dtype=np.bool_),
        filter_deconvolution="modeled",
        n_bins_lost_to_notches=int(np.sum(frf.valid_mask & ~usable)),
    )


#: Points sampled across each frequency bin when testing whether a notch lies
#: inside it. A notch 40 dB deep can be only a fraction of a Hz wide at its
#: bottom, so a single sample at the bin centre steps straight over it.
_BIN_SUBSAMPLES = 9


def _chain_usable_across_bin(
    chain: FilterChain,
    f_hz: FloatArray,
    op: OperatingPoint | None,
    floor: float,
) -> BoolArray:
    """Which bins the chain can safely be divided out of.

    A bin is unusable if the chain drops below ``floor`` *anywhere inside it*, not
    merely at its centre. This matters because the measurement is averaged over
    the bin while the model is evaluated at a point: a deep, narrow notch sitting
    between two grid points is invisible to the model and has already flattened
    the measurement, so dividing there produces a plausible-looking number with
    no basis in the data.
    """
    if f_hz.size < 2:
        return np.asarray(np.abs(chain.sensor_response(f_hz, op)) >= floor, dtype=np.bool_)

    half = 0.5 * np.gradient(f_hz)
    worst = np.full(f_hz.shape, np.inf)
    for frac in np.linspace(-1.0, 1.0, _BIN_SUBSAMPLES):
        probe = np.clip(f_hz + frac * half, 0.0, None)
        worst = np.minimum(worst, np.abs(chain.sensor_response(probe, op)))
    return np.asarray(worst >= floor, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class FilterModelCheck:
    """Whether the modeled chain actually matches what the log shows.

    ``checkable`` is a first-class outcome, not a failure. A 0.5-5 Hz sweep
    excites nothing at all near a 90 Hz notch, so on most identification records
    there is simply no evidence either way -- and reporting "verified" in that
    case would be a lie that the rest of the pipeline would then rely on.
    """

    checkable: bool
    max_magnitude_error_db: float
    n_bins_compared: int
    reason: str = ""

    @property
    def agrees(self) -> bool:
        """True only when there was evidence *and* it matched."""
        return self.checkable and self.max_magnitude_error_db < np.inf


def check_filter_model(
    plant: EffectivePlant,
    chain: FilterChain,
    airframe_guess: AirframeModel | None = None,
    *,
    op: OperatingPoint | None = None,
    band_hz: tuple[float, float] | None = None,
) -> FilterModelCheck:
    """Compare measured effective-plant shape against the modeled chain (spec 5.3.3).

    The comparison is only meaningful where the measurement is coherent, so where
    the excitation never reached, this reports ``checkable=False`` and the caller
    downgrades confidence instead of trusting an unverified chain.

    Args:
        airframe_guess: A preliminary airframe fit, used to remove the plant's own
            roll-off before comparing. Without it only the notch *shape* can be
            compared, not its depth.
    """
    frf = plant.frf
    sel = frf.valid_mask.copy()
    if band_hz is not None:
        sel &= (frf.f_hz >= band_hz[0]) & (frf.f_hz <= band_hz[1])
    if not sel.any():
        return FilterModelCheck(
            checkable=False,
            max_magnitude_error_db=float("nan"),
            n_bins_compared=0,
            reason="no coherent bins in the comparison band; the sweep never went there",
        )

    predicted = chain.sensor_response(frf.f_hz, op)
    if airframe_guess is not None:
        predicted = predicted * airframe_response(airframe_guess, frf.f_hz)
    else:
        # Without a plant model, compare only the *relative* shape: normalize both
        # to their own median so a constant gain difference is not read as a
        # filter mismatch.
        predicted = predicted / np.median(np.abs(predicted[sel]))

    measured = frf.H
    scale = np.median(np.abs(measured[sel])) if airframe_guess is None else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        err_db = 20.0 * np.log10(np.abs(measured[sel] / scale) / np.abs(predicted[sel]))
    err_db = err_db[np.isfinite(err_db)]
    if err_db.size == 0:
        return FilterModelCheck(
            checkable=False,
            max_magnitude_error_db=float("nan"),
            n_bins_compared=0,
            reason="comparison produced no finite bins",
        )

    return FilterModelCheck(
        checkable=True,
        max_magnitude_error_db=float(np.max(np.abs(err_db))),
        n_bins_compared=int(err_db.size),
    )


def fit_weights(f_hz: FloatArray, coherence: FloatArray) -> FloatArray:
    """Per-bin fit weights: coherence times the bin's width in log-frequency.

    The point of the log-frequency term is to give every *decade* the same say in
    the fit, whatever grid the bins happen to sit on. On a linear FFT grid that
    works out to the familiar ``1/f``, since a bin of constant width covers less
    and less of a decade as ``f`` rises.

    Computing it as an actual bin width rather than hard-coding ``1/f`` is what
    makes it correct on a log-spaced grid too. Applying ``1/f`` to a grid that is
    already log-spaced weights the decades a second time: in a 0.5-60 Hz fit that
    put roughly three quarters of the weight below 2 Hz, and the resonance --
    which is the whole point of the identification -- was fitted on the leftovers.
    """
    valid = f_hz > 0.0
    if valid.sum() < 2:
        return np.asarray(coherence * valid, dtype=np.float64)
    dlnf = np.zeros_like(f_hz)
    dlnf[valid] = np.gradient(np.log(f_hz[valid]))
    w = coherence * np.abs(dlnf)
    total = float(np.sum(w))
    return np.asarray(w / total if total > 0.0 else w, dtype=np.float64)


def _residuals(
    theta: FloatArray,
    structure: str,
    axis: Axis,
    f_hz: FloatArray,
    G: ComplexArray,
    weights: FloatArray,
) -> FloatArray:
    model = _model_from_theta(theta, structure, axis)
    H = airframe_response(model, f_hz)

    with np.errstate(divide="ignore", invalid="ignore"):
        mag_db = 20.0 * np.log10(np.abs(G) / np.abs(H))
    # Wrapped phase difference: always in (-180, 180], so no unwrapping choice can
    # silently add a full turn of error.
    phase_deg = np.degrees(np.angle(G * np.conj(H)))

    mag_db = np.nan_to_num(mag_db, nan=0.0, posinf=0.0, neginf=0.0)
    root_w = np.sqrt(weights)
    return np.concatenate([root_w * mag_db, root_w * _PHASE_WEIGHT * phase_deg])


def _model_from_theta(theta: FloatArray, structure: str, axis: Axis) -> AirframeModel:
    if structure == "so_delay":
        params = {"K": theta[0], "wn": theta[1], "zeta": theta[2], "tau": theta[3]}
    elif structure == "fo_delay":
        params = {"K": theta[0], "T": theta[1], "tau": theta[2]}
    else:  # pragma: no cover - guarded by fit_airframe
        raise ValueError(f"unsupported structure {structure!r}")
    return AirframeModel(
        axis=axis,
        structure=structure,  # type: ignore[arg-type]
        params=params,
        fit_rms_db=0.0,
        fit_rms_deg=0.0,
        valid_band_hz=(0.0, 0.0),
        coherence_mean=0.0,
        filter_deconvolution="none",
    )


def _initial_gain(f_hz: FloatArray, G: ComplexArray) -> float:
    """DC gain from the lowest decade of usable data."""
    low = f_hz <= max(f_hz[0] * 3.0, f_hz[0] + 1e-9)
    sel = low if low.any() else np.ones_like(f_hz, dtype=bool)
    return float(np.median(np.abs(G[sel])))


def fit_airframe(
    plant: DeconvolvedPlant,
    *,
    structure: str = "so_delay",
    wn_bounds_hz: tuple[float, float],
    zeta_bounds: tuple[float, float],
    tau_bounds_ms: tuple[float, float],
) -> AirframeModel:
    """Weighted nonlinear least-squares fit of a parametric airframe model.

    Multi-start over a coarse grid, because the delay term makes the cost surface
    multi-modal: a fit can trade a full turn of delay phase against a shifted
    resonance and land in a local minimum that looks fine in magnitude and is
    badly wrong in ``tau``.

    ``tau`` here is what is left *after* the filter chain was divided out --
    actuator, ESC and motor lag. It is never filter lag, and nothing downstream
    may add filter delay to it (spec 5.4).

    Raises:
        ValueError: if fewer usable bins than free parameters, or an unknown
            structure is requested.
    """
    if structure not in ("so_delay", "fo_delay"):
        raise ValueError(f"unsupported structure {structure!r}; use so_delay or fo_delay")

    sel = plant.valid_mask
    f = plant.f_hz[sel]
    G = plant.G[sel]
    weights = fit_weights(f, plant.coherence[sel])
    n_free = 4 if structure == "so_delay" else 3
    if f.size <= n_free:
        raise ValueError(
            f"{f.size} usable frequency bins is not enough to fit {n_free} parameters; "
            "the coherence gate rejected almost everything"
        )

    K0 = _initial_gain(f, G)
    tau_lo, tau_hi = tau_bounds_ms[0] / 1000.0, tau_bounds_ms[1] / 1000.0
    wn_lo, wn_hi = 2.0 * np.pi * wn_bounds_hz[0], 2.0 * np.pi * wn_bounds_hz[1]

    if structure == "so_delay":
        lower = np.array([K0 * 1e-3, wn_lo, zeta_bounds[0], tau_lo])
        upper = np.array([K0 * 1e3, wn_hi, zeta_bounds[1], tau_hi])
        starts = [
            np.array([K0, wn, zeta, tau])
            for wn in np.geomspace(wn_lo * 1.2, wn_hi * 0.8, 4)
            for zeta in (0.3, 0.8)
            for tau in (tau_lo * 1.5, np.sqrt(tau_lo * tau_hi), tau_hi * 0.7)
        ]
    else:
        T_lo, T_hi = 1.0 / wn_hi, 1.0 / wn_lo
        lower = np.array([K0 * 1e-3, T_lo, tau_lo])
        upper = np.array([K0 * 1e3, T_hi, tau_hi])
        starts = [
            np.array([K0, T, tau])
            for T in np.geomspace(T_lo * 1.2, T_hi * 0.8, 5)
            for tau in (tau_lo * 1.5, np.sqrt(tau_lo * tau_hi), tau_hi * 0.7)
        ]

    best: FloatArray | None = None
    best_cost = np.inf
    for theta0 in starts:
        theta0 = np.clip(theta0, lower * (1.0 + 1e-9), upper * (1.0 - 1e-9))
        result = least_squares(
            _residuals,
            theta0,
            bounds=(lower, upper),
            args=(structure, plant.axis, f, G, weights),
            xtol=1e-12,
            ftol=1e-12,
            max_nfev=2000,
        )
        if result.cost < best_cost:
            best_cost, best = float(result.cost), np.asarray(result.x, dtype=np.float64)

    assert best is not None  # starts is never empty
    model = _model_from_theta(best, structure, plant.axis)

    H = airframe_response(model, f)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag_db = np.nan_to_num(20.0 * np.log10(np.abs(G) / np.abs(H)))
    phase_deg = np.degrees(np.angle(G * np.conj(H)))
    w = weights / np.sum(weights)

    return AirframeModel(
        axis=plant.axis,
        structure=structure,  # type: ignore[arg-type]
        params=model.params,
        fit_rms_db=float(np.sqrt(np.sum(w * mag_db**2))),
        fit_rms_deg=float(np.sqrt(np.sum(w * phase_deg**2))),
        valid_band_hz=(float(f[0]), float(f[-1])),
        coherence_mean=float(np.mean(plant.coherence[sel])),
        filter_deconvolution=plant.filter_deconvolution,  # type: ignore[arg-type]
    )
