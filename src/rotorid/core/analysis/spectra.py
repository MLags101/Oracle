"""Non-parametric frequency response estimation (spec section 5.4).

Everything downstream -- the airframe fit, the margins, the recommendation -- is
only as good as the FRF, and the FRF is only trustworthy where coherence says it
is. So the estimator never returns a bare transfer function: it returns the
spectra it was built from, the coherence, and an explicit mask of which bins may
be used.

Averaging across segments and across files is done by summing the *spectra*, not
by averaging finished transfer functions. Averaging ``H`` directly throws away the
information about how much each estimate should be trusted, and averaging
coherences is simply wrong -- coherence is a ratio of averaged quantities, so it
has to be recomputed from the sums.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import csd, get_window, welch

from rotorid.core.types import BoolArray, ComplexArray, FloatArray, FrequencyResponse

__all__ = [
    "SpectralEstimate",
    "choose_nperseg",
    "combine",
    "estimate_frf",
    "log_smooth",
    "power_spectrum",
]

#: Welch overlap fraction. One half is the standard choice for a Hann window: it
#: makes the window sum to a constant, so no part of the record is under-weighted.
_OVERLAP_FRACTION = 0.5

#: Cycles of the lowest frequency of interest that must fit inside one Welch
#: segment for that frequency to be resolved at all.
_CYCLES_TO_RESOLVE = 2.0


@dataclass(frozen=True, slots=True)
class SpectralEstimate:
    """Averaged auto- and cross-spectra for one input/output pair.

    Kept as spectra rather than as a finished FRF so that several segments or
    files can be merged correctly (:func:`combine`) before anything is divided.
    """

    f_hz: FloatArray
    Pxx: FloatArray
    Pyy: FloatArray
    Pxy: ComplexArray
    n_segments: int
    input_signal: str
    output_signal: str

    @property
    def H(self) -> ComplexArray:
        """The transfer-function estimate ``Pxy / Pxx`` (the H1 estimator).

        H1 is biased low by noise on the *input*; on both stacks the input is a
        logged command rather than a measurement, so that noise is negligible and
        H1 is the right choice.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            H = np.where(self.Pxx > 0.0, self.Pxy / self.Pxx, 0.0)
        return np.asarray(H, dtype=np.complex128)

    @property
    def coherence(self) -> FloatArray:
        """Ordinary coherence, recomputed from the averaged spectra."""
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = self.Pxx * self.Pyy
            gamma2 = np.where(denom > 0.0, np.abs(self.Pxy) ** 2 / denom, 0.0)
        return np.asarray(np.clip(gamma2, 0.0, 1.0), dtype=np.float64)

    def to_frf(
        self,
        *,
        coherence_threshold: float,
        band_hz: tuple[float, float] | None = None,
    ) -> FrequencyResponse:
        """Gate the estimate and hand back the downstream contract.

        Args:
            coherence_threshold: ``[coherence].threshold``.
            band_hz: Excited band. Bins outside it are invalid even if coherent,
                because coherence can be high on a frequency nothing excited.
        """
        gamma2 = self.coherence
        valid: BoolArray = (gamma2 >= coherence_threshold) & (self.f_hz > 0.0)
        if band_hz is not None:
            valid = valid & (self.f_hz >= band_hz[0]) & (self.f_hz <= band_hz[1])
        return FrequencyResponse(
            f_hz=self.f_hz,
            H=self.H,
            coherence=gamma2,
            valid_mask=np.asarray(valid, dtype=np.bool_),
            input_signal=self.input_signal,
            output_signal=self.output_signal,
            n_segments_averaged=self.n_segments,
        )


def choose_nperseg(
    n_samples: int,
    sample_rate_hz: float,
    *,
    f_lowest_hz: float,
    min_averages: int,
) -> int:
    """Pick a Welch segment length trading resolution against variance.

    Two competing requirements: the segment must be long enough that
    ``f_lowest_hz`` fits inside it a few times, and short enough that at least
    ``min_averages`` of them fit in the record. Resolution wins -- an FRF averaged
    only three times is noisy but usable, whereas one that cannot see the lowest
    excited frequency is useless -- and the shortfall is visible to the caller as a
    low ``n_segments_averaged``.

    Returns:
        A power of two, at most the record length.

    Raises:
        ValueError: if the record is too short to resolve ``f_lowest_hz`` at all.
    """
    if f_lowest_hz <= 0.0:
        raise ValueError("f_lowest_hz must be positive")

    needed = _CYCLES_TO_RESOLVE * sample_rate_hz / f_lowest_hz
    if needed > n_samples:
        raise ValueError(
            f"record of {n_samples / sample_rate_hz:.1f} s cannot resolve {f_lowest_hz:g} Hz; "
            f"need at least {needed / sample_rate_hz:.1f} s"
        )

    # With 50% overlap, k segments of length L span L*(k+1)/2 samples.
    allowed = 2.0 * n_samples / (min_averages + 1)
    target = max(needed, min(allowed, float(n_samples)))
    nperseg = 1 << int(np.floor(np.log2(target)))
    if nperseg < needed:  # rounding down to a power of two lost the low frequency
        nperseg <<= 1
    return int(min(nperseg, n_samples))


def estimate_frf(
    u: FloatArray,
    y: FloatArray,
    sample_rate_hz: float,
    *,
    nperseg: int,
    input_signal: str,
    output_signal: str,
    detrend: str = "linear",
) -> SpectralEstimate:
    """Welch/CSD estimate of the response from ``u`` to ``y``.

    Detrending defaults to linear: a rate signal with a slow drift in it puts
    energy at DC that leaks across the whole low-frequency end, which is exactly
    where the airframe gain is identified.

    Raises:
        ValueError: if the two signals differ in length.
    """
    if u.shape != y.shape:
        raise ValueError(f"input and output lengths differ: {u.shape} vs {y.shape}")

    window = get_window("hann", nperseg)
    noverlap = int(nperseg * _OVERLAP_FRACTION)
    kwargs = {
        "fs": sample_rate_hz,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "detrend": detrend,
    }
    f, Pxx = welch(u, **kwargs)
    _, Pyy = welch(y, **kwargs)
    _, Pxy = csd(u, y, **kwargs)

    step = nperseg - noverlap
    n_segments = max(1, (u.size - noverlap) // step)

    return SpectralEstimate(
        f_hz=np.asarray(f, dtype=np.float64),
        Pxx=np.asarray(Pxx, dtype=np.float64),
        Pyy=np.asarray(Pyy, dtype=np.float64),
        Pxy=np.asarray(Pxy, dtype=np.complex128),
        n_segments=n_segments,
        input_signal=input_signal,
        output_signal=output_signal,
    )


def power_spectrum(
    y: FloatArray, sample_rate_hz: float, *, nperseg: int, detrend: str = "linear"
) -> tuple[FloatArray, FloatArray]:
    """One-sided PSD of a single signal, for noise analysis.

    Returns:
        ``(f_hz, psd)`` in units-squared per Hz.
    """
    f, pxx = welch(
        y,
        fs=sample_rate_hz,
        window=get_window("hann", nperseg),
        nperseg=nperseg,
        noverlap=int(nperseg * _OVERLAP_FRACTION),
        detrend=detrend,
    )
    return np.asarray(f, dtype=np.float64), np.asarray(pxx, dtype=np.float64)


def combine(estimates: Sequence[SpectralEstimate]) -> SpectralEstimate:
    """Merge several estimates of the same response by summing their spectra.

    Segments and whole flights are merged the same way, which is what makes the
    ArduPilot practice of averaging repeated sweeps work here. Each estimate is
    weighted by how many Welch averages went into it, so a 30 s segment does not
    carry the same weight as a 130 s one.

    Raises:
        ValueError: if the estimates are empty, disagree about which signals they
            describe, or sit on different frequency grids.
    """
    if not estimates:
        raise ValueError("nothing to combine")
    first = estimates[0]
    if len(estimates) == 1:
        return first

    for e in estimates[1:]:
        if e.f_hz.shape != first.f_hz.shape or not np.allclose(e.f_hz, first.f_hz):
            raise ValueError(
                "cannot combine estimates on different frequency grids; "
                "use the same nperseg and sample rate for every segment"
            )
        if (e.input_signal, e.output_signal) != (first.input_signal, first.output_signal):
            raise ValueError(
                f"cannot combine {e.input_signal}->{e.output_signal} with "
                f"{first.input_signal}->{first.output_signal}"
            )

    weights = np.array([float(e.n_segments) for e in estimates])
    total = float(weights.sum())
    Pxx = sum(w * e.Pxx for w, e in zip(weights, estimates, strict=True)) / total
    Pyy = sum(w * e.Pyy for w, e in zip(weights, estimates, strict=True)) / total
    Pxy = sum(w * e.Pxy for w, e in zip(weights, estimates, strict=True)) / total

    return SpectralEstimate(
        f_hz=first.f_hz,
        Pxx=np.asarray(Pxx, dtype=np.float64),
        Pyy=np.asarray(Pyy, dtype=np.float64),
        Pxy=np.asarray(Pxy, dtype=np.complex128),
        n_segments=int(total),
        input_signal=first.input_signal,
        output_signal=first.output_signal,
    )


def log_smooth(estimate: SpectralEstimate, *, points_per_decade: int = 40) -> SpectralEstimate:
    """Re-bin onto a log-spaced grid, averaging the spectra inside each bin.

    A linear FFT grid puts almost all of its points in the top octave, where they
    are noisiest, and almost none at the bottom, where the airframe gain lives.
    Averaging into log-spaced bins fixes both problems at once: it reduces
    high-frequency variance and gives the fit a grid whose density matches how the
    weighting already treats frequency.

    Bins containing no FFT point are dropped rather than interpolated -- an
    invented bin would carry a coherence it never earned.

    **Do not smooth before fitting a model with a delay in it.** Averaging the
    cross-spectrum inside a bin averages complex numbers whose phase is rotating
    at ``360 * tau`` degrees per Hz. At 50 Hz with 18 ms of delay and 30 bins per
    decade that is roughly 23 degrees of rotation across a single bin, which
    biases the averaged magnitude down and drags the identified delay with it --
    measurably, in the third significant figure of ``tau``. Fitting runs on the
    raw grid, where :func:`rotorid.core.analysis.sysid.fit_weights` handles the
    decade balance instead. Smoothing is for display and for noise work.
    """
    f = estimate.f_hz
    positive = f > 0.0
    if not positive.any():
        raise ValueError("estimate has no positive frequencies to smooth")

    f_pos = f[positive]
    edges = np.logspace(
        np.log10(f_pos[0]),
        np.log10(f_pos[-1]),
        int(points_per_decade * np.log10(f_pos[-1] / f_pos[0])) + 2,
    )
    index = np.digitize(f_pos, edges) - 1

    f_out: list[float] = []
    pxx_out: list[float] = []
    pyy_out: list[float] = []
    pxy_out: list[complex] = []
    for b in range(len(edges) - 1):
        sel = index == b
        if not sel.any():
            continue
        f_out.append(float(np.mean(f_pos[sel])))
        pxx_out.append(float(np.mean(estimate.Pxx[positive][sel])))
        pyy_out.append(float(np.mean(estimate.Pyy[positive][sel])))
        pxy_out.append(complex(np.mean(estimate.Pxy[positive][sel])))

    return SpectralEstimate(
        f_hz=np.asarray(f_out, dtype=np.float64),
        Pxx=np.asarray(pxx_out, dtype=np.float64),
        Pyy=np.asarray(pyy_out, dtype=np.float64),
        Pxy=np.asarray(pxy_out, dtype=np.complex128),
        n_segments=estimate.n_segments,
        input_signal=estimate.input_signal,
        output_signal=estimate.output_signal,
    )
