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
    "InstrumentedEstimate",
    "SpectralEstimate",
    "choose_nperseg",
    "combine",
    "combine_iv",
    "estimate_frf",
    "estimate_frf_iv",
    "log_smooth",
    "power_spectrum",
]

#: Welch overlap fraction. One half is the standard choice for a Hann window: it
#: makes the window sum to a constant, so no part of the record is under-weighted.
_OVERLAP_FRACTION = 0.5

#: Cycles of the lowest frequency of interest that must fit inside one Welch
#: segment for that frequency to be resolved at all.
_CYCLES_TO_RESOLVE = 2.0

#: How wide a gap in log frequency may be bridged when deciding which coherent
#: bins form one band. A tenth of a decade spans a notch's coherence dip without
#: reaching across the dead zone above where the excitation stopped.
_MAX_COHERENCE_GAP_DECADES = 0.1

#: How far below its own peak the excitation's power may fall before a frequency
#: stops counting as excited at all. Coherence cannot answer this question: it
#: asks whether the output is *explained* by the input, and two signals that are
#: both essentially zero can be explained by each other perfectly. A chirp is flat
#: across its sweep so this never bites; a pilot's stick falls away above a couple
#: of Hz, and this is what stops the identification claiming a band the pilot
#: never excited.
_EXCITATION_FLOOR_DB = -40.0


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

        H1 is biased low by noise on the *input*. On an open-loop measurement the
        input is a logged command rather than a measurement, so that noise is
        negligible and H1 is the right choice.

        **In closed loop it is not.** The mixer command is the controller's own
        output, so it contains the gyro noise fed back through the controller, and
        H1 is then biased towards ``-1/C`` -- an estimate of the inverse
        controller, wearing the coherence of a good measurement. That is what
        :func:`estimate_frf_iv` exists to avoid, and it is why every plant now
        records which estimator produced it.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            H = np.where(self.Pxx > 0.0, self.Pxy / self.Pxx, 0.0)
        return np.asarray(H, dtype=np.complex128)

    @property
    def coherence(self) -> FloatArray:
        """Ordinary coherence, recomputed from the averaged spectra."""
        return _coherence(self.Pxx, self.Pyy, self.Pxy)

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
        return _gate(
            f_hz=self.f_hz,
            H=self.H,
            coherence=self.coherence,
            coherence_threshold=coherence_threshold,
            band_hz=band_hz,
            input_signal=self.input_signal,
            output_signal=self.output_signal,
            n_segments=self.n_segments,
            excitation_power=self.Pxx,
        )


@dataclass(frozen=True, slots=True)
class InstrumentedEstimate:
    """Spectra for the instrument-variable (Joint Input-Output) estimator.

    Three signals rather than two: an exogenous instrument ``r``, the plant input
    ``u``, and the response ``y``. The plant is then

    .. code-block:: text

        G(jw) = (r -> y) / (r -> u) = Pry / Pru

    which is unbiased under feedback for *any* ``r`` that is uncorrelated with the
    measurement noise, whatever the controller is and wherever ``r`` enters the
    loop. Both ArduPilot injection points fall out of the same expression:

    ==================================  ============  =============  =====
    injection                           ``y/r``       ``u/r``        ratio
    ==================================  ============  =============  =====
    rate target (``SID_AXIS`` 7-9)      ``GC/(1+GC)`` ``C/(1+GC)``   ``G``
    mixer (``SID_AXIS`` 10-12)          ``G/(1+GC)``  ``1/(1+GC)``   ``G``
    ==================================  ============  =============  =====

    That generality is the point. It means the same estimator serves an injected
    chirp and a pilot's stick, and it means nothing here has to model the
    controller in order to divide it out.
    """

    f_hz: FloatArray
    Prr: FloatArray
    Puu: FloatArray
    Pyy: FloatArray
    Pru: ComplexArray
    Pry: ComplexArray
    n_segments: int
    instrument_signal: str
    input_signal: str
    output_signal: str

    @property
    def H(self) -> ComplexArray:
        """``Pry / Pru`` -- the plant, with the loop divided out."""
        with np.errstate(divide="ignore", invalid="ignore"):
            H = np.where(np.abs(self.Pru) > 0.0, self.Pry / self.Pru, 0.0)
        return np.asarray(H, dtype=np.complex128)

    @property
    def coherence_ru(self) -> FloatArray:
        """How much of the plant input the instrument accounts for."""
        return _coherence(self.Prr, self.Puu, self.Pru)

    @property
    def coherence_ry(self) -> FloatArray:
        """How much of the response the instrument accounts for."""
        return _coherence(self.Prr, self.Pyy, self.Pry)

    @property
    def coherence(self) -> FloatArray:
        """The weaker of the two, elementwise.

        The estimate is a *ratio*, so it is only as good as its worse half. A
        frequency where the stick clearly moved the aircraft but not measurably
        the mixer command has a well-determined numerator over a numerically
        meaningless denominator, and the quotient is noise -- which the ordinary
        coherence of either half on its own would not reveal.
        """
        return np.asarray(np.minimum(self.coherence_ru, self.coherence_ry), dtype=np.float64)

    def to_frf(
        self,
        *,
        coherence_threshold: float,
        band_hz: tuple[float, float] | None = None,
    ) -> FrequencyResponse:
        """Gate the estimate and hand back the downstream contract."""
        return _gate(
            f_hz=self.f_hz,
            H=self.H,
            coherence=self.coherence,
            coherence_threshold=coherence_threshold,
            band_hz=band_hz,
            input_signal=self.input_signal,
            output_signal=self.output_signal,
            n_segments=self.n_segments,
            excitation_power=self.Prr,
        )


def _was_excited(f_hz: FloatArray, power: FloatArray, floor_db: float) -> BoolArray:
    """Where the excitation actually had power, relative to its own peak.

    The companion to the coherence gate, answering the question coherence cannot:
    not "is the output explained by the input" but "was there an input at all".
    Without it, a flight whose stick stopped at 3 Hz can report a coherent band at
    50 Hz, because up there both signals are noise and the noise in one genuinely
    does explain the noise in the other -- they arrived through the same loop.
    """
    positive = f_hz > 0.0
    if not positive.any():
        return np.zeros_like(f_hz, dtype=np.bool_)
    peak = float(np.max(power[positive]))
    if peak <= 0.0:
        return np.zeros_like(f_hz, dtype=np.bool_)
    return np.asarray(power >= peak * 10.0 ** (floor_db / 10.0), dtype=np.bool_)


def _one_coherent_band(
    f_hz: FloatArray, valid: BoolArray, coherence: FloatArray, max_gap_decades: float
) -> BoolArray:
    """Keep the coherent band, and drop the islands above it.

    Coherence is scale-free: it asks whether the output is *explained* by the
    input, not whether either had any energy. Above the band an instrument
    actually excited, both spectra are noise, and a handful of bins will pass any
    threshold by luck. Individually they look like ordinary valid bins. Together
    they do real damage, because
    :attr:`~rotorid.core.types.FrequencyResponse.valid_band_hz` reads the first
    and last valid bin, so one lucky bin at 75 Hz reports a band four times wider
    than the flight supports -- and the fit then weights those bins against the
    real ones. On a pilot-flown log, where coherence dies above a few Hz, that
    was enough to drag the identified natural frequency onto its bound.

    So the valid set is reduced to a single contiguous band: bins are grouped by
    their spacing in log frequency, and gaps narrower than ``max_gap_decades`` are
    bridged. Bridging matters -- a notch inside the excited band produces a
    legitimate coherence dip, and splitting the band there would throw away
    everything above it. Bins inside the dip stay invalid; only the *span* is
    bridged.

    Groups are scored by coherence-weighted width in decades, not by how many
    bins they hold. A linear FFT grid has ten times as many bins in the decade
    from 20 to 200 Hz as in the one from 2 to 20, so counting bins hands the
    decision to the top of the spectrum -- which is exactly where the noise
    islands are. Scoring by decades, weighted by how well each bin is actually
    explained, is the same measure
    :func:`rotorid.core.analysis.sysid.fit_weights` uses to decide what the fit
    listens to, so the band and the fit agree about what matters.
    """
    indices = np.nonzero(valid)[0]
    if indices.size <= 1:
        return valid

    positive = f_hz[indices] > 0.0
    if not positive.all():
        indices = indices[positive]
        if indices.size <= 1:
            return valid

    log_f = np.log10(f_hz[indices])
    splits = np.nonzero(np.diff(log_f) > max_gap_decades)[0]
    groups = np.split(indices, splits + 1)

    def _evidence(group: np.ndarray) -> float:
        if group.size < 2:
            return 0.0
        widths = np.gradient(np.log10(f_hz[group]))
        return float(np.sum(coherence[group] * np.abs(widths)))

    best = max(groups, key=_evidence)

    kept = np.zeros_like(valid)
    kept[best[0] : best[-1] + 1] = valid[best[0] : best[-1] + 1]
    return np.asarray(kept, dtype=np.bool_)


def _coherence(Pxx: FloatArray, Pyy: FloatArray, Pxy: ComplexArray) -> FloatArray:
    """Ordinary coherence from averaged spectra, clipped to ``[0, 1]``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = Pxx * Pyy
        gamma2 = np.where(denom > 0.0, np.abs(Pxy) ** 2 / denom, 0.0)
    return np.asarray(np.clip(gamma2, 0.0, 1.0), dtype=np.float64)


def _gate(
    *,
    f_hz: FloatArray,
    H: ComplexArray,
    coherence: FloatArray,
    coherence_threshold: float,
    band_hz: tuple[float, float] | None,
    input_signal: str,
    output_signal: str,
    n_segments: int,
    excitation_power: FloatArray | None = None,
    max_gap_decades: float = _MAX_COHERENCE_GAP_DECADES,
    excitation_floor_db: float = _EXCITATION_FLOOR_DB,
) -> FrequencyResponse:
    """Apply the coherence, excitation and band gates. Shared by both estimators."""
    valid: BoolArray = (coherence >= coherence_threshold) & (f_hz > 0.0)
    if band_hz is not None:
        valid = valid & (f_hz >= band_hz[0]) & (f_hz <= band_hz[1])
    if excitation_power is not None:
        valid = valid & _was_excited(f_hz, excitation_power, excitation_floor_db)
    valid = _one_coherent_band(f_hz, valid, coherence, max_gap_decades)
    return FrequencyResponse(
        f_hz=f_hz,
        H=H,
        coherence=coherence,
        valid_mask=np.asarray(valid, dtype=np.bool_),
        input_signal=input_signal,
        output_signal=output_signal,
        n_segments_averaged=n_segments,
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

    kwargs = _welch_kwargs(sample_rate_hz, nperseg, detrend)
    f, Pxx = welch(u, **kwargs)
    _, Pyy = welch(y, **kwargs)
    _, Pxy = csd(u, y, **kwargs)

    return SpectralEstimate(
        f_hz=np.asarray(f, dtype=np.float64),
        Pxx=np.asarray(Pxx, dtype=np.float64),
        Pyy=np.asarray(Pyy, dtype=np.float64),
        Pxy=np.asarray(Pxy, dtype=np.complex128),
        n_segments=_n_segments(u.size, nperseg),
        input_signal=input_signal,
        output_signal=output_signal,
    )


def estimate_frf_iv(
    r: FloatArray,
    u: FloatArray,
    y: FloatArray,
    sample_rate_hz: float,
    *,
    nperseg: int,
    instrument_signal: str,
    input_signal: str,
    output_signal: str,
    detrend: str = "linear",
) -> InstrumentedEstimate:
    """Instrument-variable estimate of the plant from ``u`` to ``y``.

    Uses the exogenous signal ``r`` as an instrument, giving ``Pry / Pru``. See
    :class:`InstrumentedEstimate` for why that is the plant and not something
    else.

    The instrument has to be genuinely exogenous -- uncorrelated with the gyro
    noise -- or the bias it removes comes straight back. In descending order of
    how well they satisfy that: an injected chirp, the pilot's commanded lean
    angle, the rate setpoint. The last is the weakest because in Stabilize it is
    ``ATC_ANG_*_P`` times the attitude error, so it carries gyro noise round
    through the outer loop; the caller is responsible for saying which rung it
    used.

    Args:
        r: The instrument. Same length and sample rate as ``u`` and ``y``.

    Raises:
        ValueError: if the three signals differ in length.
    """
    if not (r.shape == u.shape == y.shape):
        raise ValueError(
            f"instrument, input and output lengths differ: {r.shape} vs {u.shape} vs {y.shape}"
        )

    kwargs = _welch_kwargs(sample_rate_hz, nperseg, detrend)
    f, Prr = welch(r, **kwargs)
    _, Puu = welch(u, **kwargs)
    _, Pyy = welch(y, **kwargs)
    _, Pru = csd(r, u, **kwargs)
    _, Pry = csd(r, y, **kwargs)

    return InstrumentedEstimate(
        f_hz=np.asarray(f, dtype=np.float64),
        Prr=np.asarray(Prr, dtype=np.float64),
        Puu=np.asarray(Puu, dtype=np.float64),
        Pyy=np.asarray(Pyy, dtype=np.float64),
        Pru=np.asarray(Pru, dtype=np.complex128),
        Pry=np.asarray(Pry, dtype=np.complex128),
        n_segments=_n_segments(r.size, nperseg),
        instrument_signal=instrument_signal,
        input_signal=input_signal,
        output_signal=output_signal,
    )


def _welch_kwargs(sample_rate_hz: float, nperseg: int, detrend: str) -> dict[str, object]:
    """Shared Welch/CSD settings, so both estimators land on the same grid."""
    return {
        "fs": sample_rate_hz,
        "window": get_window("hann", nperseg),
        "nperseg": nperseg,
        "noverlap": int(nperseg * _OVERLAP_FRACTION),
        "detrend": detrend,
    }


def _n_segments(n_samples: int, nperseg: int) -> int:
    """How many overlapping Welch windows fit in a record of this length."""
    noverlap = int(nperseg * _OVERLAP_FRACTION)
    return max(1, (n_samples - noverlap) // (nperseg - noverlap))


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


def combine_iv(estimates: Sequence[InstrumentedEstimate]) -> InstrumentedEstimate:
    """Merge instrument-variable estimates, exactly as :func:`combine` does.

    The five spectra are averaged and the ratio taken afterwards, never the other
    way round. Averaging finished ``H`` values from several segments would weight
    a segment where the instrument barely moved the same as one where it moved a
    lot, which is the whole reason this module keeps spectra rather than transfer
    functions.

    Raises:
        ValueError: if the estimates are empty, describe different signals, or
            sit on different frequency grids.
    """
    if not estimates:
        raise ValueError("nothing to combine")
    first = estimates[0]
    if len(estimates) == 1:
        return first

    names = (first.instrument_signal, first.input_signal, first.output_signal)
    for e in estimates[1:]:
        if e.f_hz.shape != first.f_hz.shape or not np.allclose(e.f_hz, first.f_hz):
            raise ValueError(
                "cannot combine estimates on different frequency grids; "
                "use the same nperseg and sample rate for every segment"
            )
        if (e.instrument_signal, e.input_signal, e.output_signal) != names:
            raise ValueError(
                f"cannot combine {e.instrument_signal}->({e.input_signal}, {e.output_signal}) "
                f"with {names[0]}->({names[1]}, {names[2]})"
            )

    weights = np.array([float(e.n_segments) for e in estimates])
    total = float(weights.sum())

    def _mean(attr: str) -> np.ndarray:
        stacked = sum(w * getattr(e, attr) for w, e in zip(weights, estimates, strict=True))
        return np.asarray(stacked) / total

    return InstrumentedEstimate(
        f_hz=first.f_hz,
        Prr=np.asarray(_mean("Prr"), dtype=np.float64),
        Puu=np.asarray(_mean("Puu"), dtype=np.float64),
        Pyy=np.asarray(_mean("Pyy"), dtype=np.float64),
        Pru=np.asarray(_mean("Pru"), dtype=np.complex128),
        Pry=np.asarray(_mean("Pry"), dtype=np.complex128),
        n_segments=int(total),
        instrument_signal=first.instrument_signal,
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
