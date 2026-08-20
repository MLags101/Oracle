"""Data contracts. The interface between every layer of RotorID (spec section 3).

Two rules govern everything here:

1. **Canonical units** (see :mod:`rotorid.core.units`): rad/s, rad, s, and Hz for
   user-facing frequency. Enforced at the IO boundary.
2. **Effective plant vs. airframe** (spec section 5.3). :class:`EffectivePlant` is
   what we measure and *includes* the vehicle filter chain, because the gyro that
   both stacks log is post-filter. :class:`AirframeModel` is what we fit, with the
   chain divided out. Confusing the two double-counts filter phase lag and
   silently produces bad gains -- it was the central error of spec revision 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - import cycle broken deliberately
    import control

    from rotorid.core.filters.chain import FilterChain

__all__ = [
    "AXES",
    "AirframeModel",
    "Axis",
    "BatchSamples",
    "EffectivePlant",
    "ExcitationSegment",
    "FilterRecommendation",
    "Finding",
    "FlightTestPlan",
    "FlightTestStage",
    "FrequencyResponse",
    "GainSet",
    "LatencyBudget",
    "LogBundle",
    "MarginReport",
    "NoiseProfile",
    "Session",
    "Signal",
    "SpectralPeak",
    "StepMetrics",
    "TuneRecommendation",
]

Axis = Literal["roll", "pitch", "yaw"]
Stack = Literal["ardupilot", "px4"]
Severity = Literal["blocker", "warning", "info", "good"]
Confidence = Literal["high", "medium", "low"]

#: Canonical axis order. Iterate this, never a bare set, so output ordering is stable.
AXES: tuple[Axis, ...] = ("roll", "pitch", "yaw")

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]


# --------------------------------------------------------------------------- #
# Log ingestion
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Signal:
    """Uniformly sampled time series. All times in seconds since log start.

    Attributes:
        filtered: ``True`` if this signal is downstream of the vehicle's filter
            chain. Both stacks log a post-filter gyro as the rate measurement, so
            this is ``True`` for ``rate.*.measured`` and ``False`` only for
            pre-filter batch data. ``None`` where the distinction does not apply.
        native_rate_hz: The rate this signal was *logged* at, before resampling.
            This is not the same number as :attr:`rate_hz` once the signal is on
            the common grid, and the difference is the whole point: interpolation
            moves samples onto a faster time base but creates no information, so
            everything above the native Nyquist is spline opinion. A log with
            ``LOG_BITMASK`` bit 0 clear puts ``RATE`` on the 10 Hz medium-rate
            schedule while ``SCHED_LOOP_RATE`` still says 400, and nothing else in
            the file admits it. ``None`` when the reader could not establish it.
    """

    name: str
    t: FloatArray
    y: FloatArray
    units: str
    source_msg: str
    filtered: bool | None = None
    native_rate_hz: float | None = None

    @property
    def dt(self) -> float:
        """Sample interval in seconds."""
        if self.t.size < 2:
            raise ValueError(f"{self.name}: need at least 2 samples to define dt")
        return float(self.t[1] - self.t[0])

    @property
    def rate_hz(self) -> float:
        """Sample rate in Hz."""
        return 1.0 / self.dt

    @property
    def duration_s(self) -> float:
        """Span of the signal in seconds."""
        if self.t.size == 0:
            return 0.0
        return float(self.t[-1] - self.t[0])

    @property
    def native_nyquist_hz(self) -> float | None:
        """Highest frequency this signal can carry real information about.

        Above it there is nothing to recover: the vehicle sampled the quantity
        this slowly and no reconstruction adds back what was never written down.
        """
        if self.native_rate_hz is None:
            return None
        return 0.5 * self.native_rate_hz


@dataclass(frozen=True, slots=True)
class BatchSamples:
    """High-rate gyro blocks from batch/FIFO logging.

    These are the only source of genuinely pre-filter gyro data, and they are the
    ground truth for validating the filter engine (spec section 5.5). Blocks may be
    discontinuous and run at a different rate from the control loop, so they are
    never resampled onto the main uniform grid.

    Attributes:
        kind: ``"both"`` means pre- *and* post-filter blocks are present, which is
            what ``INS_LOG_BAT_OPT = 4`` / ``INS_RAW_LOG_OPT = 9`` produce.
        blocks: Per axis, a list of ``(t_start, samples)``. Post-filter when
            ``kind`` includes post-filter data, otherwise the only available trace.
        blocks_pre: Pre-filter blocks, present only when ``kind == "both"``.
    """

    kind: Literal["pre_filter", "post_filter", "both", "raw"]
    rate_hz: float
    blocks: dict[Axis, list[tuple[float, FloatArray]]]
    blocks_pre: dict[Axis, list[tuple[float, FloatArray]]] | None = None

    @property
    def has_pre_filter(self) -> bool:
        """Whether pre-filter samples are available for filter validation."""
        return self.kind in ("pre_filter", "raw") or self.blocks_pre is not None


@dataclass(frozen=True, slots=True)
class LogBundle:
    """Everything extracted from one log file, on a common uniform grid.

    Attributes:
        sample_rate_hz: Rate of the uniform grid the signals sit on (spec 5.1).
        loop_rate_hz: Controller loop rate (``SCHED_LOOP_RATE`` / ``IMU_GYRO_RATEMAX``).
        gyro_sample_rate_hz: Sensor rate. The rate the vehicle's biquads run at,
            and therefore the rate our filter models must be designed at.
        signals: Canonical keys only (spec section 6.3). Missing signals are absent,
            never zero-filled.
    """

    path: Path
    stack: Stack
    firmware_version: str | None
    board_id: str | None
    frame_info: dict[str, str]
    sample_rate_hz: float
    loop_rate_hz: float
    gyro_sample_rate_hz: float
    signals: dict[str, Signal]
    params: dict[str, float]
    batch: BatchSamples | None = None
    warnings: tuple[str, ...] = ()

    def signal(self, key: str) -> Signal:
        """Return one signal by canonical key.

        Raises:
            KeyError: if absent. Callers that can tolerate absence should use
                ``key in bundle.signals`` and emit a ``LOG_MISSING_MSG`` finding.
        """
        try:
            return self.signals[key]
        except KeyError:
            raise KeyError(
                f"signal {key!r} not in log; available: {sorted(self.signals)}"
            ) from None

    def param(self, name: str, default: float | None = None) -> float | None:
        """Return one vehicle parameter, or ``default`` if it was not logged."""
        return self.params.get(name, default)


# --------------------------------------------------------------------------- #
# Excitation and frequency response
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ExcitationSegment:
    """One window of the flight suitable for identification.

    Attributes:
        kind: Where the excitation came from. Drives confidence: a
            ``systemid_chirp`` is worth far more than ``pilot_input``.
        injection_point: For ArduPilot SYSTEMID, decoded from ``SID_AXIS``:
            ``"rate"`` (7-9) or ``"mixer"`` (10-12).
        f_start_hz: Known exactly for chirps, ``None`` otherwise.
    """

    axis: Axis
    t_start: float
    t_end: float
    kind: Literal["systemid_chirp", "px4_autotune", "autotune_twitch", "pilot_input", "unknown"]
    amplitude_estimate: float
    confidence: float
    injection_point: str | None = None
    f_start_hz: float | None = None
    f_stop_hz: float | None = None

    @property
    def duration_s(self) -> float:
        """Length of the segment in seconds."""
        return self.t_end - self.t_start


@dataclass(frozen=True, slots=True)
class FrequencyResponse:
    """Non-parametric frequency response estimate with coherence gating."""

    f_hz: FloatArray
    H: ComplexArray
    coherence: FloatArray
    valid_mask: BoolArray
    input_signal: str
    output_signal: str
    n_segments_averaged: int

    @property
    def valid_band_hz(self) -> tuple[float, float]:
        """Lowest and highest frequency passing the coherence gate.

        Raises:
            ValueError: if no bin is valid.
        """
        if not self.valid_mask.any():
            raise ValueError("no frequency bin passed the coherence gate")
        valid = self.f_hz[self.valid_mask]
        return float(valid[0]), float(valid[-1])

    @property
    def coherence_mean(self) -> float:
        """Mean coherence over the valid band, or 0.0 if nothing is valid."""
        if not self.valid_mask.any():
            return 0.0
        return float(np.mean(self.coherence[self.valid_mask]))


@dataclass(frozen=True, slots=True)
class EffectivePlant:
    """What the controller actually sees: filters INCLUDED.

    This is the measurement. ``EffectivePlant = F_current * G_air * exp(-tau s)``.
    See spec section 5.3 -- do not treat this as a bare-airframe model.
    """

    axis: Axis
    frf: FrequencyResponse
    filters_included: bool
    source: Literal["mixer_cmd", "injected_chirp", "raw_gyro"]


@dataclass(frozen=True, slots=True)
class AirframeModel:
    """Identified bare airframe (plus motors/ESC) for one axis. Filters DIVIDED OUT.

    Attributes:
        params: Structure-dependent. ``so_delay`` uses ``K``, ``wn`` (rad/s),
            ``zeta``, ``tau`` (s).
        filter_deconvolution: How the filter chain was removed. ``"modeled"`` is
            the normal path; ``"raw_gyro"`` means pre-filter data was used directly;
            ``"none"`` means no filters were present to remove.
        gain_spread_pct: Variation of ``K`` across operating points (spec 5.9).
    """

    axis: Axis
    structure: Literal["so_delay", "fo_delay", "so_zero_delay"]
    params: dict[str, float]
    fit_rms_db: float
    fit_rms_deg: float
    valid_band_hz: tuple[float, float]
    coherence_mean: float
    filter_deconvolution: Literal["modeled", "raw_gyro", "none"]
    gain_spread_pct: float | None = None

    def tf(self) -> control.TransferFunction:
        """Return the model as a ``control`` transfer function.

        The delay is approximated by a Pade expansion for use with ``control``;
        the frequency-domain code paths use the exact ``exp(-tau s)`` instead and
        should call :meth:`response`.
        """
        from rotorid.core.analysis.model_eval import airframe_tf

        return airframe_tf(self)

    def response(self, f_hz: FloatArray) -> ComplexArray:
        """Exact complex response at the given frequencies, delay included."""
        from rotorid.core.analysis.model_eval import airframe_response

        return airframe_response(self, f_hz)


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


# ``NotchSpec`` from the specification is deliberately absent: the executable
# model in :class:`rotorid.core.filters.harmonic.HarmonicNotch` is strictly richer
# (it knows about composite notches, per-motor tracking and the minimum-frequency
# fade), and two overlapping descriptions of the same notch would drift apart.


@dataclass(frozen=True, slots=True)
class LatencyBudget:
    """Itemized phase lag at one frequency.

    The items sum to :attr:`total_deg` and nothing is counted twice: filter terms
    come from the modeled chain, ``airframe_tau_deg`` is the residual delay left in
    the airframe model *after* the chain was divided out (spec section 0, rule 6).
    """

    at_hz: float
    gyro_lpf_deg: float = 0.0
    notches_deg: float = 0.0
    dterm_lpf_deg: float = 0.0
    error_lpf_deg: float = 0.0
    zoh_deg: float = 0.0
    compute_deg: float = 0.0
    actuator_deg: float = 0.0
    airframe_tau_deg: float = 0.0

    @property
    def total_deg(self) -> float:
        """Sum of every contribution, in degrees of lag (positive = lag).

        A diagnostic total, not the phase of ``L(jw)``: the D-term filter sits in
        the derivative branch only. Margins always come from evaluating the full
        loop; this breakdown explains the answer rather than producing it.
        """
        return self.common_path_deg + self.dterm_lpf_deg

    @property
    def common_path_deg(self) -> float:
        """Lag from the terms genuinely in series with everything in the loop."""
        return (
            self.gyro_lpf_deg
            + self.notches_deg
            + self.error_lpf_deg
            + self.zoh_deg
            + self.compute_deg
            + self.actuator_deg
            + self.airframe_tau_deg
        )

    def items(self) -> tuple[tuple[str, float], ...]:
        """Contributions as ordered ``(label, degrees)`` pairs, for the stacked bar."""
        return (
            ("gyro LPF", self.gyro_lpf_deg),
            ("notches", self.notches_deg),
            ("D-term LPF", self.dterm_lpf_deg),
            ("error LPF", self.error_lpf_deg),
            ("ZOH", self.zoh_deg),
            ("compute", self.compute_deg),
            ("actuator", self.actuator_deg),
            ("airframe delay", self.airframe_tau_deg),
        )


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SpectralPeak:
    """One peak in the gyro spectrum, classified by how it behaves over the flight.

    The classification is what makes filter recommendations honest: a peak that
    tracks RPM wants a dynamic notch, a peak that sits still is a structural
    resonance that a tracking notch will chase uselessly.
    """

    f_hz: float
    magnitude_db: float
    width_hz: float
    kind: Literal["motor_fundamental", "motor_harmonic", "structural", "broadband", "unknown"]
    tracks_rpm: bool = False
    harmonic_index: int | None = None
    motor_index: int | None = None


@dataclass(frozen=True, slots=True)
class NoiseProfile:
    """Gyro noise characterization for one axis.

    Attributes:
        pre_filter_source: Where :attr:`psd_pre` came from. ``"measured"`` means
            batch-logged pre-filter gyro; ``"reconstructed"`` means the flown chain
            was divided out of the post-filter spectrum, which cannot see inside a
            notch deeper than the deconvolution floor. Design code has to know
            which, because "no peak here" means different things in the two cases.
    """

    axis: Axis
    f_hz: FloatArray
    psd_post: FloatArray
    noise_floor_db: float
    peaks: tuple[SpectralPeak, ...] = ()
    psd_pre: FloatArray | None = None
    pre_filter_source: Literal["measured", "reconstructed", "none"] = "none"
    psd_vs_throttle: FloatArray | None = None
    motor_fundamental_track: FloatArray | None = None

    @property
    def has_pre_filter(self) -> bool:
        """Whether a pre-filter spectrum is available (spec 5.5 validation path)."""
        return self.psd_pre is not None


@dataclass(frozen=True, slots=True)
class FilterRecommendation:
    """A proposed filter configuration, with its cost and its evidence.

    Attributes:
        params: Stack-specific parameter names to values, ready for export.
        phase_cost_deg: Chain phase lag at the design crossover. The number the
            joint optimizer trades against attenuation.
        psd_f_hz: Frequency axis shared by :attr:`psd_pre` and
            :attr:`predicted_psd_post`. Carried here so the recommendation can be
            plotted and argued with on its own, without the noise profile it came
            from having to be kept alongside it.
        rejected: ``(alternative, why it lost)`` pairs, surfaced by the
            "Why this number?" affordance.
    """

    stack: Stack
    chain: FilterChain
    baseline_chain: FilterChain
    params: dict[str, float]
    phase_cost_deg: float
    cpu_cost_rel: float
    rationale: str
    psd_f_hz: FloatArray | None = None
    psd_pre: FloatArray | None = None
    predicted_psd_post: FloatArray | None = None
    attenuation_at_peaks_db: dict[float, float] = field(default_factory=dict)
    rejected: tuple[tuple[str, str], ...] = ()


# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MarginReport:
    """Stability margins and disturbance-rejection performance for one loop.

    ``disturbance_rejection_bw_hz`` (DRB) and ``disturbance_rejection_peak_db``
    (DRP) are the rotorcraft performance metrics that pair with the ADS-33
    45 deg / 6 dB stability convention.
    """

    gain_margin_db: float
    phase_margin_deg: float
    crossover_hz: float
    delay_margin_ms: float
    peak_sensitivity_db: float
    disturbance_rejection_bw_hz: float
    disturbance_rejection_peak_db: float


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Closed-loop step response characteristics."""

    rise_time_s: float
    overshoot_pct: float
    settling_time_s: float
    peak_time_s: float
    steady_state_error: float


@dataclass(frozen=True, slots=True)
class GainSet:
    """Rate-loop gains for one axis, in effective (not stack-parameterized) form.

    PX4's ``K`` scaling is resolved at the IO boundary, so ``kp`` here always means
    the effective proportional gain -- never ``MC_ROLLRATE_P`` before ``K``.
    """

    axis: Axis
    kp: float
    ki: float
    kd: float
    kff: float
    imax: float | None = None
    dterm_lpf_hz: float | None = None
    error_lpf_hz: float | None = None
    target_lpf_hz: float | None = None


@dataclass(frozen=True, slots=True)
class TuneRecommendation:
    """A complete recommendation for one axis: filters and gains together.

    Filters are not optional here. On most real vehicles the filter configuration,
    not the gain arithmetic, is what limits achievable bandwidth, so a
    recommendation that changed gains alone would be incomplete by construction.
    """

    axis: Axis
    gains: GainSet
    baseline_gains: GainSet
    filters: FilterRecommendation
    model: AirframeModel
    margins: MarginReport
    latency: LatencyBudget
    predicted_step: StepMetrics
    dterm_noise_rms_pct: float
    rationale: str
    confidence: Confidence
    conservatism: float
    binding_constraint: str


# --------------------------------------------------------------------------- #
# Guidance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing the tool noticed, with the numbers that back it.

    Attributes:
        code: Stable identifier, e.g. ``"NOTCH_MISTRACKING"``. Never reword these;
            they are referenced by tests, reports and the next-steps generator.
        action: What the user should do about it. A finding without an action is
            not worth showing.
    """

    severity: Severity
    code: str
    title: str
    detail: str
    action: str
    evidence: dict[str, float] = field(default_factory=dict)
    plot_hint: str | None = None
    doc_link: str | None = None


@dataclass(frozen=True, slots=True)
class FlightTestStage:
    """One rung of the staged tuning ladder (spec section 8.2)."""

    index: int
    title: str
    changes: dict[str, float]
    watch_in_flight: tuple[str, ...]
    check_in_log: tuple[str, ...]
    motivating_findings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FlightTestPlan:
    """The ordered ladder of changes to apply, one flight at a time.

    Filters and gains are deliberately never applied in the same flight: doing so
    makes a bad outcome undiagnosable.
    """

    stages: tuple[FlightTestStage, ...]
    preamble: str = ""


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Session:
    """One complete analysis, serializable to a ``.rotorid`` bundle."""

    log: LogBundle
    segments: tuple[ExcitationSegment, ...]
    effective: dict[Axis, EffectivePlant]
    models: dict[Axis, AirframeModel]
    noise: dict[Axis, NoiseProfile]
    recommendations: dict[Axis, TuneRecommendation]
    findings: tuple[Finding, ...]
    config_hash: str
    tool_version: str
    created_utc: datetime
    next_steps: FlightTestPlan | None = None
    acknowledgements: dict[str, str] = field(default_factory=dict)

    @property
    def blockers(self) -> tuple[Finding, ...]:
        """Findings that must be acknowledged before anything may be exported."""
        return tuple(f for f in self.findings if f.severity == "blocker")

    @property
    def unacknowledged_blockers(self) -> tuple[Finding, ...]:
        """Blockers the user has not yet explicitly accepted."""
        return tuple(f for f in self.blockers if f.code not in self.acknowledgements)
