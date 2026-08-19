# RotorID — Offline Multirotor Filter & PID Identification / Tuning Tool

**Implementation specification for a coding agent. Revision 2.**

Build a cross-platform Qt6 desktop application that ingests ArduPilot `.bin` and PX4 `.ulg` flight logs, performs frequency-domain system identification on the vehicle's rate loops, recommends **a joint filter + PID configuration** against explicit stability margins and a shared phase-lag budget, and guides the user through the process with an explanatory, interactive wizard and concrete flight-test next steps.

> **What changed in revision 2** (read this if you saw r1):
> 1. **Plant identification was wrong in r1.** The routinely-logged gyro on both stacks is *post-filter*. The pipeline now distinguishes an `EffectivePlant` (filters included — what is actually measured) from an `AirframeModel` (filters divided out). See §5.3. Getting this wrong double-counts filter phase lag and silently produces bad gains.
> 2. **Filter design is now a first-class output**, not a constraint. Harmonic-notch source/frequency/bandwidth/attenuation/harmonics and gyro & D-term low-pass cutoffs are recommended for both stacks, **co-optimized with the gains** against one phase budget. See §5.6 and §5.7.
> 3. **Firmware-exact discrete filter models** in `core/filters/`, validated against the aircraft's own pre/post-filter logging. See §5.5.
> 4. **Controller structure is modeled per stack** — ArduPilot takes D on the filtered error, PX4 takes D on the measurement. Same gains, same margins, different step response. See §5.8.
> 5. **Milestones restructured around an early vertical slice** (ArduPilot + CLI, end to end, at M1). See §12.

---

## 0. Ground rules for the implementing agent

1. **Build the core library first, GUI second.** Every analysis capability must be usable headlessly via `rotorid.core` and a CLI. The GUI is a thin presentation layer over the core. If a feature can only be reached through a widget, it is wrong.
2. **Never block the Qt event loop.** All parsing, resampling, fitting, and optimization run in worker threads. See §9.
3. **Every recommended number must be traceable.** Each output records the model it came from, the margins achieved, the crossover chosen, the coherence-valid band, and — for filters — the measured peak it targets and the phase it costs. No unexplained numbers. The GUI exposes this trace through a "Why this number?" affordance (§10.5); if a value cannot answer that question, it must not be recommended.
4. **Fail loudly on bad data.** Poor coherence, missing messages, high fit residual, a filter model that disagrees with the log — surface each as a blocking or warning `Finding` (§8). Silent degradation to a garbage tune is the failure mode this whole tool exists to prevent.
5. **Model filters in discrete time, exactly as the firmware does.** Analog approximations get the phase wrong at exactly the frequencies that matter. See §5.5.
6. **Never double-count delay or filter phase.** There is exactly one place in the codebase where filters are divided out of a measurement (§5.3 step 4) and exactly one place where they are multiplied back in (§5.7). Any third site is a bug.
7. **Determinism.** Same log + same settings ⇒ same numbers. Seed anything stochastic. Pin dependency versions. Record the resolved config hash in the session (`Session.config_hash`).
8. **No magic numbers in code.** Every threshold lives in `rotorid.toml` (§4) with a comment naming its source — a firmware document, a published methodology, or "project default, chosen because…".
9. **Write tests against synthetic data with known ground truth** (§11). Do not use real logs as the primary correctness check — you have no ground truth for them. The one exception is the filter engine, which *does* have ground truth from the aircraft (§5.5).

---

## 1. Technology stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python ≥ 3.11 | Uses `X \| None`, `dataclass(slots=True)`, `tomllib` |
| GUI | **PySide6** (Qt 6.6+) | LGPL. **Do not use PyQt6** — GPL licensing is a distribution blocker for a tool users may ship internally. |
| Plotting | **pyqtgraph** ≥ 0.13 | Native Qt, fast, interactive pan/zoom on 100k+ point traces. Matplotlib is too slow for interactive log scrubbing; use it only for PDF report export. |
| Numerics | numpy, scipy | `scipy.signal` for Welch/CSD/coherence/resample/`sosfreqz`, `scipy.optimize` for fitting and design |
| Control | `python-control` ≥ 0.10 | Margins, frequency response, loop shaping. Used for margin computation and cross-checks; the interactive solver works directly on precomputed complex arrays for speed. |
| ArduPilot logs | `pymavlink` (`DFReader`) | `mavlogdump` internals; read `.bin` directly |
| PX4 logs | `pyulog` ≥ 1.0 | `.ulg` reader |
| Packaging | `pyproject.toml` + hatchling; PyInstaller for binaries | |
| Testing | pytest, pytest-qt, hypothesis (optional) | |
| Lint/format | ruff + mypy (strict on `core/`) | |

Optional, behind feature flags: `pyarrow` (fast session caching), `reportlab` or `weasyprint` (PDF report).

---

## 2. Repository layout

```
rotorid/
  pyproject.toml
  rotorid.toml                 # default thresholds & tunables (§4)
  README.md
  src/rotorid/
    __init__.py
    cli.py                     # headless entry point
    config.py                  # load/merge/hash rotorid.toml
    core/
      types.py                 # all dataclasses / data contracts (§3)
      units.py                 # unit conversion + canonical unit enforcement
      io/
        base.py                # LogReader ABC
        ardupilot.py           # .bin reader
        px4.py                 # .ulg reader
        registry.py            # format sniffing / dispatch
      filters/                 # firmware-exact digital filter models (§5.5)
        biquad.py              # DigitalBiquadFilter, NotchFilter, 1-pole IIR
        harmonic.py            # harmonic notch stack, per-motor, double/triple
        chain.py               # FilterChain assembly + discrete response
        latency.py             # LatencyBudget: itemized phase/delay accounting
      preprocess/
        resample.py            # async -> uniform grid
        segment.py             # excitation segment detection (§5.2)
        params.py              # parameter snapshot -> FilterChain / GainSet
      analysis/
        spectra.py             # Welch, CSD, coherence, FRF, multi-segment averaging
        noise.py               # gyro noise, peak tracking & classification (§5.6)
        sysid.py               # effective plant -> airframe model (§5.3, §5.4)
        deconv.py              # Wiener-deconvolution step response (fallback/validation)
        margins.py             # broken-loop, GM/PM/crossover/Ms/DRB/DRP
        operating_point.py     # gain vs throttle/voltage sensitivity (§5.9)
      design/
        controller.py          # per-stack controller models (§5.8)
        filters.py             # filter configuration recommender (§5.6)
        objectives.py          # margin-constrained gain search (§5.7)
        joint.py               # joint filter+gain optimizer (§5.7)
        recommend.py           # -> TuneRecommendation
      guidance/
        rules.py               # Finding generation engine (§8)
        explain.py             # parameterized explanations + glossary (§10.5)
        nextsteps.py           # staged flight-test plan generation (§8.2)
      export/
        params.py              # .param / PX4 param output
        profile.py             # "data collection profile" param file (§13)
        report.py              # HTML/PDF session report
        session.py             # save/load .rotorid session bundle
    gui/
      app.py                   # QApplication bootstrap
      main_window.py
      wizard/                  # one module per stage (§10)
      widgets/
        bode_plot.py
        coherence_plot.py
        timeseries_plot.py
        spectrogram_plot.py
        step_response_plot.py
        nichols_plot.py
        phase_budget_plot.py   # stacked phase-lag contributions at crossover
        prepost_spectrum.py    # measured vs predicted post-filter spectrum
        findings_panel.py
        param_diff_table.py
        why_popover.py         # "Why this number?" trace (§10.5)
      workers.py               # QThread/QRunnable wrappers (§9)
      models.py                # QAbstractTableModel implementations
      theme.py
  tests/
    synthetic/                 # ground-truth generators (§11)
    test_io_*.py
    test_filters_*.py
    test_sysid_*.py
    test_design_*.py
    test_gui_*.py
  docs/
    logging-setup-ardupilot.md
    logging-setup-px4.md
    methodology.md
    glossary.md
```

---

## 3. Data contracts

Define these in `core/types.py` as frozen dataclasses. **These are the interface between every layer — implement them before anything else.**

```python
Axis = Literal["roll", "pitch", "yaw"]
Stack = Literal["ardupilot", "px4"]

@dataclass(frozen=True, slots=True)
class Signal:
    """Uniformly sampled time series. All times in seconds since log start."""
    name: str
    t: np.ndarray          # shape (N,), strictly increasing, uniform dt
    y: np.ndarray          # shape (N,)
    units: str             # canonical: "rad/s", "rad", "normalized", "s"
    source_msg: str        # e.g. "RATE.R" or "vehicle_angular_velocity.xyz[0]"
    filtered: bool | None  # True = post vehicle filter chain, None = unknown/NA

@dataclass(frozen=True, slots=True)
class LogBundle:
    """Everything extracted from one log file, on a common uniform grid."""
    path: Path
    stack: Stack
    firmware_version: str | None
    board_id: str | None              # for CPU-class gating of filter options
    frame_info: dict[str, str]
    sample_rate_hz: float             # the uniform grid rate (§5.1)
    loop_rate_hz: float               # SCHED_LOOP_RATE / IMU_GYRO_RATEMAX
    gyro_sample_rate_hz: float        # sensor rate; the rate the biquads run at
    signals: dict[str, Signal]        # canonical keys, see §6.3
    batch: BatchSamples | None        # ISBH/ISBD or sensor_gyro_fifo, if present
    params: dict[str, float]          # onboard parameters at time of log
    warnings: list[str]

@dataclass(frozen=True, slots=True)
class BatchSamples:
    """High-rate gyro blocks. May be discontinuous; never resampled onto the main grid."""
    kind: Literal["pre_filter", "post_filter", "both", "raw"]
    rate_hz: float
    blocks: dict[Axis, list[tuple[float, np.ndarray]]]              # (t_start, samples)
    blocks_pre: dict[Axis, list[tuple[float, np.ndarray]]] | None   # when kind == "both"

@dataclass(frozen=True, slots=True)
class ExcitationSegment:
    axis: Axis
    t_start: float
    t_end: float
    kind: Literal["systemid_chirp", "px4_autotune", "autotune_twitch", "pilot_input", "unknown"]
    injection_point: str | None       # "rate" | "mixer" | ... (from SID_AXIS)
    f_start_hz: float | None          # known exactly for chirps
    f_stop_hz: float | None
    amplitude_estimate: float
    confidence: float                 # 0..1

@dataclass(frozen=True, slots=True)
class FrequencyResponse:
    f_hz: np.ndarray
    H: np.ndarray                     # complex, input -> output
    coherence: np.ndarray             # 0..1
    valid_mask: np.ndarray            # bool; coherence >= threshold AND in excited band
    input_signal: str
    output_signal: str
    n_segments_averaged: int

@dataclass(frozen=True, slots=True)
class EffectivePlant:
    """What the controller actually sees: filters INCLUDED. This is what we measure."""
    axis: Axis
    frf: FrequencyResponse
    filters_included: bool            # True in every normal path; False only for raw-gyro ID
    source: Literal["mixer_cmd", "injected_chirp", "raw_gyro"]

@dataclass(frozen=True, slots=True)
class AirframeModel:
    """Identified bare airframe (+ motors/ESC) for one axis. Filters DIVIDED OUT."""
    axis: Axis
    structure: Literal["so_delay", "fo_delay", "so_zero_delay"]
    params: dict[str, float]          # e.g. {"K":.., "wn":.., "zeta":.., "tau":..}
    fit_rms_db: float
    fit_rms_deg: float
    valid_band_hz: tuple[float, float]
    coherence_mean: float
    filter_deconvolution: Literal["modeled", "raw_gyro", "none"]
    gain_spread_pct: float | None     # K variation across operating points (§5.9)
    def tf(self) -> control.TransferFunction: ...

@dataclass(frozen=True, slots=True)
class NotchSpec:
    center_hz: float
    bandwidth_hz: float
    attenuation_db: float
    harmonics: tuple[int, ...]        # which multiples are active
    tracking: Literal["static", "throttle", "rpm", "esc", "fft"]
    per_motor: bool = False

@dataclass(frozen=True, slots=True)
class FilterChain:
    """Reconstructed from vehicle params, or proposed by the designer."""
    stack: Stack
    gyro_lpf_hz: float | None
    dterm_lpf_hz: float | None
    error_lpf_hz: float | None        # ArduPilot FLTE — feedback path, MOVES THE MARGINS
    target_lpf_hz: float | None       # ArduPilot FLTT — reference path only
    notches: tuple[NotchSpec, ...]
    sample_rate_hz: float             # rate the biquads are designed at
    loop_rate_hz: float
    def response(self, f_hz, *, operating_point) -> np.ndarray: ...   # complex, discrete
    def phase_deg(self, f_hz, *, operating_point) -> np.ndarray: ...
    def group_delay_ms(self, f_hz, *, operating_point) -> np.ndarray: ...
    def cpu_cost(self) -> float: ...  # relative units, for OPTS gating

@dataclass(frozen=True, slots=True)
class LatencyBudget:
    """Itemized phase lag at one frequency. Sums to the total; nothing is counted twice."""
    at_hz: float
    gyro_lpf_deg: float
    notches_deg: float
    dterm_lpf_deg: float
    error_lpf_deg: float
    zoh_deg: float                    # 0.5 / loop_rate
    compute_deg: float                # controller compute delay
    actuator_deg: float               # ESC protocol + motor (MOT_PWM_TYPE class)
    airframe_tau_deg: float           # residual identified delay, filters already removed
    total_deg: float

@dataclass(frozen=True, slots=True)
class SpectralPeak:
    f_hz: float
    magnitude_db: float               # above local noise floor
    width_hz: float
    kind: Literal["motor_fundamental", "motor_harmonic", "structural", "broadband", "unknown"]
    harmonic_index: int | None
    tracks_rpm: bool
    motor_index: int | None

@dataclass(frozen=True, slots=True)
class NoiseProfile:
    axis: Axis
    f_hz: np.ndarray
    psd_pre: np.ndarray | None            # pre-filter, if batch/raw logging available
    psd_post: np.ndarray                  # post-filter (always available)
    psd_vs_throttle: np.ndarray | None    # 2D: (throttle_bin, freq)
    peaks: tuple[SpectralPeak, ...]
    motor_fundamental_track: np.ndarray | None   # f(t) from ESC/RPM/FFT/throttle
    noise_floor_db: float

@dataclass(frozen=True, slots=True)
class FilterRecommendation:
    stack: Stack
    chain: FilterChain                # the proposed chain
    baseline_chain: FilterChain       # what is on the vehicle now
    params: dict[str, float]          # stack-specific parameter names -> values
    predicted_psd_post: np.ndarray    # measured pre-filter PSD through the proposed chain
    attenuation_at_peaks_db: dict[float, float]
    phase_cost_deg: float             # chain phase lag at the design crossover
    cpu_cost_rel: float
    rationale: str
    rejected: tuple[tuple[str, str], ...]   # (alternative, why it lost)

@dataclass(frozen=True, slots=True)
class MarginReport:
    gain_margin_db: float
    phase_margin_deg: float
    crossover_hz: float
    delay_margin_ms: float
    peak_sensitivity_db: float           # ||S||_inf — the single best robustness scalar
    disturbance_rejection_bw_hz: float   # DRB
    disturbance_rejection_peak_db: float # DRP

@dataclass(frozen=True, slots=True)
class GainSet:
    axis: Axis
    kp: float; ki: float; kd: float; kff: float
    imax: float | None = None
    dterm_lpf_hz: float | None = None
    error_lpf_hz: float | None = None
    target_lpf_hz: float | None = None

@dataclass(frozen=True, slots=True)
class TuneRecommendation:
    axis: Axis
    gains: GainSet
    baseline_gains: GainSet
    filters: FilterRecommendation     # a recommendation is ALWAYS a filter+gain package
    model: AirframeModel
    margins: MarginReport
    latency: LatencyBudget
    predicted_step: StepMetrics       # rise, overshoot, settle
    dterm_noise_rms_pct: float        # predicted D-term output as % of full motor range
    rationale: str                    # human-readable, shown verbatim in GUI
    confidence: Literal["high", "medium", "low"]
    conservatism: float               # 0..1 slider position used
    binding_constraint: str           # which constraint stopped the optimizer

@dataclass(frozen=True, slots=True)
class Finding:
    severity: Literal["blocker", "warning", "info", "good"]
    code: str                          # stable ID, e.g. "NOTCH_MISTRACKING"
    title: str
    detail: str                        # what it means
    action: str                        # what the user should do about it
    evidence: dict[str, float]         # numbers backing the claim
    plot_hint: str | None              # which GUI plot to jump to
    doc_link: str | None               # docs/ anchor or upstream URL

@dataclass(frozen=True, slots=True)
class Session:
    """Serializable to a .rotorid bundle (zip: JSON manifest + npz arrays)."""
    log: LogBundle
    segments: list[ExcitationSegment]
    effective: dict[Axis, EffectivePlant]
    models: dict[Axis, AirframeModel]
    noise: dict[Axis, NoiseProfile]
    recommendations: dict[Axis, TuneRecommendation]
    findings: list[Finding]
    next_steps: FlightTestPlan
    acknowledgements: dict[str, str]   # blocker code -> user acknowledgement text
    config_hash: str
    created_utc: datetime
    tool_version: str
```

**Canonical units, enforced at the IO boundary:** angular rate in rad/s, angle in rad, time in s, frequency in Hz for user-facing values and rad/s internally for `control` objects. `units.py` owns every conversion; no ad-hoc `* np.pi/180` anywhere else.

---

## 4. Configuration — `rotorid.toml`

Every threshold in this document lives here, each with a source comment. Loaded by `config.py`, merged with an optional user override file, and hashed into `Session.config_hash` for determinism.

```toml
[coherence]
threshold = 0.6              # project default; below this an FRF bin is not trusted
min_valid_octaves = 1.0      # narrower than this around crossover -> COHERENCE_NARROW_BAND

[fit]
max_rms_db = 3.0             # project default
max_rms_deg = 20.0
tau_bounds_ms = [5.0, 80.0]  # physically plausible for multirotors
wn_bounds_hz = [3.0, 80.0]
zeta_bounds = [0.05, 2.0]

[margins]
pm_min_deg = 45.0            # ADS-33 / US Army AFDD rotorcraft convention
gm_min_db = 6.0              # ADS-33 / AFDD
ms_max_db = 6.0              # ||S||inf <= 2.0, standard robustness bound
pm_floor_deg = 25.0          # HARD floor. Flight test finds 20-23 deg gives PIO tendency.
crossover_frac_of_loop = 0.2 # f_c <= 0.2 * (loop_rate / 2)

[filters]
target_noise_floor_db = -50.0   # ArduPilot guidance: below -50 dB does not need a notch
phase_budget_deg = 25.0         # max filter-chain phase lag allowed at design crossover
max_harmonics = 3               # ArduPilot doc: "three harmonics is usually safe"
freq_bw_ratio_default = 2.0     # ArduPilot doc default (BW = FREQ/2)
freq_bw_ratio_per_motor = 4.0   # ArduPilot doc: per-motor tracking needs 4:1
att_min_db = 15.0
att_max_db = 40.0
harmonic_min_benefit_db = 3.0   # a harmonic notch must buy this much noise reduction

[noise]
dterm_output_rms_limit_pct = 5.0   # project default: D output RMS as % of full motor range
peak_prominence_db = 6.0
rpm_track_correlation_min = 0.7    # above this a peak counts as RPM-tracking

[design]
outer_loop_separation = 4.0     # inner/outer timescale separation (3-5x rule)
max_gain_step_ratio = 2.0       # beyond this -> GAINS_FAR_FROM_CURRENT
```

---

## 5. Analysis pipeline

### 5.1 Resampling (`preprocess/resample.py`)

Log messages are asynchronous and jittered. Choose the uniform grid rate as

```
fs_grid = min(gyro_sample_rate_hz, max(2.5 * f_highest_modeled, 2.0 * loop_rate_hz))
```

where `f_highest_modeled` is the highest notch center the filter model must represent (highest harmonic of the highest expected motor frequency). Never below 2.5× the highest modeled notch — the discrete notch response is wrong otherwise. Never above the slowest required signal's native rate without flagging it.

For each signal: sort by time, drop duplicate timestamps, then interpolate onto the uniform grid. Use `scipy.signal.resample_poly` where the source is near-uniform, cubic interpolation otherwise. Record jitter statistics and warn if the 99th-percentile gap exceeds 3× the median (`LOG_RATE_IRREGULAR`).

Batch/FIFO samples (`BatchSamples`) are **never** resampled onto this grid — they are processed block-wise at their own rate for spectra and filter validation.

### 5.2 Segmentation (`preprocess/segment.py`)

This tool is **chirp-first**: the primary supported workflow is a deliberate excitation flight. Auto-propose excitation windows in priority order:

1. **ArduPilot SYSTEMID** — use `SIDS`/`SIDD`: exact start/stop, axis, injection point (from `SID_AXIS`), and frequency schedule. Reconstruct the commanded chirp analytically from `SID_F_START_HZ`, `SID_F_STOP_HZ`, `SID_T_FADE_IN`, `SID_T_REC`, `SID_T_FADE_OUT` and cross-check against logged `SIDD.Targ`; disagreement means a firmware difference worth flagging. Confidence 1.0.
2. **PX4 autotune** — `autotune_attitude_control_status.state` transitions. Also ingest the onboard ARX coefficients and fitness for comparison (`PX4_ONBOARD_DISAGREES`). Confidence 0.8.
3. **ArduPilot autotune twitches** — parse `MSG`/`EV` for stage boundaries; each twitch is a short segment, consecutive twitches on one axis grouped into a composite segment for averaging. Confidence 0.5, always with `EXCITATION_TWITCH_ONLY`.
4. **Fallback — energy detection** on ordinary flight: high-passed setpoint/output variance above threshold with the other two axes quiet. Confidence ≤ 0.3, always with `EXCITATION_WEAK`, and the resulting recommendation is capped at `confidence = "low"`.

The GUI must let the user drag segment boundaries on the timeseries plot and re-run. Always show the auto-proposal and allow override.

Multiple segments and multiple log files for the same axis are supported and encouraged: FRFs are combined by coherence-weighted averaging (§5.4), matching ArduPilot's published methodology of averaging repeated sweeps.

### 5.3 The two-stage plant — effective vs. airframe

**This section is the correctness core of the tool. Read it before writing any analysis code.**

The gyro signal that is routinely logged is the *filtered* one the controller consumes:

- ArduPilot: `AC_AttitudeControl_Multi::rate_controller_run()` reads `_ahrs.get_gyro_latest()`, which is post `INS_GYRO_FILTER` and post harmonic notch. That value is what appears in `RATE.{R,P,Y}` and `PID{R,P,Y}.Act`. `RATE.{R,P,Y}Out` is the normalized mixer command.
- PX4: `vehicle_angular_velocity` is post `IMU_GYRO_CUTOFF` and post notches; `vehicle_torque_setpoint` is the normalized command.

Genuinely pre-filter gyro exists only in batch/FIFO logging that is **off by default** (§13), is block-discontinuous, and runs at a different rate from the control loop.

Therefore:

```
EffectivePlant(jw)  =  F_current(jw) * G_air(jw) * e^{-tau*jw}      <- what we measure
G_air(jw)           =  EffectivePlant(jw) / F_current(jw)           <- what we fit
```

**Pipeline:**

1. **Measure `EffectivePlant`.** FRF from `u = rate.{axis}.output → y = rate.{axis}.measured`. Where SYSTEMID data exists, prefer the **injected chirp** as the reference input (`SIDD.Targ`) — far better SNR than the total mixer command, which also contains the controller's own reaction to the vehicle. Record which was used in `EffectivePlant.source`.
2. **Build `F_current`** from the logged parameter snapshot via `core/filters/chain.py` (§5.5), evaluated at the segment's operating point (hover throttle / mean motor RPM, since notch centers track).
3. **Validate the filter model against the data before trusting it.** The measured `EffectivePlant` magnitude should show notch dips where `F_current` predicts them. Compare center and depth; if they disagree beyond threshold, emit `FILTER_MODEL_MISMATCH` (warning) and downgrade confidence. This one check catches a wrong `REF`, a wrong `FM_RAT`, a notch that never tracked, an ESC telemetry dropout, and firmware differences — all of which would otherwise silently corrupt every downstream number.
4. **Divide out.** `G_air = EffectivePlant / F_current` over the coherence-valid band only. This is the **only** place in the codebase where filters are divided out. Clamp `|F_current|` away from zero — inside deep notches the division is ill-conditioned — and mark those bins invalid rather than producing enormous values.
5. **Fit `AirframeModel`** to `G_air` (§5.4).
6. **Cross-check with raw gyro when available.** If `BatchSamples` includes pre-filter data, additionally identify directly from raw gyro over the same window and report the discrepancy between the two routes. Agreement is a strong confidence signal; disagreement points at the filter model.

`AirframeModel.filter_deconvolution` records which route produced it and must be displayed next to the model in the GUI.

### 5.4 Frequency response and model fitting (`analysis/spectra.py`, `analysis/sysid.py`)

**Non-parametric FRF.**
```
f, Pxx   = welch(u, fs, nperseg=N, noverlap=N//2, window='hann')
f, Pxy   = csd(u, y, fs, nperseg=N, ...)
H        = Pxy / Pxx
gamma2   = coherence(u, y, fs, nperseg=N, ...)
```
Choose `nperseg` for ≥ 5 averages while still resolving the lowest excited frequency. Provide log-spaced composite-window smoothing (CIFER-style multi-window averaging) to reduce variance at high frequency. Mark `valid_mask` where `gamma2 >= [coherence].threshold` **and** f is inside the excited band.

Combine multiple segments/files by coherence-weighted averaging of `H` on a common log-spaced grid, with combined coherence computed from the summed spectra (not by averaging coherences).

**Parametric fit** by weighted nonlinear least squares over the valid band, weighting each frequency by coherence and by `1/f` (log-spaced weighting, so low frequencies aren't drowned out by the denser high-frequency bins).

Default structure `so_delay`:
```
G_air(s) = K * wn^2 / (s^2 + 2*zeta*wn*s + wn^2) * exp(-tau*s)
```
Also offer `fo_delay` (`K/(T s + 1) · e^{-τs}`, often sufficient for yaw) and a discrete ARX/OE option for cross-checking against PX4's onboard identifier.

Cost = weighted sum of squared magnitude error (dB) and phase error (deg), with phase weighted ~0.5× magnitude. Initialize `tau` from the high-frequency phase slope, `wn` from the −3 dB point, `K` from the low-frequency gain. Multi-start over a small grid to avoid local minima. Report `fit_rms_db` / `fit_rms_deg`; flag `MODEL_FIT_POOR` beyond the `[fit]` thresholds. Reject fits outside `[fit]` sanity bounds and say so.

**`tau` semantics.** `tau` is fitted against `G_air`, i.e. *after* the filter chain is divided out. It therefore represents actuator/ESC/motor lag plus unmodeled dynamics — **not** filter lag. Do not add filter delay to it; `LatencyBudget` keeps the terms separate. If `tau` exceeds `[fit].tau_bounds_ms[1]`, that is a real finding (`DELAY_HIGH`) pointing at ESC protocol, motor response, or loop rate — not a modeling artifact.

**Fallback view.** `analysis/deconv.py` provides a Wiener-deconvolution estimate of the closed-loop setpoint→gyro step response (the PIDtoolbox approach), usable on any flight with broadband stick input. Present it as a *validation and teaching* view, never as the basis for a gain recommendation, and state its LTI assumption in the UI — real vehicles have rate limiting, thrust nonlinearity and motor saturation that violate it.

### 5.5 Filter engine (`core/filters/`) — firmware-exact, discrete

Both identification (dividing out) and design (multiplying in) depend on this being right, so it is its own package with its own parity tests.

**ArduPilot notch** (`libraries/Filter/NotchFilter.cpp`) — implement exactly:
```
A       = 10^(-attenuation_dB / 40)
octaves = 2 * log2( f0 / (f0 - BW/2) )         # requires f0 > BW/2, else Q = 0 (disabled)
Q       = sqrt(2^octaves) / (2^octaves - 1)
w       = 2*pi*f0/fs
alpha   = sin(w) / (2*Q)
b = [1 + alpha*A^2,  -2*cos(w),  1 - alpha*A^2]
a = [1 + alpha,      -2*cos(w),  1 - alpha]    # normalize both by a0
```
Verified numerically: `|H(f0)| = -ATT` dB exactly, and `|H(f0 ∓ BW/2)| ≈ -3 dB` — slightly asymmetric in discrete time, with the asymmetry growing as `f0/fs` grows. That asymmetry is exactly why analog approximations are banned here.

**ArduPilot low-passes.** `INS_GYRO_FILTER` is a `DigitalBiquadFilter` (2-pole). `ATC_RAT_*_FLTT/FLTE/FLTD` are 1-pole IIRs with `alpha = dt / (dt + 1/(2*pi*fc))`, run at the loop rate. These are not interchangeable and their phase differs materially: at a 400 Hz loop rate a 20 Hz `FLTD` costs ≈ 26° at 10 Hz and ≈ 40° at 20 Hz — frequently the single largest term in the budget.

**Harmonic notch stack** (`harmonic.py`): notches at `n*f0` for each enabled harmonic in `HMNCS`. Details verified against `HarmonicNotchFilter.cpp`, each of which changes the phase the loop sees:

- `A` and `Q` are computed **once**, at the fundamental, from `bandwidth / composite_notches`. Harmonics reuse them and only scale the centre, making the stack **constant-Q** — harmonic *n* has *n* times the absolute bandwidth. (Do not implement this as an explicit per-harmonic bandwidth rule; the mechanism is the shared `Q`.)
- Composite notches are not symmetric about an implied centre: `OPTS` bit0 (double) places notches at `1 ∓ spread` with **no** centre notch, while bit4 (triple) places them at `1.0` and `1 ∓ spread`, where `spread = BW / (32 · f0)`. Prefer triple per upstream guidance; never set both.
- Below `FM_RAT · FREQ` the firmware **fades** attenuation toward unity rather than switching off, and disables the notch entirely below 25% of that minimum (`NOTCHFILTER_ATTENUATION_CUTOFF`). The `TreatLowAsMin` option instead scales the minimum per harmonic.
- Notches at or above `0.48 · fs` (`HARMONIC_NYQUIST_CUTOFF`) are dropped.
- bit1 = per-motor multi-source, instantiating one notch set per motor driven by that motor's RPM.

**PX4** (`chain.py`): `IMU_GYRO_CUTOFF` 2nd-order LPF, `IMU_DGYRO_CUTOFF` applied to the derivative, static notches `IMU_GYRO_NF0/NF1_FRQ/BW`, dynamic notch bank from `IMU_GYRO_DNF_EN` (bit0 ESC RPM, bit1 FFT) with `IMU_GYRO_DNF_MIN/BW/HMC`.

**Response evaluation.** `FilterChain.response()` returns `H(e^{jωT})` on a fixed log-spaced grid at the *gyro sample rate the filters actually run at*. Cache per (chain, operating point) — the joint optimizer evaluates this thousands of times.

**`latency.py`** builds `LatencyBudget`: gyro LPF, each notch, D-term LPF, error LPF, ZOH (`0.5/f_loop`), controller compute (default `1.0/f_loop`, documented and configurable), actuator/ESC protocol (small table keyed on `MOT_PWM_TYPE` / PX4 equivalent: PWM 490 Hz, OneShot, DShot classes), and the identified residual `tau`. Displayed as a stacked bar at the design crossover (§10.4) — the tool's clearest teaching artifact, because it shows *where the phase went*.

**Worked example, to keep in `docs/methodology.md`** (computed with the formulas above at `fs = 4 kHz`): a 3-harmonic stack at 80/160/240 Hz with the default 2:1 FREQ:BW costs **−7.6° at 10 Hz, −15.7° at 20 Hz, −24.6° at 30 Hz**. The same stack at 4:1 costs **−3.5°, −7.2°, −11.4°**. That difference is typically worth 20–30% of crossover frequency — the concrete reason the tool optimizes bandwidth instead of defaulting it.

**Acceptance test — the aircraft is the ground truth.** With `INS_LOG_BAT_OPT = 4` (pre *and* post-filter batches) or `INS_RAW_LOG_OPT = 9` on H7, the vehicle logs both sides of its own filter chain. Run our `FilterChain` over the logged pre-filter samples and require agreement with the logged post-filter samples within **1 dB magnitude and 5° phase over 5–200 Hz**. No synthetic test substitutes for this.

### 5.6 Noise analysis and filter recommendation (`analysis/noise.py`, `design/filters.py`)

A large share of "autotune produced an unflyable tune" cases are D-term noise, not bad gains. This section is as much the reason the tool exists as the gain design is.

**Characterization** (`NoiseProfile`):
- PSD and spectrogram of gyro per axis over the whole flight — pre-filter where available, post-filter always.
- **Motor fundamental track** `f(t)`, per motor where possible, from — in order of preference — ESC telemetry (`ESC` / `esc_status`), RPM sensor (`RPM`), onboard FFT (`FTN1`/`FTN2`, `sensor_gyro_fft`), or the throttle model.
- PSD binned by throttle, so the throttle→frequency relationship is *fitted* rather than assumed.
- **Peak classification** — the step that makes the recommendations honest. For each prominent peak, correlate its center frequency against the motor track over the flight:
  - correlates (≥ `[noise].rpm_track_correlation_min`) and sits near `n × f_motor` → `motor_fundamental` / `motor_harmonic` → **dynamic tracking notch**
  - prominent but fixed in frequency → `structural` → **static notch (`INS_HNTC2_*` / `IMU_GYRO_NF1_*`) *and* a mechanical finding**, because a tracking notch is the wrong tool for a frame resonance and will chase it uselessly
  - no distinct peak, elevated floor → `broadband` → LPF and/or a mechanical fix (balance, soft-mounting, damaged props)
- Verify the *existing* notch: measured attenuation achieved at the fundamental. Mistracking → `NOTCH_MISTRACKING`.
- **Noise-limited Kd ceiling**: propagate the measured pre-filter gyro PSD through `Kd·s/(τ_d s+1)·F(s)` and find the Kd at which D-term output RMS exceeds `[noise].dterm_output_rms_limit_pct` of full motor range. Hard constraint in §5.7.
- Motor saturation / clipping from `RCOU` / `actuator_motors`; gyro clipping from `vehicle_imu_status`.
- Energy above the loop-rate Nyquist → `ALIASING_RISK`.

**Recommendation ladder** — deterministic and enumerable, not a free search. Each step records its rejected alternatives into `FilterRecommendation.rejected`.

1. **Source selection.** ESC telemetry (`INS_HNTCH_MODE=3` / `IMU_GYRO_DNF_EN` bit0) > RPM sensor (2 / 5) > in-flight FFT (4 — note its documented tracking lag, which can make it *worse* than a well-set throttle notch) > throttle (1) > static (0). Gate each on what the log **proves** is available, and on board CPU class for the expensive options. If a better source is available than the one in use, emit `NOTCH_SOURCE_SUBOPTIMAL` with the specific parameter changes.
2. **Center and reference.**
   - Throttle mode: `INS_HNTCH_REF = MOT_THST_HOVER`, `INS_HNTCH_FREQ = hover_freq` measured from the log, `INS_HNTCH_FM_RAT = min_freq / hover_freq` (0.7–1.0). To extend tracking below hover: `REF = hover_thrust * (min_freq / hover_freq)^2`, with `min_freq` the lowest observed motor frequency (typically in descent / prop wash).
   - ESC mode: `INS_HNTCH_REF = 1` (scaling disabled), `INS_HNTCH_FREQ` = lowest motor frequency worth tracking — below hover but above the gyro LPF corner.
   - PX4 dynamic notch: `IMU_GYRO_DNF_MIN` from the same measured minimum.
3. **Harmonics** (`HMNCS` / `IMU_GYRO_DNF_HMC`). Include harmonic *n* only if (a) its measured peak exceeds `[filters].target_noise_floor_db`, and (b) the D-term noise reduction it buys exceeds `[filters].harmonic_min_benefit_db` for the phase it costs at the design crossover. Cap at `[filters].max_harmonics`.
4. **Bandwidth and attenuation.** Smallest `(BW, ATT)` that brings each targeted peak to the noise-floor target, given the measured peak width plus an allowance for RPM-tracking jitter. Default `FREQ:BW = 2:1`; `4:1` when per-motor tracking is enabled. `ATT` within `[filters].att_min_db … att_max_db`. Reject any configuration whose chain phase at the design crossover exceeds `[filters].phase_budget_deg`.
5. **`OPTS`.** Triple (bit4) in preference to double (bit0) — never both, and never together with bit1. Per-motor (bit1) when ESC telemetry is present and the board has headroom, with the 4:1 bandwidth ratio. Loop-rate update (bit2) on H7-class boards. Flag bit3 (all IMUs) as CPU-expensive and leave it off by default.
6. **Gyro LPF and D-term LPF.** Choose `INS_GYRO_FILTER` / `IMU_GYRO_CUTOFF` and `ATC_RAT_*_FLTD` / `IMU_DGYRO_CUTOFF` from post-notch residual noise versus phase cost, inside the joint optimization. Use the established rules of thumb as **priors and sanity checks, not as the answer**: `FLTD = INS_GYRO_FILTER/2` on roll/pitch and `/4` on yaw, never above `0.75 × INS_GYRO_FILTER`; PX4's documented cutoff/latency points (30 Hz ≈ 8 ms, 60 Hz ≈ 3.8 ms, 120 Hz ≈ 1.9 ms) and typical values (`IMU_GYRO_CUTOFF` ~80 Hz larger craft, ~120 Hz racers; `IMU_DGYRO_CUTOFF` 50–80 Hz; don't spread the two far apart). Whenever the optimizer departs from a prior, it must say so in words in the rationale.
7. **Predicted post-filter spectrum.** Push the measured pre-filter PSD through the proposed chain into `FilterRecommendation.predicted_psd_post`. Where a post-filter measurement also exists, overlay measured-vs-predicted for the *current* chain as a credibility check the user can see with their own eyes.

### 5.7 Joint filter + gain design (`design/joint.py`, `design/objectives.py`)

Filters and gains are solved **together** against one phase budget. Designing gains against fixed filters leaves performance on the table; designing filters without knowing the crossover over-filters.

Broken loop:
```
L(s) = C_fb(s) * F(s) * G_air(s) * D_loop(s)
```
where `C_fb(s)` is the stack-specific feedback-path controller (§5.8), `F(s)` the candidate `FilterChain` (discrete response), `G_air(s)` the identified airframe, and `D_loop(s)` the ZOH + compute + actuator delay from `LatencyBudget`. `tau` is already inside `G_air`; it is not added again here.

**Structure:**
- **Outer loop** enumerates a small ordered candidate set: source × harmonic set × (BW, ATT) tier × LPF tier, ordered by increasing phase cost — so the search naturally reports "the cheapest configuration that meets the noise target".
- **Inner loop** runs the margin-constrained gain search for each candidate. Parameterize as (crossover frequency, Ki/Kp ratio, Kd/Kp ratio) rather than raw gains — far better-conditioned. `scipy.optimize` differential evolution for the global pass, Nelder–Mead polish.

**Objective:** maximize disturbance-rejection bandwidth (DRB), subject to
- Phase margin ≥ `PM_min` (default 45°, user-adjustable 35–60°, **hard floor 25°**)
- Gain margin ≥ `GM_min` (default 6 dB, adjustable 4–10 dB)
- Peak sensitivity `||S||∞` ≤ `Ms_max` (default 6 dB)
- Disturbance-rejection peak (DRP) within bound
- Crossover ≤ `min(0.25 · 1/(2π·tau), [margins].crossover_frac_of_loop · f_loop/2)`
- Predicted D-term output RMS ≤ `[noise].dterm_output_rms_limit_pct`
- Filter chain phase at crossover ≤ `[filters].phase_budget_deg`
- Filter CPU cost within the board's class

Report DRB **and** DRP alongside GM/PM/crossover in every `MarginReport`, and record which constraint stopped the optimizer in `TuneRecommendation.binding_constraint` — that single string is the most useful explanatory output the tool produces.

**Conservatism slider (0..1)** maps to `PM_min` and crossover target: 0 = aggressive (PM 40°, higher crossover), 1 = docile (PM 60°, crossover backed off ~40%). Default 0.5.

**Interactive budget: < 300 ms re-solve.** Precompute `G_air(jω)` once and each candidate chain's `F(jω)` once on the fixed grid; inside the inner loop only `C_fb(jω)` changes. Coalesce rapid slider changes with a 50 ms debounce.

**FF and outer loop.** Recommend `kff` for ArduPilot from the identified low-frequency gain. Recommend outer-loop `ATC_ANG_*_P` / `MC_*_P` from the achieved inner-loop bandwidth using `[design].outer_loop_separation`, capped by `ATC_ACCEL_*_MAX`.

**Refuse rather than degrade.** If the noise floor makes the target crossover unreachable with any admissible filter configuration, say so explicitly — recommend the filter change and a *lower* crossover, with a finding explaining the cause. Never quietly lower the target and present it as success.

### 5.8 Controller models (`design/controller.py`)

The two stacks differ structurally. This changes the predicted step response and, via `FLTE`, the margins.

**ArduPilot `AC_PID::update_all()`** — `FLTT` low-passes the *target*; `error = filtered_target − measurement` is then low-passed by `FLTE`; the derivative is taken **on the filtered error** and low-passed by `FLTD`; `FF = kff × filtered target`. Feedback path:
```
C_fb(s) = [Kp + Ki/s + Kd*s*L_FLTD(s)] * L_FLTE(s)
```
`FLTT` and FF are reference-path only — they shape the step, not the margins.

**PX4 `RateControl::update()`** — `torque = P*e + I − D*(filtered angular acceleration) + FF*rate_sp`, i.e. **D on the measurement**, fed by the derivative of `vehicle_angular_velocity` low-passed at `IMU_DGYRO_CUTOFF`. No error LPF. Effective gains are `K*P`, `K*I`, `K*D` (standard vs parallel form).
```
C_fb(s) = Kp + Ki/s + Kd*s*L_DGYRO(s)
```

Each model exposes `feedback_tf()` (margins) and `closed_loop_response()` (predicted step, using the correct reference path).

**Nonlinear elements are not folded into the LTI model.** `SMAX` (slew limiter scaling P and D together via `Dmod`), `PDMX` (caps |P+D|), and `IMAX` are evaluated *post hoc* against the predicted signal amplitudes, producing `SLEW_LIMITER_ACTIVE`, `PDMAX_CLIPPING`, `INTEGRATOR_WINDUP` findings. If any would be active at the recommended gains, the recommendation is not valid as designed and must say so.

### 5.9 Operating-point sensitivity (`analysis/operating_point.py`)

Airframe gain moves with throttle, battery voltage, payload, and prop condition.

- Fit `K` per segment across differing throttle and voltage; report the spread in `AirframeModel.gain_spread_pct`.
- Large spread with throttle → `THRUST_LINEARIZATION_SUSPECT`, with `MOT_THST_EXPO` / `THR_MDL_FAC` guidance.
- Large spread with voltage → `BATTERY_SAG_LARGE`, with battery-compensation guidance.
- Feed the spread into the confidence rating and into how much margin the design holds back. A vehicle whose gain moves ±30% across the throttle range must not be tuned to a 45° margin at one operating point and called done.

### 5.10 Validation mode

Given two logs (before/after), compare tracking error, overshoot, settling, and D-term noise side by side, and check the *measured* response against the *predicted* closed-loop response from the model. Also overlay the earlier session's predicted post-filter spectrum against the measured post-filter spectrum in the new log — direct evidence of whether the filter recommendation did what it claimed.

Prediction-vs-outcome agreement is the single most useful trust signal the tool can give, so this is a first-class screen, not a footnote.

---

## 6. Log ingestion

### 6.1 ArduPilot (`core/io/ardupilot.py`)

Read `.bin` via `pymavlink.DFReader.DFReader_binary`.

| Message | Fields | Purpose | Required |
|---|---|---|---|
| `RATE` | `RDes,R,PDes,P,YDes,Y,AOut,ROut,POut,YOut` | rate setpoint & **filtered** measurement, mixer output | **yes** |
| `PIDR`/`PIDP`/`PIDY` | `Tar,Act,Err,P,I,D,FF,Dmod,SRate,Limit` | controller internals, term-level diagnosis, `Dmod` reveals SMAX activity | **yes** |
| `ATT` | `DesRoll,Roll,DesPitch,Pitch,DesYaw,Yaw` | outer-loop context | yes |
| `SIDD` | `Time,Targ,F,Gx,Gy,Gz,Ax,Ay,Az` | **SYSTEMID chirp record** — `Targ` is the injected signal, the gold input | preferred |
| `SIDS` | sweep config | chirp parameters, exact segment bounds | preferred |
| `IMU` | `GyrX/Y/Z,AccX/Y/Z` | noise analysis (post-filter, loop rate) | yes |
| `ISBH`/`ISBD` | batch header / samples | **pre- and/or post-filter high-rate gyro** — filter verification (§5.5) | strongly preferred |
| `FTN1`/`FTN2` | onboard FFT peaks | motor-frequency track without ESC telemetry | optional |
| `ESC` | `RPM,Volt,Curr,Temp` per ESC | per-motor frequency track; enables per-motor notch | strongly preferred |
| `RPM` | `rpm1,rpm2` | RPM-sensor notch source | optional |
| `RCOU` | `C1..C8` | motor saturation detection | yes |
| `VIBE` | `VibeX/Y/Z,Clip0..2` | vibration health, clipping | yes |
| `BAT` | `Volt,Curr` | sag correlation, operating-point sensitivity | optional |
| `MOTB` | `LiftMax,ThrOut` | thrust-linearization state | optional |
| `PM` | CPU load, long loops | gates expensive notch options (`OPTS` bits) | yes |
| `PARM` | name/value | full parameter snapshot | **yes** |
| `MSG` / `EV` | text/events | mode changes, autotune progress | yes |

Parameters to extract into `LogBundle.params` (at minimum):
`ATC_RAT_{RLL,PIT,YAW}_{P,I,D,FF,FLTD,FLTT,FLTE,IMAX,SMAX,PDMX}`, `ATC_ANG_{RLL,PIT,YAW}_P`, `ATC_ACCEL_{R,P,Y}_MAX`, `ATC_INPUT_TC`,
`INS_GYRO_FILTER`, `INS_HNTCH_*`, `INS_HNTC2_*`, `INS_LOG_BAT_*`, `INS_RAW_LOG_OPT`, `INS_GYRO_RATE`,
`FFT_*`, `SCHED_LOOP_RATE`, `MOT_THST_EXPO`, `MOT_THST_HOVER`, `MOT_SPIN_MIN`, `MOT_SPIN_MAX`, `MOT_PWM_TYPE`, `MOT_BAT_VOLT_*`, `SID_*`, `LOG_BITMASK`.

`MOT_PWM_TYPE` maps to the actuator-latency table in `filters/latency.py`. `board_id` comes from the log header / `MSG` banner and gates the CPU-expensive `OPTS` bits.

Detect autotune activity from `MSG` strings and `EV` codes; record as segments with `kind="autotune_twitch"`.

### 6.2 PX4 (`core/io/px4.py`)

Read `.ulg` via `pyulog.ULog`.

| Topic | Fields | Purpose |
|---|---|---|
| `vehicle_angular_velocity` | `xyz[0..2]` | rate measurement (**post-filter**) |
| `vehicle_angular_acceleration` | `xyz[0..2]` | the signal the D term actually uses |
| `vehicle_rates_setpoint` | `roll,pitch,yaw` | rate setpoint |
| `vehicle_torque_setpoint` | `xyz[0..2]` | controller output (plant input) |
| `rate_ctrl_status` | integrator states | saturation / windup |
| `actuator_motors` | `control[]` | motor saturation |
| `sensor_combined` | `gyro_rad[]` | noise analysis |
| `sensor_gyro_fifo` | high-rate raw gyro | filter verification, pre-filter spectra |
| `sensor_gyro_fft` | detected peaks | motor-frequency track |
| `esc_status` | `esc_rpm` | per-motor frequency track |
| `vehicle_imu_status` | clipping, gyro rate, vibration metrics | health |
| `vehicle_status` / `vehicle_control_mode` | mode | segmentation |
| `battery_status` | voltage/current | sag correlation |
| `autotune_attitude_control_status` | ARX coefficients, fitness, state | **PX4's own sysid output — ingest and compare** |
| `cpuload` | load | gates expensive notch options |

Parameters from `initial_parameters` / `changed_parameters`: `MC_{ROLL,PITCH,YAW}RATE_{P,I,D,K,FF}`, `MC_{ROLL,PITCH,YAW}_P`, `MC_*RATE_MAX`, `IMU_GYRO_CUTOFF`, `IMU_DGYRO_CUTOFF`, `IMU_GYRO_NF0_FRQ/BW`, `IMU_GYRO_NF1_FRQ/BW`, `IMU_GYRO_DNF_EN/MIN/BW/HMC`, `IMU_GYRO_FFT_*`, `IMU_GYRO_RATEMAX`, `THR_MDL_FAC`, `MC_AIRMODE`, `MC_AT_*`.

**PX4 gain parameterization:** the effective gains are `K * P`, `K * I`, `K * D`. Handle this explicitly in `GainSet` conversion — getting it wrong silently produces gains off by the `K` factor. Dedicated regression test (§11).

### 6.3 Canonical signal keys

Both readers must produce the same keys so all downstream code is stack-agnostic:

```
rate.{axis}.setpoint      rad/s
rate.{axis}.measured      rad/s      POST-FILTER on both stacks (Signal.filtered = True)
rate.{axis}.output        normalized [-1, 1]   (mixer input / torque setpoint)
rate.{axis}.accel         rad/s^2    (PX4: vehicle_angular_acceleration; AP: absent)
rate.{axis}.p_term        normalized
rate.{axis}.i_term        normalized
rate.{axis}.d_term        normalized
rate.{axis}.ff_term       normalized
rate.{axis}.dmod          normalized (ArduPilot SMAX slew modifier, 1.0 = inactive)
att.{axis}.setpoint       rad
att.{axis}.measured       rad
gyro.{axis}.prefilter     rad/s      ONLY from batch/FIFO logging (Signal.filtered = False)
excite.{axis}             normalized (chirp injection, if present)
motor.{n}.output          normalized
motor.{n}.rpm             rev/min
batt.voltage              V
batt.current              A
cpu.load                  fraction
```

Missing signals are simply absent from the dict — **never zero-filled**. A `Finding` reports what's missing and how to enable it, referencing `docs/logging-setup-*.md`.

### 6.4 Performance

Logs run 50–500 MB. Requirements:
- Stream-parse; do not hold raw message objects.
- Two-pass: the first pass builds a message-type index and reports available signals fast (< 2 s) so the GUI can show what's in the log; the second pass extracts only requested signals.
- Report progress via callback `(fraction: float, message: str)` so the worker can drive a progress bar.
- Cache the parsed `LogBundle` to `~/.cache/rotorid/<sha1-of-file>.npz`; reload in < 1 s. Cache key includes the tool version.

---

## 7. Parameter mapping and export (`core/export/`)

`export/params.py` owns the only mapping between `GainSet` / `FilterChain` and on-vehicle parameter names. It is bidirectional (`params → objects` for ingestion, `objects → params` for export) and round-trip tested.

**ArduPilot output** — `.param` text, loadable by Mission Planner:
gains `ATC_RAT_{RLL,PIT,YAW}_{P,I,D,FF}`, filters `ATC_RAT_*_{FLTD,FLTT,FLTE}`, `INS_GYRO_FILTER`,
notch `INS_HNTCH_{ENABLE,MODE,FREQ,BW,ATT,HMNCS,REF,FM_RAT,OPTS}` (and `INS_HNTC2_*` when a static/structural notch is recommended), outer loop `ATC_ANG_*_P`.

**PX4 output** — parameter file loadable by QGroundControl:
`MC_*RATE_{P,I,D,K,FF}` (with the `K` convention stated explicitly in the header), `MC_*_P`,
`IMU_GYRO_CUTOFF`, `IMU_DGYRO_CUTOFF`, `IMU_GYRO_NF{0,1}_{FRQ,BW}`, `IMU_GYRO_DNF_{EN,MIN,BW,HMC}`.

Every exported file carries a header comment: tool version, source log filename + hash, date, per-axis achieved margins (GM/PM/crossover/Ms/DRB), confidence, the binding constraint, and any low-confidence override the user made. Exports are grouped by **stage** (§8.2) so a user can apply the filter-only stage first.

`export/profile.py` emits a **data-collection profile** — a ready-to-load `.param` file that configures the vehicle for a good next log (§13). This is the single highest-value output for a user whose current log is inadequate.

---

## 8. Guidance engine (`core/guidance/`)

A pure-function rule engine: `evaluate(session) -> list[Finding]`. Each rule is a small registered function with its own unit test. Findings render in a persistent, always-visible panel; clicking one jumps to the relevant plot with the evidence highlighted.

### 8.1 Rules

**Data quality**

| Code | Sev | Trigger |
|---|---|---|
| `LOG_MISSING_MSG` | blocker | required message absent → link to logging setup doc |
| `LOG_RATE_IRREGULAR` | warning | timestamp jitter excessive |
| `NO_RAW_IMU_DATA` | warning | no pre-filter gyro → filter model unverifiable; gives exact `INS_LOG_BAT_MASK=1`, `INS_LOG_BAT_OPT=4` (or `INS_RAW_LOG_OPT=9`) / `SDLOG_PROFILE` values |
| `EXCITATION_WEAK` | warning | insufficient excitation energy/duration (< ~20 s of sweep) |
| `EXCITATION_TWITCH_ONLY` | warning | only autotune twitches available → recommend SYSTEMID, with the profile export |
| `COHERENCE_LOW` | warning | mean coherence in band < `[coherence].threshold` |
| `COHERENCE_NARROW_BAND` | warning | valid band < `[coherence].min_valid_octaves` around intended crossover |
| `MOTOR_SATURATION` | blocker | motors clipped during excitation → data invalid |
| `GYRO_CLIPPING` | blocker | IMU clipping during excitation |
| `ALIASING_RISK` | warning | significant energy above the loop-rate Nyquist |

**Model quality**

| Code | Sev | Trigger |
|---|---|---|
| `FILTER_MODEL_MISMATCH` | warning | modeled notch dips don't match measured (§5.3 step 3) — blocks trusting `G_air` |
| `MODEL_FIT_POOR` | warning | fit residual beyond `[fit]` thresholds |
| `MODEL_OUT_OF_BOUNDS` | warning | fitted parameter outside `[fit]` sanity bounds |
| `DELAY_HIGH` | warning | identified `tau` > `[fit].tau_bounds_ms[1]` → investigate ESC protocol / loop rate / motor response |
| `RAW_VS_MODELED_DISAGREE` | info | raw-gyro identification differs from the deconvolved route |
| `AXIS_COUPLING` | info | off-axis rate response significant during single-axis excitation |
| `PX4_ONBOARD_DISAGREES` | info | offline model differs materially from logged onboard ARX identification |
| `THRUST_LINEARIZATION_SUSPECT` | warning | identified gain varies strongly with throttle (§5.9) |
| `BATTERY_SAG_LARGE` | warning | identified gain varies strongly with pack voltage |

**Noise and filters**

| Code | Sev | Trigger |
|---|---|---|
| `NOTCH_NOT_CONFIGURED` | warning | clear motor peak present, no harmonic notch active |
| `NOTCH_MISTRACKING` | blocker | motor peak not attenuated by the configured notch |
| `NOTCH_SOURCE_SUBOPTIMAL` | warning | a better tracking source is available in the log than the one configured |
| `NOTCH_BW_EXCESSIVE` | warning | notch phase cost at crossover beyond `[filters].phase_budget_deg` for the attenuation gained |
| `NOTCH_HARMONIC_MISSING` | warning | a harmonic above the noise-floor target is not being notched |
| `NOTCH_TRACKING_LAG` | info | FFT-mode notch center lags the measured peak |
| `STRUCTURAL_RESONANCE` | warning | prominent peak that does **not** track RPM → mechanical fix + static notch, not a tracking notch |
| `PROP_IMBALANCE` | info | narrow, strong 1× peak inconsistent across motors |
| `GYRO_LPF_TOO_LOW` | warning | gyro LPF phase is the binding constraint on crossover |
| `GYRO_LPF_TOO_HIGH` | warning | residual noise above target reaching the D term |
| `DTERM_NOISE_LIMITED` | warning | Kd ceiling below the margin-optimal Kd |
| `ESC_TELEM_AVAILABLE_UNUSED` | info | ESC RPM present in the log but notch not using it |
| `CPU_HEADROOM_LOW` | warning | measured CPU load precludes the recommended notch options |

**Controller and design**

| Code | Sev | Trigger |
|---|---|---|
| `SLEW_LIMITER_ACTIVE` | warning | `Dmod` < 1 during excitation, or predicted active at recommended gains |
| `PDMAX_CLIPPING` | warning | `abs(P+D)` would hit `PDMX` at recommended gains |
| `INTEGRATOR_WINDUP` | warning | I-term saturated during excitation |
| `GAINS_FAR_FROM_CURRENT` | warning | recommendation > `[design].max_gain_step_ratio` × current → advise the staged ladder |
| `CROSSOVER_UNREACHABLE` | info | requested crossover impossible with any admissible filter config; explains the binding term |
| `MARGINS_HEALTHY` | good | all constraints satisfied with slack |

### 8.2 Staged next steps (`guidance/nextsteps.py`)

The generator produces a `FlightTestPlan` as an **ordered ladder**, not one jump. Each stage has: what to change, what to watch in flight, and what to check in the resulting log. Exports are grouped to match, so the user can apply one stage at a time.

1. **Filters only.** Apply the notch and LPF changes; leave all gains untouched. Hover and do gentle attitude changes. Check afterwards: motor peak attenuation in the new log, `VIBE` levels, no new low-frequency wobble.
2. **P and D.** Apply roll/pitch first; leave yaw at current values. Test at altitude with recovery room. Listen for high-frequency motor buzz on descent (D-term noise under prop wash) — if present, reduce the D-term LPF before increasing D further.
3. **I and FF.** Check attitude hold in wind and tracking of sustained inputs.
4. **Outer loop.** Apply `ATC_ANG_*_P` / `MC_*_P`; check for overshoot on step attitude commands.
5. **Re-fly a SYSTEMID sweep** with the new configuration and load it into the Validation tab to compare predicted vs. measured.

Instructions are stack-specific and concrete, e.g. *"Set `RC6_OPTION = 21` (Rate Roll/Pitch kP) with `TUNE_MIN`/`TUNE_MAX` bracketing the recommendation ±40% so you can back off in flight without landing."*

Every stage inherits the findings that motivated it, so the plan explains *why* each change is being made.

---

## 9. Concurrency model

- One `QThreadPool` for analysis jobs; jobs are `QRunnable` subclasses in `gui/workers.py` wrapping pure core functions.
- Signals: `progress(float, str)`, `finished(object)`, `failed(str, str)` (message, traceback).
- **Core functions must never import Qt.** Progress is reported via a plain callable passed in; the worker adapts it to a Qt signal.
- Cancellation via a `threading.Event` checked in parsing and optimization loops.
- Long parse and full-pipeline runs are cancellable. The fast interactive re-solve (§5.7) runs synchronously but is bounded to < 300 ms and debounced by 50 ms.
- Guard against overlapping runs: disable the run action while a job is in flight; coalesce rapid slider changes.

---

## 10. GUI design

### 10.1 Shell

`QMainWindow` with a left **stage rail** (the wizard steps), a central stacked work area, a right **findings dock**, and a bottom **log/progress dock**. The rail marks each stage as pending / running / complete / has-blocker. Users may jump backwards freely; jumping forward is gated on prerequisites.

### 10.2 Stages

1. **Load** — drag-drop or file picker. Shows detected stack, firmware, board, frame, duration, available/missing signals in a green/red table, and the parameter snapshot. Immediate "what's in this log" feedback before the full parse. If the log lacks what's needed, offer the data-collection profile export (§13) right here.
2. **Health & Noise** — noise spectrogram per axis with the tracked motor fundamental and harmonics overlaid; peak inventory with each peak's classification (motor / structural / broadband); current-notch verification; saturation, clipping, vibration, CPU load. Blockers here must be acknowledged before continuing, with an explicit "I understand this may invalidate results" checkbox recorded into `Session.acknowledgements`. This ordering is deliberate: identifying a model from a log with mistracking notches or saturated motors produces confident nonsense.
3. **Segment** — full-flight timeseries with draggable, colored segment overlays; auto-detected segments pre-populated; per-segment axis/kind/confidence table.
4. **Identify** — Bode magnitude/phase of the measured `EffectivePlant`, the modeled filter chain, and the deconvolved `G_air` with the parametric fit overlaid; coherence plotted beneath with the valid band shaded. The filter-model validation (§5.3 step 3) is shown here as a measured-vs-modeled notch overlay — the user can *see* whether the notch did what the parameters claim. Model structure selector, coherence threshold slider, refit button. Per-axis tabs.
5. **Filters** — see §10.4.
6. **Design** — conservatism slider, PM/GM/Ms constraint spinboxes, crossover target. Live-updating Nichols or Bode-with-margins, predicted step response against the current-gain step response, margin table with GM/PM/crossover/Ms/DRB/DRP, and the predicted D-term noise. The binding constraint is displayed prominently in plain language.
7. **Review & export** — parameter diff table (current → recommended, % change, per-row apply checkbox), grouped by the staged ladder (§8.2); rationale text per axis; `.param` / PX4 param export; session save; HTML/PDF report.
8. **Next flight** — the generated staged flight-test plan, printable, with in-flight tuning-knob setup instructions.
9. **Validate** (entered separately) — load a post-change log; predicted vs. measured closed-loop response, predicted vs. measured post-filter spectrum, before/after tracking and noise.

### 10.3 Plot widget requirements

All pyqtgraph-based, all with: linked x-axes where meaningful, crosshair with value readout, right-click export to PNG/CSV, and an "explain this plot" info button opening a short methodology popover. The explanatory content is a product requirement, not decoration — the tool's stated purpose includes teaching the user what the analysis means.

### 10.4 The Filters stage

The centerpiece of the added filter functionality. Four linked panels:

- **Spectrogram** of gyro over the flight with the tracked motor fundamental and its harmonics overlaid, and the configured/proposed notch centers drawn on top so tracking quality is visually obvious.
- **Spectrum comparison** — measured pre-filter PSD, measured post-filter PSD, and *predicted* post-filter PSD for the candidate configuration, on one axis. Where the current chain's prediction can be checked against the measured post-filter trace, that agreement is shown as a credibility badge.
- **Phase-lag budget** — a stacked bar at the design crossover showing every contribution from `LatencyBudget` (gyro LPF, each notch, D LPF, error LPF, ZOH, compute, actuator, airframe `tau`), with the budget limit marked. This is the single most instructive graphic in the tool: it makes the cost of filtering visible and immediate.
- **Controls and diff** — tracking source, harmonics, BW/ATT, gyro & D-term cutoffs, `OPTS` toggles, with a live parameter diff. Everything re-solves the joint problem and updates all four panels together.

### 10.5 The sandbox and "Why this number?"

The teaching layer is a product requirement, not polish.

- **Live sandbox.** Every control on the Filters and Design stages re-solves within the 300 ms budget and updates margins, predicted step, predicted noise spectrum, and the phase budget *together*. The user learns the trade by moving a slider and watching four things move at once, rather than by reading that a trade exists. Include a "reset to recommendation" affordance and a "compare against baseline" ghost overlay so exploration never loses the reference point.
- **"Why this number?"** — every recommended value has an affordance opening its trace: which model and which frequency band produced it, which constraint was binding, what the alternatives were and by how much they lost (`FilterRecommendation.rejected`, `TuneRecommendation.binding_constraint`). A value that cannot answer this is a bug (§0.3).
- **`guidance/explain.py`** — a registry of parameterized explanations, filled with the user's own numbers ("your 80 Hz notch with 40 Hz bandwidth costs 15.7° of phase at your 20 Hz crossover; halving the bandwidth costs 7.2° and still attenuates your measured peak by 34 dB"). Generic prose is worth much less than the same sentence with the user's data in it.
- **Glossary** (`docs/glossary.md`, linked from every explanation): phase margin, gain margin, crossover, sensitivity peak, coherence, DRB/DRP, notch phase lag, D-term noise, group delay, aliasing.

### 10.6 Presentation conventions

- Show frequencies in Hz, never rad/s, in the UI.
- Every recommended number displays its baseline alongside it and the direction of change.
- Confidence is always visible next to any recommendation; low-confidence recommendations render in a muted/warning style and cannot be exported without an explicit override.
- Signal provenance is always visible where it matters: any plot of `rate.{axis}.measured` is labelled post-filter.
- Support light and dark themes; make plot colors theme-aware and colorblind-safe (avoid red/green as the sole distinction).

---

## 11. Testing strategy

**Synthetic ground truth is the primary correctness check**, with one important exception: the filter engine is checked against the aircraft itself.

In `tests/synthetic/`:
1. `make_airframe(K, wn, zeta, tau)` → a known `control` system.
2. `make_chain(params, stack)` → a known filter chain.
3. `simulate_chirp(airframe, chain, gains, noise_psd, f0, f1, T, fs)` → closed-loop simulation producing signals in the same canonical form as a real log, **including the post-filter measurement path**, so the effective/airframe distinction is exercised.
4. `write_fake_dflog(signals)` / `write_fake_ulog(signals)` → synthetic logs in real container formats, so the IO layer is exercised end to end.

Required tests:

- **Filter parity (firmware ground truth)**: modeled post-filter spectrum vs. logged post-filter spectrum from a real pre+post log, within 1 dB / 5° over 5–200 Hz.
- **Notch coefficient parity**: `|H(f0)| = -ATT` dB exactly; `|H(f0 ∓ BW/2)| ≈ -3 dB`; `Q = 0` path when `f0 <= BW/2`.
- **Deconvolution round trip**: simulate with a known chain and airframe, identify the effective plant, divide out the modeled chain, and recover `K, wn, zeta, tau` within 10% (5% for `tau`).
- **Double-counting guard**: identifying from a simulated *post-filter* measurement and then designing must produce the same margins as identifying from the simulated *pre-filter* measurement. If filter phase is counted twice, this test fails. Treat it as the canary for §0.6.
- **Identification accuracy**: recover parameters from clean synthetic chirps; degrade gracefully and report lower confidence as noise increases.
- **Round-trip units**: PX4 `K`-scaled gains and ArduPilot gains convert to identical effective controllers. Explicit regression test.
- **Controller structure**: D-on-error and D-on-measurement controllers with identical gains produce identical margins but different closed-loop step overshoot.
- **Margin correctness**: designed gains, when re-analyzed, produce the margins that were requested (within 1 dB / 2°).
- **Design monotonicity**: increasing conservatism never increases crossover or decreases phase margin.
- **Noise ceiling**: injecting more gyro noise reduces the recommended Kd.
- **Filter phase**: adding a low-pass, or widening a notch, reduces the achievable crossover.
- **Filter recommendation**: synthetic noise with known motor harmonics yields the expected tracking source, harmonic set, and bandwidth; a fixed-frequency peak is classified `structural` and does **not** produce a tracking notch.
- **Peak classification**: an RPM-tracking peak and a fixed peak in the same synthetic log are separated correctly.
- **Phase budget**: no recommended configuration ever exceeds `[filters].phase_budget_deg` at the design crossover.
- **IO robustness**: truncated logs, missing messages, zero-length segments, NaNs, duplicate timestamps, single-sample messages, empty batch blocks — all produce `Finding`s rather than exceptions.
- **Export round trip**: `params → objects → params` is identity for both stacks.
- **GUI smoke** (pytest-qt): each stage renders, worker signals connect, cancellation works, no analysis on the GUI thread (assert via thread-affinity check), sandbox re-solve within budget.
- **Determinism**: same input twice → byte-identical session JSON, including `config_hash`.

Collect a small set of real logs (ArduPilot SYSTEMID, ArduPilot autotune-only, ArduPilot with pre+post batch logging, PX4 autotune) as *characterization* tests: pin current outputs and detect unintended changes, without asserting truth. The pre+post log is the exception — it asserts truth for the filter engine.

---

## 12. Milestones

Vertical slice first. Each milestone must be independently demoable and leave the repo green.

| M | Deliverable | Acceptance criteria |
|---|---|---|
| **M0** | Skeleton | `pyproject.toml`, package layout, `rotorid.toml` + `config.py`, CI running ruff+mypy+pytest, all §3 dataclasses defined with docstrings |
| **M1** | **Walking skeleton — ArduPilot, CLI only, end to end**: `.bin` → SYSTEMID segment → FRF → filter-chain model → `G_air` fit → margins → gain recommendation → HTML report | One real SYSTEMID log produces a fully traceable recommendation; synthetic chirp recovers `K, wn, zeta, tau` within 10% / 5%; the double-counting guard test passes |
| **M2** | Filter engine (`core/filters/`) | Firmware parity: modeled post-filter spectrum matches logged post-filter within 1 dB / 5°, 5–200 Hz; notch coefficient parity tests pass |
| **M3** | Noise analysis + peak classification + notch recommendation | Synthetic noise with known harmonics yields the expected source/harmonics/BW; a structural peak is classified as structural and gets a static notch + mechanical finding |
| **M4** | Joint filter+gain optimizer | Margin round-trip within 1 dB / 2°; monotonicity and phase-budget tests pass; re-solve < 300 ms |
| **M5** | Guidance engine + staged flight plan | Every §8.1 rule implemented and unit-tested; `FlightTestPlan` generated with stage grouping |
| **M6** | GUI shell + stages Load / Health & Noise / Segment / Identify | Interactive, threaded, cancellable; thread-affinity assertion passes |
| **M7** | GUI Filters + Design + sandbox + "Why this number?" | All four Filters panels linked and live; re-solve holds 300 ms with real data; every recommended value answers its trace |
| **M8** | Export, report, session save/load, safety gates | Param files verified loadable by Mission Planner; blocked exports enforced; export round-trip test passes |
| **M9** | PX4 IO + PX4 controller model + PX4 filter mapping | Canonical keys identical across stacks; `K`-scaling and D-on-measurement tests pass; PX4 autotune ARX ingested and compared |
| **M10** | Validation mode | Predicted-vs-measured overlay for both closed-loop response and post-filter spectrum, on a real before/after pair |
| **M11** | Packaging & docs | PyInstaller builds for Windows/macOS/Linux; `docs/` written including the worked phase-budget example; first-run sample session bundled |

---

## 13. Logging setup docs (`docs/logging-setup-*.md`)

These must be actionable recipes, not prose, because the quality of everything downstream depends on them. The tool can emit them as a ready-to-load parameter file via `export/profile.py`.

**ArduPilot — a good identification flight**
- SYSTEMID: `SID_AXIS` = 7 / 8 / 9 for rate roll / pitch / yaw (10–12 inject at the mixer instead); `SID_MAGNITUDE` sized for a clear response without saturating — adjustable in flight by setting `TUNE = 58`; `SID_F_START_HZ` / `SID_F_STOP_HZ` — ArduPilot's own published multicopter methodology sweeps **0.05 → 5 Hz** with ~5 s `SID_T_FADE_IN` / `SID_T_FADE_OUT` and ~130 s total per axis, extended higher for small, fast vehicles; one axis per flight, repeated sweeps averaged.
- Logging: `LOG_BITMASK` with the PID bit set (required for `PID*` messages).
- Pre/post-filter gyro: `INS_LOG_BAT_MASK = 1`, `INS_LOG_BAT_OPT = 4` (pre *and* post-filter 1 kHz sampling) → `ISBH`/`ISBD`. On H7 boards, `INS_RAW_LOG_OPT = 9` (bit 0 + bit 3: primary gyro only, pre and post filter) is preferred. Set `INS_LOG_BAT_MASK = 0` afterwards to free RAM.
- Fly: hover, then the sweep, at altitude with recovery room, in low wind.
- Warn explicitly: raw/batch logging consumes log bandwidth and RAM; turn it off after the tuning campaign.

**PX4 — a good identification flight**
- `SDLOG_PROFILE` with high-rate logging enabled; `IMU_GYRO_FFT_EN = 1` for onboard peak detection; ESC telemetry enabled if the hardware supports it (`esc_status.esc_rpm` is what unlocks per-motor notch analysis).
- Autotune: `MC_AT_EN = 1`, `MC_AT_SYSID_AMP` raised in steps of 1 if identification fails, `MC_AT_APPLY = 0` while gathering data for offline analysis.
- Fly ~30 s of hover with deliberate roll / pitch / yaw excitation, or run autotune.

---

## 14. CLI

```
rotorid inspect LOG                     # available signals, params, duration, what's missing
rotorid analyze LOG -o session.rotorid [--axes roll,pitch] [--conservatism 0.5]
rotorid filters LOG [--json]            # filter recommendation only (fast path)
rotorid recommend session.rotorid --format ardupilot -o tune.param [--stage filters|pd|i|outer]
rotorid profile --stack ardupilot -o collect.param     # data-collection profile (§13)
rotorid report session.rotorid -o report.html
rotorid validate BEFORE.bin AFTER.bin -o comparison.html
```

Exit codes: `0` success, `2` blocking findings present, `3` unparseable log. Machine-readable `--json` output for every command so the tool can be scripted in a batch tuning workflow.

---

## 15. Non-goals (explicitly out of scope for v1)

- Fixed-wing, VTOL transition, and helicopter tuning (design the model structure enums to leave room, but do not implement).
- Live MAVLink telemetry connection or writing parameters to the vehicle. Export files only — the human loads them deliberately.
- Position/velocity loop tuning.
- Automatic in-flight iteration.
- Accelerometer / EKF filter tuning. The gyro path only.

---

## 16. Safety requirements

These are not optional polish; the tool's output goes onto a flying aircraft.

1. **Never write parameters to a vehicle, and never apply filters or gains automatically.** File export only. Do not add vehicle-write capability without a separate safety review.
2. Every exported param file carries a header comment: tool version, source log, date, achieved margins per axis, confidence, and the binding constraint.
3. Low-confidence recommendations require an explicit user override to export, and the override is recorded in the export header and in `Session.acknowledgements`.
4. The export dialog and every report include: a reminder to back up current parameters, and the instruction to test at altitude with recovery room.
5. The report states plainly that identified models vary with payload, battery state, and prop condition (§5.9 quantifies this for the specific vehicle), and that the recommendation is a well-justified starting point with designed margins — not a validated final tune.
6. If blocking findings exist, exports are disabled until acknowledged, and the acknowledgement text is recorded in the session.
7. Filter changes and gain changes are exported as **separate staged files** by default (§8.2). Changing filters and gains in one flight makes a bad outcome un-diagnosable.

---

## 17. Suggested build order for the agent

Work strictly bottom-up within each milestone, and keep tests green at every step:

`types.py` + `config.py` + `units.py` → synthetic generators → ArduPilot IO → resample/segment → spectra → **filter engine** → sysid (effective → airframe) → margins → gain design → report → **M1 demo** → noise/peak classification → filter recommender → joint optimizer → guidance → CLI completion → GUI shell → GUI stages in order → sandbox → export → PX4 → validation → packaging.

Build the CLI before the GUI. If the CLI can produce a good, traceable filter+gain recommendation from a log, the GUI is presentation work; if it can't, no amount of GUI will save it.
