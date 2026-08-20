"""Ground-truth generators: a known airframe, seen through a known filter chain.

Every identification test in the project is checked against these, because they
are the only place where the true ``K, wn, zeta, tau`` are known exactly.

The one property that matters most here is that :func:`simulate_effective`
produces the **post-filter** measurement -- the same thing an ArduPilot ``RATE.R``
or a PX4 ``vehicle_angular_velocity`` sample is. A generator that handed back the
bare airframe response would make the double-counting bug invisible to the whole
test suite, which is exactly how that bug survived into spec revision 1.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

from rotorid.core.analysis.model_eval import airframe_response
from rotorid.core.filters.chain import FilterChain, OperatingPoint
from rotorid.core.filters.harmonic import HarmonicNotch
from rotorid.core.types import AXES, AirframeModel, Axis, LogBundle, Signal, Stack

FloatArray = NDArray[np.float64]

__all__ = [
    "chirp",
    "make_airframe",
    "make_bundle",
    "make_chain",
    "make_noise_bundle",
    "motor_noise",
    "simulate_airframe",
    "simulate_effective",
    "swept_tones",
    "with_axis",
]


def make_airframe(
    axis: Axis = "roll",
    *,
    K: float = 12.0,
    wn_hz: float = 2.5,
    zeta: float = 1.0,
    tau_ms: float = 18.0,
    structure: str = "so_delay",
) -> AirframeModel:
    """A plausible small-multirotor roll axis, with exactly known parameters.

    The rate response is a couple of Hz wide, not a couple of tens. That is why
    ArduPilot's published multicopter identification sweeps 0.05-5 Hz: above a few
    Hz there is little left to identify, and the loop's phase crossing is set by
    delay rather than by airframe dynamics.

    The calibration is that stock ArduPilot gains (``P = I = 0.135``,
    ``D = 0.0036``) must come out stable and conservative against these defaults --
    roughly 90 degrees of phase margin and 11 dB of gain margin. They do. A
    fixture with a 6 Hz or 20 Hz rate response looks plausible and fails that
    check badly: the plant is still flat where the loop crosses -180 degrees, so
    the gain-margin constraint caps the proportional gain an order of magnitude
    below anything a real vehicle flies, and every design comes out degenerate.
    """
    if structure == "so_delay":
        params = {"K": K, "wn": 2.0 * np.pi * wn_hz, "zeta": zeta, "tau": tau_ms / 1000.0}
    elif structure == "fo_delay":
        params = {"K": K, "T": 1.0 / (2.0 * np.pi * wn_hz), "tau": tau_ms / 1000.0}
    else:  # pragma: no cover - guards against a silent typo in a test fixture
        raise ValueError(f"unsupported synthetic structure {structure!r}")

    return AirframeModel(
        axis=axis,
        structure=structure,  # type: ignore[arg-type]
        params=params,
        fit_rms_db=0.0,
        fit_rms_deg=0.0,
        valid_band_hz=(0.5, 100.0),
        coherence_mean=1.0,
        filter_deconvolution="none",
    )


def make_chain(
    *,
    gyro_sample_rate_hz: float = 4000.0,
    loop_rate_hz: float = 400.0,
    gyro_lpf_hz: float | None = 60.0,
    notch_freq_hz: float | None = 90.0,
    notch_bw_hz: float = 45.0,
    notch_att_db: float = 40.0,
    harmonics: tuple[int, ...] = (1, 2, 3),
    notch_opts: int = 0,
    dterm_lpf_hz: float | None = 20.0,
    error_lpf_hz: float | None = None,
    target_lpf_hz: float | None = None,
) -> FilterChain:
    """A filter chain resembling a stock ArduPilot setup with a harmonic notch."""
    notches: tuple[HarmonicNotch, ...] = ()
    if notch_freq_hz:
        notches = (
            HarmonicNotch(
                freq_hz=notch_freq_hz,
                bandwidth_hz=notch_bw_hz,
                attenuation_db=notch_att_db,
                harmonics=harmonics,
                sample_rate_hz=gyro_sample_rate_hz,
                opts=notch_opts,
            ),
        )
    return FilterChain(
        stack="ardupilot",
        sample_rate_hz=gyro_sample_rate_hz,
        loop_rate_hz=loop_rate_hz,
        gyro_lpf_hz=gyro_lpf_hz,
        notches=notches,
        dterm_lpf_hz=dterm_lpf_hz,
        error_lpf_hz=error_lpf_hz,
        target_lpf_hz=target_lpf_hz,
    )


def chirp(
    *,
    sample_rate_hz: float,
    duration_s: float,
    f_start_hz: float = 0.5,
    f_stop_hz: float = 60.0,
    amplitude: float = 0.1,
    fade_s: float = 3.0,
    log_sweep: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """A faded frequency sweep, shaped like the one ArduPilot SYSTEMID injects.

    Linear in frequency by default, matching SYSTEMID. That is not cosmetic: a
    logarithmic sweep of constant amplitude spends most of the record at the
    bottom of the band, so its spectrum falls as ``1/f`` and the low-frequency
    energy leaks upward through the analysis window. Against a plant that is
    already 30 dB down at the top of the sweep, that leakage was worth several dB
    of bias in the top octave -- with coherence still reading near unity, because
    leakage is perfectly repeatable and coherence cannot see bias.

    The fade matters for more than realism: it makes the record start and end at
    zero, which is what lets the airframe be applied by FFT without the delay term
    wrapping around.

    Returns:
        ``(t, u)`` in seconds and normalized command units.
    """
    n = round(duration_s * sample_rate_hz)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz

    # SYSTEMID structure: amplitude fades in at the *start* frequency, the sweep
    # happens over the record window, and the amplitude fades out at the stop
    # frequency. Sweeping during the fade -- the obvious simplification -- puts
    # the bottom of the band entirely inside the ramp, where it is both weak and
    # amplitude-modulated, and biases the identified gain by a few tenths of a dB.
    fade = max(0.0, min(fade_s, 0.4 * duration_s))
    record_s = duration_s - 2.0 * fade
    swept = np.clip(t - fade, 0.0, record_s)

    if log_sweep:
        k = np.log(f_stop_hz / f_start_hz)
        phase = 2.0 * np.pi * f_start_hz * record_s / k * (np.exp(k * swept / record_s) - 1.0)
    else:
        rate = (f_stop_hz - f_start_hz) / record_s
        phase = 2.0 * np.pi * (f_start_hz * swept + 0.5 * rate * swept**2)
    # The held sections continue at constant frequency rather than stopping.
    phase += 2.0 * np.pi * f_start_hz * np.minimum(t, fade)
    phase += 2.0 * np.pi * f_stop_hz * np.clip(t - fade - record_s, 0.0, None)

    envelope = np.ones(n, dtype=np.float64)
    if fade > 0.0:
        m = min(round(fade * sample_rate_hz), n // 2)
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(m) / m))
        envelope[:m] = ramp
        envelope[n - m :] = ramp[::-1]

    return t, amplitude * envelope * np.sin(phase)


def simulate_airframe(u: FloatArray, sample_rate_hz: float, airframe: AirframeModel) -> FloatArray:
    """Apply the airframe model to a command, exactly, in the frequency domain.

    Uses ``exp(-tau*s)`` rather than a Pade approximation, so a sysid routine that
    recovers ``tau`` is being checked against the real thing and not against the
    same approximation it uses internally.
    """
    n = u.size
    f = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    H = airframe_response(airframe, f)
    return np.asarray(np.fft.irfft(np.fft.rfft(u) * H, n=n), dtype=np.float64)


def simulate_effective(
    u: FloatArray,
    sample_rate_hz: float,
    airframe: AirframeModel,
    chain: FilterChain | None = None,
    *,
    op: OperatingPoint | None = None,
    noise: FloatArray | None = None,
) -> FloatArray:
    """Command in, **post-filter** gyro out -- what a log actually contains.

    Noise is injected at the sensor, before the filters, because that is where it
    physically enters; injecting it afterwards would make every filter test
    meaningless.

    Args:
        u: Normalized rate-loop output (the plant input).
        sample_rate_hz: Rate of ``u``. Must match ``chain.sample_rate_hz``, since
            the biquad coefficients depend on it.
        airframe: Ground-truth plant.
        chain: Vehicle filter chain. ``None`` returns the bare airframe response.
        op: Operating point, for tracked notch centres.
        noise: Sensor noise added pre-filter, same length as ``u``.

    Raises:
        ValueError: if the chain was designed at a different sample rate.
    """
    y = simulate_airframe(u, sample_rate_hz, airframe)
    if noise is not None:
        y = y + noise
    if chain is None:
        return y
    if not np.isclose(chain.sample_rate_hz, sample_rate_hz):
        raise ValueError(
            f"chain designed at {chain.sample_rate_hz} Hz but simulating at {sample_rate_hz} Hz; "
            "the biquad coefficients would be wrong"
        )
    for stage in chain.sensor_stages(op):
        y = np.asarray(lfilter(stage.b, stage.a, y), dtype=np.float64)
    return y


def motor_noise(
    t: FloatArray,
    *,
    fundamental_hz: float = 90.0,
    harmonics: tuple[int, ...] = (1, 2, 3),
    amplitudes: tuple[float, ...] = (0.6, 0.3, 0.15),
    broadband_rms: float = 0.02,
    seed: int = 0,
) -> FloatArray:
    """Motor tones plus a broadband floor, in rad/s, for filter and noise tests."""
    rng = np.random.default_rng(seed)
    out = rng.normal(0.0, broadband_rms, size=t.size)
    for n, a in zip(harmonics, amplitudes, strict=False):
        out += a * np.sin(2.0 * np.pi * fundamental_hz * n * t + rng.uniform(0.0, 2.0 * np.pi))
    return np.asarray(out, dtype=np.float64)


def with_axis(model: AirframeModel, axis: Axis) -> AirframeModel:
    """The same model on a different axis, for multi-axis fixtures."""
    return replace(model, axis=axis)


def make_bundle(
    airframe: AirframeModel,
    chain: FilterChain,
    *,
    axis: Axis = "roll",
    duration_s: float = 90.0,
    f_start_hz: float = 0.2,
    f_stop_hz: float = 20.0,
    amplitude: float = 0.1,
    fade_s: float = 4.0,
    loop_rate_hz: float = 400.0,
    noise: FloatArray | None = None,
    with_motor_noise: bool = False,
    hover_hz: float = 50.0,
    sweep_fraction: float = 0.12,
    gains: tuple[float, float, float] = (0.135, 0.135, 0.0036),
    stack: Stack = "ardupilot",
    path: str = "synthetic.bin",
) -> LogBundle:
    """A complete synthetic ``LogBundle``: a SYSTEMID sweep on one axis.

    Built so the end-to-end pipeline can be tested without a real log. The
    measured rate is the **post-filter** signal, the parameter snapshot describes
    the chain that filtered it, and the injected chirp appears as ``excite.*`` --
    exactly the three things that have to line up for the deconvolution to be
    correct. If a change ever breaks that alignment, the end-to-end test fails
    even though every unit test still passes.

    Args:
        with_motor_noise: Add motor tones that track a slow throttle oscillation,
            plus the ESC telemetry that would have logged them. This is what a
            real SYSTEMID flight looks like -- the sweep does not happen in
            silence -- and it is the only fixture where the filter design and the
            gain design both have something to work with at once.
        stack: Which firmware's parameter names to write. A PX4 bundle also omits
            ``excite.*``: PX4 has no SYSTEMID mode, so a sweep in a PX4 log has to
            be found by its energy rather than read off an injected-signal
            message, and the lower confidence that follows is the truth about
            that log rather than a limitation of the fixture.
    """
    from pathlib import Path

    fs = chain.sample_rate_hz
    t, u = chirp(
        sample_rate_hz=fs,
        duration_s=duration_s,
        f_start_hz=f_start_hz,
        f_stop_hz=f_stop_hz,
        amplitude=amplitude,
        fade_s=fade_s,
    )
    fundamental = hover_hz * (1.0 + sweep_fraction * np.sin(2.0 * np.pi * 0.05 * t))
    if with_motor_noise:
        tones = swept_tones(t, fundamental, harmonics=(1, 2, 3), amplitudes=(0.5, 0.25, 0.12))
        noise = tones if noise is None else noise + tones

    y = simulate_effective(
        u, fs, airframe, chain, op=OperatingPoint(motor_hz=(hover_hz,)), noise=noise
    )

    step = round(fs / loop_rate_hz)
    t_log, u_log, y_log = t[::step], u[::step], y[::step]

    kp, ki, kd = gains
    if stack == "px4":
        params = _px4_params(chain, gains, axis, loop_rate_hz=loop_rate_hz)
        signals = _sweep_signals(axis, t_log, u_log, y_log, excite=False)
        if with_motor_noise:
            params["MOT_THST_HOVER"] = 0.35
            signals.update(_esc_signals(t, fundamental, fs))
        return LogBundle(
            path=Path(path),
            stack="px4",
            firmware_version="PX4 1.14.0 (synthetic)",
            board_id=None,
            frame_info={},
            sample_rate_hz=loop_rate_hz,
            loop_rate_hz=loop_rate_hz,
            gyro_sample_rate_hz=fs,
            signals=signals,
            params=params,
        )

    params: dict[str, float] = {
        "SCHED_LOOP_RATE": loop_rate_hz,
        "INS_GYRO_RATE": float(int(np.log2(fs / 1000.0))),
        "INS_GYRO_FILTER": chain.gyro_lpf_hz or 0.0,
        "MOT_PWM_TYPE": 6.0,
        "SID_AXIS": {"roll": 7.0, "pitch": 8.0, "yaw": 9.0}[axis],
        "SID_F_START_HZ": f_start_hz,
        "SID_F_STOP_HZ": f_stop_hz,
        "SID_MAGNITUDE": amplitude,
    }
    suffix = {"roll": "RLL", "pitch": "PIT", "yaw": "YAW"}[axis]
    params[f"ATC_RAT_{suffix}_P"] = kp
    params[f"ATC_RAT_{suffix}_I"] = ki
    params[f"ATC_RAT_{suffix}_D"] = kd
    if chain.dterm_lpf_hz:
        params[f"ATC_RAT_{suffix}_FLTD"] = chain.dterm_lpf_hz
    if chain.error_lpf_hz:
        params[f"ATC_RAT_{suffix}_FLTE"] = chain.error_lpf_hz
    if chain.target_lpf_hz:
        params[f"ATC_RAT_{suffix}_FLTT"] = chain.target_lpf_hz
    for notch in chain.notches:
        params.update(
            {
                "INS_HNTCH_ENABLE": 1.0,
                "INS_HNTCH_FREQ": notch.freq_hz,
                "INS_HNTCH_BW": notch.bandwidth_hz,
                "INS_HNTCH_ATT": notch.attenuation_db,
                "INS_HNTCH_HMNCS": float(sum(1 << (h - 1) for h in notch.harmonics)),
                "INS_HNTCH_OPTS": float(notch.opts),
                "INS_HNTCH_FM_RAT": notch.freq_min_ratio,
            }
        )
        break

    signals = _sweep_signals(axis, t_log, u_log, y_log, excite=True)

    if with_motor_noise:
        params["MOT_THST_HOVER"] = 0.35
        signals.update(_esc_signals(t, fundamental, fs))

    return LogBundle(
        path=Path(path),
        stack="ardupilot",
        firmware_version="ArduCopter V4.5.0 (synthetic)",
        board_id=None,
        frame_info={},
        sample_rate_hz=loop_rate_hz,
        loop_rate_hz=loop_rate_hz,
        gyro_sample_rate_hz=fs,
        signals=signals,
        params=params,
    )


def swept_tones(
    t: FloatArray,
    fundamental_hz: FloatArray,
    *,
    harmonics: tuple[int, ...] = (1, 2, 3),
    amplitudes: tuple[float, ...] = (0.6, 0.3, 0.15),
    seed: int = 0,
) -> FloatArray:
    """Tones whose frequency follows ``fundamental_hz`` sample by sample.

    The phase is the running integral of the instantaneous frequency, not
    ``sin(2*pi*f(t)*t)``. The latter is the classic mistake and produces a tone at
    roughly twice the intended frequency once ``f`` is moving.
    """
    rng = np.random.default_rng(seed)
    dt = float(t[1] - t[0])
    phase = 2.0 * np.pi * np.cumsum(fundamental_hz) * dt
    out = np.zeros_like(t)
    for n, a in zip(harmonics, amplitudes, strict=False):
        out += a * np.sin(n * phase + rng.uniform(0.0, 2.0 * np.pi))
    return np.asarray(out, dtype=np.float64)


def make_noise_bundle(
    chain: FilterChain,
    *,
    axis: Axis = "roll",
    duration_s: float = 40.0,
    loop_rate_hz: float = 400.0,
    hover_hz: float = 50.0,
    sweep_fraction: float = 0.12,
    harmonics: tuple[int, ...] = (1, 2, 3),
    amplitudes: tuple[float, ...] = (0.5, 0.25, 0.12),
    structural_hz: float | None = 118.0,
    structural_amplitude: float = 0.3,
    broadband_rms: float = 0.01,
    with_esc_telemetry: bool = True,
    gains: tuple[float, float, float] = (0.135, 0.135, 0.0036),
    seed: int = 7,
    path: str = "synthetic-noise.bin",
) -> LogBundle:
    """A hover log whose noise sources are known exactly.

    Motor tones follow a throttle sweep so they genuinely track RPM, while the
    structural tone sits still whatever the motors do. That contrast is the whole
    point: a classifier that cannot separate these two cannot be trusted to
    recommend a tracking notch, and the failure is silent on a real log because
    nobody knows the right answer there.

    Args:
        sweep_fraction: How far motor speed swings either side of ``hover_hz``.
            Must be large enough that tracking is distinguishable from chance.
        structural_hz: A fixed-frequency resonance. ``None`` omits it.
        with_esc_telemetry: Whether to include ``motor.*.rpm``. Without it the
            classifier has to fall back on how much each peak wanders.
    """
    from pathlib import Path

    from rotorid.core.io.base import canonical_signal

    fs = chain.sample_rate_hz
    t = np.arange(0.0, duration_s, 1.0 / fs)
    rng = np.random.default_rng(seed)

    # A slow throttle oscillation: motor speed rises and falls a few times over
    # the record, which is what makes the tracking correlation meaningful.
    modulation = np.sin(2.0 * np.pi * 0.05 * t)
    fundamental = hover_hz * (1.0 + sweep_fraction * modulation)

    y = swept_tones(t, fundamental, harmonics=harmonics, amplitudes=amplitudes, seed=seed)
    y += rng.normal(0.0, broadband_rms, size=t.size)
    if structural_hz is not None:
        y += structural_amplitude * np.sin(2.0 * np.pi * structural_hz * t + 0.4)

    op = OperatingPoint(motor_hz=(hover_hz,))
    for stage in chain.sensor_stages(op):
        y = np.asarray(lfilter(stage.b, stage.a, y), dtype=np.float64)

    step = round(fs / loop_rate_hz)
    t_log, y_log = t[::step], y[::step]

    signals = {
        f"rate.{axis}.measured": canonical_signal(
            f"rate.{axis}.measured", t_log, y_log, source_msg="RATE", filtered=True
        ),
        f"rate.{axis}.output": canonical_signal(
            f"rate.{axis}.output", t_log, np.zeros_like(t_log), source_msg="RATE"
        ),
    }
    if with_esc_telemetry:
        # ESC telemetry arrives far slower than the loop; 10 Hz is typical.
        esc_step = max(1, round(fs / 10.0))
        t_esc = t[::esc_step]
        rpm = fundamental[::esc_step] * 60.0
        for motor in range(1, 5):
            signals[f"motor.{motor}.rpm"] = canonical_signal(
                f"motor.{motor}.rpm", t_esc, rpm.copy(), source_msg="ESC"
            )

    kp, ki, kd = gains
    suffix = {"roll": "RLL", "pitch": "PIT", "yaw": "YAW"}[axis]
    params: dict[str, float] = {
        "SCHED_LOOP_RATE": loop_rate_hz,
        "INS_GYRO_FILTER": chain.gyro_lpf_hz or 0.0,
        "MOT_THST_HOVER": 0.35,
        f"ATC_RAT_{suffix}_P": kp,
        f"ATC_RAT_{suffix}_I": ki,
        f"ATC_RAT_{suffix}_D": kd,
    }
    if chain.dterm_lpf_hz:
        params[f"ATC_RAT_{suffix}_FLTD"] = chain.dterm_lpf_hz

    return LogBundle(
        path=Path(path),
        stack="ardupilot",
        firmware_version="ArduCopter V4.5.0 (synthetic)",
        board_id=None,
        frame_info={},
        sample_rate_hz=loop_rate_hz,
        loop_rate_hz=loop_rate_hz,
        gyro_sample_rate_hz=fs,
        signals=signals,
        params=params,
    )


def _sweep_signals(
    axis: Axis,
    t_log: FloatArray,
    u_log: FloatArray,
    y_log: FloatArray,
    *,
    excite: bool,
) -> dict[str, Signal]:
    """The three signals a sweep produces, in canonical form.

    ``excite`` is what separates the two stacks: ArduPilot logs the injected chirp
    on its own, which is the far better identification input. PX4 has no such
    message, so the sweep has to be found in the ordinary mixer command.
    """
    from rotorid.core.io.base import canonical_signal

    signals = {
        f"rate.{axis}.output": canonical_signal(
            f"rate.{axis}.output", t_log, u_log, source_msg="RATE"
        ),
        f"rate.{axis}.measured": canonical_signal(
            f"rate.{axis}.measured", t_log, y_log, source_msg="RATE", filtered=True
        ),
    }
    if excite:
        signals[f"excite.{axis}"] = canonical_signal(
            f"excite.{axis}", t_log, u_log, source_msg="SIDD"
        )
        return signals

    # Without an injected-signal message the sweep has to be found by comparing
    # the excited axis against the quiet ones, so the quiet ones have to be there.
    for other in AXES:
        if other == axis:
            continue
        signals[f"rate.{other}.output"] = canonical_signal(
            f"rate.{other}.output", t_log, np.zeros_like(t_log), source_msg="RATE"
        )
        signals[f"rate.{other}.measured"] = canonical_signal(
            f"rate.{other}.measured",
            t_log,
            np.zeros_like(t_log),
            source_msg="RATE",
            filtered=True,
        )
    return signals


def _esc_signals(t: FloatArray, fundamental: FloatArray, fs: float) -> dict[str, Signal]:
    """Per-motor RPM, at the rate ESC telemetry actually arrives (about 10 Hz)."""
    from rotorid.core.io.base import canonical_signal

    esc_step = max(1, round(fs / 10.0))
    t_esc = t[::esc_step]
    rpm = fundamental[::esc_step] * 60.0
    return {
        f"motor.{motor}.rpm": canonical_signal(
            f"motor.{motor}.rpm", t_esc, rpm.copy(), source_msg="ESC"
        )
        for motor in range(1, 5)
    }


def _px4_params(
    chain: FilterChain,
    gains: tuple[float, float, float],
    axis: Axis,
    *,
    loop_rate_hz: float,
) -> dict[str, float]:
    """The same vehicle, described in PX4's parameter names.

    ``K`` is set to 1 so the standard-form gains and the effective gains coincide
    in this fixture. The K-scaling itself is tested where it lives, at the IO
    boundary, rather than tangled into every end-to-end assertion here.
    """
    kp, ki, kd = gains
    suffix = {"roll": "ROLL", "pitch": "PITCH", "yaw": "YAW"}[axis]
    params: dict[str, float] = {
        "IMU_GYRO_RATEMAX": loop_rate_hz,
        "IMU_GYRO_CUTOFF": chain.gyro_lpf_hz or 0.0,
        f"MC_{suffix}RATE_K": 1.0,
        f"MC_{suffix}RATE_P": kp,
        f"MC_{suffix}RATE_I": ki,
        f"MC_{suffix}RATE_D": kd,
    }
    if chain.dterm_lpf_hz:
        params["IMU_DGYRO_CUTOFF"] = chain.dterm_lpf_hz
    for notch in chain.notches:
        params.update(
            {
                "IMU_GYRO_NF0_FRQ": notch.freq_hz,
                "IMU_GYRO_NF0_BW": notch.bandwidth_hz,
            }
        )
        break
    return params
