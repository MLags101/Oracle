"""A synthetic log flown by a controller, not fed by an open-loop generator.

:mod:`tests.synthetic.generators` builds its bundles open-loop: a chirp goes into
the plant, the response comes out, and the "injected chirp" signal is literally
the same array as the mixer command. That is fine for testing arithmetic, and it
is useless for testing *identification*, because there is no feedback anywhere in
it and therefore no closed-loop bias to find. A tool that identified the wrong
system entirely would pass every test built on it -- which is exactly what
happened: the injected-chirp path was estimating ``G/(1+GC)`` and calling it the
plant, and nothing noticed.

So this module closes the loop. An attitude controller drives a rate controller
which drives the plant; gyro noise enters at the measurement and comes back round
through both. Every signal a real log carries is then produced from that
simulation rather than assumed:

* ``att.{axis}.setpoint`` is the pilot's command -- genuinely exogenous;
* ``rate.{axis}.setpoint`` is the outer loop's output, which carries gyro noise
  the long way round and is therefore a *worse* instrument, as claimed;
* ``rate.{axis}.output`` is the rate controller's output, correlated with the
  noise it was computed from -- the signal that makes the direct estimator wrong;
* ``excite.{axis}`` is the injected chirp alone, where there is one.

Everything is linear and time-invariant, so the loop is solved in the frequency
domain in closed form rather than stepped sample by sample. That is exact for an
LTI loop and it keeps the fixture fast enough to use freely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from rotorid.core.analysis.margins import LoopDelay, loop_delay, plant_path
from rotorid.core.design.controller import controller_for
from rotorid.core.filters.chain import FilterChain, OperatingPoint
from rotorid.core.io.base import canonical_signal
from rotorid.core.types import (
    AXES,
    AirframeModel,
    Axis,
    ComplexArray,
    FloatArray,
    LogBundle,
    Signal,
)
from tests.synthetic.generators import chirp, make_airframe, make_chain

__all__ = [
    "ClosedLoopFlight",
    "inner_loop_step",
    "make_closed_loop_bundle",
    "simulate_closed_loop",
]

#: Where the SYSTEMID waveform is added. ArduPilot's ``SID_AXIS`` 7-9 add it to
#: the rate target, upstream of the rate controller; 10-12 add it to the actuator
#: command, downstream. The plant input differs between the two and so does what
#: ``RATE.ROut`` contains, which is the whole reason this is a parameter.
Injection = Literal["rate", "mixer"]

#: ``SID_AXIS`` per axis and injection point.
_SID_AXIS: dict[tuple[Injection, Axis], float] = {
    ("rate", "roll"): 7.0,
    ("rate", "pitch"): 8.0,
    ("rate", "yaw"): 9.0,
    ("mixer", "roll"): 10.0,
    ("mixer", "pitch"): 11.0,
    ("mixer", "yaw"): 12.0,
}


class ClosedLoopFlight:
    """The signals one simulated flight produced, all on the same time base."""

    __slots__ = (
        "att_measured",
        "att_setpoint",
        "excite",
        "rate_measured",
        "rate_output",
        "rate_setpoint",
        "t",
    )

    def __init__(
        self,
        t: FloatArray,
        att_setpoint: FloatArray,
        att_measured: FloatArray,
        rate_setpoint: FloatArray,
        rate_output: FloatArray,
        rate_measured: FloatArray,
        excite: FloatArray,
    ) -> None:
        self.t = t
        self.att_setpoint = att_setpoint
        self.att_measured = att_measured
        self.rate_setpoint = rate_setpoint
        self.rate_output = rate_output
        self.rate_measured = rate_measured
        self.excite = excite


def simulate_closed_loop(
    airframe: AirframeModel,
    chain: FilterChain,
    *,
    duration_s: float = 90.0,
    gains: tuple[float, float, float] = (0.135, 0.135, 0.0036),
    attitude_p: float = 4.5,
    loop_rate_hz: float = 400.0,
    delay: LoopDelay | None = None,
    op: OperatingPoint | None = None,
    chirp_amplitude: float = 0.1,
    chirp_band_hz: tuple[float, float] = (0.2, 20.0),
    injection: Injection = "rate",
    stick_amplitude_deg: float = 6.0,
    stick_corner_hz: float = 1.5,
    noise_rms: float = 0.0,
    seed: int = 0,
) -> ClosedLoopFlight:
    """Fly one axis round a two-loop controller and return what the log would hold.

    The loop being solved, with ``C`` the rate controller, ``P = F G D`` the
    sensor chain times the airframe times the loop delay, ``Ktheta`` the attitude
    gain, ``a`` the pilot's attitude command, ``n`` the gyro noise and ``e`` the
    injected waveform:

    .. code-block:: text

        theta = y / (j w)
        r     = Ktheta (a - theta)
        u     = C (r - y) + e            [e added here for mixer injection]
        y     = P u + F n

    Eliminating gives ``u = (C Ktheta a - Q F n + e) / (1 + Q P)`` with
    ``Q = C (Ktheta / (j w) + 1)``, which is what is evaluated below.

    Args:
        chirp_amplitude: Zero for an ordinary flight with no injected sweep, which
            is the case the pilot's stick has to carry on its own.
        noise_rms: Gyro noise, in rad/s. This is the knob that creates the bias:
            with no noise the direct and instrument-variable estimators agree,
            because there is nothing for the controller to feed back.
        injection: Where the chirp enters. Changes what ``rate.output`` contains
            and therefore how the plant input has to be reassembled.
    """
    fs = chain.sample_rate_hz
    n_samples = round(duration_s * fs)
    t = np.arange(n_samples, dtype=np.float64) / fs

    if delay is None:
        delay = loop_delay(
            loop_rate_hz=loop_rate_hz, actuator_ms=0.1, zoh_loops=0.5, compute_loops=1.0
        )

    from rotorid.core.types import GainSet

    kp, ki, kd = gains
    controller = controller_for(
        "ardupilot",
        GainSet(
            axis="roll",
            kp=kp,
            ki=ki,
            kd=kd,
            kff=0.0,
            dterm_lpf_hz=chain.dterm_lpf_hz,
            error_lpf_hz=chain.error_lpf_hz,
            target_lpf_hz=chain.target_lpf_hz,
        ),
        chain,
    )

    f = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    # The integrator and the attitude loop are both infinite at DC. Evaluate at a
    # frequency just below the first real bin instead and zero the DC component
    # afterwards -- the signals are detrended before any of this is estimated, so
    # DC carries no information anyone is going to read.
    f_eval = f.copy()
    f_eval[0] = f[1] * 1e-3
    w = 2.0 * np.pi * f_eval

    F = chain.sensor_response(f_eval, op)
    C = controller.feedback_response(f_eval)
    P = plant_path(f_eval, controller, airframe, delay=delay, op=op)
    Q = C * (attitude_p / (1j * w) + 1.0)
    closed = 1.0 / (1.0 + Q * P)

    rng = np.random.default_rng(seed)
    stick = _pilot_stick(rng, n_samples, fs, np.radians(stick_amplitude_deg), stick_corner_hz)
    noise = rng.standard_normal(n_samples) * noise_rms

    if chirp_amplitude > 0.0:
        _, e = chirp(
            sample_rate_hz=fs,
            duration_s=duration_s,
            f_start_hz=chirp_band_hz[0],
            f_stop_hz=chirp_band_hz[1],
            amplitude=chirp_amplitude,
            fade_s=4.0,
        )
        e = e[:n_samples]
    else:
        e = np.zeros(n_samples)

    A = np.fft.rfft(stick)
    N = np.fft.rfft(noise)
    E = np.fft.rfft(e)

    # The chirp enters at the rate target for SID_AXIS 7-9 and at the actuator
    # command for 10-12, which puts it on different sides of the rate controller.
    forcing = C * attitude_p * A - Q * F * N
    U = closed * (forcing + (C * E if injection == "rate" else E))
    Y = P * U + F * N
    Theta = Y / (1j * w)
    R = attitude_p * (A - Theta)

    for spectrum in (U, Y, Theta, R):
        spectrum[0] = 0.0

    u = _real(U, n_samples)
    y = _real(Y, n_samples)
    theta = _real(Theta, n_samples)
    r = _real(R, n_samples)

    # What the firmware writes down. For mixer injection RATE.ROut is the rate
    # controller's output before the waveform is added, so the logged command is
    # the plant input minus the chirp.
    logged_output = u - e if injection == "mixer" else u

    return ClosedLoopFlight(
        t=t,
        att_setpoint=stick,
        att_measured=theta,
        rate_setpoint=r,
        rate_output=logged_output,
        rate_measured=y,
        excite=e,
    )


def make_closed_loop_bundle(
    airframe: AirframeModel | None = None,
    chain: FilterChain | None = None,
    *,
    axis: Axis = "roll",
    injection: Injection = "rate",
    with_chirp: bool = True,
    loop_rate_hz: float = 400.0,
    gains: tuple[float, float, float] = (0.135, 0.135, 0.0036),
    attitude_p: float = 4.5,
    noise_rms: float = 0.02,
    path: str = "closed-loop.bin",
    **kwargs: object,
) -> LogBundle:
    """A full ``LogBundle`` from a closed-loop simulation, in ArduPilot form.

    Args:
        with_chirp: ``False`` produces an ordinary flight -- no ``SIDD``, no
            ``SID_*`` parameters, nothing to segment on but the stick. That is the
            case the general-log work exists for.
        noise_rms: A small gyro noise floor by default. A *noiseless* loop is
            perfectly coherent at every frequency, including ones where the
            excitation had no energy at all, so it would report an identification
            band far wider than any real flight supports. Real logs are never
            noiseless; the fixture should not be either.
    """
    airframe = airframe if airframe is not None else make_airframe()
    chain = chain if chain is not None else make_chain()
    op = OperatingPoint(motor_hz=(50.0,))

    flight = simulate_closed_loop(
        airframe,
        chain,
        gains=gains,
        attitude_p=attitude_p,
        loop_rate_hz=loop_rate_hz,
        op=op,
        injection=injection,
        noise_rms=noise_rms,
        chirp_amplitude=float(kwargs.pop("chirp_amplitude", 0.1)) if with_chirp else 0.0,
        **kwargs,  # type: ignore[arg-type]
    )

    step = round(chain.sample_rate_hz / loop_rate_hz)

    def _log(values: FloatArray) -> FloatArray:
        return np.asarray(values[::step], dtype=np.float64)

    t_log = _log(flight.t)
    signals: dict[str, Signal] = {
        f"rate.{axis}.output": canonical_signal(
            f"rate.{axis}.output", t_log, _log(flight.rate_output), source_msg="RATE.ROut"
        ),
        f"rate.{axis}.measured": canonical_signal(
            f"rate.{axis}.measured",
            t_log,
            _log(flight.rate_measured),
            source_msg="RATE.R",
            filtered=True,
        ),
        f"rate.{axis}.setpoint": canonical_signal(
            f"rate.{axis}.setpoint", t_log, _log(flight.rate_setpoint), source_msg="RATE.RDes"
        ),
        f"att.{axis}.setpoint": canonical_signal(
            f"att.{axis}.setpoint", t_log, _log(flight.att_setpoint), source_msg="ATT.DesRoll"
        ),
        f"att.{axis}.measured": canonical_signal(
            f"att.{axis}.measured", t_log, _log(flight.att_measured), source_msg="ATT.Roll"
        ),
    }
    # The energy segmenter compares the excited axis against the quiet ones, so
    # on a flight with no injected chirp the quiet ones have to be present.
    for other in AXES:
        if other == axis:
            continue
        quiet = np.zeros_like(t_log)
        signals[f"rate.{other}.output"] = canonical_signal(
            f"rate.{other}.output", t_log, quiet, source_msg="RATE"
        )
        signals[f"rate.{other}.measured"] = canonical_signal(
            f"rate.{other}.measured", t_log, quiet, source_msg="RATE", filtered=True
        )

    kp, ki, kd = gains
    suffix = {"roll": "RLL", "pitch": "PIT", "yaw": "YAW"}[axis]
    params: dict[str, float] = {
        "SCHED_LOOP_RATE": loop_rate_hz,
        "INS_GYRO_RATE": float(int(np.log2(chain.sample_rate_hz / 1000.0))),
        "INS_GYRO_FILTER": chain.gyro_lpf_hz or 0.0,
        "MOT_PWM_TYPE": 6.0,
        f"ATC_ANG_{suffix}_P": attitude_p,
        f"ATC_RAT_{suffix}_P": kp,
        f"ATC_RAT_{suffix}_I": ki,
        f"ATC_RAT_{suffix}_D": kd,
    }
    if chain.dterm_lpf_hz:
        params[f"ATC_RAT_{suffix}_FLTD"] = chain.dterm_lpf_hz
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

    if with_chirp:
        signals[f"excite.{axis}"] = canonical_signal(
            f"excite.{axis}", t_log, _log(flight.excite), source_msg="SIDD.Targ"
        )
        params.update(
            {
                "SID_AXIS": _SID_AXIS[(injection, axis)],
                "SID_F_START_HZ": 0.2,
                "SID_F_STOP_HZ": 20.0,
                "SID_MAGNITUDE": 0.1,
            }
        )

    return LogBundle(
        path=Path(path),
        stack="ardupilot",
        firmware_version="ArduCopter V4.5.0 (closed-loop synthetic)",
        board_id=None,
        frame_info={},
        sample_rate_hz=loop_rate_hz,
        loop_rate_hz=loop_rate_hz,
        gyro_sample_rate_hz=chain.sample_rate_hz,
        signals=signals,
        params=params,
    )


def _pilot_stick(
    rng: np.random.Generator, n: int, fs: float, amplitude_rad: float, corner_hz: float
) -> FloatArray:
    """Band-limited noise standing in for a pilot's stick.

    Shaped rather than white, because a pilot is: most of the energy is below a
    couple of Hz, and there is essentially none above the wrist. That shape is
    also why an ordinary flight identifies a narrow band -- the instrument only
    instruments where it has energy, and this is where the honest limit on a
    general log comes from.
    """
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum *= 1.0 / (1.0 + (f / corner_hz) ** 2)
    shaped = np.fft.irfft(spectrum, n=n)
    peak = float(np.max(np.abs(shaped)))
    return np.asarray(shaped * (amplitude_rad / peak) if peak > 0.0 else shaped, dtype=np.float64)


def _real(spectrum: ComplexArray, n: int) -> FloatArray:
    return np.asarray(np.fft.irfft(spectrum, n=n), dtype=np.float64)


def inner_loop_step(
    airframe: AirframeModel | None = None,
    chain: FilterChain | None = None,
    *,
    gains: tuple[float, float, float] = (0.135, 0.135, 0.0036),
    loop_rate_hz: float = 400.0,
    duration_s: float = 0.5,
    sample_rate_hz: float | None = None,
    op: OperatingPoint | None = None,
) -> tuple[FloatArray, FloatArray]:
    """The rate-setpoint-to-rate-measured step of the loop this module simulates.

    Ground truth for anything that claims to recover a step response from these
    logs. Eliminating the measurement from ``u = C (r - y)``, ``y = P u + F n``
    gives ``y = [CP / (1 + CP)] r`` plus a noise path, so the inner closed loop is
    ``T = CP / (1 + CP)`` -- independent of the attitude loop that generates ``r``,
    which is why a deconvolution of ``r`` against ``y`` can find it at all.

    Deliberately *not* :func:`rotorid.core.analysis.step.step_response`. That
    models ArduPilot's real reference path, where the derivative term acts only on
    the measurement; this fixture applies one controller to the error, as its own
    equations say. Comparing a measurement against the thing that actually
    produced it is the point -- borrowing the production predictor here would
    leave both free to be wrong together.
    """
    airframe = airframe if airframe is not None else make_airframe()
    chain = chain if chain is not None else make_chain()
    op = op if op is not None else OperatingPoint(motor_hz=(50.0,))
    fs = sample_rate_hz if sample_rate_hz is not None else chain.sample_rate_hz

    from rotorid.core.types import GainSet

    kp, ki, kd = gains
    controller = controller_for(
        "ardupilot",
        GainSet(
            axis="roll",
            kp=kp,
            ki=ki,
            kd=kd,
            kff=0.0,
            dterm_lpf_hz=chain.dterm_lpf_hz,
            error_lpf_hz=chain.error_lpf_hz,
            target_lpf_hz=chain.target_lpf_hz,
        ),
        chain,
    )
    delay = loop_delay(loop_rate_hz=loop_rate_hz, actuator_ms=0.1, zoh_loops=0.5, compute_loops=1.0)

    n_keep = round(duration_s * fs)
    n = 8 * n_keep
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    f_eval = f.copy()
    f_eval[0] = f[1] * 1e-3

    C = controller.feedback_response(f_eval)
    P = plant_path(f_eval, controller, airframe, delay=delay, op=op)
    T = C * P / (1.0 + C * P)
    if ki > 0.0:
        T[0] = 1.0

    y = np.cumsum(np.fft.irfft(T, n=n))[:n_keep]
    t = np.arange(n_keep, dtype=np.float64) / fs
    return t, np.asarray(y, dtype=np.float64)
