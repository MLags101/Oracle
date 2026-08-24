"""ArduPilot ``.bin`` reader (spec section 6.1).

Units are read from the log, not assumed. ArduPilot writes ``FMTU``/``UNIT``
messages declaring the unit of every field, and different messages in the same
log genuinely disagree: ``RATE`` records rates in degrees per second while the
``PID*`` messages record the same quantity in radians per second, because one is
written for humans and the other is written straight out of the controller. A
reader that hard-codes one conversion is wrong by a factor of 57 on half the
signals, and the resulting airframe gain looks merely surprising rather than
obviously broken.

Where the log declares no unit -- older firmware, or a field ArduPilot never
annotated -- the fallback table below is used and the substitution is recorded in
``LogBundle.warnings`` rather than applied silently.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from rotorid.core.io.base import (
    LogReader,
    ProgressCallback,
    canonical_signal,
    gate_signal,
)
from rotorid.core.preprocess.resample import (
    grid_rate_hz,
    measure_jitter,
    resample_to_grid,
    uniform_grid,
)
from rotorid.core.types import AXES, Axis, FloatArray, LogBundle, LogKind, Signal

__all__ = ["SID_AXIS_MAP", "ArduPilotReader", "read_ardupilot"]

#: ``SID_AXIS``. 7-9 inject at the rate-controller input, 10-12 at the mixer.
SID_AXIS_MAP: dict[int, tuple[Axis, str]] = {
    1: ("roll", "input_roll"),
    2: ("pitch", "input_pitch"),
    3: ("yaw", "input_yaw"),
    4: ("roll", "recovery"),
    5: ("pitch", "recovery"),
    6: ("yaw", "recovery"),
    7: ("roll", "rate"),
    8: ("pitch", "rate"),
    9: ("yaw", "rate"),
    10: ("roll", "mixer"),
    11: ("pitch", "mixer"),
    12: ("yaw", "mixer"),
}

#: Fallback units for logs that predate ``FMTU``. Sourced from the ArduPilot log
#: message definitions; used only when the log declares nothing, and always
#: reported as an assumption.
_FALLBACK_UNITS: dict[tuple[str, str], str] = {
    **{("RATE", f): "deg/s" for f in ("RDes", "R", "PDes", "P", "YDes", "Y")},
    **{("RATE", f): "normalized" for f in ("ROut", "POut", "YOut", "AOut")},
    **{(msg, f): "rad/s" for msg in ("PIDR", "PIDP", "PIDY") for f in ("Tar", "Act", "Err")},
    **{("ATT", f): "deg" for f in ("DesRoll", "Roll", "DesPitch", "Pitch", "DesYaw", "Yaw")},
    ("BAT", "Volt"): "V",
    ("BAT", "Curr"): "A",
    ("ESC", "RPM"): "rev/min",
    **{("VIBE", f): "m/s/s" for f in ("VibeX", "VibeY", "VibeZ")},
}

#: Declared-unit string to a multiplier that reaches canonical units.
_UNIT_SCALE: dict[str, float] = {
    "deg/s": np.pi / 180.0,
    "rad/s": 1.0,
    "deg": np.pi / 180.0,
    "rad": 1.0,
    "V": 1.0,
    "A": 1.0,
    "rpm": 1.0,
    "rev/min": 1.0,
    # ArduPilot spells acceleration this way in FMTU; the canonical key spells it
    # "m/s^2". Same quantity, so the scale is one -- but the string has to be here
    # or the reader warns and refuses to convert.
    "m/s/s": 1.0,
    "m/s^2": 1.0,
    # ArduPilot's unit for a compass heading. Degrees, but spelled separately
    # because a heading is allowed to run outside 0-360 and is therefore not
    # interchangeable with a bounded angle. Missing this one is not cosmetic: it
    # put ATT.Yaw and ATT.DesYaw into the bundle in degrees under a key whose
    # contract says radians, which is a factor of 57 hiding behind a warning.
    "degheading": np.pi / 180.0,
    # Servo and motor outputs, in raw PWM microseconds. Deliberately left
    # unscaled: `motor.{n}.output` is defined as the PWM the vehicle wrote, and
    # the normalization against MOT_PWM_MIN/MAX happens where those parameters
    # are in scope rather than here.
    "us": 1.0,
    "instance": 1.0,
    "": 1.0,
    "normalized": 1.0,
    "%": 0.01,
}

_MESSAGES_WANTED = (
    "RATE",
    "PIDR",
    "PIDP",
    "PIDY",
    "ATT",
    "SIDD",
    "SIDS",
    "ESC",
    "RCOU",
    "BAT",
    "VIBE",
    "PM",
    "MSG",
    "EV",
    "GYR",
    "FTN1",
    "FTN2",
)

#: ``GYR`` carries the raw sensor samples, before the vehicle's gyro filters --
#: which is the one thing that turns filter verification from a comparison of two
#: models into a measurement (spec 5.5). It arrives per IMU instance at the
#: backend's own rate, which on an H7 board is well over a kilohertz.
#:
#: ``INS_RAW_LOG_OPT`` bit 3 asks the firmware to log the *post*-filter samples
#: here instead. When it is set the data is no longer pre-filter and must not be
#: labelled as though it were, so the bit is read rather than assumed.
_RAW_LOG_OPT_POST_FILTER_BIT = 3

#: Axis order inside ``GYR``'s ``GyrX/Y/Z``, matching the vehicle's body frame.
_GYR_FIELD: dict[Axis, str] = {"roll": "GyrX", "pitch": "GyrY", "yaw": "GyrZ"}

#: Per-axis peak and bandwidth from ArduPilot's onboard FFT. On a vehicle with no
#: ESC telemetry this is the only measurement of where the motors actually are,
#: and without it every line in the spectrum has to be called structural -- a
#: tracking notch cannot be recommended for a peak nothing has shown to move.
_FTN2_PEAK: dict[Axis, str] = {"roll": "PkX", "pitch": "PkY", "yaw": "PkZ"}

#: ArduPilot ``LogEvent`` codes that bracket an autotune run. Numeric rather than
#: read out of ``MSG`` text: the codes are enum constants that have not moved in
#: years, while the human-readable strings differ between versions and between
#: Copter and the other vehicles. Parsing the prose is how a tool comes to work
#: on the author's firmware and nowhere else.
_EV_AUTOTUNE_START = frozenset({30, 32})  # INITIALISED, RESTART
_EV_AUTOTUNE_STOP = frozenset({31, 33, 34})  # OFF, SUCCESS, FAILED

#: Codes that bracket a flight. ``NOT_LANDED`` and ``LAND_COMPLETE`` are the
#: vehicle's own verdict on whether it is off the ground, which is a far better
#: answer than any throttle threshold this tool could invent -- the firmware has
#: the accelerometers, the climb rate and the motor demand, and it is the same
#: judgement it uses to decide whether to let the pilot disarm.
#:
#: ``LAND_COMPLETE_MAYBE`` (17) deliberately does not end a flight. It is the
#: firmware saying it is not sure, and a window that ends on a maybe would throw
#: away the descent.
_EV_TAKEOFF = frozenset({28})  # NOT_LANDED
_EV_LANDED = frozenset({18})  # LAND_COMPLETE
_EV_ARMED = frozenset({10})
_EV_DISARMED = frozenset({11})

_AXIS_LETTER: dict[Axis, str] = {"roll": "R", "pitch": "P", "yaw": "Y"}
_PID_AXIS: dict[str, Axis] = {"PIDR": "roll", "PIDP": "pitch", "PIDY": "yaw"}
_ATT_FIELDS: dict[Axis, tuple[str, str]] = {
    "roll": ("DesRoll", "Roll"),
    "pitch": ("DesPitch", "Pitch"),
    "yaw": ("DesYaw", "Yaw"),
}


class ArduPilotReader(LogReader):
    """Streaming reader for a DataFlash ``.bin`` log."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._warnings: list[str] = []
        #: Canonical key to the ``MSG.Field`` it was read from. Provenance, not
        #: decoration: a finding about a signal has to be able to name the thing
        #: the user would change to fix it.
        self._source_msg: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Pass one
    # ------------------------------------------------------------------ #

    def index(self) -> dict[str, int]:
        """Count every message type in the log.

        Deliberately a full scan rather than a header read: ArduPilot writes
        format messages as it goes, so a message type that only appears once, an
        hour in, is invisible to anything cheaper.
        """
        log = self._open()
        counts: Counter[str] = Counter()
        while True:
            msg = log.recv_msg()
            if msg is None:
                break
            counts[msg.get_type()] += 1
        return dict(counts)

    # ------------------------------------------------------------------ #
    # Pass two
    # ------------------------------------------------------------------ #

    def read(self, progress: ProgressCallback | None = None) -> LogBundle:
        """Extract parameters and every canonical signal the log contains.

        Raises:
            ValueError: if the log has no ``RATE`` message. Without the rate
                measurement there is nothing to identify, and continuing would
                only produce a confident answer about a different aircraft.
        """
        self._warnings = []
        log = self._open()

        params: dict[str, float] = {}
        raw: dict[str, list[tuple[float, float]]] = {}
        units_seen: dict[str, str] = {}
        firmware: str | None = None
        n_read = 0

        while True:
            msg = log.recv_msg()
            if msg is None:
                break
            n_read += 1
            if progress is not None and n_read % 20000 == 0:
                progress(min(0.9, n_read / 2.0e6), f"read {n_read} messages")

            kind = msg.get_type()
            if kind == "PARM":
                params[str(msg.Name)] = float(msg.Value)
                continue
            if kind == "MSG" and firmware is None:
                text = str(getattr(msg, "Message", ""))
                if text.startswith(("ArduCopter", "ArduPilot", "APM:Copter")):
                    firmware = text
                continue
            if kind not in _MESSAGES_WANTED:
                continue
            self._collect(msg, kind, raw, units_seen)

        if not any(key.startswith("rate.") for key in raw):
            raise ValueError(
                f"{self.path.name}: no RATE message found. Enable the ATTITUDE_FAST "
                "and PID log bits (see docs/logging-setup-ardupilot.md) and re-fly."
            )

        params.update(self._params_from_sids(raw))
        bundle = self._assemble(raw, params, firmware, units_seen, progress)
        if progress is not None:
            progress(1.0, "done")
        return bundle

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _open(self) -> Any:
        from pymavlink import DFReader

        return DFReader.DFReader_binary(str(self.path))

    def _unit_of(self, msg: Any, kind: str, field: str) -> str:
        """Declared unit for one field, falling back to the table with a warning."""
        declared = ""
        fmt = getattr(msg, "fmt", None)
        if fmt is not None:
            try:
                declared = str(fmt.get_unit(field))
            except (KeyError, AttributeError):  # pragma: no cover - old pymavlink
                declared = ""
        if declared:
            return declared
        fallback = _FALLBACK_UNITS.get((kind, field), "")
        note = f"{kind}.{field}: log declares no unit, assuming {fallback or 'dimensionless'}"
        if note not in self._warnings:
            self._warnings.append(note)
        return fallback

    def _scale(self, unit: str, kind: str, field: str) -> float | None:
        """Multiplier to canonical units, or ``None`` if the unit is unknown.

        ``None`` drops the signal. The alternative -- passing the numbers through
        unconverted while still stamping them with the canonical unit -- produces
        a signal that lies about itself, and every consumer downstream believes
        the label rather than checking. That is how ``ATT.Yaw`` reached the
        identification in degrees under a key whose contract says radians.

        A missing signal is a condition this tool is built to report: the Load
        screen lists it, the findings name what it cost. A mislabelled one is not
        reportable by anything, because nothing can tell.
        """
        try:
            return _UNIT_SCALE[unit]
        except KeyError:
            note = (
                f"{kind}.{field}: unit {unit!r} is not one this build knows how to "
                f"convert, so the signal was dropped rather than guessed at"
            )
            if note not in self._warnings:
                self._warnings.append(note)
            return None

    def _add(
        self,
        raw: dict[str, list[tuple[float, float]]],
        units_seen: dict[str, str],
        key: str,
        t: float,
        msg: Any,
        kind: str,
        field: str,
    ) -> None:
        value = getattr(msg, field, None)
        if value is None:
            return
        unit = self._unit_of(msg, kind, field)
        scale = self._scale(unit, kind, field)
        if scale is None:
            return
        units_seen.setdefault(key, unit)
        self._source_msg.setdefault(key, f"{kind}.{field}")
        raw.setdefault(key, []).append((t, float(value) * scale))

    def _collect(
        self,
        msg: Any,
        kind: str,
        raw: dict[str, list[tuple[float, float]]],
        units_seen: dict[str, str],
    ) -> None:
        t = float(msg.TimeUS) / 1.0e6 if hasattr(msg, "TimeUS") else float(msg._timestamp)

        if kind == "RATE":
            for axis in AXES:
                letter = _AXIS_LETTER[axis]
                self._add(raw, units_seen, f"rate.{axis}.setpoint", t, msg, kind, f"{letter}Des")
                self._add(raw, units_seen, f"rate.{axis}.measured", t, msg, kind, letter)
                self._add(raw, units_seen, f"rate.{axis}.output", t, msg, kind, f"{letter}Out")
        elif kind in ("PIDR", "PIDP", "PIDY"):
            pid_axis = _PID_AXIS[kind]
            for field, suffix in (
                ("P", "p_term"),
                ("I", "i_term"),
                ("D", "d_term"),
                ("FF", "ff_term"),
                ("Dmod", "dmod"),
            ):
                self._add(raw, units_seen, f"rate.{pid_axis}.{suffix}", t, msg, kind, field)
        elif kind == "ATT":
            for att_axis, (des, act) in _ATT_FIELDS.items():
                self._add(raw, units_seen, f"att.{att_axis}.setpoint", t, msg, kind, des)
                self._add(raw, units_seen, f"att.{att_axis}.measured", t, msg, kind, act)
        elif kind == "SIDD":
            raw.setdefault("_sidd.targ", []).append((t, float(getattr(msg, "Targ", 0.0))))
            raw.setdefault("_sidd.freq", []).append((t, float(getattr(msg, "F", 0.0))))
        elif kind == "SIDS":
            raw.setdefault("_sids", []).append((t, float(getattr(msg, "Axis", 0.0))))
            for field in ("StartFreq", "StopFreq", "Magnitude", "FadeIn", "TimeRec", "FadeOut"):
                value = getattr(msg, field, None)
                if value is not None:
                    raw.setdefault(f"_sids.{field}", []).append((t, float(value)))
        elif kind == "ESC":
            index = int(getattr(msg, "Instance", getattr(msg, "I", 0)))
            self._add(raw, units_seen, f"motor.{index}.rpm", t, msg, kind, "RPM")
        elif kind == "RCOU":
            for index in range(1, 9):
                self._add(raw, units_seen, f"motor.{index}.output", t, msg, kind, f"C{index}")
        elif kind == "BAT":
            self._add(raw, units_seen, "batt.voltage", t, msg, kind, "Volt")
            self._add(raw, units_seen, "batt.current", t, msg, kind, "Curr")
        elif kind == "VIBE":
            # Two log formats, handled without branching. Modern ArduPilot writes
            # one VIBE record per IMU with an ``IMU`` instance field and a single
            # ``Clip``; before 4.0 there was no instance field and the three clip
            # counters were separate fields on one record. `_add` skips a field the
            # message does not carry, so both shapes land in the same keys.
            imu = int(getattr(msg, "IMU", 0))
            for field, component in (("VibeX", "x"), ("VibeY", "y"), ("VibeZ", "z")):
                self._add(raw, units_seen, f"imu.{imu}.vibe.{component}", t, msg, kind, field)
            self._add(raw, units_seen, f"imu.{imu}.clip", t, msg, kind, "Clip")
            for legacy in range(3):
                self._add(raw, units_seen, f"imu.{legacy}.clip", t, msg, kind, f"Clip{legacy}")
        elif kind == "PM":
            value = getattr(msg, "Load", None)
            if value is not None:
                raw.setdefault("cpu.load", []).append((t, float(value) / 1000.0))
        elif kind == "EV":
            code = getattr(msg, "Id", None)
            if code is not None:
                raw.setdefault("_ev", []).append((t, float(code)))
        elif kind == "GYR":
            # ``SampleUS`` is when the sensor was read, not when the line was
            # written. On a message logged faster than the scheduler runs, the two
            # differ by enough to smear the very spectrum this data exists for.
            sample_t = float(getattr(msg, "SampleUS", 0.0)) / 1.0e6 or t
            instance = int(getattr(msg, "I", 0))
            for axis, field in _GYR_FIELD.items():
                value = getattr(msg, field, None)
                if value is not None:
                    raw.setdefault(f"_gyr.{instance}.{axis}", []).append((sample_t, float(value)))
        elif kind == "FTN2":
            for axis, field in _FTN2_PEAK.items():
                value = getattr(msg, field, None)
                if value is not None and float(value) > 0.0:
                    raw.setdefault(f"_fft.{axis}", []).append((t, float(value)))

    def _event_windows(
        self,
        raw: dict[str, list[tuple[float, float]]],
        opens: frozenset[int],
        closes: frozenset[int],
        t_end: float,
    ) -> list[tuple[float, float]]:
        """Stretches bracketed by a pair of event codes.

        A window left open at the end of the log runs to the end rather than
        being discarded: a flight that was still going when the battery died, or
        an autotune interrupted by a failsafe, is exactly the record somebody
        wants analysed, and dropping it would silently turn it into no window at
        all.
        """
        windows: list[tuple[float, float]] = []
        start: float | None = None
        for t, code in raw.get("_ev", []):
            value = int(code)
            if value in opens and start is None:
                start = t
            elif value in closes and start is not None:
                windows.append((start, t))
                start = None
        if start is not None and t_end > start:
            windows.append((start, t_end))
        return windows

    def _params_from_sids(self, raw: dict[str, list[tuple[float, float]]]) -> dict[str, float]:
        """Recover the sweep configuration that ``SIDS`` recorded.

        The ``SID_*`` parameters can be changed in flight (``TUNE = 58`` adjusts
        the magnitude), so the ``SIDS`` record of what was actually flown is more
        trustworthy than the parameter snapshot and takes precedence.
        """
        out: dict[str, float] = {}
        mapping = {
            "_sids.StartFreq": "SID_F_START_HZ",
            "_sids.StopFreq": "SID_F_STOP_HZ",
            "_sids.Magnitude": "SID_MAGNITUDE",
            "_sids.FadeIn": "SID_T_FADE_IN",
            "_sids.TimeRec": "SID_T_REC",
            "_sids.FadeOut": "SID_T_FADE_OUT",
        }
        for key, name in mapping.items():
            if raw.get(key):
                out[name] = raw[key][-1][1]
        if raw.get("_sids"):
            out["SID_AXIS"] = raw["_sids"][-1][1]
        return out

    def _assemble(
        self,
        raw: dict[str, list[tuple[float, float]]],
        params: dict[str, float],
        firmware: str | None,
        units_seen: dict[str, str],
        progress: ProgressCallback | None,
    ) -> LogBundle:
        loop_rate = params.get("SCHED_LOOP_RATE", 400.0)
        gyro_rate = _gyro_rate_hz(params)
        highest_notch = _highest_modeled_notch_hz(params)
        rate_hz = grid_rate_hz(
            gyro_sample_rate_hz=gyro_rate,
            loop_rate_hz=loop_rate,
            highest_modeled_notch_hz=highest_notch,
            min_oversample_of_highest_notch=2.5,
        )

        series = {k: v for k, v in raw.items() if not k.startswith("_") and len(v) >= 4}
        if not series:
            raise ValueError(f"{self.path.name}: no usable time series in the log")

        t_start = max(min(t for t, _ in v) for v in series.values())
        t_end = min(max(t for t, _ in v) for v in series.values())
        grid = uniform_grid(t_start, t_end, rate_hz)

        signals: dict[str, Signal] = {}
        for key, samples in series.items():
            arr = np.asarray(samples, dtype=np.float64)
            jitter = measure_jitter(arr[:, 0])
            if jitter.is_irregular(3.0):
                self._warnings.append(
                    f"{key}: irregular logging, p99 gap is {jitter.ratio:.1f}x the median"
                )
            raw_signal = canonical_signal(
                key,
                arr[:, 0],
                arr[:, 1],
                source_msg=self._source_msg.get(key, ""),
                filtered=True if key.endswith(".measured") else None,
            )
            signals[key] = resample_to_grid(raw_signal, grid)

        if raw.get("_sidd.targ"):
            signals.update(self._excitation_signals(raw, params, grid))
        signals.update(self._raw_gyro_signals(raw, params, grid))
        signals.update(self._fft_signals(raw, grid))

        t_end = float(grid[-1])
        for key, opens, closes in (
            ("mode.autotune", _EV_AUTOTUNE_START, _EV_AUTOTUNE_STOP),
            ("mode.flying", _EV_TAKEOFF, _EV_LANDED),
            ("mode.armed", _EV_ARMED, _EV_DISARMED),
        ):
            windows = self._event_windows(raw, opens, closes, t_end)
            if windows:
                signals[key] = gate_signal(key, grid, windows, source_msg="EV.Id")

        if progress is not None:
            progress(0.95, "resampled")

        return LogBundle(
            path=self.path,
            stack="ardupilot",
            firmware_version=firmware,
            board_id=None,
            frame_info={},
            sample_rate_hz=rate_hz,
            loop_rate_hz=loop_rate,
            gyro_sample_rate_hz=gyro_rate,
            signals=signals,
            params=params,
            warnings=tuple(self._warnings),
        )

    def _raw_gyro_signals(
        self,
        raw: dict[str, list[tuple[float, float]]],
        params: dict[str, float],
        grid: FloatArray,
    ) -> dict[str, Signal]:
        """Pre-filter gyro from ``GYR``, if the firmware logged it that way.

        The primary IMU only. Logging several instances is common and averaging
        them would be wrong: the controller reads one gyro, and the whole point of
        a pre-filter trace is to be the input to the chain whose output the
        controller used. Which one is primary is not recorded in the message, so
        the instance with the most samples is taken -- the primary is the one the
        firmware logs at the full backend rate.

        Silent when ``INS_RAW_LOG_OPT`` asks for post-filter samples instead. The
        data is then perfectly good and simply is not what this key means, and a
        post-filter trace labelled pre-filter would have the chain divided out of
        it twice.
        """
        instances = {int(key.split(".")[1]) for key in raw if key.startswith("_gyr.")}
        if not instances:
            return {}
        if int(params.get("INS_RAW_LOG_OPT", 0.0)) & (1 << _RAW_LOG_OPT_POST_FILTER_BIT):
            self._warnings.append(
                "INS_RAW_LOG_OPT asks for post-filter raw gyro, so GYR is not the "
                "pre-filter trace the filter check needs. Clear bit 3 to log the "
                "unfiltered samples."
            )
            return {}

        primary = max(instances, key=lambda i: len(raw.get(f"_gyr.{i}.roll", [])))
        out: dict[str, Signal] = {}
        for axis in AXES:
            samples = raw.get(f"_gyr.{primary}.{axis}")
            if not samples or len(samples) < 4:
                continue
            arr = np.asarray(samples, dtype=np.float64)
            signal = canonical_signal(
                f"gyro.{axis}.prefilter",
                arr[:, 0],
                arr[:, 1],
                source_msg=f"GYR.{_GYR_FIELD[axis]}",
                filtered=False,
            )
            out[f"gyro.{axis}.prefilter"] = resample_to_grid(signal, grid)
        if out:
            self._warnings.append(
                f"pre-filter gyro read from GYR (IMU {primary}) at "
                f"{out[next(iter(out))].native_rate_hz or 0:.0f} Hz on the analysis grid"
            )
        return out

    def _fft_signals(
        self, raw: dict[str, list[tuple[float, float]]], grid: FloatArray
    ) -> dict[str, Signal]:
        """Motor frequency as the vehicle's own FFT measured it.

        ``FTN2`` reports the detected peak per axis, which is the motor
        fundamental whenever the FFT has a lock. Treated as a frequency track in
        exactly the same way ESC telemetry is, because it answers the same
        question -- where are the motors right now -- and a vehicle with no ESC
        telemetry has no other source at all.
        """
        out: dict[str, Signal] = {}
        for axis in AXES:
            samples = raw.get(f"_fft.{axis}")
            if not samples or len(samples) < 4:
                continue
            arr = np.asarray(samples, dtype=np.float64)
            signal = canonical_signal(
                f"fft.{axis}.peak_hz", arr[:, 0], arr[:, 1], source_msg=f"FTN2.{_FTN2_PEAK[axis]}"
            )
            out[f"fft.{axis}.peak_hz"] = resample_to_grid(signal, grid)
        return out

    def _excitation_signals(
        self,
        raw: dict[str, list[tuple[float, float]]],
        params: dict[str, float],
        grid: FloatArray,
    ) -> dict[str, Signal]:
        """The injected chirp, on the axis ``SID_AXIS`` says it was injected into.

        This is the reference input worth having: it is the excitation alone,
        without the controller's reaction to the vehicle mixed into it, so its
        coherence with the response is far better than the mixer command's.
        """
        axis_code = int(params.get("SID_AXIS", 0))
        mapped = SID_AXIS_MAP.get(axis_code)
        if mapped is None:
            self._warnings.append(
                f"SYSTEMID data present but SID_AXIS={axis_code} is not a known axis; "
                "the chirp cannot be attributed and will not be used"
            )
            return {}

        axis, _injection = mapped
        arr = np.asarray(raw["_sidd.targ"], dtype=np.float64)
        signal = canonical_signal(f"excite.{axis}", arr[:, 0], arr[:, 1], source_msg="SIDD")
        return {f"excite.{axis}": resample_to_grid(signal, grid)}


def _gyro_rate_hz(params: dict[str, float]) -> float:
    """Sensor rate the vehicle's biquads run at.

    ``INS_GYRO_RATE`` is an enum of powers of two above 1 kHz, not a frequency.
    Absent, the safe assumption is 1 kHz -- every supported board runs at least
    that -- and assuming higher would make every modeled notch shallower than the
    one actually flown.
    """
    enum = params.get("INS_GYRO_RATE")
    if enum is None:
        return 1000.0
    return 1000.0 * float(2 ** int(enum))


def _highest_modeled_notch_hz(params: dict[str, float]) -> float:
    """Highest notch centre the filter model has to reproduce.

    Taken from the configured fundamental and the highest enabled harmonic across
    both banks. Used only to size the analysis grid.
    """
    highest = 0.0
    for prefix in ("INS_HNTCH", "INS_HNTC2"):
        if not params.get(f"{prefix}_ENABLE", 0.0):
            continue
        freq = float(params.get(f"{prefix}_FREQ", 0.0))
        mask = int(params.get(f"{prefix}_HMNCS", 1))
        top = max((n + 1 for n in range(16) if mask & (1 << n)), default=1)
        highest = max(highest, freq * top)
    return highest


def read_ardupilot(
    path: Path,
    progress: ProgressCallback | None = None,
    *,
    kind: LogKind | None = None,
) -> LogBundle:
    """Convenience wrapper: read one ``.bin`` in a single call.

    Args:
        kind: What the user says this flight was. Recorded on the bundle rather
            than acted on here -- the reader's job is to say what is in the file,
            and what the file is *for* is not something the file knows.
    """
    bundle = ArduPilotReader(path).read(progress)
    return replace(bundle, declared_kind=kind) if kind is not None else bundle
