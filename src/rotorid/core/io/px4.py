"""PX4 ``.ulg`` reader (spec section 6.2).

PX4 makes this easier than ArduPilot in one way and harder in two.

Easier: uLog is self-describing and strictly SI. Angular rates are in rad/s
everywhere, there is no per-message unit disagreement to discover, and no
fallback table is needed.

Harder, first: what the log calls a signal changes with the release. The
rate-controller output has been ``actuator_controls_0``, then
``vehicle_torque_setpoint``, and the angular rate has been ``vehicle_attitude``'s
derivative, then ``vehicle_angular_velocity``. Each canonical key therefore lists
several candidate sources in preference order, and which one was used is recorded
so a surprising result can be traced back to the topic it came from.

Harder, second: **the same provenance trap as ArduPilot, for a different reason.**
``vehicle_angular_velocity`` is post-filter -- it comes out of the same filter
chain the controller reads -- so a plant identified from it includes
``IMU_GYRO_CUTOFF`` and every notch. The raw signal lives in ``sensor_gyro_fifo``
and is off by default. This reader marks the filtered signal as filtered, which
is what lets the deconvolution stage divide the chain back out exactly once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rotorid.core.io.base import LogReader, ProgressCallback, canonical_signal
from rotorid.core.preprocess.resample import (
    grid_rate_hz,
    measure_jitter,
    resample_to_grid,
    uniform_grid,
)
from rotorid.core.types import AXES, Axis, FloatArray, LogBundle, Signal

__all__ = ["PX4Reader", "read_px4"]

#: Axis order inside PX4's ``xyz[3]`` array fields.
_AXIS_INDEX: dict[Axis, int] = {"roll": 0, "pitch": 1, "yaw": 2}

#: Canonical key to the uLog topics and fields that can supply it, best first.
#: The first entry that exists in the log wins, and the choice is recorded in
#: ``Signal.source_msg``.
_SOURCES: dict[str, tuple[tuple[str, str], ...]] = {
    "rate.{axis}.measured": (("vehicle_angular_velocity", "xyz[{i}]"),),
    "rate.{axis}.setpoint": (("vehicle_rates_setpoint", "{name}"),),
    "rate.{axis}.output": (
        ("vehicle_torque_setpoint", "xyz[{i}]"),
        ("actuator_controls_0", "control[{i}]"),
    ),
    "rate.{axis}.accel": (("vehicle_angular_acceleration", "xyz[{i}]"),),
    # Attitude has no plain field to read: PX4 logs a quaternion, converted below.
    "att.{axis}.measured": (),
}

#: Rate-setpoint field names, which are spelled out rather than indexed.
_RATE_SETPOINT_FIELD: dict[Axis, str] = {"roll": "roll", "pitch": "pitch", "yaw": "yaw"}

#: Signals that come out of the filter chain the controller reads. Marking these
#: correctly is what stops the filters being counted twice or not at all.
_POST_FILTER = ("vehicle_angular_velocity", "vehicle_angular_acceleration")

#: PX4 logs its main loop rate nowhere directly. ``IMU_GYRO_RATEMAX`` is the rate
#: the controller is scheduled at, which is the number the loop model wants.
_DEFAULT_LOOP_RATE_HZ = 400.0

#: ``sensor_gyro_fifo`` is the raw, unfiltered gyro. Its presence is worth a great
#: deal -- it removes the need to reconstruct the pre-filter spectrum at all.
_FIFO_TOPIC = "sensor_gyro_fifo"


def read_px4(path: Path, progress: ProgressCallback | None = None) -> LogBundle:
    """Read a PX4 ``.ulg`` into the canonical bundle."""
    return PX4Reader(path).read(progress)


class PX4Reader(LogReader):
    """Reader for a PX4 uLog file."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._warnings: list[str] = []

    # ------------------------------------------------------------------ #
    # Pass one
    # ------------------------------------------------------------------ #

    def index(self) -> dict[str, int]:
        """Topic name to sample count.

        uLog carries this in its header, so unlike the ArduPilot reader this
        genuinely is cheap -- no full scan is needed to say what is in the file.
        """
        log = self._open()
        counts: dict[str, int] = {}
        for dataset in log.data_list:
            name = str(dataset.name)
            counts[name] = counts.get(name, 0) + len(dataset.data["timestamp"])
        return counts

    # ------------------------------------------------------------------ #
    # Pass two
    # ------------------------------------------------------------------ #

    def read(self, progress: ProgressCallback | None = None) -> LogBundle:
        """Extract parameters and every canonical signal the log contains.

        Raises:
            ValueError: if the log has no angular-velocity topic. Without the rate
                measurement there is nothing to identify.
        """
        self._warnings = []
        log = self._open()
        if progress is not None:
            progress(0.2, "reading topics")

        params = {str(k): float(v) for k, v in log.initial_parameters.items()}
        datasets = {str(d.name): d for d in log.data_list}

        if not any(name in datasets for name in ("vehicle_angular_velocity",)):
            raise ValueError(
                f"{self.path.name}: no vehicle_angular_velocity in the log, so there is "
                "no rate measurement to identify from. Raise SDLOG_PROFILE to include "
                "the high-rate topics and re-fly."
            )

        raw: dict[str, tuple[FloatArray, FloatArray, str]] = {}
        for template, candidates in _SOURCES.items():
            for axis in AXES:
                found = self._extract(datasets, template, candidates, axis)
                if found is not None:
                    raw[template.format(axis=axis)] = found

        raw.update(self._esc(datasets))
        raw.update(self._misc(datasets))
        raw.update(self._clipping(log))

        if progress is not None:
            progress(0.7, "resampling")
        return self._assemble(raw, params, datasets, log, progress)

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #

    def _open(self) -> Any:
        from pyulog import ULog

        return ULog(str(self.path))

    def _extract(
        self,
        datasets: dict[str, Any],
        template: str,
        candidates: tuple[tuple[str, str], ...],
        axis: Axis,
    ) -> tuple[FloatArray, FloatArray, str] | None:
        """First candidate topic that actually carries this signal."""
        index = _AXIS_INDEX[axis]
        for topic, field_template in candidates:
            dataset = datasets.get(topic)
            if dataset is None:
                continue
            field = field_template.format(i=index, name=_RATE_SETPOINT_FIELD[axis])
            values = dataset.data.get(field)
            if values is None:
                continue
            t = np.asarray(dataset.data["timestamp"], dtype=np.float64) / 1.0e6
            return t, np.asarray(values, dtype=np.float64), topic

        if template == "att.{axis}.measured":
            return self._attitude(datasets, axis)
        return None

    def _attitude(
        self, datasets: dict[str, Any], axis: Axis
    ) -> tuple[FloatArray, FloatArray, str] | None:
        """Euler angles, which PX4 logs only as a quaternion.

        Converted here rather than left out: the attitude loop is designed against
        an angle, and a user comparing this tool's outer-loop numbers with their
        ground station needs the same quantity the ground station shows.
        """
        dataset = datasets.get("vehicle_attitude")
        if dataset is None:
            return None
        try:
            q = np.stack(
                [np.asarray(dataset.data[f"q[{i}]"], dtype=np.float64) for i in range(4)],
                axis=1,
            )
        except KeyError:
            return None

        t = np.asarray(dataset.data["timestamp"], dtype=np.float64) / 1.0e6
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        if axis == "roll":
            angle = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        elif axis == "pitch":
            angle = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        else:
            angle = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return t, np.asarray(angle, dtype=np.float64), "vehicle_attitude"

    def _esc(self, datasets: dict[str, Any]) -> dict[str, tuple[FloatArray, FloatArray, str]]:
        """Per-motor RPM from ``esc_status``, which is what a dynamic notch needs."""
        dataset = datasets.get("esc_status")
        if dataset is None:
            return {}
        t = np.asarray(dataset.data["timestamp"], dtype=np.float64) / 1.0e6
        out: dict[str, tuple[FloatArray, FloatArray, str]] = {}
        for motor in range(8):
            values = dataset.data.get(f"esc[{motor}].esc_rpm")
            if values is None:
                continue
            out[f"motor.{motor}.rpm"] = (
                t,
                np.asarray(values, dtype=np.float64),
                "esc_status",
            )
        return out

    def _clipping(self, log: Any) -> dict[str, tuple[FloatArray, FloatArray, str]]:
        """Accelerometer clipping counts, per IMU, from ``vehicle_imu_status``.

        Read off ``log.data_list`` rather than the name-keyed ``datasets`` map,
        because this topic is multi-instance and that map keeps only one of them.
        Which IMU clipped is the whole diagnostic value: one clipping sensor is a
        mounting fault, all of them is an airframe that is shaking itself apart.

        PX4 counts clipping per axis; the three are summed, because the question a
        finding asks is whether the accelerometer was saturated at all, and an
        accelerometer saturated on any axis is not measuring the aircraft.

        Vibration itself is deliberately not read here. PX4's
        ``accel_vibration_metric`` is a filtered delta-velocity difference, which is
        not the quantity ArduPilot's ``VIBE`` reports, so the published 15/30 m/s^2
        thresholds do not transfer to it and there is no equivalent published
        threshold to use instead. Ingesting the number without a threshold would
        only let it be mistaken for the one that has one.
        """
        out: dict[str, tuple[FloatArray, FloatArray, str]] = {}
        for dataset in log.data_list:
            if str(dataset.name) != "vehicle_imu_status":
                continue
            axes = [dataset.data.get(f"accel_clipping[{i}]") for i in range(3)]
            present = [a for a in axes if a is not None]
            if not present:
                continue
            total = np.sum(
                np.stack([np.asarray(a, dtype=np.float64) for a in present], axis=1), axis=1
            )
            t = np.asarray(dataset.data["timestamp"], dtype=np.float64) / 1.0e6
            instance = int(getattr(dataset, "multi_id", 0))
            out[f"imu.{instance}.clip"] = (t, np.asarray(total), "vehicle_imu_status")
        return out

    def _misc(self, datasets: dict[str, Any]) -> dict[str, tuple[FloatArray, FloatArray, str]]:
        """Battery and CPU load: the inputs to the operating point and the CPU gate."""
        out: dict[str, tuple[FloatArray, FloatArray, str]] = {}
        for key, topic, field in (
            ("batt.voltage", "battery_status", "voltage_v"),
            ("batt.current", "battery_status", "current_a"),
            ("cpu.load", "cpuload", "load"),
        ):
            dataset = datasets.get(topic)
            if dataset is None:
                continue
            values = dataset.data.get(field)
            if values is None:
                continue
            t = np.asarray(dataset.data["timestamp"], dtype=np.float64) / 1.0e6
            out[key] = (t, np.asarray(values, dtype=np.float64), topic)
        return out

    # ------------------------------------------------------------------ #
    # Assembly
    # ------------------------------------------------------------------ #

    def _assemble(
        self,
        raw: dict[str, tuple[FloatArray, FloatArray, str]],
        params: dict[str, float],
        datasets: dict[str, Any],
        log: Any,
        progress: ProgressCallback | None,
    ) -> LogBundle:
        # PX4 runs the rate controller on each gyro publication, so the loop rate
        # and the gyro rate are the same number -- unlike ArduPilot, where
        # SCHED_LOOP_RATE and the gyro rate are set independently.
        gyro_rate = float(params.get("IMU_GYRO_RATEMAX", 0.0)) or self._measured_rate(datasets)
        loop_rate = gyro_rate
        rate_hz = grid_rate_hz(
            gyro_sample_rate_hz=gyro_rate,
            loop_rate_hz=loop_rate,
            highest_modeled_notch_hz=_highest_modeled_notch_hz(params),
            min_oversample_of_highest_notch=2.5,
        )

        series = {k: v for k, v in raw.items() if v[0].size >= 4}
        if not series:
            raise ValueError(f"{self.path.name}: no usable time series in the log")

        t_start = max(float(t[0]) for t, _, _ in series.values())
        t_end = min(float(t[-1]) for t, _, _ in series.values())
        grid = uniform_grid(t_start, t_end, rate_hz)

        signals: dict[str, Signal] = {}
        for key, (t, y, topic) in series.items():
            jitter = measure_jitter(t)
            if jitter.is_irregular(3.0):
                self._warnings.append(
                    f"{key}: irregular logging, p99 gap is {jitter.ratio:.1f}x the median"
                )
            signals[key] = resample_to_grid(
                canonical_signal(
                    key, t, y, source_msg=topic, filtered=topic in _POST_FILTER or None
                ),
                grid,
            )

        if _FIFO_TOPIC not in datasets:
            self._warnings.append(
                "no sensor_gyro_fifo in this log, so the pre-filter spectrum has to be "
                "reconstructed by dividing the modelled chain back out. Raise "
                "SDLOG_PROFILE to log it and the reconstruction becomes a measurement."
            )

        if progress is not None:
            progress(0.95, "resampled")

        return LogBundle(
            path=self.path,
            stack="px4",
            firmware_version=self._firmware(log),
            board_id=self._info(log, "ver_hw"),
            frame_info={},
            sample_rate_hz=rate_hz,
            loop_rate_hz=loop_rate,
            gyro_sample_rate_hz=gyro_rate,
            signals=signals,
            params=params,
            warnings=tuple(self._warnings),
        )

    def _measured_rate(self, datasets: dict[str, Any]) -> float:
        """Gyro rate read off the timestamps, when no parameter declares it."""
        dataset = datasets.get("vehicle_angular_velocity")
        if dataset is None:  # pragma: no cover - read() has already refused
            return _DEFAULT_LOOP_RATE_HZ
        t = np.asarray(dataset.data["timestamp"], dtype=np.float64) / 1.0e6
        if t.size < 2:
            return _DEFAULT_LOOP_RATE_HZ
        dt = float(np.median(np.diff(t)))
        rate = 1.0 / dt if dt > 0.0 else _DEFAULT_LOOP_RATE_HZ
        self._warnings.append(
            f"IMU_GYRO_RATEMAX is not in the parameter snapshot; gyro rate taken from "
            f"the logging interval as {rate:.0f} Hz"
        )
        return rate

    def _firmware(self, log: Any) -> str | None:
        version = self._info(log, "ver_sw_release") or self._info(log, "ver_sw")
        return f"PX4 {version}" if version else None

    @staticmethod
    def _info(log: Any, key: str) -> str | None:
        value = log.msg_info_dict.get(key)
        return None if value is None else str(value)


def _highest_modeled_notch_hz(params: dict[str, float]) -> float:
    """The highest notch centre the chain will contain, for grid-rate selection."""
    highest = 0.0
    for prefix in ("IMU_GYRO_NF0", "IMU_GYRO_NF1"):
        highest = max(highest, float(params.get(f"{prefix}_FRQ", 0.0)))
    if params.get("IMU_GYRO_DNF_EN", 0.0):
        harmonics = max(1.0, float(params.get("IMU_GYRO_DNF_HMC", 3)))
        highest = max(highest, float(params.get("IMU_GYRO_DNF_MIN", 0.0)) * harmonics)
    return highest
