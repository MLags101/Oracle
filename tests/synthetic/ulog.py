"""A minimal uLog writer, so the PX4 reader is tested against real bytes.

The reader's job is not arithmetic, it is *interpretation*: which topic supplies
which canonical signal, which of them are post-filter, how a quaternion becomes
an Euler angle, what happens when the topic the log was expected to have is the
older one. None of that is exercised by handing the reader a Python object -- it
is exercised by handing it a file.

So this writes the real container format (ULog v1, `docs/ulog_file_format.md`):
a header, a flag-bits message, one format and one subscription per topic, the
parameters, and the data. It is deliberately the smallest writer that produces
something `pyulog` will parse, because its purpose is to feed our reader, not to
be a general uLog library.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

__all__ = ["ULogWriter", "write_px4_log"]

_MAGIC = b"ULog\x01\x12\x35"
_VERSION = 1

_MSG_FORMAT = ord("F")
_MSG_DATA = ord("D")
_MSG_INFO = ord("I")
_MSG_PARAMETER = ord("P")
_MSG_ADD_LOGGED = ord("A")
_MSG_FLAG_BITS = ord("B")

#: uLog type name to struct format. Only the types this writer emits.
_TYPES = {"uint64_t": "Q", "float": "f", "int32_t": "i"}


class ULogWriter:
    """Builds one uLog file in memory, then writes it."""

    def __init__(self, start_us: int = 0) -> None:
        self._start_us = start_us
        self._formats: list[tuple[str, list[tuple[str, str]]]] = []
        self._subscriptions: dict[str, int] = {}
        self._info: dict[str, str] = {}
        self._params: dict[str, float] = {}
        self._data: list[tuple[int, bytes]] = []

    # ----------------------------------------------------------------- #

    def info(self, key: str, value: str) -> ULogWriter:
        self._info[key] = value
        return self

    def parameters(self, params: dict[str, float]) -> ULogWriter:
        self._params.update(params)
        return self

    def topic(
        self, name: str, fields: list[tuple[str, str]], samples: dict[str, np.ndarray]
    ) -> ULogWriter:
        """Add one topic and all of its samples.

        Args:
            fields: ``(type, name)`` pairs, **not** including the timestamp, which
                every uLog topic carries first and which is added here.
            samples: Field name to values. Must include ``"timestamp"`` in
                seconds; it is converted to the microseconds uLog stores.
        """
        all_fields = [("uint64_t", "timestamp"), *fields]
        msg_id = len(self._subscriptions)
        self._formats.append((name, all_fields))
        self._subscriptions[name] = msg_id

        t_us = np.asarray(samples["timestamp"] * 1e6, dtype=np.uint64)
        packer = struct.Struct("<" + "".join(_TYPES[t] for t, _ in all_fields))
        for i in range(t_us.size):
            values: list[float | int] = [int(t_us[i])]
            values += [float(samples[field][i]) for _, field in all_fields[1:]]
            self._data.append((msg_id, packer.pack(*values)))
        return self

    # ----------------------------------------------------------------- #

    def write(self, path: Path) -> Path:
        chunks = [_MAGIC + struct.pack("<BQ", _VERSION, self._start_us)]
        # Flag bits: no incompatible features, nothing appended.
        chunks.append(_message(_MSG_FLAG_BITS, bytes(8) + bytes(8) + struct.pack("<3Q", 0, 0, 0)))

        for name, fields in self._formats:
            definition = name + ":" + "".join(f"{t} {f};" for t, f in fields)
            chunks.append(_message(_MSG_FORMAT, definition.encode()))

        for key, value in self._info.items():
            chunks.append(_message(_MSG_INFO, _keyed(f"char[{len(value)}] {key}", value.encode())))

        for key, value in self._params.items():
            chunks.append(
                _message(_MSG_PARAMETER, _keyed(f"float {key}", struct.pack("<f", float(value))))
            )

        for name, msg_id in self._subscriptions.items():
            chunks.append(_message(_MSG_ADD_LOGGED, struct.pack("<BH", 0, msg_id) + name.encode()))

        for msg_id, payload in self._data:
            chunks.append(_message(_MSG_DATA, struct.pack("<H", msg_id) + payload))

        path.write_bytes(b"".join(chunks))
        return path


def _message(kind: int, payload: bytes) -> bytes:
    return struct.pack("<HB", len(payload), kind) + payload


def _keyed(key: str, value: bytes) -> bytes:
    return struct.pack("<B", len(key)) + key.encode() + value


def write_px4_log(
    path: Path,
    *,
    duration_s: float = 40.0,
    rate_hz: float = 400.0,
    params: dict[str, float] | None = None,
    with_esc: bool = True,
    torque_topic: str = "vehicle_torque_setpoint",
) -> Path:
    """A small but structurally real PX4 log.

    The signal content is arbitrary -- what is being tested is the reading, not
    the flying -- but the *structure* is exactly what a real log has: a
    post-filter angular velocity, a normalized torque setpoint, a quaternion
    attitude, and optional ESC telemetry.

    Args:
        torque_topic: Which name the rate-controller output goes under. Older PX4
            logged ``actuator_controls_0``; the reader has to accept both, and
            that fallback is only meaningfully tested by writing the old one.
    """
    t = np.arange(0.0, duration_s, 1.0 / rate_hz)
    phase = 2.0 * np.pi * 0.5 * t
    writer = ULogWriter()
    writer.info("ver_sw_release", "1.14.0")
    writer.info("ver_hw", "PX4_FMU_V6X")
    writer.parameters(
        params
        if params is not None
        else {
            "IMU_GYRO_RATEMAX": rate_hz,
            "IMU_GYRO_CUTOFF": 80.0,
            "IMU_DGYRO_CUTOFF": 50.0,
            "MC_ROLLRATE_P": 0.15,
            "MC_ROLLRATE_I": 0.2,
            "MC_ROLLRATE_D": 0.003,
            "MC_ROLLRATE_K": 1.0,
        }
    )

    writer.topic(
        "vehicle_angular_velocity",
        [("float", f"xyz[{i}]") for i in range(3)],
        {
            "timestamp": t,
            "xyz[0]": 0.4 * np.sin(phase),
            "xyz[1]": 0.2 * np.cos(phase),
            "xyz[2]": 0.1 * np.sin(0.5 * phase),
        },
    )

    # The rate-controller output is indexed differently under the two topic names.
    torque_field = "xyz[{i}]" if torque_topic == "vehicle_torque_setpoint" else "control[{i}]"
    writer.topic(
        torque_topic,
        [("float", torque_field.format(i=i)) for i in range(3)],
        {
            "timestamp": t,
            **{torque_field.format(i=i): 0.1 * np.sin(phase + i) for i in range(3)},
        },
    )
    writer.topic(
        "vehicle_attitude",
        [("float", f"q[{i}]") for i in range(4)],
        {
            "timestamp": t,
            "q[0]": np.cos(0.05 * np.sin(phase)),
            "q[1]": np.sin(0.05 * np.sin(phase)),
            "q[2]": np.zeros_like(t),
            "q[3]": np.zeros_like(t),
        },
    )
    if with_esc:
        t_esc = t[::40]
        writer.topic(
            "esc_status",
            [("float", f"esc[{i}].esc_rpm") for i in range(4)],
            {
                "timestamp": t_esc,
                **{f"esc[{i}].esc_rpm": np.full(t_esc.size, 3000.0) for i in range(4)},
            },
        )
    return writer.write(path)
