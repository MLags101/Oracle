"""Data-collection parameter profiles (spec sections 7 and 13).

The single biggest determinant of whether this tool can say anything useful is
how the log was recorded, and that is decided before the flight by a handful of
parameters nobody remembers. `docs/logging-setup-*.md` explains them; this module
emits them as a file the user loads, because a recipe that has to be typed in
correctly is a recipe half the users will get wrong in the one place that
matters.

Two profiles per stack, and the difference between them is the point:

* **collect** turns on the logging a good identification needs, and nothing else.
  It is safe to leave on.
* **sweep** additionally configures the excitation itself -- the SYSTEMID sweep on
  ArduPilot, the autotune on PX4 -- and is emphatically not safe to leave on,
  because the next arming injects a chirp into the rate loop.

Both are written with the same header machinery as the tune export, and for the
same reason: a parameter file outlives the session that produced it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from rotorid.core.types import Axis, Stack

__all__ = ["PROFILES", "Profile", "profile", "write_profile"]

Profile = Literal["collect", "sweep"]

#: The profiles on offer, in the order they are meant to be used.
PROFILES: tuple[Profile, ...] = ("collect", "sweep")

#: ``LOG_BITMASK`` bits the analysis depends on, by what each one buys.
#:
#: Bit 0 is the one that decides whether a log is usable at all: with it clear,
#: ``RATE`` and ``ATT`` go out on the 10 Hz medium-rate schedule however fast the
#: loop runs, and nothing else in the file admits it.
_AP_LOG_BITS: tuple[tuple[int, str], ...] = (
    (0, "ATTITUDE_FAST -- RATE and ATT at the loop rate. Without this nothing works."),
    (2, "IMU -- gives VIBE: vibration level and accelerometer clipping"),
    (12, "PID -- the PIDR/PIDP/PIDY messages, for term-level diagnosis"),
    (18, "IMU_FAST -- higher-rate IMU, better noise spectra"),
)

#: ``SID_AXIS`` for a rate-loop injection on each axis. 10-12 inject at the mixer
#: instead, which measures the plant directly but excites the airframe harder.
_SID_AXIS: dict[Axis, float] = {"roll": 7.0, "pitch": 8.0, "yaw": 9.0}


def profile(
    stack: Stack,
    which: Profile = "collect",
    *,
    axis: Axis = "roll",
    f_start_hz: float = 0.05,
    f_stop_hz: float = 20.0,
    magnitude: float = 0.05,
) -> tuple[dict[str, float], tuple[str, ...]]:
    """Parameters for one profile, and the notes that belong in its header.

    Args:
        axis: Which axis the sweep excites. One axis per flight -- the
            identification's single-axis assumption is not a formality, and a
            sweep on two axes at once produces a model of neither.
        magnitude: ``SID_MAGNITUDE``. Deliberately low: it is the one value that
            needs judgement in the air, and starting small and raising it is the
            only safe direction to approach it from.

    Raises:
        ValueError: on an unknown stack or profile name.
    """
    if which not in PROFILES:
        raise ValueError(f"unknown profile {which!r}; expected one of {PROFILES}")
    if stack == "ardupilot":
        return _ardupilot(which, axis, f_start_hz, f_stop_hz, magnitude)
    if stack == "px4":
        return _px4(which)
    raise ValueError(f"unknown stack {stack!r}")


def write_profile(
    path: Path,
    stack: Stack,
    which: Profile = "collect",
    *,
    tool_version: str,
    axis: Axis = "roll",
    f_start_hz: float = 0.05,
    f_stop_hz: float = 20.0,
    magnitude: float = 0.05,
) -> Path:
    """Write a data-collection profile as a loadable ``.param`` file.

    Returns:
        The path written, for convenience.
    """
    params, notes = profile(
        stack,
        which,
        axis=axis,
        f_start_hz=f_start_hz,
        f_stop_hz=f_stop_hz,
        magnitude=magnitude,
    )
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# RotorID {tool_version} -- {stack} data-collection profile: {which}",
        f"# Generated {when}",
        "#",
        "# BACK UP YOUR CURRENT PARAMETERS BEFORE LOADING THIS FILE.",
        "# This file changes what your vehicle records, and in the 'sweep' profile",
        "# what it does when armed. It does not change how it flies otherwise.",
        "#",
    ]
    lines += [f"# {note}" for note in notes]
    lines.append("#")
    lines += [
        f"{name},{value:.6f}".rstrip("0").rstrip(".") for name, value in sorted(params.items())
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _ardupilot(
    which: Profile, axis: Axis, f_start_hz: float, f_stop_hz: float, magnitude: float
) -> tuple[dict[str, float], tuple[str, ...]]:
    """ArduPilot: logging bits, batch gyro logging, and optionally the sweep."""
    bitmask = sum(1 << bit for bit, _ in _AP_LOG_BITS)
    params: dict[str, float] = {
        "LOG_BITMASK": float(bitmask),
        # Pre *and* post-filter batch gyro. This is what turns filter
        # verification from a model comparison into a measurement, and it is the
        # single most valuable optional thing a log can carry.
        "INS_LOG_BAT_MASK": 1.0,
        "INS_LOG_BAT_OPT": 4.0,
    }
    notes = [
        f"LOG_BITMASK = {bitmask} sets:",
        *[f"  bit {bit}: {why}" for bit, why in _AP_LOG_BITS],
        "INS_LOG_BAT_MASK/OPT turn on pre- and post-filter batch gyro logging,",
        "  which lets RotorID check its filter model against your firmware rather",
        "  than assume it. Set INS_LOG_BAT_MASK back to 0 afterwards: it consumes",
        "  log bandwidth and RAM.",
    ]
    if which == "collect":
        notes.append(
            "This profile only changes logging. It is safe to leave loaded, apart "
            "from the batch-logging cost above."
        )
        return params, tuple(notes)

    params.update(
        {
            "SID_AXIS": _SID_AXIS[axis],
            "SID_F_START_HZ": f_start_hz,
            "SID_F_STOP_HZ": f_stop_hz,
            "SID_MAGNITUDE": magnitude,
            "SID_T_FADE_IN": 5.0,
            "SID_T_REC": 120.0,
            "SID_T_FADE_OUT": 5.0,
        }
    )
    notes += [
        "",
        f"SWEEP CONFIGURED ON {axis.upper()}. Switching to SYSTEMID mode will inject a",
        f"  {f_start_hz:g}-{f_stop_hz:g} Hz chirp into the rate loop. Fly at altitude, in",
        "  low wind, with room to recover, and one axis per flight.",
        f"SID_MAGNITUDE is deliberately low at {magnitude:g}. Raise it until the response",
        "  stands clearly above the noise without saturating a motor -- TUNE = 58 lets",
        "  you adjust it from the tuning knob in flight instead of landing between tries.",
        "SET SID_AXIS BACK TO 0 WHEN YOU ARE DONE.",
    ]
    return params, tuple(notes)


def _px4(which: Profile) -> tuple[dict[str, float], tuple[str, ...]]:
    """PX4: high-rate logging, the onboard FFT, and optionally autotune.

    PX4 has no SYSTEMID equivalent, so the 'sweep' profile configures the
    multicopter autotune instead -- with ``MC_AT_APPLY`` at 0, so it identifies
    and reports without writing gains to the vehicle. That is the same division
    of labour this whole tool is built on: the machine measures, the human
    decides.
    """
    params: dict[str, float] = {
        "SDLOG_PROFILE": 3.0,  # default plus high-rate topics
        "IMU_GYRO_FFT_EN": 1.0,
    }
    notes = [
        "SDLOG_PROFILE = 3 adds the high-rate topics the identification needs.",
        "IMU_GYRO_FFT_EN = 1 turns on the onboard FFT, which is what lets a notch",
        "  track the motors on a vehicle without ESC telemetry. If your ESCs do",
        "  report RPM, enable that too -- esc_status.esc_rpm is strictly better.",
    ]
    if which == "collect":
        notes.append("This profile only changes logging. It is safe to leave loaded.")
        return params, tuple(notes)

    params.update({"MC_AT_EN": 1.0, "MC_AT_APPLY": 0.0})
    notes += [
        "",
        "MC_AT_EN = 1 arms the multicopter autotune; MC_AT_APPLY = 0 means it will",
        "  identify and report without writing gains to the vehicle, which is what",
        "  you want when the analysis is going to be done offline.",
        "Fly about 30 s of hover with deliberate roll, pitch and yaw excitation, or",
        "  run the autotune, then download the log.",
    ]
    return params, tuple(notes)
