"""Session save and load (spec section 7).

A ``.rotorid`` file is a zip holding two members:

* ``session.json`` -- the structure, as a tagged tree.
* ``arrays.npz`` -- every array in it, compressed, referenced from the json by key.

The split is what makes the format worth having. Sessions are mostly arrays by
volume and mostly structure by meaning, and putting the arrays through json would
make the file both enormous and lossy at the last bit of every float.

The design constraint that shapes the rest is **round-trip fidelity**: a reloaded
session has to produce the same recommendation, the same figures and the same
export as the session it was saved from, because the reason to save one is to
come back and finish the argument tomorrow. So the encoder is generic over the
dataclasses rather than hand-written per type -- a hand-written encoder silently
drops the field somebody adds next week, and the drop shows up as a subtly
different recommendation rather than as an error.

What is deliberately *not* preserved is authority. A session records the tool
version and the config hash that produced it, and :func:`load_session` reports
when either has moved, because numbers computed by an older version under
different thresholds are evidence about that version, not about this one.
"""

from __future__ import annotations

import dataclasses
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from rotorid.core.filters.chain import FilterChain
from rotorid.core.filters.harmonic import HarmonicNotch
from rotorid.core.types import (
    AirframeModel,
    BatchSamples,
    EffectivePlant,
    ExcitationSegment,
    FilterRecommendation,
    Finding,
    FlightTestPlan,
    FlightTestStage,
    FrequencyResponse,
    GainSet,
    LatencyBudget,
    LogBundle,
    MarginReport,
    MeasuredStep,
    NoiseProfile,
    Session,
    Signal,
    SpectralPeak,
    StepMetrics,
    TuneRecommendation,
)

__all__ = ["SessionMismatch", "load_session", "save_session"]

#: Bumped when the encoding itself changes shape, which is not the same thing as
#: the tool version changing. A file written under a different format version is
#: refused rather than guessed at.
FORMAT_VERSION = 1

_JSON_MEMBER = "session.json"
_ARRAY_MEMBER = "arrays.npz"

#: Every class the encoder will reconstruct. A tag outside this set is refused:
#: a session file is data, and data does not get to name the class it becomes.
_CLASSES: tuple[type, ...] = (
    AirframeModel,
    BatchSamples,
    EffectivePlant,
    ExcitationSegment,
    FilterChain,
    FilterRecommendation,
    Finding,
    FlightTestPlan,
    FlightTestStage,
    FrequencyResponse,
    GainSet,
    HarmonicNotch,
    LatencyBudget,
    LogBundle,
    MarginReport,
    MeasuredStep,
    NoiseProfile,
    Session,
    Signal,
    SpectralPeak,
    StepMetrics,
    TuneRecommendation,
)
_BY_NAME = {cls.__name__: cls for cls in _CLASSES}


@dataclasses.dataclass(frozen=True, slots=True)
class SessionMismatch:
    """What has moved since the session was saved.

    Not an error. A session loaded under a newer tool is still perfectly good
    evidence about the flight; it is just no longer evidence about what *this*
    build would recommend. The caller decides what to do about that, and the GUI
    says so on the screen rather than in a log line.
    """

    tool_version: tuple[str, str] | None = None
    config_hash: tuple[str, str] | None = None

    def __bool__(self) -> bool:
        return self.tool_version is not None or self.config_hash is not None

    def describe(self) -> str:
        parts = []
        if self.tool_version is not None:
            parts.append(f"saved by RotorID {self.tool_version[0]}, running {self.tool_version[1]}")
        if self.config_hash is not None:
            parts.append(f"saved under config {self.config_hash[0]}, running {self.config_hash[1]}")
        if not parts:
            return "unchanged since it was saved"
        return (
            "; ".join(parts) + ". The numbers in it are what that version recommended. Re-run the "
            "analysis to see what this one recommends."
        )


def save_session(path: Path, session: Session) -> Path:
    """Write a session to a ``.rotorid`` bundle.

    The write goes to a temporary neighbour and is renamed into place, so an
    interrupted save cannot leave a half-written file where a good one was.
    """
    arrays: dict[str, np.ndarray] = {}
    tree = {
        "format_version": FORMAT_VERSION,
        "session": _encode(session, arrays),
    }

    buffer = io.BytesIO()
    # The stub types the second positional as a flag; these are named members.
    np.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]

    tmp = path.with_suffix(path.suffix + ".partial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_JSON_MEMBER, json.dumps(tree, indent=1))
        # Already deflated by savez_compressed; storing avoids a second pass.
        zf.writestr(zipfile.ZipInfo(_ARRAY_MEMBER), buffer.getvalue(), zipfile.ZIP_STORED)
    tmp.replace(path)
    return path


def load_session(
    path: Path, *, tool_version: str | None = None, config_hash: str | None = None
) -> tuple[Session, SessionMismatch]:
    """Read a ``.rotorid`` bundle back.

    Args:
        tool_version: The running version, to compare against the saved one.
        config_hash: The running config hash, likewise. Both optional: loading a
            session to look at it does not require anything to compare it with.

    Returns:
        The session, and what has moved since it was written.

    Raises:
        ValueError: if the file is not a session bundle, was written by a
            different format version, or names a class the loader does not know.
    """
    with zipfile.ZipFile(path) as zf:
        try:
            tree = json.loads(zf.read(_JSON_MEMBER))
            with io.BytesIO(zf.read(_ARRAY_MEMBER)) as buffer, np.load(buffer) as npz:
                arrays = {key: npz[key] for key in npz.files}
        except KeyError as exc:
            raise ValueError(f"{path} is not a RotorID session bundle") from exc

    saved_format = tree.get("format_version")
    if saved_format != FORMAT_VERSION:
        raise ValueError(
            f"{path} was written in session format {saved_format}, and this build "
            f"reads format {FORMAT_VERSION}. Re-run the analysis from the log."
        )

    session = _decode(tree["session"], arrays)
    if not isinstance(session, Session):
        raise ValueError(f"{path} does not contain a session")

    return session, SessionMismatch(
        tool_version=(
            (session.tool_version, tool_version)
            if tool_version is not None and tool_version != session.tool_version
            else None
        ),
        config_hash=(
            (session.config_hash, config_hash)
            if config_hash is not None and config_hash != session.config_hash
            else None
        ),
    )


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def _encode(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    """One value as a json-safe tagged tree, with arrays lifted out to ``arrays``.

    Tags are needed for everything json flattens: tuple and list are both json
    arrays, and a dict with float keys is not expressible at all. Losing any of
    those would change what the reloaded objects are.
    """
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, np.ndarray):
        key = f"a{len(arrays)}"
        arrays[key] = value
        return {"$array": key}
    if isinstance(value, np.floating | np.integer | np.bool_):
        return _encode(value.item(), arrays)
    if isinstance(value, int | float):
        return value
    if isinstance(value, Path):
        return {"$path": str(value)}
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        if cls.__name__ not in _BY_NAME:
            raise TypeError(f"{cls.__name__} is not registered for session storage")
        return {
            "$type": cls.__name__,
            "fields": {
                f.name: _encode(getattr(value, f.name), arrays) for f in dataclasses.fields(value)
            },
        }
    if isinstance(value, tuple):
        return {"$tuple": [_encode(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_encode(item, arrays) for item in value]
    if isinstance(value, dict):
        return {"$dict": [[_encode(k, arrays), _encode(v, arrays)] for k, v in value.items()]}
    raise TypeError(f"{type(value).__name__} cannot be stored in a session")


def _decode(node: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(node, list):
        return [_decode(item, arrays) for item in node]
    if not isinstance(node, dict):
        return node
    if "$array" in node:
        return arrays[node["$array"]]
    if "$path" in node:
        return Path(node["$path"])
    if "$datetime" in node:
        return datetime.fromisoformat(node["$datetime"])
    if "$tuple" in node:
        return tuple(_decode(item, arrays) for item in node["$tuple"])
    if "$dict" in node:
        return {_decode(k, arrays): _decode(v, arrays) for k, v in node["$dict"]}
    if "$type" in node:
        name = node["$type"]
        cls = _BY_NAME.get(name)
        if cls is None:
            raise ValueError(f"session names an unknown type {name!r}")
        return cls(**{k: _decode(v, arrays) for k, v in node["fields"].items()})
    raise ValueError(f"unrecognised node in session file: {sorted(node)}")
