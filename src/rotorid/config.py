"""Configuration loading, merging and hashing.

Every threshold in RotorID lives in ``rotorid.toml`` (spec section 4), never in
code. This module loads the packaged defaults, merges an optional user override
on top, and hashes the resolved result into :attr:`Config.hash` so that
``Session.config_hash`` pins the numbers a session was produced with.

Lookups are strict: asking for a key that does not exist raises, rather than
silently falling back to a default that nobody wrote down.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_CONFIG_PATH", "Config", "load_config"]

#: Packaged defaults. Shipped next to the package so an installed wheel works.
DEFAULT_CONFIG_PATH = Path(__file__).with_name("rotorid.toml")

# In a source checkout the file lives at the repo root instead.
if not DEFAULT_CONFIG_PATH.exists():  # pragma: no cover - depends on install layout
    _repo_root = Path(__file__).resolve().parents[2]
    DEFAULT_CONFIG_PATH = _repo_root / "rotorid.toml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    out = dict(base)
    for key, value in override.items():
        existing = out.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            out[key] = _deep_merge(existing, value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True, slots=True)
class Config:
    """Resolved configuration.

    Attributes:
        data: The merged configuration tree.
        hash: SHA-256 over the canonical JSON form of ``data``, truncated to 16
            hex characters. Recorded in every session for determinism.
        sources: Paths that contributed, in application order.
    """

    data: dict[str, Any]
    hash: str
    sources: tuple[Path, ...]

    def section(self, name: str) -> dict[str, Any]:
        """Return one top-level section.

        Raises:
            KeyError: if the section is absent.
        """
        try:
            value = self.data[name]
        except KeyError:
            raise KeyError(
                f"config section [{name}] not found; known sections: {sorted(self.data)}"
            ) from None
        if not isinstance(value, dict):
            raise KeyError(f"config entry [{name}] is a value, not a section")
        return value

    def get(self, section: str, key: str) -> Any:
        """Return one value.

        Raises:
            KeyError: if the section or key is absent. Deliberately strict -- a
                missing threshold is a bug in the config, not a reason to guess.
        """
        sect = self.section(section)
        try:
            return sect[key]
        except KeyError:
            raise KeyError(
                f"config key [{section}].{key} not found; known keys: {sorted(sect)}"
            ) from None

    def float_(self, section: str, key: str) -> float:
        """Return one value as a float."""
        value = self.get(section, key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"config [{section}].{key} = {value!r} is not a number")
        return float(value)

    def int_(self, section: str, key: str) -> int:
        """Return one value as an int."""
        value = self.get(section, key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"config [{section}].{key} = {value!r} is not an integer")
        return value

    def pair(self, section: str, key: str) -> tuple[float, float]:
        """Return one value as exactly a ``(low, high)`` bound.

        Raises:
            ValueError: if the list is not a pair. A three-element "range" would
                otherwise be silently truncated to its first two entries.
        """
        values = self.floats(section, key)
        if len(values) != 2:
            raise ValueError(f"[{section}].{key} must be a [low, high] pair, got {list(values)}")
        return values[0], values[1]

    def floats(self, section: str, key: str) -> tuple[float, ...]:
        """Return one value as a tuple of floats."""
        value = self.get(section, key)
        if not isinstance(value, list) or not value:
            raise TypeError(f"config [{section}].{key} = {value!r} is not a non-empty list")
        return tuple(float(v) for v in value)


def load_config(user_override: Path | str | None = None) -> Config:
    """Load packaged defaults, optionally merged with a user override file.

    Args:
        user_override: Path to a TOML file whose keys win over the defaults.

    Returns:
        The resolved :class:`Config`.

    Raises:
        FileNotFoundError: if the packaged defaults or the named override are missing.
    """
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"packaged defaults not found at {DEFAULT_CONFIG_PATH}")

    with DEFAULT_CONFIG_PATH.open("rb") as handle:
        data: dict[str, Any] = tomllib.load(handle)
    sources: list[Path] = [DEFAULT_CONFIG_PATH]

    if user_override is not None:
        override_path = Path(user_override)
        if not override_path.exists():
            raise FileNotFoundError(f"config override not found: {override_path}")
        with override_path.open("rb") as handle:
            data = _deep_merge(data, tomllib.load(handle))
        sources.append(override_path)

    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return Config(data=data, hash=digest, sources=tuple(sources))
