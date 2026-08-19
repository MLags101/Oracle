"""Headless entry point (spec section 14).

The CLI is built before the GUI on purpose. If it can produce a traceable
recommendation from a log, the GUI is presentation work; if it cannot, no amount
of GUI will rescue it.

Exit codes are part of the interface, so this can be scripted over a directory of
logs: ``0`` success, ``2`` blocking findings, ``3`` unparseable log.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

from rotorid import __version__
from rotorid.config import load_config
from rotorid.core.types import AXES, Axis, LogBundle

__all__ = ["main"]

EXIT_OK = 0
EXIT_BLOCKED = 2
EXIT_UNREADABLE = 3


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code rather than calling ``sys.exit``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK

    try:
        return int(args.handler(args))
    except FileNotFoundError as exc:
        print(f"rotorid: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE
    except ValueError as exc:
        # Analysis refusing to guess is a normal, expected outcome and must read
        # as an explanation rather than as a crash.
        print(f"rotorid: {exc}", file=sys.stderr)
        return EXIT_BLOCKED


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rotorid", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=f"rotorid {__version__}")
    sub = parser.add_subparsers(dest="command")

    inspect = sub.add_parser("inspect", help="what is in a log, and what is missing")
    inspect.add_argument("log", type=Path)
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(handler=_cmd_inspect)

    analyze = sub.add_parser("analyze", help="identify and recommend gains")
    analyze.add_argument("log", type=Path)
    analyze.add_argument("--axes", default="roll,pitch,yaw", help="comma-separated axis list")
    analyze.add_argument("--conservatism", type=float, default=0.5, help="0 aggressive, 1 docile")
    analyze.add_argument("--config", type=Path, default=None, help="override rotorid.toml")
    analyze.add_argument("-o", "--report", type=Path, default=None, help="write an HTML report")
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(handler=_cmd_analyze)

    return parser


def _read(path: Path) -> LogBundle:
    from rotorid.core.io.ardupilot import read_ardupilot

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if path.suffix.lower() != ".bin":
        raise ValueError(
            f"{path.name}: only ArduPilot .bin logs are supported so far; "
            "PX4 .ulg support arrives with milestone M9"
        )
    return read_ardupilot(path)


def _cmd_inspect(args: argparse.Namespace) -> int:
    bundle = _read(args.log)
    from rotorid.core.preprocess.segment import propose_segments

    segments = propose_segments(bundle)
    payload = {
        "path": str(bundle.path),
        "stack": bundle.stack,
        "firmware": bundle.firmware_version,
        "sample_rate_hz": bundle.sample_rate_hz,
        "loop_rate_hz": bundle.loop_rate_hz,
        "gyro_sample_rate_hz": bundle.gyro_sample_rate_hz,
        "signals": sorted(bundle.signals),
        "n_params": len(bundle.params),
        "warnings": list(bundle.warnings),
        "segments": [
            {
                "axis": s.axis,
                "kind": s.kind,
                "t_start": round(s.t_start, 2),
                "t_end": round(s.t_end, 2),
                "confidence": s.confidence,
            }
            for s in segments
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return EXIT_OK

    print(f"{bundle.path.name}  [{bundle.stack}]  {bundle.firmware_version or 'firmware unknown'}")
    print(
        f"  grid {bundle.sample_rate_hz:.0f} Hz   loop {bundle.loop_rate_hz:.0f} Hz   "
        f"gyro {bundle.gyro_sample_rate_hz:.0f} Hz   {len(bundle.params)} params"
    )
    print(f"  signals: {len(bundle.signals)}")
    for key in sorted(bundle.signals):
        print(f"    {key}")
    if segments:
        print("  excitation:")
        for s in segments:
            print(
                f"    {s.axis:<6} {s.kind:<15} {s.t_start:8.1f}-{s.t_end:8.1f}s  "
                f"confidence {s.confidence:.1f}"
            )
    else:
        print(
            "  excitation: none found. Fly a SYSTEMID sweep -- see docs/logging-setup-ardupilot.md"
        )
    for warning in bundle.warnings:
        print(f"  ! {warning}")
    return EXIT_OK


def _cmd_analyze(args: argparse.Namespace) -> int:
    from rotorid.core.design.recommend import analyze_axis
    from rotorid.core.export.report import write_report

    bundle = _read(args.log)
    config = load_config(args.config)
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    unknown = [a for a in axes if a not in AXES]
    if unknown:
        raise ValueError(f"unknown axes {unknown}; choose from {list(AXES)}")

    results = {}
    failures: dict[str, str] = {}
    for axis in axes:
        try:
            results[axis] = analyze_axis(
                bundle,
                cast("Axis", axis),
                config,
                conservatism=args.conservatism,
            )
        except ValueError as exc:
            failures[axis] = str(exc)

    if args.report is not None and results:
        write_report(
            args.report,
            bundle,
            results,
            config_hash=config.hash,
            tool_version=__version__,
        )

    if args.json:
        print(
            json.dumps(
                {"axes": {a: _jsonable(r) for a, r in results.items()}, "failed": failures},
                indent=2,
            )
        )
    else:
        for axis, rec in results.items():
            _print_axis(axis, rec)
        for axis, why in failures.items():
            print(f"\n{axis}: not analysed -- {why}")
        if args.report is not None and results:
            print(f"\nreport written to {args.report}")

    if not results:
        return EXIT_BLOCKED
    return EXIT_OK


def _print_axis(axis: str, rec: Any) -> None:
    m = rec.margins
    print(f"\n{axis.upper()}   confidence {rec.confidence}")
    print(
        f"  model     {rec.model.structure}  "
        + "  ".join(f"{k}={v:.4g}" for k, v in sorted(rec.model.params.items()))
    )
    print(
        f"  fit       {rec.model.fit_rms_db:.2f} dB / {rec.model.fit_rms_deg:.1f} deg over "
        f"{rec.model.valid_band_hz[0]:.2f}-{rec.model.valid_band_hz[1]:.1f} Hz"
    )
    print(
        f"  margins   PM {m.phase_margin_deg:.0f} deg   GM {m.gain_margin_db:.1f} dB   "
        f"wc {m.crossover_hz:.2f} Hz   Ms {m.peak_sensitivity_db:.1f} dB   "
        f"DRB {m.disturbance_rejection_bw_hz:.2f} Hz"
    )
    print(
        f"  gains     P {rec.baseline_gains.kp:.4g} -> {rec.gains.kp:.4g}   "
        f"I {rec.baseline_gains.ki:.4g} -> {rec.gains.ki:.4g}   "
        f"D {rec.baseline_gains.kd:.4g} -> {rec.gains.kd:.4g}"
    )
    print(f"  limited by {rec.binding_constraint}")


def _jsonable(value: Any) -> Any:
    """Recursively convert dataclasses and numpy scalars for ``json.dumps``."""
    import numpy as np

    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return f"<array shape={value.shape}>"
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
