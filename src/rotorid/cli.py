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
import math
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
    analyze.add_argument(
        "--export",
        type=Path,
        default=None,
        metavar="DIR",
        help="write staged .param files, one per flight, into this directory",
    )
    analyze.add_argument(
        "--session",
        type=Path,
        default=None,
        help="save the whole analysis to a .rotorid bundle for reopening later",
    )
    analyze.add_argument(
        "--acknowledge",
        default="",
        help="comma-separated finding codes to accept, unblocking the export",
    )
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(handler=_cmd_analyze)

    gui = sub.add_parser("gui", help="open the interactive window")
    gui.add_argument("log", type=Path, nargs="?", default=None)
    gui.add_argument("--theme", choices=("light", "dark"), default="light")
    gui.set_defaults(handler=_cmd_gui)

    session = sub.add_parser("session", help="reopen a saved .rotorid analysis")
    session.add_argument("session", type=Path)
    session.add_argument("--config", type=Path, default=None, help="override rotorid.toml")
    session.set_defaults(handler=_cmd_session)

    return parser


def _read(path: Path) -> LogBundle:
    """Read a log, choosing the reader by extension.

    Extension, not content sniffing: a file named ``.bin`` that is really a uLog
    is a mistake worth surfacing rather than papering over, because the parameter
    names the whole analysis then uses depend on which stack this is.
    """
    from rotorid.core.io.ardupilot import read_ardupilot
    from rotorid.core.io.px4 import read_px4

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    suffix = path.suffix.lower()
    if suffix == ".bin":
        return read_ardupilot(path)
    if suffix == ".ulg":
        return read_px4(path)
    raise ValueError(
        f"{path.name}: expected an ArduPilot .bin or a PX4 .ulg log, not {suffix or 'no'} extension"
    )


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
        print(f"  excitation: none found -- see docs/logging-setup-{bundle.stack}.md")
    for warning in bundle.warnings:
        print(f"  ! {warning}")
    return EXIT_OK


def _cmd_analyze(args: argparse.Namespace) -> int:
    from rotorid.core.export.params import write_param_files
    from rotorid.core.export.report import write_report
    from rotorid.core.export.session import save_session
    from rotorid.core.pipeline import analyze

    bundle = _read(args.log)
    config = load_config(args.config)
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    unknown = [a for a in axes if a not in AXES]
    if unknown:
        raise ValueError(f"unknown axes {unknown}; choose from {list(AXES)}")

    acknowledgements = {
        code.strip(): "accepted on the command line"
        for code in args.acknowledge.split(",")
        if code.strip()
    }
    result = analyze(
        bundle,
        tuple(cast("Axis", a) for a in axes),
        config,
        tool_version=__version__,
        conservatism=args.conservatism,
        acknowledgements=acknowledgements,
    )
    session = result.session
    results = session.recommendations
    findings = session.findings
    plan = session.next_steps
    unresolved = result.blockers

    if args.report is not None and results:
        write_report(
            args.report,
            bundle,
            {str(a): r for a, r in results.items()},
            config_hash=config.hash,
            tool_version=__version__,
            findings=findings,
            plan=plan,
            measured_steps={str(a): m for a, m in session.measured_steps.items()},
        )

    if args.session is not None:
        save_session(args.session, session)

    exported: list[Path] = []
    if args.export is not None and plan is not None and plan.stages and not unresolved:
        exported = write_param_files(
            args.export,
            plan,
            log_name=bundle.path.name,
            tool_version=__version__,
            config_hash=config.hash,
            findings=findings,
            acknowledgements=session.acknowledgements,
        )

    if args.json:
        print(
            json.dumps(
                {
                    "axes": {a: _jsonable(r) for a, r in results.items()},
                    "exported": [str(p) for p in exported],
                    "failed": result.failures,
                    "findings": [_jsonable(f) for f in findings],
                    "blocked": list(unresolved),
                },
                indent=2,
            )
        )
    else:
        for axis, rec in results.items():
            _print_axis(axis, rec)
        for axis, why in result.failures.items():
            print(f"\n{axis}: not analysed -- {why}")
        _print_findings(findings)
        if args.report is not None and results:
            print(f"\nreport written to {args.report}")
        if args.session is not None:
            print(f"session written to {args.session}")

    if not results:
        return EXIT_BLOCKED
    if unresolved:
        print(
            "\nblocked by "
            + ", ".join(unresolved)
            + ". Re-run with --acknowledge "
            + ",".join(unresolved)
            + " to accept the stated risk; the acknowledgement is recorded in the export.",
            file=sys.stderr,
        )
        return EXIT_BLOCKED
    return EXIT_OK


def _cmd_gui(args: argparse.Namespace) -> int:
    """Open the window. Imported here so the CLI works without PySide6 installed."""
    try:
        from rotorid.gui.app import run
    except ImportError as exc:  # pragma: no cover - depends on the install extras
        raise ValueError(
            "the GUI needs the optional dependencies: pip install 'rotorid[gui]'"
        ) from exc
    return run(args.log, theme=args.theme)


def _cmd_session(args: argparse.Namespace) -> int:
    """Reopen a saved analysis without the log it came from."""
    from rotorid.core.export.session import load_session

    config = load_config(args.config)
    session, mismatch = load_session(
        args.session, tool_version=__version__, config_hash=config.hash
    )
    print(
        f"{args.session.name}: {session.log.path.name}, "
        f"{len(session.recommendations)} axis/axes, "
        f"saved {session.created_utc:%Y-%m-%d %H:%M UTC} by RotorID {session.tool_version}"
    )
    if mismatch:
        print(f"note: {mismatch.describe()}")
    for axis, rec in session.recommendations.items():
        _print_axis(axis, rec)
    _print_findings(session.findings)
    if session.acknowledgements:
        print("\nacknowledged")
        for code, why in sorted(session.acknowledgements.items()):
            print(f"  {code}: {why}")
    return EXIT_OK


def _print_findings(findings: Sequence[Any]) -> None:
    """Findings after the numbers, because they qualify what was just printed."""
    if not findings:
        return
    print("\nfindings")
    for finding in findings:
        print(f"  [{finding.severity:<7}] {finding.code:<28} {finding.title}")
        print(f"              {finding.action}")


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
    _print_filters(rec)


def _print_filters(rec: Any) -> None:
    """The filter half of the recommendation, as a diff or as a stated decision."""
    filters = rec.filters
    if filters.params:
        print(f"  filters   {filters.chain.describe()}")
        for name, value in sorted(filters.params.items()):
            print(f"    {name:<24} {value:g}")
        print(
            "    ! fly the filter changes on their own first, then the gains -- "
            "changing both at once makes a bad outcome undiagnosable"
        )
    else:
        print(f"  filters   unchanged -- {filters.rationale.split('.')[0]}.")
    noise = (
        f"D-term output {rec.dterm_noise_rms_pct:.1f}% of full scale"
        if math.isfinite(rec.dterm_noise_rms_pct)
        else "D-term output noise not measured -- no usable noise spectrum in this log"
    )
    print(f"  noise     {noise}, at {filters.phase_cost_deg:.1f} deg of chain phase")


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
