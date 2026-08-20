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
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, cast

from rotorid import __version__
from rotorid.config import load_config
from rotorid.core.export.profile import PROFILES, Profile
from rotorid.core.logkind import KINDS
from rotorid.core.types import AXES, Axis, LogBundle, LogKind, Stack

__all__ = ["main"]

EXIT_OK = 0
EXIT_BLOCKED = 2
EXIT_UNREADABLE = 3


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code rather than calling ``sys.exit``.

    With no arguments at all this opens the window rather than printing usage.
    Loading a log is something the GUI does perfectly well on its own -- there is
    a file picker, a drag target and a File menu on the first screen -- so
    requiring the path on the command line only turned the graphical tool into
    one that had to be started from a terminal. Printing help stays the fallback
    for an install without the GUI extra.
    """
    argv = _rewrite_for_a_packaged_build(argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        # ``argv is None`` means the arguments came from the process, so an empty
        # one is a user who typed ``rotorid`` and wants the tool. An explicitly
        # passed empty list is a caller asking what the arguments are, and gets
        # told rather than having a window opened at it.
        if argv is None and len(sys.argv) <= 1:
            opened = _open_window_without_a_log()
            if opened is not None:
                return opened
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
    _add_kind_argument(inspect)
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(handler=_cmd_inspect)

    analyze = sub.add_parser("analyze", help="identify and recommend gains")
    analyze.add_argument("log", type=Path)
    analyze.add_argument("--axes", default="roll,pitch,yaw", help="comma-separated axis list")
    _add_kind_argument(analyze)
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

    filters = sub.add_parser(
        "filters",
        help="filter recommendation only, without identifying the airframe",
    )
    filters.add_argument("log", type=Path)
    filters.add_argument("--axes", default="roll,pitch,yaw", help="comma-separated axis list")
    _add_kind_argument(filters)
    filters.add_argument("--config", type=Path, default=None, help="override rotorid.toml")
    filters.add_argument("--json", action="store_true")
    filters.set_defaults(handler=_cmd_filters)

    validate = sub.add_parser(
        "validate",
        help="compare two flights: did the change do what the tool said it would",
    )
    validate.add_argument("before", type=Path, help="the log the recommendation came from")
    validate.add_argument("after", type=Path, help="a log flown after applying it")
    validate.add_argument(
        "--session",
        type=Path,
        default=None,
        help="the .rotorid saved from the 'before' analysis. Without it this is an "
        "outcome comparison rather than a validation: nothing records what was predicted",
    )
    validate.add_argument("--axes", default="roll,pitch,yaw", help="comma-separated axis list")
    validate.add_argument("--config", type=Path, default=None, help="override rotorid.toml")
    validate.add_argument("-o", "--report", type=Path, default=None, help="write an HTML report")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(handler=_cmd_validate)

    report = sub.add_parser("report", help="re-render the HTML report from a saved session")
    report.add_argument("session", type=Path)
    report.add_argument("-o", "--report", type=Path, required=True)
    report.add_argument("--config", type=Path, default=None, help="override rotorid.toml")
    report.add_argument("--json", action="store_true")
    report.set_defaults(handler=_cmd_report)

    recommend = sub.add_parser("recommend", help="write .param files from a saved session")
    recommend.add_argument("session", type=Path)
    recommend.add_argument(
        "-o", "--export", type=Path, required=True, metavar="DIR", help="directory to write into"
    )
    recommend.add_argument(
        "--stage",
        type=int,
        default=None,
        help="write only this flight from the staged plan, by its number",
    )
    recommend.add_argument(
        "--acknowledge",
        default="",
        help="comma-separated finding codes to accept, unblocking the export",
    )
    recommend.add_argument("--config", type=Path, default=None, help="override rotorid.toml")
    recommend.add_argument("--json", action="store_true")
    recommend.set_defaults(handler=_cmd_recommend)

    profile = sub.add_parser(
        "profile", help="write a data-collection parameter file to load before flying"
    )
    profile.add_argument("--stack", choices=("ardupilot", "px4"), required=True)
    profile.add_argument(
        "--which",
        choices=PROFILES,
        default="collect",
        help="'collect' turns on the logging the analysis needs; 'sweep' also configures "
        "the excitation, and must be turned off again afterwards",
    )
    profile.add_argument("--axis", choices=AXES, default="roll", help="axis for the sweep")
    profile.add_argument("-o", "--out", type=Path, required=True)
    profile.add_argument("--json", action="store_true")
    profile.set_defaults(handler=_cmd_profile)

    gui = sub.add_parser("gui", help="open the interactive window")
    gui.add_argument("log", type=Path, nargs="?", default=None)
    gui.add_argument("--theme", choices=("light", "dark"), default="light")
    gui.set_defaults(handler=_cmd_gui)

    selftest = sub.add_parser("selftest", help="check that this build actually works")
    selftest.add_argument(
        "log",
        type=Path,
        nargs="?",
        default=None,
        help="a log to read. Without one the message definitions are never exercised, "
        "which is the part of a packaged build most likely to be missing",
    )
    selftest.add_argument(
        "--out", type=Path, default=None, help="write the result here as well as to stdout"
    )
    selftest.add_argument(
        "--no-gui", action="store_true", help="skip the window, for a headless install"
    )
    selftest.add_argument("--json", action="store_true")
    selftest.set_defaults(handler=_cmd_selftest)

    session = sub.add_parser("session", help="reopen a saved .rotorid analysis")
    session.add_argument("session", type=Path)
    session.add_argument("--config", type=Path, default=None, help="override rotorid.toml")
    session.set_defaults(handler=_cmd_session)

    return parser


def _add_kind_argument(parser: argparse.ArgumentParser) -> None:
    """The one question the tool cannot answer for the user (spec 5.2).

    ``auto`` is the default rather than ``general`` because guessing wrong in
    either direction is worse than reading the file: declaring a sweep flight as
    ordinary throws away the sweep, and declaring an ordinary flight as a tuning
    one refuses it outright. Detection is right whenever the excitation is
    actually recorded, which is the only case where the distinction has teeth.
    """
    parser.add_argument(
        "--kind",
        choices=("auto", *KINDS),
        default="auto",
        help=(
            "what this flight was. 'tuning' identifies from an injected sweep or an "
            "autotune run only; 'general' identifies from ordinary stick input, caps "
            "confidence at medium and holds the design back. Default: detect from the log"
        ),
    )


def _declared_kind(args: argparse.Namespace) -> LogKind | None:
    """The ``--kind`` flag as the readers want it: ``None`` for detect."""
    kind = getattr(args, "kind", "auto")
    return None if kind == "auto" else cast("LogKind", kind)


def _read(path: Path, kind: LogKind | None = None) -> LogBundle:
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
        return read_ardupilot(path, kind=kind)
    if suffix == ".ulg":
        return read_px4(path, kind=kind)
    raise ValueError(
        f"{path.name}: expected an ArduPilot .bin or a PX4 .ulg log, not {suffix or 'no'} extension"
    )


def _cmd_inspect(args: argparse.Namespace) -> int:
    bundle = _read(args.log, _declared_kind(args))
    from rotorid.core.logkind import capabilities, detect_kind, kind_evidence
    from rotorid.core.preprocess.segment import propose_segments

    segments = propose_segments(bundle)
    caps = capabilities(bundle.kind)
    payload = {
        "path": str(bundle.path),
        "stack": bundle.stack,
        "kind": bundle.kind,
        "kind_declared": bundle.kind_was_declared,
        "kind_detected": detect_kind(bundle),
        "kind_evidence": list(kind_evidence(bundle)),
        "offers": list(caps.offers),
        "limits": list(caps.limits),
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
    source = "you said so" if bundle.kind_was_declared else "detected"
    print(f"  read as a {caps.label.lower()} log ({source})")
    for line in kind_evidence(bundle) or ("no deliberate excitation recorded",):
        print(f"    - {line}")
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
    for limit in caps.limits:
        print(f"  . {limit}")
    for warning in bundle.warnings:
        print(f"  ! {warning}")
    return EXIT_OK


def _cmd_analyze(args: argparse.Namespace) -> int:
    from rotorid.core.export.params import write_param_files
    from rotorid.core.export.report import write_report
    from rotorid.core.export.session import save_session
    from rotorid.core.pipeline import analyze

    bundle = _read(args.log, _declared_kind(args))
    config = load_config(args.config)
    axes = _axes_from(args)

    acknowledgements = {
        code.strip(): "accepted on the command line"
        for code in args.acknowledge.split(",")
        if code.strip()
    }
    result = analyze(
        bundle,
        axes,
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
            print(f"\n{axis}: not analysed -- {_unprefixed(why, axis)}")
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


def _rewrite_for_a_packaged_build(argv: Sequence[str] | None) -> Sequence[str] | None:
    """Let the packaged executable be used the way an application is used.

    Dropping a log onto the icon, or opening one with it from a file manager,
    hands the program a bare path. Argparse would reject that as an unknown
    subcommand -- and a windowed executable has nowhere to print the complaint,
    so from the user's side the program would simply fail to start. A single
    argument that is a file is what "open this" looks like, so it is read as one.

    Only under a frozen build. From a shell, ``rotorid flight.bin`` should still
    be told that the command is ``rotorid gui flight.bin``, because there the
    error is visible and the correction sticks.
    """
    if not getattr(sys, "frozen", False):
        return argv
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) == 1 and not values[0].startswith("-") and Path(values[0]).is_file():
        return ["gui", values[0]]
    return argv


def _axes_from(args: argparse.Namespace) -> tuple[Axis, ...]:
    """The ``--axes`` list, validated once rather than in each command."""
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    unknown = [a for a in axes if a not in AXES]
    if unknown:
        raise ValueError(f"unknown axes {unknown}; choose from {list(AXES)}")
    return tuple(cast("Axis", a) for a in axes)


def _cmd_filters(args: argparse.Namespace) -> int:
    """Filter recommendation only -- the fast path (spec 14).

    Deliberately reports the noise and the flown chain rather than a designed
    tune. Noise analysis needs a spectrum and a filter chain; neither needs a
    model of the aircraft, so a user whose log has no usable excitation can still
    be told that their notch is chasing the wrong line. Refusing to answer that
    because the *gain* half is impossible would be refusing the half that was
    possible.
    """
    from rotorid.core.analysis.noise import noise_profile
    from rotorid.core.preprocess.params import chain_from_bundle

    bundle = _read(args.log, _declared_kind(args))
    config = load_config(args.config)
    payload: dict[str, Any] = {"path": str(bundle.path), "stack": bundle.stack, "axes": {}}
    failures: dict[str, str] = {}

    for axis in _axes_from(args):
        signal = bundle.signals.get(f"rate.{axis}.measured")
        if signal is None or signal.t.size < 2:
            failures[axis] = "no gyro measurement in this log"
            continue
        chain = chain_from_bundle(bundle, axis)
        try:
            noise = noise_profile(
                bundle,
                axis,
                t_start=float(signal.t[0]),
                t_end=float(signal.t[-1]),
                chain=chain,
                prominence_db=config.float_("noise", "peak_prominence_db"),
                track_margin_db=config.float_("noise", "rpm_track_margin_db"),
                deconv_floor_db=config.float_("filters", "deconv_floor_db"),
                evidence_ceiling_hz=signal.native_nyquist_hz,
            )
        except (ValueError, KeyError) as exc:
            failures[axis] = str(exc)
            continue
        payload["axes"][axis] = {
            "peaks": [
                {
                    "f_hz": round(p.f_hz, 1),
                    "kind": p.kind,
                    "magnitude_db": round(p.magnitude_db, 1),
                    "tracks_rpm": p.tracks_rpm,
                }
                for p in noise.peaks
            ],
            "flown_chain": _jsonable(chain),
            "pre_filter_source": noise.pre_filter_source,
            "noise_floor_db": round(noise.noise_floor_db, 1),
        }
    payload["failures"] = failures

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return EXIT_OK if payload["axes"] else EXIT_BLOCKED

    print(f"{bundle.path.name}  [{bundle.stack}]")
    for name, found in payload["axes"].items():
        print(f"\n{name}: gyro noise, {found['pre_filter_source']} pre-filter spectrum")
        for peak in found["peaks"]:
            print(f"    {peak['f_hz']:7.1f} Hz  {peak['magnitude_db']:+6.1f} dB  {peak['kind']}")
        if not found["peaks"]:
            print("    no peaks stand above the floor")
    for name, why in failures.items():
        print(f"\n{name}: {_unprefixed(why, name)}")
    return EXIT_OK if payload["axes"] else EXIT_BLOCKED


def _unprefixed(message: str, axis: str) -> str:
    """A message with its own leading ``"roll: "`` removed.

    Most of these are written to be read on their own -- in a finding, in a
    traceback, in a GUI dialog -- so they name their axis. Printing them under a
    heading that names it too produces "roll: roll: ...".
    """
    prefix = f"{axis}: "
    return message[len(prefix) :] if message.startswith(prefix) else message


def _cmd_validate(args: argparse.Namespace) -> int:
    """Compare two flights, and say whether the prediction held (spec 5.10)."""
    from rotorid.core.analysis.compare import compare_logs
    from rotorid.core.export.comparison import write_comparison
    from rotorid.core.export.session import load_session
    from rotorid.core.guidance.validation import validation_findings

    config = load_config(args.config)
    before = _read(args.before)
    after = _read(args.after)
    session = None
    if args.session is not None:
        session, mismatch = load_session(
            args.session, tool_version=__version__, config_hash=config.hash
        )
        if mismatch:
            print(f"note: {mismatch.describe()}", file=sys.stderr)

    report = compare_logs(
        before,
        after,
        config,
        tool_version=__version__,
        session=session,
        axes=_axes_from(args),
    )
    findings = validation_findings(report)
    if args.report is not None:
        write_comparison(args.report, report)

    if args.json:
        print(
            json.dumps(
                {
                    "before": str(before.path),
                    "after": str(after.path),
                    "validated": report.has_predictions,
                    "axes": {
                        axis: {
                            "tracking_change": c.tracking_change,
                            "dterm_change": c.dterm_change,
                            "rise_ratio": c.rise_ratio,
                            "prediction_holds": c.prediction_holds,
                            "filter_prediction_error_db": c.filter_prediction_error_db,
                            "applied": c.applied,
                        }
                        for axis, c in report.axes.items()
                    },
                    "findings": [_jsonable(f) for f in findings],
                },
                indent=2,
                default=str,
            )
        )
    else:
        print(f"{before.path.name} -> {after.path.name}  [{after.stack}]")
        if not report.has_predictions:
            print(
                "  outcome comparison only: pass --session to check what was predicted "
                "against what happened"
            )
        for axis, c in report.axes.items():
            print(
                f"  {axis:<6} tracking {_pct(c.tracking_change):>9}   "
                f"D-term {_pct(c.dterm_change):>9}   prediction {_verdict_word(c)}"
            )
        for note in report.notes:
            print(f"  . {note}")
        _print_findings(findings)
        if args.report is not None:
            print(f"\nwrote {args.report}")

    # A missed prediction is not a broken run, but it is a result a script should
    # be able to branch on without parsing prose.
    missed = any(f.code in ("PREDICTION_MISSED", "FILTER_PREDICTION_MISSED") for f in findings)
    return EXIT_BLOCKED if missed else EXIT_OK


def _wrote(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    """Say what was written, in whichever form the caller asked for.

    Every command is meant to be scriptable over a directory of logs (spec 14),
    and a command whose only output is prose is one a batch workflow has to parse
    prose out of. The human form stays the default because it is what a person
    typing the command wants.
    """
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
        return
    for value in payload.get("files", []) or [payload.get("file") or payload.get("report")]:
        if value:
            print(f"wrote {value}")


def _pct(change: float | None) -> str:
    return "n/a" if change is None else f"{change * 100:+.0f}%"


def _verdict_word(comparison: Any) -> str:
    """One word for the prediction column, or why there is not one."""
    if comparison.predicted_step is None:
        return "not tested"
    if comparison.applied is False:
        return "gains not applied"
    holds = comparison.prediction_holds
    return "not measurable" if holds is None else ("confirmed" if holds else "MISSED")


def _cmd_report(args: argparse.Namespace) -> int:
    """Re-render the HTML report from a saved session, without the log."""
    from rotorid.core.export.report import write_report
    from rotorid.core.export.session import load_session

    config = load_config(args.config)
    session, mismatch = load_session(
        args.session, tool_version=__version__, config_hash=config.hash
    )
    if mismatch:
        print(f"note: {mismatch.describe()}", file=sys.stderr)
    if not session.recommendations:
        raise ValueError(f"{args.session.name} has no recommendation in it to report on")

    write_report(
        args.report,
        session.log,
        {str(a): r for a, r in session.recommendations.items()},
        config_hash=session.config_hash,
        tool_version=session.tool_version,
        findings=session.findings,
        plan=session.next_steps,
        measured_steps={str(a): m for a, m in session.measured_steps.items()},
    )
    _wrote(args, {"report": str(args.report), "log": session.log.path.name})
    return EXIT_OK


def _cmd_recommend(args: argparse.Namespace) -> int:
    """Write the staged .param files from a saved session.

    Split out from ``analyze`` so an export can be redone -- with a finding
    acknowledged, or into a different directory -- without re-reading the log.
    Re-analysing in order to re-export would also risk producing different
    numbers under a different tool version, which is the one thing a saved
    session exists to prevent.
    """
    from rotorid.core.export.params import write_param_files
    from rotorid.core.export.session import load_session

    config = load_config(args.config)
    session, mismatch = load_session(
        args.session, tool_version=__version__, config_hash=config.hash
    )
    if mismatch:
        print(f"note: {mismatch.describe()}", file=sys.stderr)
    plan = session.next_steps
    if plan is None or not plan.stages:
        raise ValueError(f"{args.session.name} carries no flight plan to export")

    acknowledgements = {
        **session.acknowledgements,
        **{
            code.strip(): "accepted on the command line"
            for code in args.acknowledge.split(",")
            if code.strip()
        },
    }
    if args.stage is not None:
        stages = tuple(s for s in plan.stages if s.index == args.stage)
        if not stages:
            available = ", ".join(str(s.index) for s in plan.stages)
            raise ValueError(f"no flight {args.stage} in this plan; it has {available}")
        plan = replace(plan, stages=stages)

    args.export.mkdir(parents=True, exist_ok=True)
    written = write_param_files(
        args.export,
        plan,
        log_name=session.log.path.name,
        tool_version=session.tool_version,
        config_hash=session.config_hash,
        findings=session.findings,
        acknowledgements=acknowledgements,
    )
    _wrote(
        args,
        {
            "files": [str(path) for path in written],
            "stages": [stage.index for stage in plan.stages],
            "acknowledged": sorted(acknowledgements),
        },
    )
    return EXIT_OK


def _cmd_profile(args: argparse.Namespace) -> int:
    """Write the parameter file to load *before* the flight (spec 13)."""
    from rotorid.core.export.profile import profile, write_profile

    write_profile(
        args.out,
        cast("Stack", args.stack),
        cast("Profile", args.which),
        tool_version=__version__,
        axis=cast("Axis", args.axis),
    )
    params, notes = profile(
        cast("Stack", args.stack), cast("Profile", args.which), axis=cast("Axis", args.axis)
    )
    _wrote(
        args,
        {
            "file": str(args.out),
            "stack": args.stack,
            "profile": args.which,
            "arms_excitation": args.which == "sweep",
            "params": params,
            "notes": list(notes),
        },
    )
    if args.which == "sweep" and not args.json:
        print(
            "this profile arms a deliberate excitation. Read the header before loading "
            "it, and turn it off again when the tuning campaign is over."
        )
    return EXIT_OK


def _cmd_selftest(args: argparse.Namespace) -> int:
    """Exercise every layer of this build and say which ones worked."""
    from rotorid.selftest import run_selftest

    result = run_selftest(args.log, gui=not args.no_gui)
    text = result.to_json() if args.json else result.describe()
    print(text)
    if args.out is not None:
        # Written as well as printed, because the packaged executable is windowed
        # and has no stdout for anyone to read.
        args.out.write_text(result.to_json(), encoding="utf-8")
    return EXIT_OK if result.ok else EXIT_BLOCKED


def _open_window_without_a_log() -> int | None:
    """Open the window with nothing loaded, or ``None`` if there is no window.

    ``None`` rather than an error: an install without the ``gui`` extra is a
    perfectly good headless install, and the right thing to show someone who
    typed ``rotorid`` there is the list of commands they do have.
    """
    try:
        from rotorid.gui.app import run
    except ImportError:  # pragma: no cover - depends on the install extras
        return None
    return run(None)


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
