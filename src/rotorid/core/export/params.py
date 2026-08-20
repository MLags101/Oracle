"""Parameter file export, staged and gated (spec sections 7 and 16).

Three rules, all of them safety rules rather than conveniences:

1. **Nothing is ever written to a vehicle.** This module writes files. A human
   loads them, deliberately, having read what is in them.
2. **Filters and gains are separate files**, because they are separate flights.
   Handing over one file containing both invites exactly the change-everything
   -at-once flight that makes a bad outcome impossible to attribute.
3. **Blocking findings stop the export** until the user acknowledges them by
   name, and the acknowledgement is written into the file header. Someone reading
   the file later can see what was accepted and by whom.

The format is the ArduPilot ``.param`` text format that Mission Planner and QGC
both load: one ``NAME,VALUE`` per line, ``#`` for comments.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rotorid.core.types import Finding, FlightTestPlan, FlightTestStage

__all__ = ["ExportBlockedError", "write_param_files", "write_stage_file"]

#: Decimal places. ArduPilot parameters are single-precision floats and gains of
#: 1e-5 are meaningful, so this has to be generous enough not to round a small D
#: gain away.
_PRECISION = 6


class ExportBlockedError(Exception):
    """Raised when an export is attempted over unacknowledged blocking findings.

    Carries the codes so a caller can present them, and so a GUI can offer the
    acknowledgement in the same place it shows the reason.
    """

    def __init__(self, codes: tuple[str, ...]) -> None:
        self.codes = codes
        super().__init__(
            "export blocked by unacknowledged findings: "
            + ", ".join(codes)
            + ". Acknowledge each by code to proceed; the acknowledgement is recorded "
            "in the exported file."
        )


def write_param_files(
    directory: Path,
    plan: FlightTestPlan,
    *,
    log_name: str,
    tool_version: str,
    config_hash: str,
    findings: tuple[Finding, ...] = (),
    acknowledgements: dict[str, str] | None = None,
    prefix: str = "rotorid",
) -> list[Path]:
    """Write one ``.param`` file per flight in the plan.

    Args:
        acknowledgements: Finding code to the reason the user gave for accepting
            it. Written into every file header.
        prefix: Filename prefix. Files are named
            ``{prefix}-1-filters-only.param`` and so on, so that the order they
            should be flown in survives being sorted alphabetically in a folder.

    Returns:
        The files written, in flight order.

    Raises:
        ExportBlockedError: if any blocking finding is unacknowledged. Nothing is
            written in that case -- a partial export is worse than none, because
            the files that did appear look complete.
        ValueError: if the plan has no stages.
    """
    acknowledged = acknowledgements or {}
    unresolved = tuple(
        f.code for f in findings if f.severity == "blocker" and f.code not in acknowledged
    )
    if unresolved:
        raise ExportBlockedError(unresolved)
    if not plan.stages:
        raise ValueError("the plan has no stages, so there is nothing to export")

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for stage in plan.stages:
        path = directory / f"{prefix}-{stage.index}-{_slug(stage.title)}.param"
        write_stage_file(
            path,
            stage,
            log_name=log_name,
            tool_version=tool_version,
            config_hash=config_hash,
            acknowledgements=acknowledged,
            total_stages=len(plan.stages),
        )
        written.append(path)
    return written


def write_stage_file(
    path: Path,
    stage: FlightTestStage,
    *,
    log_name: str,
    tool_version: str,
    config_hash: str,
    acknowledgements: dict[str, str] | None = None,
    total_stages: int | None = None,
) -> Path:
    """Write one stage as a loadable ``.param`` file.

    The header is long on purpose. A parameter file outlives the session that
    produced it, gets emailed around, and is loaded months later by someone who
    does not remember which log it came from -- so it carries the log, the tool
    version, the config hash, what to watch for, and what was acknowledged.
    """
    lines = _header_lines(
        stage,
        log_name=log_name,
        tool_version=tool_version,
        config_hash=config_hash,
        acknowledgements=acknowledgements or {},
        total_stages=total_stages,
    )
    for name, value in sorted(stage.changes.items()):
        lines.append(f"{name},{_format(value)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _header_lines(
    stage: FlightTestStage,
    *,
    log_name: str,
    tool_version: str,
    config_hash: str,
    acknowledgements: dict[str, str],
    total_stages: int | None,
) -> list[str]:
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    of_total = f" of {total_stages}" if total_stages else ""
    lines = [
        f"# RotorID {tool_version} -- flight {stage.index}{of_total}: {stage.title}",
        f"# Generated {when} from {log_name}, config {config_hash}",
        "#",
        "# BACK UP YOUR CURRENT PARAMETERS BEFORE LOADING THIS FILE.",
        "# This is one stage of a staged plan. Load this file, fly it, download the",
        "# log, and check it before moving to the next stage. These are designed",
        "# starting points with stated stability margins, identified at one",
        "# operating point -- not a validated tune.",
        "#",
    ]
    if stage.watch_in_flight:
        lines.append("# Watch for in flight:")
        lines += [f"#   - {item}" for item in stage.watch_in_flight]
        lines.append("#")
    if stage.check_in_log:
        lines.append("# Then check in the log:")
        lines += [f"#   - {item}" for item in stage.check_in_log]
        lines.append("#")
    if acknowledgements:
        lines.append("# Exported over acknowledged findings:")
        lines += [f"#   - {code}: {why}" for code, why in sorted(acknowledgements.items())]
        lines.append("#")
    return lines


def _format(value: float) -> str:
    """A parameter value, without exponent notation and without trailing noise."""
    text = f"{value:.{_PRECISION}f}".rstrip("0").rstrip(".")
    return text or "0"


def _slug(title: str) -> str:
    """A filename-safe version of a stage title."""
    return "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-").replace("--", "-")
