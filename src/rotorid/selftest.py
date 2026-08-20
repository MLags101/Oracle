"""Does this build actually work? (spec section 14, milestone M11.)

Building an executable and having one that runs are different events, and on a
one-file PyInstaller build they come apart in a specific, well-known way: a
dependency that reaches its data through the filesystem rather than through an
import gets left behind, and the failure only appears at run time on a machine
with no Python to fall back on. ``pymavlink`` is exactly that -- its message
definitions are XML files, not modules -- so a build with the imports right and
the data wrong reads every log fine on the developer's machine and cannot open a
single one anywhere else.

A windowed executable cannot tell you which you have. It has no console to print
to, so the only channels out of it are its exit code and any file it writes. This
carries the check itself for that reason: run it, look at the exit code, read the
file if you want the detail.

It is not only a build check. "Is my install working, and if not, which part"
is a reasonable thing for a user to ask, and the answer is more useful than a
traceback from whichever stage happened to fail first.
"""

from __future__ import annotations

import importlib
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Result", "Step", "run_selftest"]

#: Everything that has ever been left out of a frozen build of this tool, plus
#: the ones that would be. Checked by import rather than by inspecting the
#: bundle, because an import is what will actually happen in anger.
_MODULES: tuple[tuple[str, str], ...] = (
    ("numpy", "the arrays everything is built on"),
    ("scipy.signal", "filters, spectra and peak finding"),
    ("pymavlink.DFReader", "reading ArduPilot .bin logs"),
    ("pyulog", "reading PX4 .ulg logs"),
)

#: Checked only when the GUI is expected to be present.
_GUI_MODULES: tuple[tuple[str, str], ...] = (
    ("PySide6.QtWidgets", "the window itself"),
    ("pyqtgraph", "every plot in it"),
)


@dataclass(frozen=True, slots=True)
class Step:
    """One thing that was checked, and what happened."""

    name: str
    ok: bool
    detail: str = ""
    seconds: float = 0.0


@dataclass
class Result:
    """The whole check.

    ``ok`` is about the *machinery*, not about the aircraft. A log the analysis
    refuses is a working build reaching a correct conclusion, and is recorded as
    such rather than as a failure -- otherwise the only logs that could ever
    verify a build would be the ones with nothing wrong with them.
    """

    ok: bool = True
    frozen: bool = False
    version: str = ""
    steps: list[Step] = field(default_factory=list)

    def add(self, step: Step) -> None:
        self.steps.append(step)
        self.ok = self.ok and step.ok

    def describe(self) -> str:
        lines = [f"rotorid {self.version}" + (" (frozen build)" if self.frozen else "")]
        for step in self.steps:
            mark = "ok  " if step.ok else "FAIL"
            timing = f"  [{step.seconds * 1000:.0f} ms]" if step.seconds >= 0.001 else ""
            lines.append(f"  {mark} {step.name}{timing}")
            if step.detail:
                lines.append(f"         {step.detail}")
        lines.append("PASS" if self.ok else "FAIL")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def run_selftest(log: Path | None = None, *, gui: bool = True) -> Result:
    """Exercise every layer this build has, in the order they would fail.

    Args:
        log: A flight log to read. Optional, and the check is much weaker without
            one: reading a real log is the only step that exercises the message
            definitions, which is the thing most likely to be missing.
        gui: Whether to build the window. Turned off for a headless install,
            where the absence of Qt is not a fault.

    Returns:
        The result. Never raises: a self-test that crashes has told the user
        nothing except that something crashed.
    """
    from rotorid import __version__

    result = Result(version=__version__, frozen=_is_frozen())

    for name, why in _MODULES + (_GUI_MODULES if gui else ()):
        result.add(_timed(f"import {name}", lambda n=name: importlib.import_module(n), why))

    if log is None:
        result.add(
            Step(
                name="read a log",
                ok=True,
                detail=(
                    "skipped -- no log given. Nothing here checked the message "
                    "definitions, which is the part of a frozen build most likely "
                    "to be missing. Pass a .bin or .ulg to check it properly."
                ),
            )
        )
        return result

    bundle = _read_log(result, log)
    if bundle is None:
        return result

    analysis = _analyse(result, bundle)
    if gui and analysis is not None:
        _walk_the_wizard(result, analysis)
    return result


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


def _read_log(result: Result, log: Path) -> Any:
    """Read the log. The step that exercises the message definitions."""
    from rotorid.cli import _read

    started = time.perf_counter()
    try:
        bundle = _read(log)
    except Exception as exc:
        result.add(
            Step(
                name="read a log",
                ok=False,
                detail=f"{log.name}: {type(exc).__name__}: {exc}",
                seconds=time.perf_counter() - started,
            )
        )
        return None

    result.add(
        Step(
            name="read a log",
            ok=True,
            detail=f"{log.name}: {len(bundle.signals)} signals, {len(bundle.params)} parameters",
            seconds=time.perf_counter() - started,
        )
    )
    return bundle


def _analyse(result: Result, bundle: Any) -> Any:
    """Run the pipeline. Exercises scipy, the optimizer and the whole core."""
    from rotorid import __version__
    from rotorid.config import load_config
    from rotorid.core.pipeline import analyze

    started = time.perf_counter()
    try:
        analysis = analyze(bundle, ("roll",), load_config(), tool_version=__version__)
    except Exception as exc:
        result.add(
            Step(
                name="run the analysis",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
                seconds=time.perf_counter() - started,
            )
        )
        return None

    session = analysis.session
    # A refusal is the machinery working. Recorded, not failed.
    if session.recommendations:
        outcome = (
            f"{len(session.recommendations)} recommendation(s), {len(session.findings)} finding(s)"
        )
    else:
        why = "; ".join(f.code for f in session.blockers) or "no usable data"
        outcome = f"no recommendation -- {why}"

    result.add(
        Step(
            name="run the analysis",
            ok=True,
            detail=outcome,
            seconds=time.perf_counter() - started,
        )
    )
    return analysis


def _walk_the_wizard(result: Result, analysis: Any) -> None:
    """Build the window and draw every stage.

    Offscreen, so this works over SSH, in CI, and inside a build script. Every
    stage is refreshed rather than only constructed: a plot widget that cannot
    find its backend builds perfectly happily and fails when asked to draw.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    started = time.perf_counter()
    try:
        from PySide6.QtWidgets import QApplication

        from rotorid.gui.main_window import MainWindow
        from rotorid.gui.state import STAGES, AppState
        from rotorid.gui.wizard.base import StageWidget

        app = QApplication.instance() or QApplication([])
        # Set directly rather than run through the worker threads. What is being
        # checked is that every stage can draw the result, and going round the
        # thread pool for that would only add a way for the check itself to hang.
        state = AppState()
        state.bundle = analysis.session.log
        state.result = analysis
        window = MainWindow(state)
        window.show()
        for row in range(len(STAGES)):
            window.rail.setCurrentRow(row)
            stage = window.work.currentWidget()
            if isinstance(stage, StageWidget):
                stage.refresh()
        app.processEvents()
    except Exception as exc:
        result.add(
            Step(
                name="draw every stage",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
                seconds=time.perf_counter() - started,
            )
        )
        return

    result.add(
        Step(
            name="draw every stage",
            ok=True,
            detail=f"{len(STAGES)} stages, {STAGES[0]} through {STAGES[-1]}",
            seconds=time.perf_counter() - started,
        )
    )


# --------------------------------------------------------------------------- #
# Small pieces
# --------------------------------------------------------------------------- #


def _timed(name: str, call: Any, why: str) -> Step:
    started = time.perf_counter()
    try:
        call()
    except Exception as exc:
        return Step(
            name=name,
            ok=False,
            detail=f"{why} -- {type(exc).__name__}: {exc}",
            seconds=time.perf_counter() - started,
        )
    return Step(name=name, ok=True, detail=why, seconds=time.perf_counter() - started)


def _is_frozen() -> bool:
    """Whether this is running out of a PyInstaller bundle rather than a checkout."""
    import sys

    return bool(getattr(sys, "frozen", False))
