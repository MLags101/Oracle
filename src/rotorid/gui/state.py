"""What the window knows (spec sections 9 and 10.1).

One object holds the whole session's state and every stage reads from it. The
alternative -- each stage owning its slice and handing it forward -- is what
makes wizards drift: the user steps back two stages, changes something, and the
stage they were on is still showing the previous answer with no indication that
it is stale.

So state changes are announced, never pushed. Stages connect to the signals here
and redraw themselves; nothing reaches into a stage to update it. That also
makes the staleness rule enforceable in one place: anything that invalidates the
analysis clears it, and every stage downstream goes empty at the same moment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThreadPool, Signal

from rotorid import __version__
from rotorid.config import Config, load_config
from rotorid.core.types import AXES, Axis, Finding, LogBundle
from rotorid.gui.workers import Job

__all__ = ["STAGES", "AppState", "Stage"]

#: The wizard's stages, in order. The rail draws them from this and the gating
#: rules below are expressed against it, so adding a stage is one edit.
Stage = str
STAGES: tuple[Stage, ...] = (
    "Load",
    "Health & Noise",
    "Segment",
    "Identify",
    "Filters",
    "Design",
    "Review & Export",
    "Next Flight",
)


class AppState(QObject):
    """The session, and the one thread pool that is allowed to work on it."""

    log_loading = Signal(str)
    log_loaded = Signal(object)  # LogBundle
    log_failed = Signal(str, str)

    analysis_started = Signal()
    analysis_progress = Signal(float, str)
    analysis_finished = Signal(object)  # AnalysisResult
    analysis_failed = Signal(str, str)
    analysis_cancelled = Signal()

    acknowledgements_changed = Signal()
    busy_changed = Signal(bool)

    def __init__(self, config: Config | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.config = config if config is not None else load_config()
        self.bundle: LogBundle | None = None
        self.result: Any = None  # AnalysisResult, imported lazily to keep Qt out of core
        self.conservatism = 0.5
        self.acknowledgements: dict[str, str] = {}

        self._pool = QThreadPool(self)
        # One at a time. Two analyses of the same log racing to finish would
        # deliver whichever happened to win, which is not a thing the user asked
        # for and not a thing they could tell had happened.
        self._pool.setMaxThreadCount(1)
        self._job: Job | None = None

    # ----------------------------------------------------------------- #
    # Loading
    # ----------------------------------------------------------------- #

    def load_log(self, path: Path) -> None:
        """Read a log, off the GUI thread.

        Clears any existing analysis first: the moment a new log is chosen, every
        number on every later stage is about a different aircraft.
        """
        from rotorid.core.io.ardupilot import read_ardupilot
        from rotorid.core.io.px4 import read_px4

        reader = read_px4 if path.suffix.lower() == ".ulg" else read_ardupilot
        self._clear()
        self.log_loading.emit(str(path))
        self._start(
            Job(reader, path, wants_progress=False),
            on_finished=self._log_ready,
            on_failed=self.log_failed.emit,
        )

    def _log_ready(self, bundle: object) -> None:
        assert isinstance(bundle, LogBundle)
        self.bundle = bundle
        self.log_loaded.emit(bundle)

    # ----------------------------------------------------------------- #
    # Analysis
    # ----------------------------------------------------------------- #

    def run_analysis(self, axes: tuple[Axis, ...] = AXES) -> None:
        """Identify, design, check and plan, off the GUI thread."""
        from rotorid.core.pipeline import analyze

        if self.bundle is None:
            raise RuntimeError("no log is loaded")

        self.result = None
        self.analysis_started.emit()
        self._start(
            Job(
                analyze,
                self.bundle,
                axes,
                self.config,
                tool_version=__version__,
                conservatism=self.conservatism,
                acknowledgements=dict(self.acknowledgements),
            ),
            on_finished=self._analysis_ready,
            on_failed=self.analysis_failed.emit,
            on_progress=self.analysis_progress.emit,
            on_cancelled=self.analysis_cancelled.emit,
        )

    def _analysis_ready(self, result: object) -> None:
        self.result = result
        self.analysis_finished.emit(result)

    def cancel(self) -> None:
        if self._job is not None:
            self._job.cancel()

    @property
    def busy(self) -> bool:
        return self._job is not None

    def wait(self, timeout_ms: int = 60_000) -> bool:
        """Block until the pool is idle. For tests and for shutdown, not for the UI."""
        return bool(self._pool.waitForDone(timeout_ms))

    # ----------------------------------------------------------------- #
    # Findings
    # ----------------------------------------------------------------- #

    @property
    def findings(self) -> tuple[Finding, ...]:
        if self.result is None:
            return ()
        return tuple(self.result.session.findings)

    @property
    def unresolved(self) -> tuple[str, ...]:
        """Blocking findings still outstanding. Non-empty means no export."""
        if self.result is None:
            return ()
        return tuple(self.result.unresolved(self.acknowledgements))

    def acknowledge(self, code: str, reason: str) -> None:
        """Accept a stated risk, on the record.

        The reason is required and is written into the exported files. An
        acknowledgement with no reason is indistinguishable from a click-through,
        and the person reading the parameter file next month is usually not the
        person who clicked.
        """
        if not reason.strip():
            raise ValueError("an acknowledgement has to say why; it is recorded in the export")
        self.acknowledgements[code] = reason.strip()
        self.acknowledgements_changed.emit()

    def withdraw(self, code: str) -> None:
        if self.acknowledgements.pop(code, None) is not None:
            self.acknowledgements_changed.emit()

    # ----------------------------------------------------------------- #
    # Stage gating
    # ----------------------------------------------------------------- #

    def stage_ready(self, stage: Stage) -> bool:
        """Whether a stage has anything to show yet.

        Backwards is always allowed -- a user who wants to re-read the
        identification behind a number should not have to start again -- so this
        only gates going forward into a stage with no content.
        """
        if stage == "Load":
            return True
        if self.bundle is None:
            return False
        if stage in ("Health & Noise", "Segment"):
            return True
        return self.result is not None and bool(self.result.session.recommendations)

    # ----------------------------------------------------------------- #
    # Plumbing
    # ----------------------------------------------------------------- #

    def _clear(self) -> None:
        self.bundle = None
        self.result = None
        self.acknowledgements.clear()

    def _start(
        self,
        job: Job,
        *,
        on_finished: Any,
        on_failed: Any,
        on_progress: Any = None,
        on_cancelled: Any = None,
    ) -> None:
        if self._job is not None:
            self._job.cancel()

        self._job = job
        job.signals.finished.connect(on_finished)
        job.signals.failed.connect(on_failed)
        if on_progress is not None:
            job.signals.progress.connect(on_progress)
        if on_cancelled is not None:
            job.signals.cancelled.connect(on_cancelled)
        for signal in (job.signals.finished, job.signals.failed, job.signals.cancelled):
            signal.connect(self._job_done)

        self.busy_changed.emit(True)
        self._pool.start(job)

    def _job_done(self, *_: object) -> None:
        self._job = None
        self.busy_changed.emit(False)
