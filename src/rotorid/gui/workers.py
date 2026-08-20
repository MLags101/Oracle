"""Analysis off the GUI thread (spec section 9).

The rule the rest of the GUI is built on: **no analysis ever runs on the Qt
event loop**. A twelve-second parse on the GUI thread is not a slow program, it
is a frozen one -- the window stops repainting, the platform marks it as not
responding, and a user who has been told the tool is thinking has no reason to
believe it.

The boundary is deliberately one-way. Core takes a plain ``Callable`` for
progress and a plain ``Callable[[], bool]`` for cancellation; this module adapts
those to Qt signals and a ``threading.Event``. Core never imports Qt, which is
what keeps every analysis capability reachable from the CLI and from a test with
no display attached.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

__all__ = ["Job", "JobSignals"]


class JobSignals(QObject):
    """Signals for one job.

    On a ``QRunnable`` rather than a ``QThread`` these have to live on a separate
    ``QObject``, which is also convenient: the signals outlive the runnable, so a
    late ``finished`` cannot be delivered to a deleted object.
    """

    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str, str)  # message, traceback
    cancelled = Signal()


class Job(QRunnable):
    """One core function, run on the thread pool.

    The function is called with ``progress`` and ``should_cancel`` keyword
    arguments when it accepts them, so a job wrapping something simple does not
    have to grow a signature it has no use for.

    Exceptions are caught and reported through :attr:`signals`. Letting one
    escape a ``QRunnable`` takes the whole application down, and a failed
    analysis is a normal outcome here -- a log without a sweep in it is a thing
    users will hand us daily.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        wants_progress: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._wants_progress = wants_progress
        self._cancel = threading.Event()
        self.signals = JobSignals()

    def cancel(self) -> None:
        """Ask the job to stop at its next checkpoint.

        Cooperative, not pre-emptive: a job is stopped between steps, never in
        the middle of one. Killing a thread mid-numpy leaves the interpreter in a
        state nobody can reason about.
        """
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @Slot()
    def run(self) -> None:
        from rotorid.core.pipeline import AnalysisCancelled

        kwargs = dict(self._kwargs)
        if self._wants_progress:
            kwargs["progress"] = self._emit_progress
            kwargs["should_cancel"] = self._cancel.is_set

        try:
            result = self._fn(*self._args, **kwargs)
        except AnalysisCancelled:
            self.signals.cancelled.emit()
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
        else:
            if self._cancel.is_set():
                # It finished anyway, but the user has moved on and is looking at
                # something else. Delivering the result now would replace what
                # they are reading with an answer to a question they withdrew.
                self.signals.cancelled.emit()
            else:
                self.signals.finished.emit(result)

    def _emit_progress(self, fraction: float, message: str) -> None:
        self.signals.progress.emit(float(fraction), str(message))
