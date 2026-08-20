"""Stage 1: Load (spec section 10.2).

The job of this screen is to answer "can this log support the analysis?" before
the user invests any time in it. Which signals are present is not a technicality:
a log without a rate target cannot be identified at all, and a log without ESC
telemetry can be analysed but not given a tracking notch. Saying so here, next
to the file that was just opened, is worth more than saying it in a finding
after a two-minute run.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from rotorid.core.types import LogBundle
from rotorid.gui.state import AppState
from rotorid.gui.theme import severity_colour
from rotorid.gui.wizard.base import StageWidget

__all__ = ["LoadStage"]

#: What the analysis needs, and what it can only do if it is there. Ordered by
#: how much is lost without it.
_WANTED: tuple[tuple[str, str, bool], ...] = (
    ("rate.{axis}.measured", "the gyro the controller actually used", True),
    ("rate.{axis}.output", "the mixer command, which is the identification input", True),
    ("rate.{axis}.target", "the commanded rate, for closed-loop views", False),
    ("motor.0.output", "throttle, for the operating point and notch reference", False),
    ("esc.0.rpm", "ESC telemetry, without which a notch cannot track the motors", False),
    ("cpu.load", "scheduler load, which gates the expensive filter options", False),
)


class LoadStage(StageWidget):
    title = "Load"

    def __init__(self, state: AppState, parent: object = None) -> None:
        super().__init__(state)
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        heading = QLabel("Load a flight log")
        heading.setObjectName("Heading")
        layout.addWidget(heading)

        hint = QLabel(
            "Drop an ArduPilot <code>.bin</code> or PX4 <code>.ulg</code> anywhere on this "
            "window, or choose one below. Nothing is read from the vehicle and nothing is "
            "written to it -- RotorID only ever reads a file you hand it."
        )
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        row = QHBoxLayout()
        self._choose = QPushButton("Choose log...")
        self._choose.setDefault(True)
        self._choose.setAutoDefault(True)
        self._choose.clicked.connect(self.open_log_dialog)
        row.addWidget(self._choose)
        self._status = QLabel("No log loaded.")
        self._status.setObjectName("Muted")
        row.addWidget(self._status, 1)
        layout.addLayout(row)

        self._summary_card = QFrame()
        self._summary_card.setObjectName("Card")
        self._summary = QFormLayout(self._summary_card)
        layout.addWidget(self._summary_card)

        self._signals = QTreeWidget()
        self._signals.setColumnCount(3)
        self._signals.setHeaderLabels(("Signal", "Present", "What it is for"))
        layout.addWidget(self._signals, 1)

        # An empty screen with a button on it reads as a screen that is broken.
        # Saying what the next seven stages will do with the file turns the wait
        # into an explanation.
        self._empty = QLabel(
            "Once a log is open this page lists every signal the analysis wants and says "
            "which of them the log actually carries, before any time is spent on it. "
            "The stages down the left unlock in order: what the gyro is hearing, which "
            "stretches of the flight are worth identifying from, the airframe recovered "
            "from them, then the filters and gains designed together against it."
        )
        self._empty.setObjectName("Muted")
        self._empty.setWordWrap(True)
        layout.addWidget(self._empty)

        state.log_loaded.connect(self._on_loaded)
        state.log_failed.connect(self._on_failed)
        state.log_loading.connect(lambda path: self._status.setText(f"Reading {path}..."))
        self.refresh()

    # ----------------------------------------------------------------- #

    def open_log_dialog(self) -> None:
        """Ask for a log. Public because the File menu opens the same dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open flight log", "", "Flight logs (*.bin *.BIN *.ulg);;All files (*)"
        )
        if path:
            self.state.load_log(Path(path))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            self.state.load_log(Path(urls[0].toLocalFile()))
            event.acceptProposedAction()

    # ----------------------------------------------------------------- #

    def _on_failed(self, message: str, _traceback: str) -> None:
        self._status.setText(message)
        self._status.setStyleSheet(f"color: {severity_colour('blocker')};")

    def _on_loaded(self, bundle: object) -> None:
        assert isinstance(bundle, LogBundle)
        self._status.setStyleSheet("")
        self._status.setText(f"{bundle.path.name} loaded.")
        self.refresh()

    def refresh(self) -> None:
        bundle = self.state.bundle
        while self._summary.rowCount():
            self._summary.removeRow(0)
        self._signals.clear()
        self._empty.setVisible(bundle is None)
        self._signals.setVisible(bundle is not None)
        self._summary_card.setVisible(bundle is not None)
        if bundle is None:
            return

        for label, value in (
            ("File", bundle.path.name),
            ("Stack", bundle.stack),
            ("Firmware", bundle.firmware_version or "not recorded"),
            ("Board", bundle.board_id or "not recorded"),
            ("Loop rate", f"{bundle.loop_rate_hz:.0f} Hz"),
            ("Gyro rate", f"{bundle.gyro_sample_rate_hz:.0f} Hz"),
            ("Analysis grid", f"{bundle.sample_rate_hz:.0f} Hz"),
            ("Parameters", f"{len(bundle.params)}"),
            ("Duration", self._duration(bundle)),
        ):
            self._summary.addRow(label, QLabel(value))

        for warning in bundle.warnings:
            row = QLabel(warning)
            row.setWordWrap(True)
            row.setStyleSheet(f"color: {severity_colour('warning')};")
            self._summary.addRow("Note", row)

        self._fill_signals(bundle)

    def _fill_signals(self, bundle: LogBundle) -> None:
        for pattern, why, required in _WANTED:
            names = (
                [pattern.format(axis=a) for a in ("roll", "pitch", "yaw")]
                if "{axis}" in pattern
                else [pattern]
            )
            present = [n for n in names if n in bundle.signals]
            item = QTreeWidgetItem(
                (
                    pattern,
                    self._presence(present, names),
                    why if present else f"missing -- {why}",
                )
            )
            if not present:
                item.setForeground(
                    1,
                    Qt.GlobalColor.red
                    if required
                    else Qt.GlobalColor.darkYellow,  # word in column 1 carries it too
                )
            self._signals.addTopLevelItem(item)
        self._signals.resizeColumnToContents(0)

    @staticmethod
    def _presence(present: list[str], names: list[str]) -> str:
        if len(present) == len(names):
            return "yes"
        if not present:
            return "no"
        return f"{len(present)} of {len(names)}"

    @staticmethod
    def _duration(bundle: LogBundle) -> str:
        spans = [sig.t[-1] - sig.t[0] for sig in bundle.signals.values() if sig.t.size > 1]
        return f"{max(spans):.0f} s" if spans else "unknown"
