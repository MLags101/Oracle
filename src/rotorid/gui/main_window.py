"""The shell (spec section 10.1).

A left rail of stages, a central work area, findings on the right and progress
at the bottom. The rail is the tool's argument about how tuning works: you look
at the noise before you identify, you identify before you design, and you design
filters and gains together. Making that order visible and steppable is most of
the teaching this application does.

Going backwards is always allowed. Going forwards is gated on there being
something to see -- not on approval. A wizard that refuses to let you look is a
wizard people learn to fight.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QWidget,
)

from rotorid import __version__
from rotorid.gui.state import STAGES, AppState
from rotorid.gui.theme import Palette, palette
from rotorid.gui.widgets.findings_panel import FindingsPanel
from rotorid.gui.wizard.base import StageWidget
from rotorid.gui.wizard.design import DesignStage
from rotorid.gui.wizard.filters import FiltersStage
from rotorid.gui.wizard.health import HealthStage
from rotorid.gui.wizard.identify import IdentifyStage
from rotorid.gui.wizard.load import LoadStage
from rotorid.gui.wizard.nextflight import NextFlightStage
from rotorid.gui.wizard.review import ReviewStage
from rotorid.gui.wizard.segment import SegmentStage

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None, theme: Palette | None = None) -> None:
        super().__init__()
        self.state = state if state is not None else AppState()
        self.palette_ = theme if theme is not None else palette("light")
        self.setWindowTitle(f"RotorID {__version__}")
        self.setStyleSheet(self.palette_.stylesheet())
        self.resize(1280, 860)

        self._stages: dict[str, StageWidget] = {}
        self._build_rail()
        self._build_stages()
        self._build_findings_dock()
        self._build_progress_dock()
        self._build_actions()

        self.state.log_loaded.connect(self._state_changed)
        self.state.log_failed.connect(self._failed)
        self.state.analysis_finished.connect(self._analysis_done)
        self.state.analysis_failed.connect(self._failed)
        self.state.analysis_cancelled.connect(lambda: self._say("Cancelled."))
        self.state.analysis_progress.connect(self._progress)
        self.state.analysis_started.connect(lambda: self._say("Analysing..."))
        self.state.busy_changed.connect(self._busy)
        self.state.acknowledgements_changed.connect(self._refresh_findings)

        self._select(0)

    # ----------------------------------------------------------------- #
    # Construction
    # ----------------------------------------------------------------- #

    def _build_rail(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.rail = QListWidget()
        self.rail.setFixedWidth(210)
        self.rail.currentRowChanged.connect(self._select)
        layout.addWidget(self.rail)

        self.work = QStackedWidget()
        layout.addWidget(self.work, 1)
        self.setCentralWidget(central)

    def _build_stages(self) -> None:
        self.load_stage = LoadStage(self.state)
        self.design_stage = DesignStage(self.state)
        self.filters_stage = FiltersStage(self.state)
        self.review_stage = ReviewStage(self.state)
        self.identify_stage = IdentifyStage(self.state)
        self.health_stage = HealthStage(self.state)
        self.segment_stage = SegmentStage(self.state)
        self.nextflight_stage = NextFlightStage(self.state)
        built: dict[str, StageWidget] = {
            "Load": self.load_stage,
            "Health & Noise": self.health_stage,
            "Segment": self.segment_stage,
            "Identify": self.identify_stage,
            "Filters": self.filters_stage,
            "Design": self.design_stage,
            "Review & Export": self.review_stage,
            "Next Flight": self.nextflight_stage,
        }
        for name in STAGES:
            stage = built[name]
            self._stages[name] = stage
            self.work.addWidget(stage)
            self.rail.addItem(QListWidgetItem(name))
        self._update_rail()

    def _build_findings_dock(self) -> None:
        self.findings = FindingsPanel()
        self.findings.acknowledge_requested.connect(self.state.acknowledge)
        self.findings.withdraw_requested.connect(self.state.withdraw)
        dock = QDockWidget("Findings", self)
        dock.setWidget(self.findings)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.findings_dock = dock

    def _build_progress_dock(self) -> None:
        body = QWidget()
        row = QHBoxLayout(body)
        self.status = QLabel("Ready.")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setVisible(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.state.cancel)
        row.addWidget(self.status, 1)
        row.addWidget(self.bar)
        row.addWidget(self.cancel_button)

        dock = QDockWidget("Progress", self)
        dock.setWidget(body)
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _build_actions(self) -> None:
        run = QAction("&Analyse", self)
        run.setShortcut(QKeySequence("Ctrl+R"))
        run.triggered.connect(self._run)
        self.run_action = run

        file_menu = self.menuBar().addMenu("&File")
        open_action = QAction("&Open log...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.load_stage.open_log_dialog)
        file_menu.addAction(open_action)
        file_menu.addAction(run)

    # ----------------------------------------------------------------- #
    # Reactions
    # ----------------------------------------------------------------- #

    def _run(self) -> None:
        if self.state.bundle is None:
            self._say("Load a log first.")
            return
        self.state.run_analysis()

    def _select(self, row: int) -> None:
        if 0 <= row < self.work.count():
            self.work.setCurrentIndex(row)
            stage = self._stages[STAGES[row]]
            stage.refresh()

    def _state_changed(self, *_: object) -> None:
        self._update_rail()
        self._stages[STAGES[self.work.currentIndex()]].refresh()

    def _analysis_done(self, *_: object) -> None:
        self._say("Analysis complete.")
        self._refresh_findings()
        self._state_changed()

    def _refresh_findings(self) -> None:
        self.findings.show_findings(self.state.findings, dict(self.state.acknowledgements))
        blocked = self.state.unresolved
        title = "Findings" if not blocked else f"Findings -- {len(blocked)} blocking"
        self.findings_dock.setWindowTitle(title)

    def _update_rail(self) -> None:
        for row, name in enumerate(STAGES):
            item = self.rail.item(row)
            ready = self.state.stage_ready(name)
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsEnabled
                if ready
                else item.flags() & ~Qt.ItemFlag.ItemIsEnabled
            )
            item.setText(name if ready else f"{name}  (not yet)")

    def _progress(self, fraction: float, message: str) -> None:
        self.bar.setValue(round(100 * fraction))
        self._say(message)

    def _busy(self, busy: bool) -> None:
        self.bar.setVisible(busy)
        self.cancel_button.setEnabled(busy)
        self.run_action.setEnabled(not busy)

    def _failed(self, message: str, detail: str = "") -> None:
        self._say(message)
        box = QMessageBox(self)
        box.setWindowTitle("That did not work")
        box.setText(message)
        if detail:
            box.setDetailedText(detail)
        box.exec()

    def _say(self, message: str) -> None:
        self.status.setText(message)

    def closeEvent(self, event: object) -> None:
        self.state.cancel()
        self.state.wait(2000)
        super().closeEvent(event)  # type: ignore[arg-type]
