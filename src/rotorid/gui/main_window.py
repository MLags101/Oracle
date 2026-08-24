"""The shell (spec section 10.1).

A toolbar of verbs, a left rail of stages, a central work area, findings on the
right and progress along the bottom. The rail is the tool's argument about how
tuning works: you look at the noise before you identify, you identify before you
design, and you design filters and gains together. Making that order visible and
steppable is most of the teaching this application does.

Going backwards is always allowed. Going forwards is gated on there being
something to see -- not on approval. A wizard that refuses to let you look is a
wizard people learn to fight.

Three decisions here are worth stating, because each replaced something that
tested badly:

* **Opening a log runs the analysis.** The verb the user came for is not
  "analyse", it is "tell me what is wrong with this flight", and the tool can
  answer that without being asked twice. The button is still there, prominently,
  for a re-run after the conservatism slider moves.
* **The verbs are on a toolbar, not in the File menu.** ``Analyse`` was the
  single most important action in the program and it was three clicks deep, in a
  menu named after files, under a heading that gives no hint that the answer to
  the user's question is behind it.
* **Every stage scrolls.** A stage lays out for the content it has, and some of
  that content -- a signal inventory, a verdict written in whole sentences, a
  table of peaks -- is longer than a laptop screen. Content that runs off the
  bottom with no scrollbar is content the user is entitled to think does not
  exist.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from rotorid import __version__
from rotorid.gui.state import STAGE_BLURBS, STAGES, AppState
from rotorid.gui.theme import Mode, Palette, palette
from rotorid.gui.widgets.findings_panel import FindingsPanel
from rotorid.gui.widgets.pipeline_rail import PipelineRail, Step, StepState
from rotorid.gui.widgets.stage_page import StagePage
from rotorid.gui.wizard.base import StageWidget
from rotorid.gui.wizard.design import DesignStage
from rotorid.gui.wizard.filters import FiltersStage
from rotorid.gui.wizard.health import HealthStage
from rotorid.gui.wizard.identify import IdentifyStage
from rotorid.gui.wizard.load import LoadStage
from rotorid.gui.wizard.nextflight import NextFlightStage
from rotorid.gui.wizard.review import ReviewStage
from rotorid.gui.wizard.segment import SegmentStage
from rotorid.gui.wizard.validate import ValidateStage

__all__ = ["MainWindow"]


def _literal(text: str) -> str:
    """Escape a stage name for use as button text.

    ``Health & Noise`` on a button is ``Health _N_oise`` with a mnemonic, because
    Qt reads a single ampersand as "the next letter is the accelerator". The stage
    names are data, not markup, so they are escaped wherever they become a label.
    """
    return text.replace("&", "&&")


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None, theme: Palette | None = None) -> None:
        super().__init__()
        self.state = state if state is not None else AppState()
        self.palette_ = theme if theme is not None else palette("light")
        self.setWindowTitle(f"RotorID {__version__}")
        self.setStyleSheet(self.palette_.stylesheet())
        self.resize(1440, 920)
        # 900 rather than nothing: below about this the rail, the work area and
        # the findings dock stop being three columns and start being three
        # slivers, and the user's fix for that is to resize, which they can only
        # do if the window admits it is too small.
        self.setMinimumSize(1024, 700)
        # On the window rather than only on the Load page, so a log dropped while
        # the user happens to be looking at some other stage still opens.
        self.setAcceptDrops(True)

        #: Which stages the user has actually looked at. The rail ticks them off,
        #: which is what turns it from a map into a record of the session.
        self._visited: set[str] = set()
        self._stages: dict[str, StageWidget] = {}
        self._scrollers: list[QScrollArea] = []

        self._build_shell()
        self._build_stages()
        self._build_findings_dock()
        self._build_actions()
        self._build_toolbar()
        self._build_statusbar()

        self.state.log_loaded.connect(self._log_loaded)
        self.state.log_failed.connect(self._failed)
        self.state.analysis_finished.connect(self._analysis_done)
        self.state.analysis_failed.connect(self._failed)
        self.state.analysis_cancelled.connect(lambda: self._say("Cancelled."))
        self.state.analysis_progress.connect(self._progress)
        self.state.analysis_started.connect(self._analysis_started)
        self.state.busy_changed.connect(self._busy)
        self.state.acknowledgements_changed.connect(self._refresh_findings)
        self.state.comparison_failed.connect(self._failed)

        self.rail.setCurrentRow(0)
        self._select(0)
        self._update_rail()

    # ----------------------------------------------------------------- #
    # Construction
    # ----------------------------------------------------------------- #

    def _build_shell(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.rail = PipelineRail(self.palette_)
        self.rail.currentRowChanged.connect(self._select)
        layout.addWidget(self.rail)

        right = QWidget()
        column = QVBoxLayout(right)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        self.work = QStackedWidget()
        column.addWidget(self.work, 1)
        column.addWidget(self._build_footer())

        layout.addWidget(right, 1)
        self.setCentralWidget(central)

    def _build_footer(self) -> QWidget:
        """Back and Next, named.

        A rail alone leaves the sequence implicit -- the user can see nine steps
        but not which one they are supposed to do after this one. A Next button
        that says where it goes, and says why it cannot when it cannot, is the
        cheapest way to make a nine-step process feel like a path rather than a
        set of nine choices.
        """
        bar = QFrame()
        bar.setObjectName("Footer")
        bar.setStyleSheet(
            f"QFrame#Footer {{ background: {self.palette_.surface_alt};"
            f" border-top: 1px solid {self.palette_.grid}; }}"
        )
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 8, 14, 8)

        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(lambda: self._step_by(-1))
        row.addWidget(self.back_button)

        self.stage_hint = QLabel()
        self.stage_hint.setObjectName("Muted")
        row.addWidget(self.stage_hint, 1)

        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("Primary")
        self.next_button.clicked.connect(lambda: self._step_by(1))
        row.addWidget(self.next_button)
        return bar

    def _build_stages(self) -> None:
        self.load_stage = LoadStage(self.state, self.palette_)
        self.design_stage = DesignStage(self.state, self.palette_)
        self.filters_stage = FiltersStage(self.state, self.palette_)
        self.review_stage = ReviewStage(self.state, self.palette_)
        self.identify_stage = IdentifyStage(self.state, self.palette_)
        self.health_stage = HealthStage(self.state, self.palette_)
        self.segment_stage = SegmentStage(self.state, self.palette_)
        self.nextflight_stage = NextFlightStage(self.state, self.palette_)
        self.validate_stage = ValidateStage(self.state, self.palette_)
        built: dict[str, StageWidget] = {
            "Load": self.load_stage,
            "Health & Noise": self.health_stage,
            "Segment": self.segment_stage,
            "Identify": self.identify_stage,
            "Filters": self.filters_stage,
            "Design": self.design_stage,
            "Review & Export": self.review_stage,
            "Next Flight": self.nextflight_stage,
            "Validate": self.validate_stage,
        }
        for name in STAGES:
            stage = built[name]
            self._stages[name] = stage
            self.work.addWidget(self._scrolled(stage))

    def _scrolled(self, stage: StageWidget) -> QScrollArea:
        """Wrap a stage so it can be taller than the window.

        Done here rather than in each stage so that no stage can forget, and so
        that a stage author never has to think about it: they lay out for the
        content, and the shell handles the case where the content does not fit.
        See :class:`StagePage` for why a plain scroll area is not enough.
        """
        area = StagePage(stage)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scrollers.append(area)
        return area

    def _build_findings_dock(self) -> None:
        self.findings = FindingsPanel(self.palette_)
        self.findings.acknowledge_requested.connect(self.state.acknowledge)
        self.findings.withdraw_requested.connect(self.state.withdraw)
        dock = QDockWidget("Findings", self)
        dock.setObjectName("FindingsDock")
        dock.setWidget(self.findings)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setMinimumWidth(320)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.findings_dock = dock
        # Qt gives a fresh dock whatever its contents ask for, and findings cards
        # are wrapped paragraphs that will happily take half the window. The work
        # area is what the user came to look at, so the dock is told its size.
        self.resizeDocks([dock], [380], Qt.Orientation.Horizontal)
        self._refresh_findings()

    def _build_actions(self) -> None:
        run = QAction("&Analyse this log", self)
        run.setShortcut(QKeySequence("Ctrl+R"))
        run.setToolTip("Identify, design and check every axis (Ctrl+R)")
        run.triggered.connect(self._run)
        self.run_action = run

        open_action = QAction("&Open log...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_log_dialog)
        self.open_action = open_action

        compare_action = QAction("Open an &after-flight log to validate against...", self)
        compare_action.triggered.connect(self._open_after)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_action)
        file_menu.addAction(compare_action)
        file_menu.addSeparator()
        file_menu.addAction(run)

        view_menu = self.menuBar().addMenu("&View")
        self.dark_action = QAction("&Dark theme", self)
        self.dark_action.setCheckable(True)
        self.dark_action.setChecked(self.palette_.mode == "dark")
        self.dark_action.toggled.connect(lambda on: self.apply_theme("dark" if on else "light"))
        view_menu.addAction(self.dark_action)
        view_menu.addAction(self.findings_dock.toggleViewAction())

    def _build_toolbar(self) -> None:
        """The verbs, where a verb belongs.

        Real buttons rather than tool-button actions, because the primary action
        has to *look* primary and a ``QToolButton`` styled to fill with the accent
        stops looking like the rest of the toolbar on every platform in a slightly
        different way.
        """
        bar = QToolBar("Actions")
        bar.setMovable(False)
        bar.setFloatable(False)
        self.addToolBar(bar)

        open_button = QPushButton("Open log...")
        open_button.setToolTip("Open an ArduPilot .bin or PX4 .ulg (Ctrl+O)")
        open_button.clicked.connect(self.open_log_dialog)
        bar.addWidget(open_button)

        self.run_button = QPushButton("Analyse")
        self.run_button.setObjectName("Primary")
        self.run_button.setToolTip("Identify, design and check every axis (Ctrl+R)")
        self.run_button.clicked.connect(self._run)
        bar.addWidget(self.run_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.state.cancel)
        bar.addWidget(self.cancel_button)

        bar.addSeparator()

        self.auto_box = QCheckBox("Analyse on open")
        self.auto_box.setChecked(self.state.auto_analyse)
        self.auto_box.setToolTip(
            "Opening a log starts the analysis straight away. Turn this off to open "
            "a log, look at what it contains, and decide for yourself."
        )
        self.auto_box.toggled.connect(self._set_auto)
        bar.addWidget(self.auto_box)

        spacer = QWidget()
        spacer.setStyleSheet("background: transparent;")
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)

        self.file_label = QLabel("No log open")
        self.file_label.setToolTip("The log these nine steps are about.")
        self.file_label.setObjectName("Muted")
        bar.addWidget(self.file_label)
        self._sync_actions()

    def _build_statusbar(self) -> None:
        bar = QStatusBar()
        self.status = QLabel("Ready. Open a log to begin.")
        bar.addWidget(self.status, 1)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFixedWidth(220)
        self.bar.setVisible(False)
        bar.addPermanentWidget(self.bar)
        self.setStatusBar(bar)

    # ----------------------------------------------------------------- #
    # Theme
    # ----------------------------------------------------------------- #

    def apply_theme(self, mode: Mode) -> None:
        """Swap the palette, rebuilding the stages so the plots follow.

        Rebuilding rather than restyling is only affordable because stages are
        passive views over :class:`AppState` -- they hold no analysis of their own,
        so a new one drawn from the same state shows exactly what the old one
        showed. Restyling instead would leave every pyqtgraph canvas on the old
        background, which is the half-switched look that makes a theme toggle feel
        broken.
        """
        if mode == self.palette_.mode:
            return
        self.palette_ = palette(mode)
        self.setStyleSheet(self.palette_.stylesheet())

        row = max(0, self.work.currentIndex())
        while self.work.count():
            widget = self.work.widget(0)
            if widget is None:  # pragma: no cover - the count says otherwise
                break
            self.work.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        self._stages.clear()
        self._scrollers.clear()
        self._build_stages()

        self.rail.set_theme(self.palette_)
        self.findings.set_theme(self.palette_)
        self._refresh_findings()
        self._update_rail()
        self.rail.setCurrentRow(row)
        self._select(row)

    # ----------------------------------------------------------------- #
    # Reactions
    # ----------------------------------------------------------------- #

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """A dropped log opens as a new before-log, and takes the user to it.

        Except on the Validate stage, which has its own drop handler and its own
        meaning for a second file. Qt delivers to the child first, so this is
        only reached when the drop landed on the window rather than on that
        stage -- but the stage is where a user with two logs in mind will be
        standing, and getting this backwards would silently discard their
        analysis.
        """
        urls = event.mimeData().urls()
        if not urls:
            return
        self._select(0)
        self.state.load_log(Path(urls[0].toLocalFile()))
        event.acceptProposedAction()

    def _open_after(self) -> None:
        """Jump to the Validate stage and ask for the second log there.

        Rather than loading it from wherever the user happens to be: the file
        only means anything next to the screen that explains what is being
        compared, and arriving at a filled-in table with no context is how a
        comparison gets read as a verdict.
        """
        self._select(STAGES.index("Validate"))
        self.validate_stage.open_after_dialog()

    def open_log_dialog(self) -> None:
        """Ask for a log, through whichever Load stage is current.

        Indirected through the window because a theme change rebuilds the stages,
        and a toolbar button still wired to the previous stage object is a
        crash waiting for the first user who switches to dark mode.
        """
        self.load_stage.open_log_dialog()

    def _run(self) -> None:
        if self.state.bundle is None:
            self._say("Open a log first -- there is nothing to analyse yet.")
            self.open_log_dialog()
            return
        self.state.run_analysis()

    def _set_auto(self, on: bool) -> None:
        self.state.auto_analyse = on
        self._say(
            "Opening a log will analyse it straight away."
            if on
            else "Logs will open without being analysed. Press Analyse when you want it."
        )

    def _step_by(self, delta: int) -> None:
        row = self.work.currentIndex() + delta
        if 0 <= row < len(STAGES):
            self.rail.setCurrentRow(row)

    def _select(self, row: int) -> None:
        if not (0 <= row < self.work.count()):
            return
        self.work.setCurrentIndex(row)
        name = STAGES[row]
        self._stages[name].refresh()
        self._scrollers[row].verticalScrollBar().setValue(0)
        self._update_rail()
        self._update_footer(row)

    def _update_footer(self, row: int) -> None:
        self.back_button.setEnabled(row > 0)
        self.back_button.setText(f"Back to {_literal(STAGES[row - 1])}" if row > 0 else "Back")
        self.stage_hint.setText(f"Step {row + 1} of {len(STAGES)} -- {STAGE_BLURBS[STAGES[row]]}")

        if row + 1 >= len(STAGES):
            self.next_button.setText("Done")
            self.next_button.setEnabled(False)
            self.next_button.setToolTip("This is the last step.")
            return

        nxt = STAGES[row + 1]
        reason = self.state.stage_block_reason(nxt)
        self.next_button.setEnabled(not reason)
        self.next_button.setText(f"Next: {_literal(nxt)}")
        self.next_button.setToolTip(reason or STAGE_BLURBS[nxt])

    def _log_loaded(self, *_: object) -> None:
        bundle = self.state.bundle
        if bundle is not None:
            self.file_label.setText(bundle.path.name)
            self._say(
                f"{bundle.path.name} opened."
                + (" Analysing..." if self.state.auto_analyse else " Press Analyse when ready.")
            )
        self._state_changed()

    def _analysis_started(self) -> None:
        self._say("Analysing...")
        self._update_rail()

    def _state_changed(self, *_: object) -> None:
        self._update_rail()
        self._update_footer(max(0, self.work.currentIndex()))
        self._sync_actions()
        self.current_stage().refresh()

    def _analysis_done(self, *_: object) -> None:
        blocking = len(self.state.unresolved)
        findings = len(self.state.findings)
        self._say(
            f"Analysis complete -- {findings} finding(s)"
            + (f", {blocking} blocking export." if blocking else ", none blocking.")
        )
        self._refresh_findings()
        self._state_changed()

    def _refresh_findings(self) -> None:
        self.findings.show_findings(self.state.findings, dict(self.state.acknowledgements))
        blocked = self.state.unresolved
        count = len(self.state.findings)
        if blocked:
            title = f"Findings -- {len(blocked)} blocking"
        elif count:
            title = f"Findings -- {count}"
        else:
            title = "Findings"
        self.findings_dock.setWindowTitle(title)

    def _sync_actions(self) -> None:
        has_log = self.state.bundle is not None
        busy = self.state.busy
        self.run_action.setEnabled(has_log and not busy)
        if hasattr(self, "run_button"):
            self.run_button.setEnabled(has_log and not busy)
            self.run_button.setText("Re-analyse" if self.state.result is not None else "Analyse")
            self.cancel_button.setEnabled(busy)

    def current_stage(self) -> StageWidget:
        """The stage the user is looking at.

        Not ``work.currentWidget()``: that is the scroll area the stage is
        wrapped in. Anything wanting the stage itself asks for it here rather
        than knowing about the wrapper.
        """
        return self._stages[STAGES[max(0, self.work.currentIndex())]]

    def _has_content(self, name: str) -> bool:
        """Whether this step would show the user anything if they went to it.

        Not the same question as :meth:`AppState.stage_ready`, which asks whether
        the door is open. Load is open from the first frame and has nothing on it
        until a file arrives, and Validate is open with one log and has nothing to
        compare until there are two. Ticking either of those on arrival would put
        a tick against work nobody did.
        """
        if name == "Load":
            return self.state.bundle is not None
        if name == "Validate":
            return self.state.comparison is not None
        return self.state.result is not None

    def _update_rail(self) -> None:
        """Redraw the nine steps: where each one stands and why."""
        current = STAGES[max(0, self.work.currentIndex())]
        if self._has_content(current):
            self._visited.add(current)

        steps: list[Step] = []
        for name in STAGES:
            reason = self.state.stage_block_reason(name)
            state: StepState
            if reason:
                state, note = "waiting", reason
            elif self.state.busy and name != "Load":
                state, note = "running", "Working..."
            elif name in self._visited:
                state, note = "reviewed", STAGE_BLURBS[name]
            else:
                state, note = "ready", STAGE_BLURBS[name]
            steps.append(Step(name=name, blurb=STAGE_BLURBS[name], state=state, note=note))
        self.rail.set_steps(tuple(steps))

    def _progress(self, fraction: float, message: str) -> None:
        self.bar.setValue(round(100 * fraction))
        self._say(message)

    def _busy(self, busy: bool) -> None:
        self.bar.setVisible(busy)
        self._sync_actions()
        self._update_rail()

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
