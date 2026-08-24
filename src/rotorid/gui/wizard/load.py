"""Stage 1: Load (spec section 10.2).

The job of this screen is to answer "can this log support the analysis?" before
the user invests any time in it. Which signals are present is not a technicality:
a log without a rate target cannot be identified at all, and a log without ESC
telemetry can be analysed but not given a tracking notch. Saying so here, next
to the file that was just opened, is worth more than saying it in a finding
after a two-minute run.

The screen has two faces and they are laid out separately, because they are
answering different questions.

*Empty*, it is asking for a file, so it is almost nothing: a target to drop on, a
button, and a short account of what the nine steps will do with the log. An empty
page with a lone button reads as a page that failed to load.

*Loaded*, it is reporting, and reporting is where the old version came apart. It
stacked a summary, several paragraphs of verdict and a six-row inventory in one
column and ran off the bottom of the screen with nothing to say it had -- so the
signal inventory, which is the whole point of the screen, was below the fold on a
laptop and most users never saw it. The facts now sit in a two-column grid beside
the verdict, and everything is sized to its content inside a page that scrolls.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rotorid.core.logkind import capabilities, detect_kind, kind_evidence
from rotorid.core.types import LogBundle, LogKind
from rotorid.gui.state import AppState
from rotorid.gui.theme import Palette
from rotorid.gui.widgets.layouts import clear
from rotorid.gui.widgets.responsive import ResponsiveRow
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


#: The declaration, in the order it is offered. ``None`` is first and is the
#: default: detection is right whenever the excitation is actually recorded,
#: which is the only case where the choice changes anything. Offering it first
#: also means the user who does not yet know the difference is not made to guess
#: before they have seen the descriptions.
_KIND_CHOICES: tuple[tuple[LogKind | None, str, str], ...] = (
    (
        None,
        "Detect from the log",
        "Read the file and decide: a recorded sweep or autotune run makes it a "
        "tuning flight, anything else is a general flight.",
    ),
    ("general", "General flight", capabilities("general").summary),
    ("tuning", "Tuning flight", capabilities("tuning").summary),
)

#: The facts about the file, in the order somebody checking a log looks for
#: them: which file, what it is, what flew it, then the rates the analysis has to
#: work with.
_SUMMARY_FIELDS: tuple[str, ...] = (
    "File",
    "Kind",
    "Stack",
    "Firmware",
    "Board",
    "Duration",
    "Loop rate",
    "Gyro rate",
    "Analysis grid",
    "Parameters",
)

#: What the user is buying with the wait, in the order it happens. Shown on the
#: empty screen so that the pause after dropping a file is an explanation rather
#: than a blank.
_WHAT_HAPPENS: tuple[tuple[str, str], ...] = (
    ("Read", "The log is opened and checked for the signals the analysis needs."),
    (
        "Listen",
        "The gyro spectrum is measured before anything is identified from it -- a "
        "model fitted to a shaking frame is confident nonsense.",
    ),
    (
        "Identify",
        "The airframe is recovered from the stretches of flight that can support it.",
    ),
    (
        "Design",
        "Filters and gains are designed together against that airframe, with the "
        "phase each filter costs subtracted from the margin the gains get to spend.",
    ),
    ("Report", "Findings, a parameter file, and what to fly next to confirm it."),
)


class LoadStage(StageWidget):
    title = "Load"

    def __init__(
        self,
        state: AppState,
        theme: Palette | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(state, theme, parent)
        self.setAcceptDrops(True)

        layout = self.page()
        layout.addWidget(
            self.header(
                "Load",
                subtitle=(
                    "Open an ArduPilot .bin or PX4 .ulg. Nothing is read from the vehicle "
                    "and nothing is written to it -- RotorID only ever reads a file you "
                    "hand it."
                ),
            )
        )

        self._empty = self._build_empty()
        layout.addWidget(self._empty)

        self._loaded = self._build_loaded()
        layout.addWidget(self._loaded)

        layout.addWidget(self._build_kind_card(state))
        layout.addStretch(1)

        state.log_loaded.connect(self._on_loaded)
        state.log_failed.connect(self._on_failed)
        state.log_loading.connect(lambda path: self._status.setText(f"Reading {path}..."))
        self.refresh()

    # ----------------------------------------------------------------- #
    # The empty face
    # ----------------------------------------------------------------- #

    def _build_empty(self) -> QWidget:
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(14)

        zone = QFrame()
        zone.setObjectName("DropZone")
        zone.setMinimumHeight(180)
        inner = QVBoxLayout(zone)
        inner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.setSpacing(8)

        drop = QLabel("Drop a flight log here")
        drop.setObjectName("Heading")
        drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(drop)

        formats = QLabel("ArduPilot .bin    ·    PX4 .ulg")
        formats.setObjectName("Muted")
        formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(formats)

        self._choose = QPushButton("Choose log...")
        self._choose.setObjectName("Primary")
        self._choose.setDefault(True)
        self._choose.setAutoDefault(True)
        self._choose.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._choose.clicked.connect(self.open_log_dialog)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._choose)
        row.addStretch(1)
        inner.addLayout(row)
        column.addWidget(zone)

        column.addWidget(self._build_what_happens())
        return box

    def _build_what_happens(self) -> QWidget:
        card = self.card()
        grid = QGridLayout(card)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        heading = QLabel("What happens when you open one")
        heading.setObjectName("Subheading")
        grid.addWidget(heading, 0, 0, 1, 2)

        for row, (name, blurb) in enumerate(_WHAT_HAPPENS, start=1):
            step = QLabel(f"{row}.  {name}")
            step.setObjectName("Subheading")
            step.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(step, row, 0)

            note = QLabel(blurb)
            note.setObjectName("Muted")
            note.setWordWrap(True)
            grid.addWidget(note, row, 1)

        grid.setColumnStretch(1, 1)
        return card

    # ----------------------------------------------------------------- #
    # The loaded face
    # ----------------------------------------------------------------- #

    def _build_loaded(self) -> QWidget:
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(14)

        row = QHBoxLayout()
        self._status = QLabel("No log loaded.")
        self._status.setObjectName("Subheading")
        self._status.setWordWrap(True)
        row.addWidget(self._status, 1)
        self._reopen = QPushButton("Open a different log...")
        self._reopen.clicked.connect(self.open_log_dialog)
        row.addWidget(self._reopen)
        column.addLayout(row)

        # Side by side on a wide window, stacked on a narrow one. Two cards in
        # a plain row would give this page a floor of about 900 pixels and a
        # horizontal scrollbar below it.
        facts = ResponsiveRow(threshold=760)
        self._summary_card = self.card()
        self._summary = QGridLayout(self._summary_card)
        self._summary.setContentsMargins(16, 14, 16, 14)
        self._summary.setHorizontalSpacing(16)
        self._summary.setVerticalSpacing(7)
        facts.add(self._summary_card)

        verdict_card = self.card()
        verdict_box = QVBoxLayout(verdict_card)
        verdict_box.setContentsMargins(16, 14, 16, 14)
        verdict_box.setSpacing(8)
        verdict_heading = QLabel("What this log can prove")
        verdict_heading.setObjectName("Subheading")
        verdict_box.addWidget(verdict_heading)
        self._verdict = QLabel()
        self._verdict.setWordWrap(True)
        verdict_box.addWidget(self._verdict)
        verdict_box.addStretch(1)
        facts.add(verdict_card)
        column.addWidget(facts)

        signals_card = self.card()
        signals_box = QVBoxLayout(signals_card)
        signals_box.setContentsMargins(16, 14, 16, 14)
        signals_box.setSpacing(8)
        signals_heading = QLabel("Signals the analysis wants")
        signals_heading.setObjectName("Subheading")
        signals_box.addWidget(signals_heading)
        signals_note = QLabel(
            "Two of these are required. Without them there is nothing to identify from, "
            "and the tool will say so rather than fitting a model to whatever is there."
        )
        signals_note.setObjectName("Muted")
        signals_note.setWordWrap(True)
        signals_box.addWidget(signals_note)

        self._signals = QTreeWidget()
        self._signals.setColumnCount(3)
        self._signals.setHeaderLabels(("Signal", "In this log", "What it is for"))
        self._signals.setRootIsDecorated(False)
        self._signals.setAlternatingRowColors(True)
        self._signals.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        self._signals.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # The page scrolls; the tree must not, or the user hits a scroll region
        # inside a scroll region and one of the two swallows their wheel.
        self._signals.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self._signals.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        signals_box.addWidget(self._signals)
        column.addWidget(signals_card)
        return box

    def _build_kind_card(self, state: AppState) -> QFrame:
        """The one question the file cannot answer for itself (spec 5.2).

        Asked on this screen rather than after loading, because it decides what
        the load is for: a user who has to open a log, read a refusal and then go
        hunting for a setting has already been told the tool does not work on
        their flight.

        Three radios in a row with one description underneath, rather than three
        radios each with their own paragraph. The old arrangement was the single
        tallest thing on the page, and it was an advanced option that the default
        answers correctly nearly every time.
        """
        card = self.card()
        box = QVBoxLayout(card)
        box.setContentsMargins(16, 14, 16, 14)
        box.setSpacing(8)

        heading = QLabel("What kind of flight is this?")
        heading.setObjectName("Subheading")
        box.addWidget(heading)

        # Three across when they fit, stacked when they do not. Radio buttons
        # do not shrink -- their minimum is their label -- so a plain row of
        # three sets a floor for this card and, through it, for the page.
        row = ResponsiveRow(threshold=620, spacing=18)
        self._kind_group = QButtonGroup(self)
        self._kind_buttons: dict[LogKind | None, QRadioButton] = {}
        for kind, label, _blurb in _KIND_CHOICES:
            button = QRadioButton(label)
            button.setChecked(kind == state.declared_kind)
            button.toggled.connect(
                lambda checked, k=kind: self._kind_chosen(k) if checked else None
            )
            self._kind_group.addButton(button)
            self._kind_buttons[kind] = button
            row.add(button)
        box.addWidget(row)

        self._kind_note = QLabel()
        self._kind_note.setObjectName("Muted")
        self._kind_note.setWordWrap(True)
        box.addWidget(self._kind_note)
        self._describe_kind(state.declared_kind)
        return card

    def _kind_chosen(self, kind: LogKind | None) -> None:
        self._describe_kind(kind)
        self.state.declare_kind(kind)

    def _describe_kind(self, kind: LogKind | None) -> None:
        for choice, _label, blurb in _KIND_CHOICES:
            if choice == kind:
                self._kind_note.setText(blurb)
                return

    # ----------------------------------------------------------------- #

    def open_log_dialog(self) -> None:
        """Ask for a log. Public because the toolbar opens the same dialog."""
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
        self.banner(self._status, "blocker")
        self._loaded.setVisible(True)

    def _on_loaded(self, bundle: object) -> None:
        assert isinstance(bundle, LogBundle)
        self.banner(self._status, None)
        self._status.setText(f"{bundle.path.name} loaded.")
        self.refresh()

    def refresh(self) -> None:
        bundle = self.state.bundle
        clear(self._summary)
        self._signals.clear()
        self._empty.setVisible(bundle is None)
        self._loaded.setVisible(bundle is not None)
        # Kept as separate handles because the tests -- and the eye -- read the
        # inventory and the verdict as the two things this page is for.
        self._verdict.setVisible(bundle is not None)
        self._signals.setVisible(bundle is not None)
        self._summary_card.setVisible(bundle is not None)
        if bundle is None:
            return

        # Set here rather than only in the ``log_loaded`` handler: stages redraw
        # from state, and one that shows "No log loaded" beside a full summary
        # because it missed a signal is exactly the drift this design forbids.
        self.banner(self._status, None)
        self._status.setText(f"{bundle.path.name} loaded.")
        self._fill_summary(bundle)
        self._fill_verdict(bundle)
        self._fill_signals(bundle)

    def _fill_summary(self, bundle: LogBundle) -> None:
        heading = QLabel("About this flight")
        heading.setObjectName("Subheading")
        self._summary.addWidget(heading, 0, 0, 1, 2)

        values = {
            "File": bundle.path.name,
            "Kind": self._kind_label(bundle),
            "Stack": bundle.stack,
            "Firmware": bundle.firmware_version or "not recorded",
            "Board": bundle.board_id or "not recorded",
            "Duration": self._duration(bundle),
            "Loop rate": f"{bundle.loop_rate_hz:.0f} Hz",
            "Gyro rate": f"{bundle.gyro_sample_rate_hz:.0f} Hz",
            "Analysis grid": f"{bundle.sample_rate_hz:.0f} Hz",
            "Parameters": f"{len(bundle.params)}",
        }
        for row, label in enumerate(_SUMMARY_FIELDS, start=1):
            value = values[label]
            key = QLabel(label)
            key.setObjectName("Muted")
            self._summary.addWidget(key, row, 0)
            # Wrapped, because a firmware string is long and a label that will
            # not wrap sets a minimum width for the card, the row, and through
            # them the whole page.
            shown = QLabel(value)
            shown.setWordWrap(True)
            self._summary.addWidget(shown, row, 1)

        if bundle.warnings:
            self._summary.addWidget(self._notes(bundle.warnings), len(_SUMMARY_FIELDS) + 1, 0, 1, 2)
        self._summary.setColumnStretch(1, 1)

    def _notes(self, warnings: tuple[str, ...]) -> QLabel:
        """Every note the reader raised, as one banner rather than a stack of them.

        A real log produces ten of these -- one per message whose units the
        firmware did not declare -- and ten identically coloured boxes down the
        page read as ten problems. They are one observation about how the log was
        written, and none of them stops anything, so they are counted and listed
        rather than shouted individually.
        """
        lines = [f"{len(warnings)} note(s) about how this log was written:"]
        lines.extend(f"    - {warning}" for warning in warnings)
        note = QLabel("\n".join(lines))
        note.setWordWrap(True)
        self.banner(note, "warning")
        return note

    @staticmethod
    def _kind_label(bundle: LogBundle) -> str:
        """The kind in force, and where it came from."""
        label = capabilities(bundle.kind).label
        return f"{label} (you said so)" if bundle.kind_was_declared else f"{label} (detected)"

    def _fill_verdict(self, bundle: LogBundle) -> None:
        """What the declaration costs or buys, said before the analysis runs.

        A capped confidence rating discovered on the Review screen reads as the
        tool being unsure of itself. The same cap, stated here as a consequence of
        a choice the user just made, reads as the tool being precise about what
        the flight can prove.
        """
        caps = capabilities(bundle.kind)
        evidence = kind_evidence(bundle)
        disagrees = bundle.kind_was_declared and detect_kind(bundle) != bundle.kind

        lines = [f"<b>Analysed as a {caps.label.lower()}.</b> {caps.summary}"]
        if evidence:
            lines.append("Deliberate excitation in this log: " + "; ".join(evidence) + ".")
        else:
            lines.append("No injected sweep or autotune run is recorded in this log.")
        if disagrees:
            lines.append(
                "<b>That disagrees with the file.</b> "
                + (
                    "Nothing in it was deliberately excited, so the axes below will be "
                    "refused rather than identified."
                    if bundle.kind == "tuning"
                    else "The recorded excitation is not being used."
                )
            )
        lines.extend(caps.limits)
        self._verdict.setText("<br><br>".join(lines))
        self.banner(self._verdict, "warning" if disagrees else None)
        self._verdict.setVisible(True)

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
                    self._presence(present, names, required=required),
                    why if present else f"missing -- {why}",
                )
            )
            if not present:
                band = self.theme.band("blocker" if required else "warning")
                for column in range(3):
                    item.setForeground(column, _colour(band.foreground))
                    item.setBackground(column, _colour(band.background))
            self._signals.addTopLevelItem(item)
        self._fit(self._signals, len(_WANTED))

    @staticmethod
    def _fit(tree: QTreeWidget, rows: int) -> None:
        """Size the tree to all of its rows.

        It lives inside a page that scrolls, so a tree with its own scrollbar
        would be a scroll region inside a scroll region -- and the inner one
        swallows the wheel whenever the pointer happens to be over six rows of
        the most important table on the screen.
        """
        row_height = tree.sizeHintForRow(0) if rows else 22
        header = tree.header().height()
        tree.setFixedHeight(header + rows * max(row_height, 22) + 8)

    @staticmethod
    def _presence(present: list[str], names: list[str], *, required: bool) -> str:
        """A word, never a colour alone. See :mod:`rotorid.gui.theme`."""
        if len(present) == len(names):
            return "yes"
        if not present:
            return "missing -- required" if required else "missing"
        return f"{len(present)} of {len(names)}"

    @staticmethod
    def _duration(bundle: LogBundle) -> str:
        spans = [sig.t[-1] - sig.t[0] for sig in bundle.signals.values() if sig.t.size > 1]
        return f"{max(spans):.0f} s" if spans else "unknown"


def _colour(value: str) -> QBrush:
    """A palette colour as a brush, for the item views that take one."""
    return QBrush(QColor(value))
