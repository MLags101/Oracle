"""Stage 7: Review and export (spec sections 10.2 and 16).

The only screen whose output ends up on an aircraft, so it is built around
refusal rather than convenience:

* **Nothing is written to a vehicle, ever.** These buttons write files. A human
  loads them, deliberately, having read them.
* **Blocking findings disable the export**, and the only way past is to
  acknowledge each one by name with a reason -- which is then written into the
  file header, where somebody who was not in the room will read it.
* **The plan is shown as flights, not as a parameter list.** The single most
  common way a tuning session goes wrong is changing several things at once and
  then being unable to attribute the result, and a screen that offers one big
  "export everything" button is a screen that encourages exactly that.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rotorid import __version__
from rotorid.core.types import FlightTestStage
from rotorid.gui.state import AppState
from rotorid.gui.theme import Palette, severity_colour
from rotorid.gui.widgets.layouts import clear
from rotorid.gui.widgets.param_diff_table import ParamDiffTable
from rotorid.gui.widgets.responsive import FlowLayout
from rotorid.gui.wizard.base import StageWidget

__all__ = ["ReviewStage"]


class ReviewStage(StageWidget):
    """What to change, in what order, and the files that say so."""

    title = "Review & Export"

    def __init__(
        self,
        state: AppState,
        theme: Palette | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(state, theme, parent)

        layout = self.page()
        layout.addWidget(
            self.header(
                "Review & Export",
                subtitle=(
                    "Every change the analysis is proposing, staged in the order it is "
                    "safe to fly, and the files that record it."
                ),
            )
        )

        self._safety = QLabel(
            "RotorID never writes to a vehicle. These buttons write files that you "
            "load yourself. Back up your current parameters first, fly one stage per "
            "flight, and download the log after each one."
        )
        self._safety.setWordWrap(True)
        self.banner(self._safety, "info")
        layout.addWidget(self._safety)

        self._gate = QLabel("")
        self._gate.setWordWrap(True)
        layout.addWidget(self._gate)

        # Wrapping, not a fixed row: three buttons with names this long give the
        # page a floor of about a thousand pixels, which is more than a 1200-wide
        # window has left once the rail and the findings dock have taken theirs.
        buttons = FlowLayout(spacing=10)
        self._export_button = QPushButton("Export staged .param files...")
        self._export_button.setObjectName("Primary")
        self._export_button.clicked.connect(self._export_params)
        self._report_button = QPushButton("Write HTML report...")
        self._report_button.clicked.connect(self._export_report)
        self._session_button = QPushButton("Save session...")
        self._session_button.clicked.connect(self._save_session)
        for button in (self._export_button, self._report_button, self._session_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        # No scroll area of its own: the shell wraps every stage in one, and
        # a scroll region inside a scroll region swallows the wheel wherever the
        # pointer happens to be sitting.
        self._body = QWidget()
        self._stages = QVBoxLayout(self._body)
        self._stages.setContentsMargins(0, 0, 0, 0)
        self._stages.setSpacing(12)
        layout.addWidget(self._body, 1)

        state.analysis_finished.connect(lambda *_: self.refresh())
        state.acknowledgements_changed.connect(self.refresh)

    # ----------------------------------------------------------------- #
    # Drawing
    # ----------------------------------------------------------------- #

    def refresh(self) -> None:
        clear(self._stages)

        result = self.state.result
        plan = result.session.next_steps if result is not None else None
        self._update_gate()

        if plan is None or not plan.stages:
            self._stages.addWidget(
                _muted(
                    "No plan yet. Run the analysis, or -- if it has run -- nothing in "
                    "this log supports a change worth flying for."
                )
            )
            self._stages.addStretch(1)
            return

        preamble = QLabel(plan.preamble)
        preamble.setWordWrap(True)
        self._stages.addWidget(preamble)
        for stage in plan.stages:
            self._stages.addWidget(self._card(stage))
        # A trailing stretch rather than the layout's alignment flag: an aligned
        # layout is sized to its size hint, and a size hint cannot say "this tall
        # at this width", so cards of wrapped text come out clipped.
        self._stages.addStretch(1)

    def _card(self, stage: FlightTestStage) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)

        title = QLabel(f"Flight {stage.index}: {stage.title}")
        title.setObjectName("Heading")

        title.setWordWrap(True)
        layout.addWidget(title)

        if stage.motivating_findings:
            layout.addWidget(_muted("Prompted by: " + ", ".join(stage.motivating_findings)))

        table = ParamDiffTable()
        flown = self.state.bundle.params if self.state.bundle is not None else {}
        table.show_diff(stage.changes, flown)
        layout.addWidget(table)

        for label, items in (
            ("Watch for in flight", stage.watch_in_flight),
            ("Then check in the log", stage.check_in_log),
        ):
            if not items:
                continue
            layout.addWidget(_muted(label + ":"))
            for item in items:
                line = QLabel(f"    - {item}")
                line.setWordWrap(True)
                layout.addWidget(line)
        return card

    def _update_gate(self) -> None:
        blocked = self.state.unresolved
        allowed = bool(self.state.result is not None and not blocked)
        self._export_button.setEnabled(allowed)

        if not blocked:
            self._gate.setText("")
            self._gate.setStyleSheet("")
            return
        self._gate.setStyleSheet(f"color: {severity_colour('blocker')};")
        self._gate.setText(
            "Export is disabled: "
            + ", ".join(blocked)
            + ". Acknowledge each one in the findings panel -- with a reason, which is "
            "written into the exported file -- or fly again and re-analyse. This is not "
            "a formality: acting on this analysis means accepting a risk the tool has "
            "already said it cannot stand behind."
        )

    # ----------------------------------------------------------------- #
    # Writing files
    # ----------------------------------------------------------------- #

    def _export_params(self) -> None:
        from rotorid.core.export.params import ExportBlockedError, write_param_files

        result = self.state.result
        if result is None or result.session.next_steps is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Where should the files go?")
        if not directory:
            return

        try:
            written = write_param_files(
                Path(directory),
                result.session.next_steps,
                log_name=result.session.log.path.name,
                tool_version=__version__,
                config_hash=self.state.config.hash,
                findings=result.session.findings,
                acknowledgements=dict(self.state.acknowledgements),
            )
        except (ExportBlockedError, ValueError) as exc:
            QMessageBox.warning(self, "Not exported", str(exc))
            return

        QMessageBox.information(
            self,
            "Exported",
            "Written, one file per flight:\n\n"
            + "\n".join(p.name for p in written)
            + "\n\nBack up your current parameters before loading any of them.",
        )

    def _export_report(self) -> None:
        from rotorid.core.export.report import write_report

        result = self.state.result
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Write report", "report.html", "HTML (*.html)")
        if not path:
            return
        write_report(
            Path(path),
            result.session.log,
            {str(a): r for a, r in result.session.recommendations.items()},
            config_hash=self.state.config.hash,
            tool_version=__version__,
            findings=result.session.findings,
            plan=result.session.next_steps,
            measured_steps={str(a): m for a, m in result.session.measured_steps.items()},
        )

    def _save_session(self) -> None:
        import dataclasses

        from rotorid.core.export.session import save_session

        result = self.state.result
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", "session.rotorid", "RotorID session (*.rotorid)"
        )
        if not path:
            return
        # The acknowledgements live on the window until they are saved, so the
        # session on disk has to pick up whatever has been accepted since the
        # analysis ran -- otherwise reopening it would silently drop them.
        save_session(
            Path(path),
            dataclasses.replace(result.session, acknowledgements=dict(self.state.acknowledgements)),
        )


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Muted")
    label.setWordWrap(True)
    return label
