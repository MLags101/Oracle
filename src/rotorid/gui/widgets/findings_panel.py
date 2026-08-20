"""The findings dock (spec sections 8 and 10.1).

Always visible, never modal. A finding is a qualification on the numbers the user
is currently reading, so it has to be beside them rather than behind an alert
they dismissed three stages ago.

Blocking findings carry an acknowledgement control right here, in the same place
the reason is displayed. Making someone read the risk on one screen and accept it
on another is how acknowledgements turn into clicking through.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from rotorid.core.types import Finding
from rotorid.gui.theme import SEVERITY_MARK, severity_colour

__all__ = ["FindingsPanel"]


class FindingsPanel(QScrollArea):
    """Findings, worst first, each with what to do about it."""

    acknowledge_requested = Signal(str, str)  # code, reason
    withdraw_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._body = QWidget()
        self._layout = QVBoxLayout(self._body)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setWidget(self._body)
        self._empty = QLabel("No findings yet.")
        self._empty.setObjectName("Muted")
        self._layout.addWidget(self._empty)

    def show_findings(
        self, findings: tuple[Finding, ...], acknowledged: dict[str, str] | None = None
    ) -> None:
        accepted = acknowledged or {}
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()

        if not findings:
            label = QLabel("Nothing to flag. That is a result, not an absence of one.")
            label.setObjectName("Muted")
            label.setWordWrap(True)
            self._layout.addWidget(label)
            return

        for finding in findings:
            self._layout.addWidget(self._card(finding, accepted.get(finding.code)))

    def _card(self, finding: Finding, accepted: str | None) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)

        header = QLabel(f"{SEVERITY_MARK[finding.severity]}  {finding.title}")
        header.setStyleSheet(f"color: {severity_colour(finding.severity)}; font-weight: 600;")
        header.setWordWrap(True)
        layout.addWidget(header)

        for text in (finding.detail, f"What to do: {finding.action}"):
            label = QLabel(text)
            label.setWordWrap(True)
            layout.addWidget(label)

        if finding.evidence:
            evidence = QLabel(
                "  ".join(f"{k} = {v:.4g}" for k, v in sorted(finding.evidence.items()))
            )
            evidence.setObjectName("Muted")
            evidence.setWordWrap(True)
            layout.addWidget(evidence)

        code = QLabel(finding.code)
        code.setObjectName("Muted")
        layout.addWidget(code)

        if finding.severity == "blocker":
            layout.addLayout(self._acknowledgement_row(finding, accepted))
        return card

    def _acknowledgement_row(self, finding: Finding, accepted: str | None) -> QHBoxLayout:
        row = QHBoxLayout()
        if accepted is not None:
            note = QLabel(f"Accepted: {accepted}")
            note.setWordWrap(True)
            withdraw = QPushButton("Withdraw")
            withdraw.clicked.connect(lambda: self.withdraw_requested.emit(finding.code))
            row.addWidget(note, 1)
            row.addWidget(withdraw)
            return row

        button = QPushButton("Acknowledge and continue")
        button.clicked.connect(lambda: self._ask(finding))
        row.addWidget(QLabel("Exports stay disabled until this is acknowledged."), 1)
        row.addWidget(button)
        return row

    def _ask(self, finding: Finding) -> None:
        """Acknowledging requires typing a reason, which is exported with the file."""
        reason, ok = QInputDialog.getText(
            self,
            f"Acknowledge {finding.code}",
            "This is recorded in the exported parameter file, where somebody who was "
            "not here will read it.\n\nWhy are you accepting this risk?",
        )
        if ok and reason.strip():
            self.acknowledge_requested.emit(finding.code, reason.strip())
