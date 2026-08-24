"""The findings dock (spec sections 8 and 10.1).

Always visible, never modal. A finding is a qualification on the numbers the user
is currently reading, so it has to be beside them rather than behind an alert
they dismissed three stages ago.

Blocking findings carry an acknowledgement control right here, in the same place
the reason is displayed. Making somebody read the risk on one screen and accept
it on another is how acknowledgements turn into clicking through.

The panel opens with a one-line tally -- so many blocking, so many warnings --
because a dock holding fourteen cards is a dock nobody scrolls to the end of, and
the number of things that will actually stop an export is the fact the user needs
before they decide whether to read any of them.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QInputDialog,
    QLabel,
    QLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rotorid.core.types import Finding, Severity
from rotorid.gui.theme import SEVERITY_MARK, Palette, palette
from rotorid.gui.widgets.layouts import clear
from rotorid.gui.widgets.responsive import FlowLayout
from rotorid.gui.widgets.stage_page import FittedScrollArea

__all__ = ["FindingsPanel"]

#: Worst first. Findings arrive in the order the analysis made them, which is
#: the order the code runs in rather than the order that matters to a reader.
_ORDER: dict[Severity, int] = {"blocker": 0, "warning": 1, "info": 2, "good": 3}


class FindingsPanel(FittedScrollArea):
    """Findings, worst first, each with what to do about it."""

    acknowledge_requested = Signal(str, str)  # code, reason
    withdraw_requested = Signal(str)

    def __init__(self, theme: Palette | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme = theme if theme is not None else palette("light")
        self._last: tuple[tuple[Finding, ...], dict[str, str]] = ((), {})
        self._body = QWidget()
        self._layout = QVBoxLayout(self._body)
        # Top-aligned with a trailing stretch rather than with the layout's own
        # alignment flag. An alignment set on a layout makes Qt size that layout
        # to its *size hint* inside the space it was given -- and a size hint
        # cannot express "this tall at this width", so every card of wrapped text
        # came out short. A stretch achieves the same look with none of that.
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(10)
        self.setWidget(self._body)
        # Drawn through the same path as everything else, so there is exactly one
        # empty-state message rather than a placeholder and a real one racing to
        # be deleted.
        self.show_findings(())

    def set_theme(self, theme: Palette) -> None:
        self.theme = theme
        findings, accepted = self._last
        self.show_findings(findings, accepted)

    def show_findings(
        self, findings: tuple[Finding, ...], acknowledged: dict[str, str] | None = None
    ) -> None:
        accepted = acknowledged or {}
        self._last = (findings, dict(accepted))
        clear(self._layout)

        if not findings:
            label = QLabel("Nothing to flag. That is a result, not an absence of one.")
            label.setObjectName("Muted")
            label.setWordWrap(True)
            self._layout.addWidget(label)
            self._layout.addStretch(1)
            return

        ordered = sorted(findings, key=lambda f: _ORDER.get(f.severity, 9))
        self._layout.addWidget(self._tally(ordered, accepted))
        for finding in ordered:
            self._layout.addWidget(self._card(finding, accepted.get(finding.code)))
        self._layout.addStretch(1)

    # ------------------------------------------------------------------ #

    def _tally(self, findings: list[Finding], accepted: dict[str, str]) -> QWidget:
        """One line saying how much of this the user has to care about."""
        counts: dict[Severity, int] = {}
        for finding in findings:
            severity: Severity = finding.severity
            counts[severity] = counts.get(severity, 0) + 1
        blocking = sum(1 for f in findings if f.severity == "blocker" and f.code not in accepted)
        wording: tuple[tuple[Severity, str], ...] = (
            ("blocker", "blocking"),
            ("warning", "to weigh"),
            ("info", "for information"),
            ("good", "confirmed good"),
        )
        parts = [f"{counts[sev]} {word}" for sev, word in wording if counts.get(sev)]
        label = QLabel(", ".join(parts) or "nothing to flag")
        label.setWordWrap(True)
        band = self.theme.band("blocker" if blocking else "good")
        band.apply(label, padding=8)
        if blocking:
            label.setText(
                f"{blocking} finding(s) must be acknowledged before anything can be "
                f"exported. " + ", ".join(parts) + "."
            )
        return label

    def _card(self, finding: Finding, accepted: str | None) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setSpacing(6)

        band = self.theme.band(finding.severity)
        header = QLabel(f"{SEVERITY_MARK[finding.severity]}   {finding.title}")
        band.apply(header, padding=7, bold=True)
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

    def _acknowledgement_row(self, finding: Finding, accepted: str | None) -> QLayout:
        """The accept control, beside its warning or under it.

        Wrapping, because the dock is the narrowest column in the window and a
        fixed row of "Exports stay disabled until this is acknowledged" next to
        an "Acknowledge and continue" button is wider than the dock ever gets --
        which made the whole panel scroll sideways and every card in it come out
        short.
        """
        row = FlowLayout(spacing=8)
        if accepted is not None:
            note = QLabel(f"Accepted: {accepted}")
            note.setWordWrap(True)
            withdraw = QPushButton("Withdraw")
            withdraw.clicked.connect(lambda: self.withdraw_requested.emit(finding.code))
            row.addWidget(note)
            row.addWidget(withdraw)
            return row

        note = QLabel("Exports stay disabled until this is acknowledged.")
        note.setWordWrap(True)
        button = QPushButton("Acknowledge and continue")
        button.clicked.connect(lambda: self._ask(finding))
        row.addWidget(note)
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
