""" "Why this number?" (spec section 10.5).

Attached to a value, not to a screen. The affordance sits next to the number it
explains, because a trace that has to be hunted for in a help menu is a trace
nobody reads -- and a recommendation nobody can interrogate is one they either
take on faith or ignore, which are the two outcomes this tool exists to avoid.

If :func:`rotorid.core.guidance.explain.explain` has nothing filed under a key,
:func:`why_button` returns ``None`` and the caller shows no affordance at all. A
"why?" link that opens onto an empty box is worse than no link: it teaches the
user that the button does not work.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from rotorid.core.guidance.explain import Explanation, glossary_for
from rotorid.core.types import TuneRecommendation

__all__ = ["WhyDialog", "why_button"]


def why_button(
    key: str, rec: TuneRecommendation, parent: QWidget | None = None
) -> QPushButton | None:
    """A small button that opens the trace behind ``key``, or None if there is none."""
    from rotorid.core.guidance.explain import explain

    explanation = explain(key, rec)
    if explanation is None:
        return None

    button = QPushButton("why?")
    button.setFlat(True)
    button.setToolTip(explanation.headline)
    button.clicked.connect(lambda: WhyDialog(explanation, parent).exec())
    return button


class WhyDialog(QDialog):
    """The whole trace: what the number does, why it is this, what lost."""

    def __init__(self, explanation: Explanation, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{explanation.title} = {explanation.value}")
        self.resize(560, 520)

        body = QWidget()
        inner = QVBoxLayout(body)
        inner.addWidget(_wrapped(explanation.headline, heading=True))

        inner.addWidget(_wrapped("Why this number:", heading=True))
        for line in explanation.because:
            inner.addWidget(_wrapped(f"•  {line}"))

        if explanation.binding is not None:
            inner.addWidget(
                _wrapped(f"Binding constraint: {explanation.binding.replace('_', ' ')}", muted=True)
            )

        if explanation.alternatives:
            inner.addWidget(_wrapped("What was rejected, and why:", heading=True))
            for what, why in explanation.alternatives:
                inner.addWidget(_wrapped(f"•  {what} — {why}", muted=True))

        entries = glossary_for(explanation)
        if entries:
            inner.addWidget(_wrapped("Terms used here:", heading=True))
            for entry in entries:
                inner.addWidget(_wrapped(f"{entry.term}: {entry.detail}", muted=True))

        inner.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


def _wrapped(text: str, *, heading: bool = False, muted: bool = False) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(label.textInteractionFlags().TextSelectableByMouse)
    if heading:
        label.setObjectName("Heading")
    elif muted:
        label.setObjectName("Muted")
    return label
