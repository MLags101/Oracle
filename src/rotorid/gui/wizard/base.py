"""What every wizard stage is (spec section 10.2)."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from rotorid.core.types import Severity
from rotorid.gui.state import STAGE_BLURBS, AppState
from rotorid.gui.theme import Palette, palette

__all__ = ["StageWidget"]


class StageWidget(QWidget):
    """One step of the wizard.

    Stages are passive: they read :class:`AppState` when told to refresh and
    never hold their own copy of an analysis result. A stage caching a
    recommendation is a stage that will still be showing it after the user
    changes the conservatism slider.

    They are handed the palette rather than reaching for a default one. The old
    arrangement -- every plot quietly falling back to the light palette -- meant
    the dark theme restyled the chrome and left nine white rectangles in the
    middle of it.
    """

    #: Shown in the rail.
    title = "Stage"

    def __init__(
        self,
        state: AppState,
        theme: Palette | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.theme = theme if theme is not None else palette("light")

    def refresh(self) -> None:
        """Redraw from current state. Called on entry and on every state change."""

    # ----------------------------------------------------------------- #
    # Shared furniture
    # ----------------------------------------------------------------- #

    def page(self, *, margin: int = 18, spacing: int = 14) -> QVBoxLayout:
        """The stage's own layout, with the spacing every stage shares.

        Consistent gutters are most of what separates a screen that looks
        designed from one that looks assembled, and they are exactly the thing
        that drifts when each stage picks its own.
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(spacing)
        return layout

    def header(self, name: str, *, subtitle: str = "") -> QWidget:
        """A titled header for the stage, with what the step is for underneath.

        Every stage gets one, so a user who arrived by clicking the rail rather
        than by stepping through knows which of the nine they are looking at
        without counting discs.
        """
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        eyebrow = QLabel(f"STEP {self._number(name)} OF 9")
        eyebrow.setObjectName("Eyebrow")
        column.addWidget(eyebrow)

        heading = QLabel(name)
        heading.setObjectName("Title")
        column.addWidget(heading)

        note = QLabel(subtitle or STAGE_BLURBS.get(name, ""))
        note.setObjectName("Muted")
        note.setWordWrap(True)
        column.addWidget(note)
        return box

    def card(self) -> QFrame:
        """An empty surface with the shared border and radius."""
        frame = QFrame()
        frame.setObjectName("Card")
        return frame

    def banner(self, label: QLabel, severity: Severity | None) -> None:
        """Paint ``label`` as a banner at this severity, or clear it when ``None``.

        Takes the label rather than returning a stylesheet, because a banner is
        two things -- colours and room to breathe -- and a caller who applies only
        the first gets text with its last line cut off. See :meth:`Band.apply`.
        """
        if severity is None:
            label.setStyleSheet("")
            label.setContentsMargins(0, 0, 0, 0)
            return
        self.theme.band(severity).apply(label)

    @staticmethod
    def _number(name: str) -> int:
        from rotorid.gui.state import STAGES

        return STAGES.index(name) + 1 if name in STAGES else 0
