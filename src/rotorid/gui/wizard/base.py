"""What every wizard stage is (spec section 10.2)."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from rotorid.gui.state import AppState

__all__ = ["StageWidget"]


class StageWidget(QWidget):
    """One step of the wizard.

    Stages are passive: they read :class:`AppState` when told to refresh and
    never hold their own copy of an analysis result. A stage caching a
    recommendation is a stage that will still be showing it after the user
    changes the conservatism slider.
    """

    #: Shown in the rail.
    title = "Stage"

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state

    def refresh(self) -> None:
        """Redraw from current state. Called on entry and on every state change."""
