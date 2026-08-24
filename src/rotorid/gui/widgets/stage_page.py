"""Scroll areas that are honest about how tall their contents are.

A plain ``QScrollArea`` with ``setWidgetResizable(True)`` sizes its contents to
``max(viewport height, minimum size hint)``, and that is not good enough here,
because of how Qt sizes wrapped text.

A ``QLabel`` with word wrap has no fixed height -- its height depends on how wide
it is allowed to be -- and Qt expresses that through ``heightForWidth``. But a
widget's *minimum size hint* cannot ask a question about width, so a wrapped
label reports a minimum of about one line however many paragraphs it holds. A
page built from a dozen such labels therefore reports that it is happy at a few
hundred pixels, the scroll area believes it, no scrollbar appears, and the text
is drawn clipped inside a box too short for it. That is the "runs off the screen
with nothing to say it has" failure, and it is a Qt default rather than anything
the pages did wrong.

So this asks the layout the question the minimum size hint cannot: given the
width the viewport actually is, how tall does this need to be? The answer becomes
the contents' minimum height, and from there ``setWidgetResizable`` does the
right thing -- filling the viewport when the content is short, and growing a
scrollbar when it is not.

Both scrolling surfaces in the application need this and both had the bug: the
stage pages, and the findings dock, whose cards are nothing *but* wrapped
paragraphs.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QFrame, QScrollArea, QWidget

__all__ = ["FittedScrollArea", "StagePage"]


class FittedScrollArea(QScrollArea):
    """A scroll area whose contents are measured at the width they will be drawn at."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fitting = False
        self.setWidgetResizable(True)

    def setWidget(self, widget: QWidget) -> None:
        super().setWidget(widget)
        # Contents rebuild themselves whenever the state changes, and something
        # that has just grown three cards needs re-measuring. A layout request is
        # Qt saying exactly that.
        widget.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.widget() and event.type() == QEvent.Type.LayoutRequest:
            self._fit()
        return False

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        contents = self.widget()
        layout = contents.layout() if contents is not None else None
        if contents is None or layout is None or self._fitting:
            return
        # The width the contents will actually be drawn at, which is not always
        # the viewport's: when something inside refuses to be narrower, the
        # widget is laid out wider and the viewport scrolls sideways to it.
        # Measuring the height at the viewport width would then answer a question
        # about a layout that never happens, and the answer is short.
        width = max(self.viewport().width(), layout.totalMinimumSize().width())
        if width <= 0:
            return

        # Before the guard, not inside it. Teaching a card to answer invalidates
        # its cached size, and the answer measured in the same breath is still
        # the old one -- so when anything changed, this pass is thrown away and
        # another is scheduled for once the invalidation has settled.
        if _let_containers_answer(contents):
            QTimer.singleShot(0, self._fit)
            return

        # Re-entrancy guard: setting the minimum height posts another layout
        # request, which arrives back here. It settles on the second pass because
        # the answer stops changing, but the guard keeps that from depending on
        # a coincidence.
        self._fitting = True
        try:
            wanted = (
                layout.totalHeightForWidth(width)
                if layout.hasHeightForWidth()
                else layout.totalSizeHint().height()
            )
            contents.setMinimumHeight(max(0, wanted))
        finally:
            self._fitting = False


def _let_containers_answer(root: QWidget) -> bool:
    """Let every card in the tree be asked how tall it is at a given width.

    ``QWidget`` implements ``heightForWidth`` by forwarding to its layout, but a
    parent layout only *asks* when the child's size policy says the child has an
    answer -- and the default policy says it does not. ``QLabel`` sets the flag
    for itself when word wrap is on; a plain ``QFrame`` holding four wrapped
    labels does not, so the measurement stops at the card and the card is sized
    for one line of each.

    That is why findings cards were drawn with their first paragraph sliced
    through. Setting the flag wherever the layout genuinely has an answer lets
    the question reach the bottom of the tree.

    Returns whether anything changed, because a caller that measured in the same
    pass would be reading sizes cached before the change.
    """
    changed = False
    for child in root.findChildren(QWidget):
        layout = child.layout()
        if layout is None or not layout.hasHeightForWidth():
            continue
        policy = child.sizePolicy()
        if policy.hasHeightForWidth():
            continue
        policy.setHeightForWidth(True)
        child.setSizePolicy(policy)
        changed = True
    return changed


class StagePage(FittedScrollArea):
    """One stage, scrollable, sized to its content at the current width."""

    def __init__(self, stage: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidget(stage)
