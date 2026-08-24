"""Emptying a layout, done once and done correctly.

Ten places in this GUI redraw themselves by throwing away every widget they built
last time and building new ones, and all ten had written the same loop. Nine of
them had also written the same bug.

``deleteLater`` does not remove a widget from the screen. It schedules a deferred
deletion, which Qt runs when the event loop next gets round to ``DeferredDelete``
events -- and ``QApplication.processEvents()`` on its own does not always get
round to them. In between, the widget is still a child of its old parent and
still has its old geometry, so it keeps painting exactly where it was. The
symptom is a stale card sitting under a freshly drawn one, or an empty-state
message that appears twice.

Unparenting first is what actually takes it off the screen. The deferred delete
then frees it whenever Qt likes, which is fine, because by then nobody can see
it.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLayout

__all__ = ["clear"]


def clear(layout: QLayout) -> None:
    """Remove and destroy everything in ``layout``, nested layouts included."""
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        widget = item.widget()
        if widget is not None:
            # Order matters: off the screen first, freed afterwards.
            widget.setParent(None)
            widget.deleteLater()
            continue
        child = item.layout()
        if child is not None:
            clear(child)
            child.deleteLater()
