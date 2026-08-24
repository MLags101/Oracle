"""Current against recommended, parameter by parameter (spec section 10.2).

Three columns and nothing clever. The one decision worth arguing about is that
the flown value is always shown, even when the tool is not asking to change it:
a diff that hides what a parameter is now is a diff you cannot check against
your own aircraft, and checking it against your own aircraft is exactly what the
user should be doing before loading anything.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem, QWidget

__all__ = ["ParamDiffTable"]


class ParamDiffTable(QTableWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 4, parent)
        self.setHorizontalHeaderLabels(("Parameter", "Now", "Proposed", "Change"))
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        # Resize modes rather than a call to ``resizeColumnsToContents``. That
        # call measures once, at a moment when the table may not yet have been
        # laid out, and leaves ``ATC_RAT_RLL_D`` reading ``ATC_RAT_...`` next to
        # 300 pixels of empty table. Modes are re-evaluated whenever anything
        # changes, so the parameter name gets the width it needs and the three
        # numeric columns share what is left.
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

    def show_diff(self, proposed: dict[str, float], flown: dict[str, float]) -> None:
        self.setRowCount(len(proposed))
        for row, (name, value) in enumerate(sorted(proposed.items())):
            current = flown.get(name)
            self.setItem(row, 0, _cell(name))
            self.setItem(row, 1, _cell("not set" if current is None else f"{current:g}"))
            self.setItem(row, 2, _cell(f"{value:g}"))

            change = _change(current, value)
            item = _cell(change)
            if change != "unchanged":
                item.setForeground(Qt.GlobalColor.darkYellow)
                item.setToolTip("This value is being asked to change.")
            self.setItem(row, 3, item)
        self.fit_to_rows()

    def show_nothing(self, why: str) -> None:
        """No proposal is a result, and it needs to say why rather than be blank."""
        self.setRowCount(1)
        self.setSpan(0, 0, 1, 4)
        item = _cell(why)
        item.setForeground(Qt.GlobalColor.darkGray)
        self.setItem(0, 0, item)
        self.fit_to_rows()

    def fit_to_rows(self) -> None:
        """Take exactly the height of the rows in it.

        These tables sit inside cards on a page that scrolls. A table with its
        own scrollbar there would hide half of a two-row diff behind a control
        the user has no reason to expect, on the one screen where seeing every
        proposed change at once is the entire point.
        """
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rows = self.rowCount()
        height = self.horizontalHeader().height() + 2 * self.frameWidth()
        for row in range(rows):
            height += self.rowHeight(row)
        self.setFixedHeight(height)


def _cell(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    return item


def _change(current: float | None, proposed: float) -> str:
    if current is None:
        return "new"
    if abs(proposed - current) <= 1e-9:
        return "unchanged"
    if current == 0.0:
        return "up from zero"
    return f"{proposed / current:.2f}x"
