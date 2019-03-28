from typing import TYPE_CHECKING, List

from PyQt5 import QtWidgets as qw, QtCore as qc
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut

from corrscope.gui.util import find_ranges

if TYPE_CHECKING:
    from corrscope.gui import ChannelModel


class ChannelTableView(qw.QTableView):
    def append_channels(self, wavs: List[str]):
        model: "ChannelModel" = self.model()

        begin_row = model.rowCount()
        count_rows = len(wavs)

        col = model.idx_of_key["wav_path"]

        # Insert N empty rows into model (mutates cfg.channels).
        model.insertRows(begin_row, count_rows)

        # Fill N empty rows with wav_path.
        for row, wav_path in enumerate(wavs, begin_row):
            index = model.index(row, col)
            model.setData(index, wav_path)

    def delete_selected(self):
        model: "ChannelModel" = self.model()
        rows = self.selected_rows()
        row_ranges = find_ranges(rows)

        for first_row, nrow in reversed(list(row_ranges)):
            model.removeRows(first_row, nrow)

    def on_channel_up(self):
        self.move_selection(-1)

    def on_channel_down(self):
        self.move_selection(1)

    def move_selection(self, delta: int):
        model: "ChannelModel" = self.model()
        rows = self.selected_rows()
        row_ranges = find_ranges(rows)

        # If we hit the end, cancel all other moves.
        # If moving up, move top first.
        if delta > 0:
            # If moving down, move bottom first.
            row_ranges = reversed(list(row_ranges))

        parent = qc.QModelIndex()
        for first_row, nrow in row_ranges:
            if delta > 0:
                dest_row = first_row + nrow + delta
            else:
                dest_row = first_row + delta

            if not model.moveRows(parent, first_row, nrow, parent, dest_row):
                break

    def selected_rows(self) -> List[int]:
        sel: qc.QItemSelectionModel = self.selectionModel()
        inds: List[qc.QModelIndex] = sel.selectedIndexes()
        rows: List[int] = sorted({ind.row() for ind in inds})
        return rows


class ShortcutButton(qw.QPushButton):
    scoped_shortcut: QShortcut

    def add_shortcut(self, scope: qw.QWidget, shortcut: str) -> None:
        """ Adds shortcut and tooltip. """
        keys = QKeySequence(shortcut, QKeySequence.PortableText)

        self.scoped_shortcut = qw.QShortcut(keys, scope)
        self.scoped_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.scoped_shortcut.activated.connect(self.click)

        self.setToolTip(keys.toString(QKeySequence.NativeText))
