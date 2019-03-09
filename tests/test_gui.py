import PyQt5.QtWidgets as qw

from corrscope import gui
from corrscope.corrscope import default_config


def test_gui_init():
    app = qw.QApplication([])
    cfg = default_config()
    gui.MainWindow(cfg)
