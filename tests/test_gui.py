import PyQt6.QtWidgets as qw

from corrscope import gui
from corrscope.corrscope import template_config


def test_gui_init():
    app = qw.QApplication([])
    cfg = template_config()
    gui.MainWindow(cfg)
