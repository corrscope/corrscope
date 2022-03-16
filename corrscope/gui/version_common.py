import qtpy
from packaging.version import parse

QT6 = parse(qtpy.QT_VERSION) >= parse("6.0.0")
