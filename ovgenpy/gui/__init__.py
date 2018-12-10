import sys
from typing import *
from pathlib import Path

import attr
from PyQt5 import uic
import PyQt5.QtCore as qc
import PyQt5.QtWidgets as qw
from PyQt5.QtCore import QModelIndex, Qt

from ovgenpy.channel import ChannelConfig
from ovgenpy.config import OvgenError
from ovgenpy.gui.data_bind import PresentationModel, map_gui, rgetattr, behead
from ovgenpy.ovgenpy import Config
from ovgenpy.triggers import CorrelationTriggerConfig, ITriggerConfig
from ovgenpy.util import perr, obj_name

APP_NAME = 'ovgenpy'
APP_DIR = Path(__file__).parent

def res(file: str) -> str:
    return str(APP_DIR / file)


def gui_main(cfg: Config, cfg_dir: str):
    app = qw.QApplication(sys.argv)
    app.setAttribute(qc.Qt.AA_EnableHighDpiScaling)

    window = MainWindow(cfg, cfg_dir)
    sys.exit(app.exec_())


class MainWindow(qw.QMainWindow):
    """
    Main window.

    Control flow:
    __init__
        load_cfg

    # Opening a document
    load_cfg
    """

    def __init__(self, cfg: Config, cfg_dir: str):
        super().__init__()
        uic.loadUi(res('mainwindow.ui'), self)   # sets windowTitle
        self.setWindowTitle(APP_NAME)

        self.cfg_dir = cfg_dir
        self.load_cfg(cfg)

        self.master_audio_browse: qw.QPushButton
        self.master_audio_browse.clicked.connect(self.on_master_audio_browse)
        self.show()
        # Complex changes are done in the presentation model's setters.

        # Abstract Item Model.data[idx] == QVariant (impl in subclass, wraps data)
        # Standard Item Model.item[r,c] == QStandardItem (it IS the data)
        # Explanation: https://doc.qt.io/qt-5/modelview.html#3-3-predefined-models

    model: 'ConfigModel'
    channel_model: 'ChannelModel'

    def load_cfg(self, cfg: Config):
        # TODO unbind current model's slots if exists
        # or maybe disconnect ALL connections??
        self.model = ConfigModel(cfg)
        map_gui(self, self.model)

        self.channel_model = ChannelModel(cfg.channels)
        self.channel_widget: qw.QTableView
        self.channel_widget.setModel(self.channel_model)

    def on_master_audio_browse(self):
        # TODO add default file-open dir, initialized to yaml path and remembers prev
        # useless if people don't reopen old projects
        name, file_type = qw.QFileDialog.getOpenFileName(
            self, "Open master audio file", filter="WAV files (*.wav)"
        )
        if name != '':
            master_audio = 'master_audio'
            self.model[master_audio] = name
            self.model.update_widget[master_audio]()


def nrow_ncol_property(altered: str, unaltered: str) -> property:
    def get(self: 'ConfigModel'):
        val = getattr(self._cfg.layout, altered)
        if val is None:
            return 0
        else:
            return val

    def set(self: 'ConfigModel', val: int):
        perr(altered)
        if val > 0:
            setattr(self._cfg.layout, altered, val)
            setattr(self._cfg.layout, unaltered, None)
            self.update_widget['layout__' + unaltered]()
        elif val == 0:
            setattr(self._cfg.layout, altered, None)
        else:
            raise OvgenError(f"invalid input: {altered} < 0, should never happen")

    return property(get, set)


class ConfigModel(PresentationModel):
    _cfg: Config
    combo_symbols = {}
    combo_text = {}

    @property
    def render_video_size(self) -> str:
        render = self._cfg.render
        w, h = render.width, render.height
        return f'{w}x{h}'

    @render_video_size.setter
    def render_video_size(self, value: str):
        error = OvgenError(f"invalid video size {value}, must be WxH")

        for sep in 'x*,':
            width_height = value.split(sep)
            if len(width_height) == 2:
                break
        else:
            raise error

        render = self._cfg.render
        width, height = width_height
        try:
            render.width = int(width)
            render.height = int(height)
        except ValueError:
            raise error

    layout__nrows = nrow_ncol_property('nrows', unaltered='ncols')
    layout__ncols = nrow_ncol_property('ncols', unaltered='nrows')
    combo_symbols['layout__orientation'] = ['h', 'v']
    combo_text['layout__orientation'] = ['Horizontal', 'Vertical']

    # TODO mutate _cfg and convert all colors to #rrggbb on access


T = TypeVar('T')


nope = qc.QVariant()


@attr.dataclass
class Column:
    key: str
    cls: Type
    default: Any

    def _display_name(self) -> str:
        return (self.key
                .replace('__', '\n')
                .replace('_', ' ')
                .title())
    display_name: str = attr.Factory(_display_name, takes_self=True)


class ChannelModel(qc.QAbstractTableModel):
    """ Design based off http://doc.qt.io/qt-5/model-view-programming.html#a-read-only-example-model """

    def __init__(self, channels: List[ChannelConfig]):
        """ Mutates `channels` for convenience. """
        super().__init__()
        self.channels = channels
        self.triggers: List[dict] = []

        for cfg in self.channels:
            t = cfg.trigger
            if isinstance(t, ITriggerConfig):
                if not isinstance(t, CorrelationTriggerConfig):
                    raise OvgenError(
                        f'Loading per-channel {obj_name(t)} not supported')
                trigger_dict = attr.asdict(t)
            else:
                trigger_dict = dict(t or {})

            cfg.trigger = trigger_dict
            self.triggers.append(trigger_dict)

    # columns
    col_data = [
        Column('wav_path', str, '', 'WAV Path'),
        Column('trigger_width', int, None, 'Trigger Width ×'),
        Column('render_width', int, None, 'Render Width ×'),
        Column('line_color', str, None, 'Line Color'),
        # TODO move from table view to sidebar QDataWidgetMapper?
        Column('trigger__edge_strength', float, None),
        Column('trigger__responsiveness', float, None),
        Column('trigger__buffer_falloff', float, None),
    ]

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return len(self.col_data)

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                col = section
                try:
                    return self.col_data[col].display_name
                except IndexError:
                    return nope
            else:
                return str(section + 1)
        return nope

    # rows
    def rowCount(self, parent: QModelIndex = ...) -> int:
        return len(self.channels)

    # data
    TRIGGER = 'trigger__'

    def data(self, index: QModelIndex, role=Qt.DisplayRole) -> qc.QVariant:
        col = index.column()
        row = index.row()

        if role in [Qt.DisplayRole, Qt.EditRole] and index.isValid() and row < self.rowCount():
            data = self.col_data[col]
            key = data.key
            if key.startswith(self.TRIGGER):
                key = behead(key, self.TRIGGER)
                value = self.triggers[row].get(key, '')

            else:
                value = getattr(self.channels[row], key)

            if value == data.default:
                return ''
            else:
                return str(value)

        return nope

    def setData(self, index: QModelIndex, value: str, role=Qt.EditRole) -> bool:
        col = index.column()
        row = index.row()

        if index.isValid() and role == Qt.EditRole:
            # type(value) == str

            data = self.col_data[col]
            key = data.key
            if value and not value.isspace():
                try:
                    value = data.cls(value)
                except ValueError as e:
                    # raise OvgenError(e)
                    return False
            else:
                value = data.default

            if key.startswith(self.TRIGGER):
                key = behead(key, self.TRIGGER)
                self.triggers[row][key] = value

            else:
                setattr(self.channels[row], key, value)

            return True
        return False

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.ItemIsEnabled
        return qc.QAbstractItemModel.flags(self, index) | Qt.ItemIsEditable
