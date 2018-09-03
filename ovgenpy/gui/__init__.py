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
from ovgenpy.ovgenpy import Config, default_config
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
        # items.setData(items.index(0,0), item)

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


class ChannelWidget(qw.QWidget):
    """ Widget bound to a single ChannelModel. """

    def __init__(self, cfg: ChannelConfig):
        super().__init__()
        uic.loadUi(res('channel_widget.ui'), self)

        self.model = ChannelModel(cfg)
        map_gui(self, self.model)

        # FIXME uncomment?
        # self.show()


T = TypeVar('T')


# def coalesce_property(key: str, default: T) -> property:
#     def get(self: 'ChannelModel') -> T:
#         val: Optional[T] = getattr(self._cfg, key)
#         if val is None:
#             return default
#         return val
#
#     def set(self: 'ChannelModel', val: T):
#         if val == default:
#             val = None
#         setattr(self._cfg, key, val)
#
#     return property(get, set)


class ChannelModel(qc.QAbstractTableModel):

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
    col_keys = ['wav_path', 'trigger_width', 'render_width', 'line_color',
                'trigger__']

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return len(self.col_keys)

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role=Qt.DisplayRole):
        nope = qc.QVariant()
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            col = section
            try:
                return self.col_keys[col]
            except IndexError:
                return nope
        return nope

    # rows
    def rowCount(self, parent: QModelIndex = ...) -> int:
        return len(self.channels)

    # data
    TRIGGER = 'trigger__'

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            col = index.column()
            row = index.row()

            key = self.col_keys[col]
            if key.startswith(self.TRIGGER):
                key = behead(key, self.TRIGGER)
                return self.triggers[row].get(key, '')

            else:
                return getattr(self.channels[row], key)

            # try:
            #     return getattr(self, 'cfg__' + key)
            # except AttributeError:
            #     pass
            # if key.startswith('trigger__') and cfg.trigger is None:
            #     return rgetattr(cfg, key, '')
            # else:
            #     return rgetattr(cfg, key)

            # DEFAULT = object()
            # if val is DEFAULT:
            #     # Trigger attributes can be missing, all others must be present.
            #     assert key.startswith('trigger__')
            #     return ''
            # else:
            #     return val
        return super().data(index, role)

    def setData(self, index: QModelIndex, value, role=Qt.EditRole) -> bool:
        if role == Qt.EditRole:
            # FIXME what type is value? str or not?
            perr(repr(role))
            return True
        return super().setData(index, value, role)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        return super().flags(index)

    # _cfg: ChannelConfig
    # combo_symbols = {}
    # combo_text = {}
    #
    # trigger_width = coalesce_property('trigger_width', 1)
    # render_width = coalesce_property('render_width', 1)
    # line_color = coalesce_property('line_color', '')  # TODO

