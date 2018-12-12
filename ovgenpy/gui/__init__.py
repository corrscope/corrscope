import os
import sys
from pathlib import Path
from typing import *

import PyQt5.QtCore as qc
import PyQt5.QtWidgets as qw
import attr
from PyQt5 import uic
from PyQt5.QtCore import QModelIndex, Qt

from ovgenpy import cli
from ovgenpy.channel import ChannelConfig
from ovgenpy.config import OvgenError, copy_config, yaml
from ovgenpy.gui.data_bind import PresentationModel, map_gui, behead, rgetattr, rsetattr
from ovgenpy.gui.util import color2hex, Locked
from ovgenpy.outputs import IOutputConfig, FFplayOutputConfig, FFmpegOutputConfig
from ovgenpy.ovgenpy import Ovgen, Config, Arguments
from ovgenpy.triggers import CorrelationTriggerConfig, ITriggerConfig
from ovgenpy.util import perr, obj_name

APP_NAME = 'ovgenpy'
APP_DIR = Path(__file__).parent

def res(file: str) -> str:
    return str(APP_DIR / file)


def gui_main(cfg: Config, cfg_path: Optional[Path]):
    app = qw.QApplication(sys.argv)
    app.setAttribute(qc.Qt.AA_EnableHighDpiScaling)

    window = MainWindow(cfg, cfg_path)
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

    def __init__(self, cfg: Config, cfg_path: Optional[Path]):
        super().__init__()

        # Load UI.
        uic.loadUi(res('mainwindow.ui'), self)   # sets windowTitle

        # Bind UI buttons, etc. Functions block main thread, avoiding race conditions.
        self.master_audio_browse.clicked.connect(self.on_master_audio_browse)
        self.actionExit.triggered.connect(qw.QApplication.quit)
        self.actionSave.triggered.connect(self.on_action_save)
        self.actionPlay.triggered.connect(self.on_action_play)
        self.actionRender.triggered.connect(self.on_action_render)

        # Initialize ovgen-thread attribute.
        self.ovgen_thread: Locked[Optional[OvgenThread]] = Locked(None)

        # Bind config to UI.
        self._cfg_path = cfg_path
        self.load_cfg(cfg)

        self.load_title()
        self.show()

    @property
    def ever_saved(self):
        return self._cfg_path is not None

    master_audio_browse: qw.QPushButton
    # Loading mainwindow.ui changes menuBar from a getter to an attribute.
    menuBar: qw.QMenuBar
    actionExit: qw.QAction
    actionSave: qw.QAction
    actionPlay: qw.QAction
    actionRender: qw.QAction

    def on_master_audio_browse(self):
        # TODO add default file-open dir, initialized to yaml path and remembers prev
        # useless if people don't reopen old projects
        name, file_type = qw.QFileDialog.getOpenFileName(
            self, "Open master audio file", self.cfg_dir, "WAV files (*.wav)"
        )
        if name != '':
            master_audio = 'master_audio'
            self.model[master_audio] = name
            self.model.update_widget[master_audio]()

    def on_action_save(self):
        if self._cfg_path is None:
            raise NotImplementedError
        yaml.dump(self.cfg, self._cfg_path)

    def on_action_play(self):
        """ Launch ovgen and ffplay. """
        arg = self._get_args([FFplayOutputConfig()])
        error_msg = 'Cannot play, another play/render is active'
        self.play_thread(arg, error_msg)

    def on_action_render(self):
        """ Get file name. Then show a progress dialog while rendering to file. """
        video_path = os.path.join(self.cfg_dir, self.file_stem) + cli.VIDEO_NAME

        name, file_type = qw.QFileDialog.getSaveFileName(
            self, "Render to Video", video_path, "MP4 files (*.mp4);;All files (*)"
        )
        if name != '':
            dlg = OvgenProgressDialog(self)
            arg = self._get_args([FFmpegOutputConfig(name)], dlg)
            error_msg = 'Cannot render to file, another play/render is active'
            self.play_thread(arg, error_msg)

    def _get_args(self, outputs: List[IOutputConfig],
                  dlg: Optional['OvgenProgressDialog'] = None):
        arg = Arguments(
            cfg_dir=self.cfg_dir,
            outputs=outputs,
        )
        if dlg:
            arg = attr.evolve(arg,
                on_begin=dlg.on_begin,
                progress=dlg.setValue,
                is_aborted=dlg.wasCanceled,
                on_end=dlg.reset,
            )

        return arg

    def play_thread(self, arg: Arguments, error_msg: str):
        with self.ovgen_thread as t:
            if t is not None:
                self.ovgen_thread.unlock()
                qw.QMessageBox.critical(self, 'Error', error_msg)
                return

            cfg = copy_config(self.model.cfg)

            t = self.ovgen_thread.obj = OvgenThread(self, cfg, arg)
            # Assigns self.ovgen_thread.set(None) when finished.
            t.start()

    # Config models
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

    # File paths
    def load_title(self):
        self.setWindowTitle(f'{self.title} - {APP_NAME}')

    @property
    def cfg_dir(self) -> str:
        maybe_path = self._cfg_path or self.cfg.master_audio
        if maybe_path:
            return str(Path(maybe_path).resolve().parent)

        return '.'

    UNTITLED = 'Untitled'

    @property
    def title(self) -> str:
        if self._cfg_path:
            return self._cfg_path.name
        return self.UNTITLED

    @property
    def file_stem(self) -> str:
        return cli.get_name(self._cfg_path or self.cfg.master_audio)

    @property
    def cfg(self):
        return self.model.cfg


class OvgenThread(qc.QThread):
    def __init__(self, parent: MainWindow, cfg: Config, arg: Arguments):
        qc.QThread.__init__(self)

        def run() -> None:
            Ovgen(cfg, arg).play()
        self.run = run

        def finished():
            parent.ovgen_thread.set(None)
        self.finished.connect(finished)


class OvgenProgressDialog(qw.QProgressDialog):
    def __init__(self, parent: Optional[qw.QWidget]):
        super().__init__(parent)

        # If set to 0, the dialog is always shown as soon as any progress is set.
        self.setMinimumDuration(0)

        # Don't reset when rendering is approximately finished.
        self.setAutoReset(False)

        # Close after ovgen finishes.
        self.setAutoClose(True)

    def on_begin(self, begin_time, end_time):
        self.setRange(int(round(begin_time)), int(round(end_time)))
        # self.setValue is called by Ovgen, on the first frame.


def nrow_ncol_property(altered: str, unaltered: str) -> property:
    def get(self: 'ConfigModel'):
        val = getattr(self.cfg.layout, altered)
        if val is None:
            return 0
        else:
            return val

    def set(self: 'ConfigModel', val: int):
        perr(altered)
        if val > 0:
            setattr(self.cfg.layout, altered, val)
            setattr(self.cfg.layout, unaltered, None)
            self.update_widget['layout__' + unaltered]()
        elif val == 0:
            setattr(self.cfg.layout, altered, None)
        else:
            raise OvgenError(f"invalid input: {altered} < 0, should never happen")

    return property(get, set)


def default_property(path: str, default):
    def getter(self: 'ConfigModel'):
        val = rgetattr(self.cfg, path)
        if val is None:
            return default
        else:
            return val

    def setter(self: 'ConfigModel', val: int):
        rsetattr(self.cfg, path, val)

    return property(getter, setter)

class ConfigModel(PresentationModel):
    cfg: Config
    combo_symbols = {}
    combo_text = {}

    def __init__(self, cfg: Config):
        """ Mutates colors for convenience. """
        super().__init__(cfg)

        for key in ['bg_color', 'init_line_color']:
            color = getattr(cfg.render, key)
            setattr(cfg.render, key, color2hex(color))

    @property
    def render_video_size(self) -> str:
        render = self.cfg.render
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

        render = self.cfg.render
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

    render__line_width = default_property('render__line_width', 1.5)

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
        """ Mutates `channels` and `line_color` for convenience. """
        super().__init__()
        self.channels = channels
        self.triggers: List[dict] = []

        line_color = 'line_color'

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
            if line_color in trigger_dict:
                trigger_dict[line_color] = color2hex(trigger_dict[line_color])

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
