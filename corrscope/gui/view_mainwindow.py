# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from corrscope.gui.view_stack import (
    LayoutStack,
    set_layout,
    central_widget,
    append_widget,
    add_row,
    add_tab,
    set_attr_objectName,
    append_stretch,
    add_grid_col,
    Both,
    set_menu_bar,
    append_menu,
    add_toolbar,
)

NBSP = "\xa0"


class HLine(QFrame):
    def __init__(self, parent):
        super(HLine, self).__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class VLine(QFrame):
    def __init__(self, parent):
        super(VLine, self).__init__(parent)
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)


def fixed_size_policy():
    return QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


# noinspection PyAttributeOutsideInit
class MainWindow(QWidget):
    @staticmethod
    def tr(*args, **kwargs) -> str:
        """Only at runtime, not at pylupdate5 time."""
        # noinspection PyCallByClass,PyTypeChecker
        return QCoreApplication.translate("MainWindow", *args, **kwargs)

    left_tabs: QTabWidget

    def setupUi(self, MainWindow: QMainWindow):
        MainWindow.resize(1160, 0)

        s = LayoutStack(MainWindow)

        # Window contents
        with central_widget(s, QWidget) as self.centralWidget:
            horizontalLayout = set_layout(s, QHBoxLayout)

            # Left-hand config tabs
            with append_widget(s, QTabWidget) as self.left_tabs:
                self.tabGeneral = self.add_general_tab(s)
                self.tabStereo = self.add_stereo_tab(s)
                self.tabPerf = self.add_performance_tab(s)

            # Right-hand channel list
            with append_widget(s, QVBoxLayout) as self.audioColumn:

                # Top bar (master audio, trigger)
                self.add_top_bar(s)

                # Channel list (group box)
                self.channelsGroup = self.add_channels_list(s)

            # Right-hand channel list expands to fill space.
            horizontalLayout.setStretch(1, 1)

        self.add_actions(s, MainWindow)

        # Creates references to labels
        set_attr_objectName(self, s)

        # Initializes labels by reference
        self.retranslateUi(MainWindow)

        # Depends on objectName
        QMetaObject.connectSlotsByName(MainWindow)

    def add_general_tab(self, s: LayoutStack) -> QWidget:
        tr = self.tr
        with self._add_tab(s, tr("&General")) as tab:
            set_layout(s, QVBoxLayout)

            # Global group
            with append_widget(s, QGroupBox) as self.optionGlobal:
                set_layout(s, QFormLayout)

                with add_row(s, "", BoundSpinBox) as self.fps:
                    self.fps.setMinimum(1)
                    self.fps.setMaximum(999)
                    self.fps.setSingleStep(10)

                with add_row(s, "", BoundSpinBox) as self.trigger_ms:
                    self.trigger_ms.setMinimum(5)
                    self.trigger_ms.setSingleStep(5)

                with add_row(s, "", BoundSpinBox) as self.render_ms:
                    self.render_ms.setMinimum(5)
                    self.render_ms.setSingleStep(5)

                with add_row(s, "", BoundDoubleSpinBox) as self.amplification:
                    self.amplification.setSingleStep(0.1)

                with add_row(s, "", BoundDoubleSpinBox) as self.begin_time:
                    self.begin_time.setMaximum(9999.0)

            with append_widget(s, QGroupBox) as self.optionAppearance:
                set_layout(s, QFormLayout)

                with add_row(s, "", BoundLineEdit) as self.render_resolution:
                    pass

                with add_row(s, "", BoundColorWidget) as self.render__bg_color:
                    pass

                with add_row(s, "", BoundColorWidget) as self.render__init_line_color:
                    pass

                with add_row(s, "", BoundDoubleSpinBox) as self.render__line_width:
                    self.render__line_width.setMinimum(0.5)
                    self.render__line_width.setSingleStep(0.5)

                with add_row(s, "", OptionalColorWidget) as self.render__grid_color:
                    pass

                with add_row(s, "", OptionalColorWidget) as self.render__midline_color:
                    pass

                with add_row(s, BoundCheckBox, BoundCheckBox) as (
                    self.render__v_midline,
                    self.render__h_midline,
                ):
                    pass

            with append_widget(s, QGroupBox) as self.optionLayout:
                set_layout(s, QFormLayout)

                with add_row(s, "", BoundComboBox) as self.layout__orientation:
                    pass

                with add_row(s, tr("Columns"), QHBoxLayout) as self.layoutDims:
                    with append_widget(s, BoundSpinBox) as self.layout__ncols:
                        self.layout__ncols.setSpecialValueText(NBSP)

                    with append_widget(s, QLabel) as self.layout__nrowsL:
                        pass

                    with append_widget(s, BoundSpinBox) as self.layout__nrows:
                        self.layout__nrows.setSpecialValueText(NBSP)

            append_stretch(s)

        return tab

    def add_stereo_tab(self, s: LayoutStack) -> QWidget:
        tr = self.tr
        with self._add_tab(s, tr("&Stereo")) as tab:
            set_layout(s, QVBoxLayout)

            with append_widget(s, QGroupBox) as self.optionStereo:
                set_layout(s, QFormLayout)
                with add_row(s, "", BoundComboBox) as self.trigger_stereo:
                    pass

                with add_row(s, "", BoundComboBox) as self.render_stereo:
                    pass

            with append_widget(s, QGroupBox) as self.dockStereo_2:
                set_layout(s, QFormLayout)

                with add_row(s, "", BoundComboBox) as self.layout__stereo_orientation:
                    pass

                with add_row(s, "", BoundDoubleSpinBox) as (
                    self.render__stereo_grid_opacity
                ):
                    self.render__stereo_grid_opacity.setMaximum(1.0)
                    self.render__stereo_grid_opacity.setSingleStep(0.25)

            append_stretch(s)

        return tab

    def add_performance_tab(self, s: LayoutStack) -> QWidget:
        tr = self.tr
        with self._add_tab(s, tr("&Performance")) as tab:
            set_layout(s, QVBoxLayout)

            with append_widget(s, QGroupBox) as self.perfAll:
                set_layout(s, QFormLayout)

                with add_row(s, "", BoundSpinBox) as self.trigger_subsampling:
                    self.trigger_subsampling.setMinimum(1)

                with add_row(s, "", BoundSpinBox) as self.render_subsampling:
                    self.render_subsampling.setMinimum(1)

            with append_widget(s, QGroupBox) as self.perfPreview:
                set_layout(s, QFormLayout)

                with add_row(s, "", BoundSpinBox) as self.render_subfps:
                    self.render_subfps.setMinimum(1)

                with add_row(s, "", BoundDoubleSpinBox) as self.render__res_divisor:
                    self.render__res_divisor.setMinimum(1.0)
                    self.render__res_divisor.setSingleStep(0.5)

            append_stretch(s)

        return tab

    @staticmethod
    def _add_tab(s: LayoutStack, label: str = ""):
        return add_tab(s, QWidget, label)

    def add_top_bar(self, s):
        tr = self.tr
        with append_widget(s, QHBoxLayout):
            with append_widget(s, QVBoxLayout):

                with append_widget(s, QGroupBox):
                    s.widget.setTitle(tr("FFmpeg Options"))
                    set_layout(s, QFormLayout)

                    # Master audio
                    with add_row(s, tr("Master Audio"), QHBoxLayout):
                        with append_widget(s, BoundLineEdit) as self.master_audio:
                            pass
                        with append_widget(s, QPushButton) as self.master_audio_browse:
                            pass

                    with add_row(
                        s, tr("Video Template"), BoundLineEdit
                    ) as self.ffmpeg_cli__video_template:
                        pass
                    with add_row(
                        s, tr("Audio Template"), BoundLineEdit
                    ) as self.ffmpeg_cli__audio_template:
                        pass

            # Trigger config
            with append_widget(s, QGroupBox) as self.optionTrigger:
                set_layout(s, QVBoxLayout)

                # Top row
                with append_widget(s, QGridLayout):
                    with add_grid_col(s, "", BoundComboBox) as (
                        self.trigger__edge_direction
                    ):
                        pass

                    with add_grid_col(s, "", BoundDoubleSpinBox) as (
                        self.trigger__edge_strength
                    ):
                        self.trigger__edge_strength.setMinimum(0.0)

                    with add_grid_col(s, "", BoundDoubleSpinBox) as (
                        self.trigger__responsiveness
                    ):
                        self.trigger__responsiveness.setMaximum(1.0)
                        self.trigger__responsiveness.setSingleStep(0.1)

                    with add_grid_col(s, "", BoundDoubleSpinBox) as (
                        self.trigger__buffer_falloff
                    ):
                        self.trigger__buffer_falloff.setSingleStep(0.5)

                with append_widget(s, HLine):
                    pass

                # Bottom row
                with append_widget(s, QGridLayout):
                    with add_grid_col(s, BoundCheckBox, Both) as (
                        self.trigger__pitch_tracking
                    ):
                        assert isinstance(self.trigger__pitch_tracking, QWidget)

                    with add_grid_col(
                        s, tr("Post Trigger"), TypeComboBox
                    ) as self.trigger__post_trigger:
                        pass

                    with add_grid_col(
                        s, tr("Post Trigger Radius"), BoundSpinBox
                    ) as self.trigger__post_radius:
                        pass
                        # self.trigger__post_radius: BoundSpinBox
                        # self.trigger__post_radius.setMinimum(0)

    channel_tabs: QTabWidget

    def add_channels_list(self, s):
        tr = self.tr
        with append_widget(s, QTabWidget) as out:
            self.channel_tabs = out

            # Channels list
            with self._add_tab(s, tr("Oscilloscope Channels")):
                set_layout(s, QVBoxLayout)

                # Button toolbar
                with append_widget(s, QHBoxLayout) as self.channelBar:
                    append_stretch(s)

                    with append_widget(s, ShortcutButton) as self.channelAdd:
                        pass
                    with append_widget(s, ShortcutButton) as self.channelDelete:
                        pass
                    with append_widget(s, ShortcutButton) as self.channelUp:
                        pass
                    with append_widget(s, ShortcutButton) as self.channelDown:
                        pass

                # Spreadsheet grid
                with append_widget(s, ChannelTableView) as self.channel_view:
                    pass

            # FFmpeg output config
            with self._add_tab(s, tr("FFmpeg encoding flags")):
                pass

        return out

    def add_actions(self, s: LayoutStack, MainWindow):
        tr = self.tr
        # Setup actions
        self.actionOpen = QAction(MainWindow)
        self.actionSave = QAction(MainWindow)
        self.actionNew = QAction(MainWindow)
        self.actionSaveAs = QAction(MainWindow)
        self.actionExit = QAction(MainWindow)
        self.actionPreview = QAction(MainWindow)
        self.actionRender = QAction(MainWindow)
        self.action_separate_render_dir = QAction(MainWindow)
        self.action_separate_render_dir.setCheckable(True)

        # Setup menu_bar
        assert s.widget is MainWindow
        with set_menu_bar(s) as self.menuBar:
            with append_menu(s) as self.menuFile:
                self.menuFile.addAction(self.actionNew)
                self.menuFile.addAction(self.actionOpen)
                self.menuFile.addAction(self.actionSave)
                self.menuFile.addAction(self.actionSaveAs)
                self.menuFile.addSeparator()
                self.menuFile.addAction(self.actionPreview)
                self.menuFile.addAction(self.actionRender)
                self.menuFile.addSeparator()
                self.menuFile.addAction(self.actionExit)

            with append_menu(s) as self.menuTools:
                self.menuTools.addAction(self.action_separate_render_dir)

        # Setup toolbar
        with add_toolbar(s, Qt.TopToolBarArea) as self.toolBar:
            self.toolBar.addAction(self.actionNew)
            self.toolBar.addAction(self.actionOpen)
            self.toolBar.addAction(self.actionSave)
            self.toolBar.addAction(self.actionSaveAs)
            self.toolBar.addSeparator()
            self.toolBar.addAction(self.actionPreview)
            self.toolBar.addAction(self.actionRender)

    # noinspection PyUnresolvedReferences
    def retranslateUi(self, MainWindow):
        tr = self.tr

        MainWindow.setWindowTitle(tr("MainWindow"))

        self.optionGlobal.setTitle(tr("Global"))
        self.fpsL.setText(tr("FPS"))
        self.trigger_msL.setText(tr("Trigger Width"))
        self.render_msL.setText(tr("Render Width"))
        self.amplificationL.setText(tr("Amplification"))
        self.begin_timeL.setText(tr("Begin Time"))
        self.optionAppearance.setTitle(tr("Appearance"))
        self.render_resolutionL.setText(tr("Resolution"))
        self.render_resolution.setText(tr("vs"))
        self.render__bg_colorL.setText(tr("Background"))
        self.render__init_line_colorL.setText(tr("Line Color"))
        self.render__line_widthL.setText(tr("Line Width"))
        self.render__grid_colorL.setText(tr("Grid Color"))
        self.render__midline_colorL.setText(tr("Midline Color"))
        self.render__v_midline.setText(tr("Vertical"))
        self.render__h_midline.setText(tr("Horizontal Midline"))
        self.optionLayout.setTitle(tr("Layout"))
        self.layout__orientationL.setText(tr("Orientation"))
        self.layout__nrowsL.setText(tr("Rows"))
        self.optionStereo.setTitle(tr("Stereo Enable"))
        self.trigger_stereoL.setText(tr("Trigger Stereo"))
        self.render_stereoL.setText(tr("Render Stereo"))
        self.dockStereo_2.setTitle(tr("Stereo Appearance"))
        self.layout__stereo_orientationL.setText(tr("Stereo Orientation"))
        self.render__stereo_grid_opacityL.setText(tr("Grid Opacity"))

        self.perfAll.setTitle(tr("Preview and Render"))
        self.trigger_subsamplingL.setText(tr("Trigger Subsampling"))
        self.render_subsamplingL.setText(tr("Render Subsampling"))
        self.perfPreview.setTitle(tr("Preview Only"))
        self.render_subfpsL.setText(tr("Render FPS Divisor"))
        self.render__res_divisorL.setText(tr("Resolution Divisor"))
        self.master_audio.setText(tr("/"))
        self.master_audio_browse.setText(tr("&Browse..."))
        self.optionTrigger.setTitle(tr("Trigger"))
        self.trigger__edge_strengthL.setText(tr("Edge Strength"))
        self.trigger__responsivenessL.setText(tr("Responsiveness"))
        self.trigger__buffer_falloffL.setText(tr("Buffer Falloff"))
        self.trigger__pitch_tracking.setText(tr("Pitch Tracking"))
        self.trigger__edge_directionL.setText(tr("Edge Direction"))

        # self.channelsGroup.setTitle()
        self.channelAdd.setText(tr("&Add..."))
        self.channelDelete.setText(tr("&Delete"))
        self.channelUp.setText(tr("Up"))
        self.channelDown.setText(tr("Down"))
        self.menuFile.setTitle(tr("&File"))
        self.menuTools.setTitle(tr("&Tools"))
        self.toolBar.setWindowTitle(tr("toolBar"))
        self.actionOpen.setText(tr("&Open"))
        self.actionOpen.setShortcut(tr("Ctrl+O"))
        self.actionSave.setText(tr("&Save"))
        self.actionSave.setShortcut(tr("Ctrl+S"))
        self.actionNew.setText(tr("&New"))
        self.actionNew.setShortcut(tr("Ctrl+N"))
        self.actionSaveAs.setText(tr("Save &As"))
        self.actionSaveAs.setShortcut(tr("Ctrl+Shift+S"))
        self.actionExit.setText(tr("E&xit"))
        self.actionExit.setShortcut(tr("Ctrl+Q"))
        self.actionPreview.setText(tr("&Preview"))
        self.actionPreview.setShortcut(tr("Ctrl+P"))
        self.actionRender.setText(tr("&Render to Video"))
        self.actionRender.setShortcut(tr("Ctrl+R"))
        self.action_separate_render_dir.setText(tr("&Separate Render Folder"))


from corrscope.gui.__init__ import ChannelTableView, ShortcutButton
from corrscope.gui.model_bind import (
    BoundLineEdit,
    BoundSpinBox,
    BoundDoubleSpinBox,
    BoundCheckBox,
    BoundComboBox,
    TypeComboBox,
    BoundColorWidget,
    OptionalColorWidget,
)

# Delete unbound widgets, so they cannot accidentally be used.
del QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox
