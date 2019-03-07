# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1160, 0)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralWidget)
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setObjectName("tabWidget")
        self.tabGeneral = QtWidgets.QWidget()
        self.tabGeneral.setObjectName("tabGeneral")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tabGeneral)
        self.verticalLayout_2.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.optionGlobal = QtWidgets.QGroupBox(self.tabGeneral)
        self.optionGlobal.setObjectName("optionGlobal")
        self.formLayout = QtWidgets.QFormLayout(self.optionGlobal)
        self.formLayout.setContentsMargins(11, 11, 11, 11)
        self.formLayout.setSpacing(6)
        self.formLayout.setObjectName("formLayout")
        self.fpsL = QtWidgets.QLabel(self.optionGlobal)
        self.fpsL.setObjectName("fpsL")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.fpsL)
        self.fps = BoundSpinBox(self.optionGlobal)
        self.fps.setMinimum(1)
        self.fps.setMaximum(999)
        self.fps.setSingleStep(10)
        self.fps.setObjectName("fps")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.fps)
        self.trigger_msL = QtWidgets.QLabel(self.optionGlobal)
        self.trigger_msL.setObjectName("trigger_msL")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.trigger_msL)
        self.trigger_ms = BoundSpinBox(self.optionGlobal)
        self.trigger_ms.setMinimum(5)
        self.trigger_ms.setSingleStep(5)
        self.trigger_ms.setObjectName("trigger_ms")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.trigger_ms)
        self.render_msL = QtWidgets.QLabel(self.optionGlobal)
        self.render_msL.setObjectName("render_msL")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.render_msL)
        self.render_ms = BoundSpinBox(self.optionGlobal)
        self.render_ms.setMinimum(5)
        self.render_ms.setSingleStep(5)
        self.render_ms.setObjectName("render_ms")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.render_ms)
        self.amplificationL = QtWidgets.QLabel(self.optionGlobal)
        self.amplificationL.setObjectName("amplificationL")
        self.formLayout.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.amplificationL
        )
        self.amplification = BoundDoubleSpinBox(self.optionGlobal)
        self.amplification.setSingleStep(0.1)
        self.amplification.setObjectName("amplification")
        self.formLayout.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.amplification
        )
        self.begin_timeL = QtWidgets.QLabel(self.optionGlobal)
        self.begin_timeL.setObjectName("begin_timeL")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.begin_timeL)
        self.begin_time = BoundDoubleSpinBox(self.optionGlobal)
        self.begin_time.setMaximum(9999.0)
        self.begin_time.setObjectName("begin_time")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.begin_time)
        self.verticalLayout_2.addWidget(self.optionGlobal)
        self.optionAppear = QtWidgets.QGroupBox(self.tabGeneral)
        self.optionAppear.setObjectName("optionAppear")
        self.formLayout_4 = QtWidgets.QFormLayout(self.optionAppear)
        self.formLayout_4.setContentsMargins(11, 11, 11, 11)
        self.formLayout_4.setSpacing(6)
        self.formLayout_4.setObjectName("formLayout_4")
        self.render_resolutionL = QtWidgets.QLabel(self.optionAppear)
        self.render_resolutionL.setObjectName("render_resolutionL")
        self.formLayout_4.setWidget(
            0, QtWidgets.QFormLayout.LabelRole, self.render_resolutionL
        )
        self.render_resolution = BoundLineEdit(self.optionAppear)
        self.render_resolution.setObjectName("render_resolution")
        self.formLayout_4.setWidget(
            0, QtWidgets.QFormLayout.FieldRole, self.render_resolution
        )
        self.render__bg_colorL = QtWidgets.QLabel(self.optionAppear)
        self.render__bg_colorL.setObjectName("render__bg_colorL")
        self.formLayout_4.setWidget(
            1, QtWidgets.QFormLayout.LabelRole, self.render__bg_colorL
        )
        self.render__bg_color = BoundColorWidget(self.optionAppear)
        self.render__bg_color.setObjectName("render__bg_color")
        self.formLayout_4.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.render__bg_color
        )
        self.render__init_line_colorL = QtWidgets.QLabel(self.optionAppear)
        self.render__init_line_colorL.setObjectName("render__init_line_colorL")
        self.formLayout_4.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.render__init_line_colorL
        )
        self.render__init_line_color = BoundColorWidget(self.optionAppear)
        self.render__init_line_color.setObjectName("render__init_line_color")
        self.formLayout_4.setWidget(
            2, QtWidgets.QFormLayout.FieldRole, self.render__init_line_color
        )
        self.render__line_widthL = QtWidgets.QLabel(self.optionAppear)
        self.render__line_widthL.setObjectName("render__line_widthL")
        self.formLayout_4.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.render__line_widthL
        )
        self.render__line_width = BoundDoubleSpinBox(self.optionAppear)
        self.render__line_width.setMinimum(0.5)
        self.render__line_width.setSingleStep(0.5)
        self.render__line_width.setObjectName("render__line_width")
        self.formLayout_4.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.render__line_width
        )
        self.render__grid_colorL = QtWidgets.QLabel(self.optionAppear)
        self.render__grid_colorL.setObjectName("render__grid_colorL")
        self.formLayout_4.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.render__grid_colorL
        )
        self.render__grid_color = OptionalColorWidget(self.optionAppear)
        self.render__grid_color.setObjectName("render__grid_color")
        self.formLayout_4.setWidget(
            4, QtWidgets.QFormLayout.FieldRole, self.render__grid_color
        )
        self.render__midline_colorL = QtWidgets.QLabel(self.optionAppear)
        self.render__midline_colorL.setObjectName("render__midline_colorL")
        self.formLayout_4.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.render__midline_colorL
        )
        self.render__midline_color = OptionalColorWidget(self.optionAppear)
        self.render__midline_color.setObjectName("render__midline_color")
        self.formLayout_4.setWidget(
            5, QtWidgets.QFormLayout.FieldRole, self.render__midline_color
        )
        self.render__v_midline = BoundCheckBox(self.optionAppear)
        self.render__v_midline.setObjectName("render__v_midline")
        self.formLayout_4.setWidget(
            6, QtWidgets.QFormLayout.LabelRole, self.render__v_midline
        )
        self.render__h_midline = BoundCheckBox(self.optionAppear)
        self.render__h_midline.setObjectName("render__h_midline")
        self.formLayout_4.setWidget(
            6, QtWidgets.QFormLayout.FieldRole, self.render__h_midline
        )
        self.verticalLayout_2.addWidget(self.optionAppear)
        self.optionLayout = QtWidgets.QGroupBox(self.tabGeneral)
        self.optionLayout.setObjectName("optionLayout")
        self.formLayout_2 = QtWidgets.QFormLayout(self.optionLayout)
        self.formLayout_2.setContentsMargins(11, 11, 11, 11)
        self.formLayout_2.setSpacing(6)
        self.formLayout_2.setObjectName("formLayout_2")
        self.layout__orientationL = QtWidgets.QLabel(self.optionLayout)
        self.layout__orientationL.setObjectName("layout__orientationL")
        self.formLayout_2.setWidget(
            0, QtWidgets.QFormLayout.LabelRole, self.layout__orientationL
        )
        self.layout__orientation = BoundComboBox(self.optionLayout)
        self.layout__orientation.setObjectName("layout__orientation")
        self.formLayout_2.setWidget(
            0, QtWidgets.QFormLayout.FieldRole, self.layout__orientation
        )
        self.layout__ncolsL = QtWidgets.QLabel(self.optionLayout)
        self.layout__ncolsL.setObjectName("layout__ncolsL")
        self.formLayout_2.setWidget(
            1, QtWidgets.QFormLayout.LabelRole, self.layout__ncolsL
        )
        self.layoutDims = QtWidgets.QHBoxLayout()
        self.layoutDims.setSpacing(6)
        self.layoutDims.setObjectName("layoutDims")
        self.layout__ncols = BoundSpinBox(self.optionLayout)
        self.layout__ncols.setSpecialValueText(" ")
        self.layout__ncols.setObjectName("layout__ncols")
        self.layoutDims.addWidget(self.layout__ncols)
        self.layout__nrowsL = QtWidgets.QLabel(self.optionLayout)
        self.layout__nrowsL.setObjectName("layout__nrowsL")
        self.layoutDims.addWidget(self.layout__nrowsL)
        self.layout__nrows = BoundSpinBox(self.optionLayout)
        self.layout__nrows.setSpecialValueText(" ")
        self.layout__nrows.setObjectName("layout__nrows")
        self.layoutDims.addWidget(self.layout__nrows)
        self.formLayout_2.setLayout(1, QtWidgets.QFormLayout.FieldRole, self.layoutDims)
        self.verticalLayout_2.addWidget(self.optionLayout)
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.verticalLayout_2.addItem(spacerItem)
        self.tabWidget.addTab(self.tabGeneral, "")
        self.tabStereo = QtWidgets.QWidget()
        self.tabStereo.setObjectName("tabStereo")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tabStereo)
        self.verticalLayout_3.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.optionStereo = QtWidgets.QGroupBox(self.tabStereo)
        self.optionStereo.setObjectName("optionStereo")
        self.formLayout_8 = QtWidgets.QFormLayout(self.optionStereo)
        self.formLayout_8.setContentsMargins(11, 11, 11, 11)
        self.formLayout_8.setSpacing(6)
        self.formLayout_8.setObjectName("formLayout_8")
        self.trigger_stereoL = QtWidgets.QLabel(self.optionStereo)
        self.trigger_stereoL.setObjectName("trigger_stereoL")
        self.formLayout_8.setWidget(
            0, QtWidgets.QFormLayout.LabelRole, self.trigger_stereoL
        )
        self.trigger_stereo = BoundComboBox(self.optionStereo)
        self.trigger_stereo.setObjectName("trigger_stereo")
        self.formLayout_8.setWidget(
            0, QtWidgets.QFormLayout.FieldRole, self.trigger_stereo
        )
        self.render_stereoL = QtWidgets.QLabel(self.optionStereo)
        self.render_stereoL.setObjectName("render_stereoL")
        self.formLayout_8.setWidget(
            1, QtWidgets.QFormLayout.LabelRole, self.render_stereoL
        )
        self.render_stereo = BoundComboBox(self.optionStereo)
        self.render_stereo.setObjectName("render_stereo")
        self.formLayout_8.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.render_stereo
        )
        self.verticalLayout_3.addWidget(self.optionStereo)
        self.dockStereo_2 = QtWidgets.QGroupBox(self.tabStereo)
        self.dockStereo_2.setObjectName("dockStereo_2")
        self.formLayout_7 = QtWidgets.QFormLayout(self.dockStereo_2)
        self.formLayout_7.setContentsMargins(11, 11, 11, 11)
        self.formLayout_7.setSpacing(6)
        self.formLayout_7.setObjectName("formLayout_7")
        self.layout__stereo_orientationL = QtWidgets.QLabel(self.dockStereo_2)
        self.layout__stereo_orientationL.setObjectName("layout__stereo_orientationL")
        self.formLayout_7.setWidget(
            0, QtWidgets.QFormLayout.LabelRole, self.layout__stereo_orientationL
        )
        self.layout__stereo_orientation = BoundComboBox(self.dockStereo_2)
        self.layout__stereo_orientation.setObjectName("layout__stereo_orientation")
        self.formLayout_7.setWidget(
            0, QtWidgets.QFormLayout.FieldRole, self.layout__stereo_orientation
        )
        self.render__stereo_grid_opacityL = QtWidgets.QLabel(self.dockStereo_2)
        self.render__stereo_grid_opacityL.setObjectName("render__stereo_grid_opacityL")
        self.formLayout_7.setWidget(
            1, QtWidgets.QFormLayout.LabelRole, self.render__stereo_grid_opacityL
        )
        self.render__stereo_grid_opacity = BoundDoubleSpinBox(self.dockStereo_2)
        self.render__stereo_grid_opacity.setMaximum(1.0)
        self.render__stereo_grid_opacity.setSingleStep(0.25)
        self.render__stereo_grid_opacity.setObjectName("render__stereo_grid_opacity")
        self.formLayout_7.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.render__stereo_grid_opacity
        )
        self.verticalLayout_3.addWidget(self.dockStereo_2)
        spacerItem1 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.verticalLayout_3.addItem(spacerItem1)
        self.tabWidget.addTab(self.tabStereo, "")
        self.tabPerf = QtWidgets.QWidget()
        self.tabPerf.setObjectName("tabPerf")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tabPerf)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.perfAll = QtWidgets.QGroupBox(self.tabPerf)
        self.perfAll.setObjectName("perfAll")
        self.formLayout_5 = QtWidgets.QFormLayout(self.perfAll)
        self.formLayout_5.setContentsMargins(11, 11, 11, 11)
        self.formLayout_5.setSpacing(6)
        self.formLayout_5.setObjectName("formLayout_5")
        self.trigger_subsamplingL = QtWidgets.QLabel(self.perfAll)
        self.trigger_subsamplingL.setObjectName("trigger_subsamplingL")
        self.formLayout_5.setWidget(
            0, QtWidgets.QFormLayout.LabelRole, self.trigger_subsamplingL
        )
        self.trigger_subsampling = BoundSpinBox(self.perfAll)
        self.trigger_subsampling.setMinimum(1)
        self.trigger_subsampling.setObjectName("trigger_subsampling")
        self.formLayout_5.setWidget(
            0, QtWidgets.QFormLayout.FieldRole, self.trigger_subsampling
        )
        self.render_subsamplingL = QtWidgets.QLabel(self.perfAll)
        self.render_subsamplingL.setObjectName("render_subsamplingL")
        self.formLayout_5.setWidget(
            1, QtWidgets.QFormLayout.LabelRole, self.render_subsamplingL
        )
        self.render_subsampling = BoundSpinBox(self.perfAll)
        self.render_subsampling.setMinimum(1)
        self.render_subsampling.setObjectName("render_subsampling")
        self.formLayout_5.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.render_subsampling
        )
        self.verticalLayout.addWidget(self.perfAll)
        self.perfPreview = QtWidgets.QGroupBox(self.tabPerf)
        self.perfPreview.setObjectName("perfPreview")
        self.formLayout_3 = QtWidgets.QFormLayout(self.perfPreview)
        self.formLayout_3.setContentsMargins(11, 11, 11, 11)
        self.formLayout_3.setSpacing(6)
        self.formLayout_3.setObjectName("formLayout_3")
        self.render_subfpsL = QtWidgets.QLabel(self.perfPreview)
        self.render_subfpsL.setObjectName("render_subfpsL")
        self.formLayout_3.setWidget(
            0, QtWidgets.QFormLayout.LabelRole, self.render_subfpsL
        )
        self.render_subfps = BoundSpinBox(self.perfPreview)
        self.render_subfps.setMinimum(1)
        self.render_subfps.setObjectName("render_subfps")
        self.formLayout_3.setWidget(
            0, QtWidgets.QFormLayout.FieldRole, self.render_subfps
        )
        self.render__res_divisorL = QtWidgets.QLabel(self.perfPreview)
        self.render__res_divisorL.setObjectName("render__res_divisorL")
        self.formLayout_3.setWidget(
            1, QtWidgets.QFormLayout.LabelRole, self.render__res_divisorL
        )
        self.render__res_divisor = BoundDoubleSpinBox(self.perfPreview)
        self.render__res_divisor.setMinimum(1.0)
        self.render__res_divisor.setSingleStep(0.5)
        self.render__res_divisor.setObjectName("render__res_divisor")
        self.formLayout_3.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.render__res_divisor
        )
        self.verticalLayout.addWidget(self.perfPreview)
        spacerItem2 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.verticalLayout.addItem(spacerItem2)
        self.tabWidget.addTab(self.tabPerf, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        self.audioColumn = QtWidgets.QVBoxLayout()
        self.audioColumn.setSpacing(6)
        self.audioColumn.setObjectName("audioColumn")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.audioGroup = QtWidgets.QGroupBox(self.centralWidget)
        self.audioGroup.setObjectName("audioGroup")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.audioGroup)
        self.horizontalLayout_3.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.master_audio = BoundLineEdit(self.audioGroup)
        self.master_audio.setObjectName("master_audio")
        self.horizontalLayout_3.addWidget(self.master_audio)
        self.master_audio_browse = QtWidgets.QPushButton(self.audioGroup)
        self.master_audio_browse.setObjectName("master_audio_browse")
        self.horizontalLayout_3.addWidget(self.master_audio_browse)
        self.horizontalLayout_4.addWidget(self.audioGroup)
        self.optionAudio = QtWidgets.QGroupBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.optionAudio.sizePolicy().hasHeightForWidth())
        self.optionAudio.setSizePolicy(sizePolicy)
        self.optionAudio.setObjectName("optionAudio")
        self.gridLayout = QtWidgets.QGridLayout(self.optionAudio)
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.trigger__edge_strengthL = QtWidgets.QLabel(self.optionAudio)
        self.trigger__edge_strengthL.setObjectName("trigger__edge_strengthL")
        self.gridLayout.addWidget(self.trigger__edge_strengthL, 0, 0, 1, 1)
        self.trigger__responsivenessL = QtWidgets.QLabel(self.optionAudio)
        self.trigger__responsivenessL.setObjectName("trigger__responsivenessL")
        self.gridLayout.addWidget(self.trigger__responsivenessL, 0, 1, 1, 1)
        self.trigger__buffer_falloffL = QtWidgets.QLabel(self.optionAudio)
        self.trigger__buffer_falloffL.setObjectName("trigger__buffer_falloffL")
        self.gridLayout.addWidget(self.trigger__buffer_falloffL, 0, 2, 1, 1)
        self.trigger__edge_strength = BoundDoubleSpinBox(self.optionAudio)
        self.trigger__edge_strength.setMinimum(-99.0)
        self.trigger__edge_strength.setObjectName("trigger__edge_strength")
        self.gridLayout.addWidget(self.trigger__edge_strength, 1, 0, 1, 1)
        self.trigger__responsiveness = BoundDoubleSpinBox(self.optionAudio)
        self.trigger__responsiveness.setMaximum(1.0)
        self.trigger__responsiveness.setSingleStep(0.1)
        self.trigger__responsiveness.setObjectName("trigger__responsiveness")
        self.gridLayout.addWidget(self.trigger__responsiveness, 1, 1, 1, 1)
        self.trigger__buffer_falloff = BoundDoubleSpinBox(self.optionAudio)
        self.trigger__buffer_falloff.setSingleStep(0.5)
        self.trigger__buffer_falloff.setObjectName("trigger__buffer_falloff")
        self.gridLayout.addWidget(self.trigger__buffer_falloff, 1, 2, 1, 1)
        self.trigger__pitch_tracking = BoundCheckBox(self.optionAudio)
        self.trigger__pitch_tracking.setObjectName("trigger__pitch_tracking")
        self.gridLayout.addWidget(self.trigger__pitch_tracking, 0, 3, 2, 1)
        self.horizontalLayout_4.addWidget(self.optionAudio)
        self.audioColumn.addLayout(self.horizontalLayout_4)
        self.channelsGroup = QtWidgets.QGroupBox(self.centralWidget)
        self.channelsGroup.setObjectName("channelsGroup")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.channelsGroup)
        self.verticalLayout_4.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_4.setSpacing(6)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.channelBar = QtWidgets.QHBoxLayout()
        self.channelBar.setSpacing(6)
        self.channelBar.setObjectName("channelBar")
        spacerItem3 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.channelBar.addItem(spacerItem3)
        self.channelAdd = ShortcutButton(self.channelsGroup)
        self.channelAdd.setObjectName("channelAdd")
        self.channelBar.addWidget(self.channelAdd)
        self.channelDelete = ShortcutButton(self.channelsGroup)
        self.channelDelete.setObjectName("channelDelete")
        self.channelBar.addWidget(self.channelDelete)
        self.channelUp = ShortcutButton(self.channelsGroup)
        self.channelUp.setObjectName("channelUp")
        self.channelBar.addWidget(self.channelUp)
        self.channelDown = ShortcutButton(self.channelsGroup)
        self.channelDown.setObjectName("channelDown")
        self.channelBar.addWidget(self.channelDown)
        self.verticalLayout_4.addLayout(self.channelBar)
        self.channel_view = ChannelTableView(self.channelsGroup)
        self.channel_view.setObjectName("channel_view")
        self.verticalLayout_4.addWidget(self.channel_view)
        self.audioColumn.addWidget(self.channelsGroup)
        self.horizontalLayout.addLayout(self.audioColumn)
        self.horizontalLayout.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setObjectName("menuBar")
        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")
        self.menuTools = QtWidgets.QMenu(self.menuBar)
        self.menuTools.setObjectName("menuTools")
        MainWindow.setMenuBar(self.menuBar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionNew = QtWidgets.QAction(MainWindow)
        self.actionNew.setObjectName("actionNew")
        self.actionSaveAs = QtWidgets.QAction(MainWindow)
        self.actionSaveAs.setObjectName("actionSaveAs")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionPreview = QtWidgets.QAction(MainWindow)
        self.actionPreview.setObjectName("actionPreview")
        self.actionRender = QtWidgets.QAction(MainWindow)
        self.actionRender.setObjectName("actionRender")
        self.action_separate_render_dir = QtWidgets.QAction(MainWindow)
        self.action_separate_render_dir.setCheckable(True)
        self.action_separate_render_dir.setObjectName("action_separate_render_dir")
        self.menuFile.addAction(self.actionNew)
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSaveAs)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionPreview)
        self.menuFile.addAction(self.actionRender)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuTools.addAction(self.action_separate_render_dir)
        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuTools.menuAction())
        self.toolBar.addAction(self.actionNew)
        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addAction(self.actionSaveAs)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionPreview)
        self.toolBar.addAction(self.actionRender)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.optionGlobal.setTitle(_translate("MainWindow", "Global"))
        self.fpsL.setText(_translate("MainWindow", "FPS"))
        self.trigger_msL.setText(_translate("MainWindow", "Trigger Width"))
        self.render_msL.setText(_translate("MainWindow", "Render Width"))
        self.amplificationL.setText(_translate("MainWindow", "Amplification"))
        self.begin_timeL.setText(_translate("MainWindow", "Begin Time"))
        self.optionAppear.setTitle(_translate("MainWindow", "Appearance"))
        self.render_resolutionL.setText(_translate("MainWindow", "Resolution"))
        self.render_resolution.setText(_translate("MainWindow", "vs"))
        self.render__bg_colorL.setText(_translate("MainWindow", "Background"))
        self.render__init_line_colorL.setText(_translate("MainWindow", "Line Color"))
        self.render__line_widthL.setText(_translate("MainWindow", "Line Width"))
        self.render__grid_colorL.setText(_translate("MainWindow", "Grid Color"))
        self.render__midline_colorL.setText(_translate("MainWindow", "Midline Color"))
        self.render__v_midline.setText(_translate("MainWindow", "Vertical"))
        self.render__h_midline.setText(_translate("MainWindow", "Horizontal Midline"))
        self.optionLayout.setTitle(_translate("MainWindow", "Layout"))
        self.layout__orientationL.setText(_translate("MainWindow", "Orientation"))
        self.layout__ncolsL.setText(_translate("MainWindow", "Columns"))
        self.layout__nrowsL.setText(_translate("MainWindow", "Rows"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tabGeneral),
            _translate("MainWindow", "&General"),
        )
        self.optionStereo.setTitle(_translate("MainWindow", "Stereo Enable"))
        self.trigger_stereoL.setText(_translate("MainWindow", "Trigger Stereo"))
        self.render_stereoL.setText(_translate("MainWindow", "Render Stereo"))
        self.dockStereo_2.setTitle(_translate("MainWindow", "Stereo Appearance"))
        self.layout__stereo_orientationL.setText(
            _translate("MainWindow", "Stereo Orientation")
        )
        self.render__stereo_grid_opacityL.setText(
            _translate("MainWindow", "Grid Opacity")
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tabStereo), _translate("MainWindow", "&Stereo")
        )
        self.perfAll.setTitle(_translate("MainWindow", "Preview and Render"))
        self.trigger_subsamplingL.setText(
            _translate("MainWindow", "Trigger Subsampling")
        )
        self.render_subsamplingL.setText(_translate("MainWindow", "Render Subsampling"))
        self.perfPreview.setTitle(_translate("MainWindow", "Preview Only"))
        self.render_subfpsL.setText(_translate("MainWindow", "Render FPS Divisor"))
        self.render__res_divisorL.setText(
            _translate("MainWindow", "Resolution Divisor")
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tabPerf),
            _translate("MainWindow", "&Performance"),
        )
        self.audioGroup.setTitle(_translate("MainWindow", "Master Audio"))
        self.master_audio.setText(_translate("MainWindow", "/"))
        self.master_audio_browse.setText(_translate("MainWindow", "&Browse..."))
        self.optionAudio.setTitle(_translate("MainWindow", "Trigger"))
        self.trigger__edge_strengthL.setText(_translate("MainWindow", "Edge Strength"))
        self.trigger__responsivenessL.setText(
            _translate("MainWindow", "Responsiveness")
        )
        self.trigger__buffer_falloffL.setText(
            _translate("MainWindow", "Buffer Falloff")
        )
        self.trigger__pitch_tracking.setText(_translate("MainWindow", "Pitch Tracking"))
        self.channelsGroup.setTitle(_translate("MainWindow", "Oscilloscope Channels"))
        self.channelAdd.setText(_translate("MainWindow", "&Add..."))
        self.channelDelete.setText(_translate("MainWindow", "&Delete"))
        self.channelUp.setText(_translate("MainWindow", "Up"))
        self.channelDown.setText(_translate("MainWindow", "Down"))
        self.menuFile.setTitle(_translate("MainWindow", "&File"))
        self.menuTools.setTitle(_translate("MainWindow", "&Tools"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionOpen.setText(_translate("MainWindow", "&Open"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionSave.setText(_translate("MainWindow", "&Save"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionNew.setText(_translate("MainWindow", "&New"))
        self.actionNew.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.actionSaveAs.setText(_translate("MainWindow", "Save &As"))
        self.actionSaveAs.setShortcut(_translate("MainWindow", "Ctrl+Shift+S"))
        self.actionExit.setText(_translate("MainWindow", "E&xit"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionPreview.setText(_translate("MainWindow", "&Preview"))
        self.actionPreview.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self.actionRender.setText(_translate("MainWindow", "&Render to Video"))
        self.actionRender.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.action_separate_render_dir.setText(
            _translate("MainWindow", "&Separate Render Folder")
        )


from corrscope.gui.__init__ import ChannelTableView, ShortcutButton
from corrscope.gui.data_bind import (
    BoundCheckBox,
    BoundColorWidget,
    BoundComboBox,
    BoundDoubleSpinBox,
    BoundLineEdit,
    BoundSpinBox,
    OptionalColorWidget,
)
