# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/pyQtKinect.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PyQtKinect(object):
    def setupUi(self, PyQtKinect):
        PyQtKinect.setObjectName("PyQtKinect")
        PyQtKinect.resize(1362, 975)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PyQtKinect.sizePolicy().hasHeightForWidth())
        PyQtKinect.setSizePolicy(sizePolicy)
        self.centralWidget = QtWidgets.QWidget(PyQtKinect)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralWidget.sizePolicy().hasHeightForWidth())
        self.centralWidget.setSizePolicy(sizePolicy)
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout_9.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_9.setSpacing(6)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.streamFrame = QtWidgets.QFrame(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.streamFrame.sizePolicy().hasHeightForWidth())
        self.streamFrame.setSizePolicy(sizePolicy)
        self.streamFrame.setMinimumSize(QtCore.QSize(642, 242))
        self.streamFrame.setFrameShape(QtWidgets.QFrame.Box)
        self.streamFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.streamFrame.setObjectName("streamFrame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.streamFrame)
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.rgbWidget = QtWidgets.QWidget(self.streamFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rgbWidget.sizePolicy().hasHeightForWidth())
        self.rgbWidget.setSizePolicy(sizePolicy)
        self.rgbWidget.setObjectName("rgbWidget")
        self.verticalTopLayout = QtWidgets.QVBoxLayout(self.rgbWidget)
        self.verticalTopLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalTopLayout.setSpacing(6)
        self.verticalTopLayout.setObjectName("verticalTopLayout")
        self.rgbTitle = QtWidgets.QLabel(self.rgbWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rgbTitle.sizePolicy().hasHeightForWidth())
        self.rgbTitle.setSizePolicy(sizePolicy)
        self.rgbTitle.setMinimumSize(QtCore.QSize(0, 0))
        self.rgbTitle.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.rgbTitle.setFont(font)
        self.rgbTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.rgbTitle.setObjectName("rgbTitle")
        self.verticalTopLayout.addWidget(self.rgbTitle)
        self.rgbLabel = QtWidgets.QLabel(self.rgbWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rgbLabel.sizePolicy().hasHeightForWidth())
        self.rgbLabel.setSizePolicy(sizePolicy)
        self.rgbLabel.setMinimumSize(QtCore.QSize(640, 480))
        self.rgbLabel.setMaximumSize(QtCore.QSize(640, 480))
        self.rgbLabel.setAutoFillBackground(True)
        self.rgbLabel.setScaledContents(True)
        self.rgbLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.rgbLabel.setObjectName("rgbLabel")
        self.verticalTopLayout.addWidget(self.rgbLabel)
        self.horizontalLayout.addWidget(self.rgbWidget)
        self.depthWidget = QtWidgets.QWidget(self.streamFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.depthWidget.sizePolicy().hasHeightForWidth())
        self.depthWidget.setSizePolicy(sizePolicy)
        self.depthWidget.setObjectName("depthWidget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.depthWidget)
        self.verticalLayout_5.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_5.setSpacing(6)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.depthTitle = QtWidgets.QLabel(self.depthWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.depthTitle.sizePolicy().hasHeightForWidth())
        self.depthTitle.setSizePolicy(sizePolicy)
        self.depthTitle.setMinimumSize(QtCore.QSize(0, 0))
        self.depthTitle.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.depthTitle.setFont(font)
        self.depthTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.depthTitle.setObjectName("depthTitle")
        self.verticalLayout_5.addWidget(self.depthTitle)
        self.depthLabel = QtWidgets.QLabel(self.depthWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.depthLabel.sizePolicy().hasHeightForWidth())
        self.depthLabel.setSizePolicy(sizePolicy)
        self.depthLabel.setMinimumSize(QtCore.QSize(640, 480))
        self.depthLabel.setMaximumSize(QtCore.QSize(640, 480))
        self.depthLabel.setAutoFillBackground(True)
        self.depthLabel.setScaledContents(True)
        self.depthLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.depthLabel.setObjectName("depthLabel")
        self.verticalLayout_5.addWidget(self.depthLabel)
        self.horizontalLayout.addWidget(self.depthWidget)
        self.verticalLayout_9.addWidget(self.streamFrame)
        self.middleButtonFrame = QtWidgets.QFrame(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.middleButtonFrame.sizePolicy().hasHeightForWidth())
        self.middleButtonFrame.setSizePolicy(sizePolicy)
        self.middleButtonFrame.setFrameShape(QtWidgets.QFrame.Box)
        self.middleButtonFrame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.middleButtonFrame.setObjectName("middleButtonFrame")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.middleButtonFrame)
        self.verticalLayout_6.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_6.setSpacing(6)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label = QtWidgets.QLabel(self.middleButtonFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_6.addWidget(self.label)
        self.horizontalWidget = QtWidgets.QWidget(self.middleButtonFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalWidget.sizePolicy().hasHeightForWidth())
        self.horizontalWidget.setSizePolicy(sizePolicy)
        self.horizontalWidget.setObjectName("horizontalWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalWidget)
        self.horizontalLayout_2.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pauseButton = QtWidgets.QPushButton(self.horizontalWidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.pauseButton.setFont(font)
        self.pauseButton.setObjectName("pauseButton")
        self.horizontalLayout_2.addWidget(self.pauseButton)
        self.resumeButton = QtWidgets.QPushButton(self.horizontalWidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.resumeButton.setFont(font)
        self.resumeButton.setObjectName("resumeButton")
        self.horizontalLayout_2.addWidget(self.resumeButton)
        self.tiltUpButton = QtWidgets.QPushButton(self.horizontalWidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.tiltUpButton.setFont(font)
        self.tiltUpButton.setObjectName("tiltUpButton")
        self.horizontalLayout_2.addWidget(self.tiltUpButton)
        self.tiltDownButton = QtWidgets.QPushButton(self.horizontalWidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.tiltDownButton.setFont(font)
        self.tiltDownButton.setObjectName("tiltDownButton")
        self.horizontalLayout_2.addWidget(self.tiltDownButton)
        self.resetButton = QtWidgets.QPushButton(self.horizontalWidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.resetButton.setFont(font)
        self.resetButton.setObjectName("resetButton")
        self.horizontalLayout_2.addWidget(self.resetButton)
        self.verticalLayout_6.addWidget(self.horizontalWidget)
        self.verticalLayout_9.addWidget(self.middleButtonFrame)
        self.logFrame = QtWidgets.QFrame(self.centralWidget)
        self.logFrame.setFrameShape(QtWidgets.QFrame.Box)
        self.logFrame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.logFrame.setObjectName("logFrame")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.logFrame)
        self.verticalLayout_3.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.programLogLabel = QtWidgets.QLabel(self.logFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.programLogLabel.sizePolicy().hasHeightForWidth())
        self.programLogLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.programLogLabel.setFont(font)
        self.programLogLabel.setObjectName("programLogLabel")
        self.verticalLayout_3.addWidget(self.programLogLabel)
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.logFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plainTextEdit.sizePolicy().hasHeightForWidth())
        self.plainTextEdit.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.verticalLayout_3.addWidget(self.plainTextEdit)
        self.verticalLayout_9.addWidget(self.logFrame)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalWidget1 = QtWidgets.QWidget(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalWidget1.sizePolicy().hasHeightForWidth())
        self.horizontalWidget1.setSizePolicy(sizePolicy)
        self.horizontalWidget1.setObjectName("horizontalWidget1")
        self.bottomHorizontalLayout = QtWidgets.QHBoxLayout(self.horizontalWidget1)
        self.bottomHorizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.bottomHorizontalLayout.setSpacing(6)
        self.bottomHorizontalLayout.setObjectName("bottomHorizontalLayout")
        self.verticalLayout.addWidget(self.horizontalWidget1)
        self.verticalLayout_9.addLayout(self.verticalLayout)
        PyQtKinect.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(PyQtKinect)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1362, 21))
        self.menuBar.setObjectName("menuBar")
        self.menuMain = QtWidgets.QMenu(self.menuBar)
        self.menuMain.setObjectName("menuMain")
        PyQtKinect.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(PyQtKinect)
        self.mainToolBar.setObjectName("mainToolBar")
        PyQtKinect.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(PyQtKinect)
        self.statusBar.setObjectName("statusBar")
        PyQtKinect.setStatusBar(self.statusBar)
        self.actionSettings = QtWidgets.QAction(PyQtKinect)
        self.actionSettings.setObjectName("actionSettings")
        self.actionInformation = QtWidgets.QAction(PyQtKinect)
        self.actionInformation.setObjectName("actionInformation")
        self.actionExit = QtWidgets.QAction(PyQtKinect)
        self.actionExit.setObjectName("actionExit")
        self.menuMain.addAction(self.actionSettings)
        self.menuMain.addAction(self.actionInformation)
        self.menuMain.addSeparator()
        self.menuMain.addAction(self.actionExit)
        self.menuBar.addAction(self.menuMain.menuAction())

        self.retranslateUi(PyQtKinect)
        QtCore.QMetaObject.connectSlotsByName(PyQtKinect)

    def retranslateUi(self, PyQtKinect):
        _translate = QtCore.QCoreApplication.translate
        PyQtKinect.setWindowTitle(_translate("PyQtKinect", "PyQtKinect"))
        self.rgbTitle.setText(_translate("PyQtKinect", "RGB Stream"))
        self.rgbLabel.setText(_translate("PyQtKinect", "(RGB)"))
        self.depthTitle.setText(_translate("PyQtKinect", "Depth Stream"))
        self.depthLabel.setText(_translate("PyQtKinect", "(Depth)"))
        self.label.setText(_translate("PyQtKinect", "Sensor Control"))
        self.pauseButton.setText(_translate("PyQtKinect", "Pause Stream"))
        self.resumeButton.setText(_translate("PyQtKinect", "Resume Stream"))
        self.tiltUpButton.setText(_translate("PyQtKinect", "Tilt Up"))
        self.tiltDownButton.setText(_translate("PyQtKinect", "Tilt Down"))
        self.resetButton.setText(_translate("PyQtKinect", "Reset"))
        self.programLogLabel.setText(_translate("PyQtKinect", "Program Logs"))
        self.menuMain.setTitle(_translate("PyQtKinect", "Menu"))
        self.actionSettings.setText(_translate("PyQtKinect", "Settings"))
        self.actionInformation.setText(_translate("PyQtKinect", "About"))
        self.actionExit.setText(_translate("PyQtKinect", "Exit"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    PyQtKinect = QtWidgets.QMainWindow()
    ui = Ui_PyQtKinect()
    ui.setupUi(PyQtKinect)
    PyQtKinect.show()
    sys.exit(app.exec_())

