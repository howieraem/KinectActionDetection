# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/settingsDialog.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_settingsDialog(object):
    def setupUi(self, settingsDialog):
        settingsDialog.setObjectName("settingsDialog")
        settingsDialog.resize(432, 139)
        self.verticalLayout = QtWidgets.QVBoxLayout(settingsDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalWidget = QtWidgets.QWidget(settingsDialog)
        self.verticalWidget.setObjectName("verticalWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.setSkeletonConfidenceWidget = QtWidgets.QWidget(self.verticalWidget)
        self.setSkeletonConfidenceWidget.setObjectName("setSkeletonConfidenceWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.setSkeletonConfidenceWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.setSkeletonConfidenceLabel = QtWidgets.QLabel(self.setSkeletonConfidenceWidget)
        self.setSkeletonConfidenceLabel.setObjectName("setSkeletonConfidenceLabel")
        self.horizontalLayout.addWidget(self.setSkeletonConfidenceLabel)
        self.setSkeletonConfidenceLineEdit = QtWidgets.QLineEdit(self.setSkeletonConfidenceWidget)
        self.setSkeletonConfidenceLineEdit.setObjectName("setSkeletonConfidenceLineEdit")
        self.horizontalLayout.addWidget(self.setSkeletonConfidenceLineEdit)
        self.verticalLayout_2.addWidget(self.setSkeletonConfidenceWidget)
        self.verticalLayout.addWidget(self.verticalWidget)
        self.buttonBox = QtWidgets.QDialogButtonBox(settingsDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(settingsDialog)
        self.buttonBox.accepted.connect(settingsDialog.accept)
        self.buttonBox.rejected.connect(settingsDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(settingsDialog)

    def retranslateUi(self, settingsDialog):
        _translate = QtCore.QCoreApplication.translate
        settingsDialog.setWindowTitle(_translate("settingsDialog", "Dialog"))
        self.setSkeletonConfidenceLabel.setText(_translate("settingsDialog", "Skeleton Confidence Threshold (0-1)"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    settingsDialog = QtWidgets.QDialog()
    ui = Ui_settingsDialog()
    ui.setupUi(settingsDialog)
    settingsDialog.show()
    sys.exit(app.exec_())

