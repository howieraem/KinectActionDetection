import datetime
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDialog
from PyQt5.QtGui import QImage, QPixmap
from ui.pyQtKinect import Ui_PyQtKinect
from ui.aboutDialog import Ui_AboutDialog
from ui.settingsDialog import Ui_settingsDialog
# from PyQt5 import uic
from sensor.abstract import Sensor


FRAME_WIDTH = 640
FRAME_HEIGHT = 480


class UiMainWindow(QMainWindow, Ui_PyQtKinect):  # uic.loadUiType('pyQtKinect.ui') returns QMainWindow, Ui_PyQtKinect
    def __init__(self, parent=None):
        super(UiMainWindow, self).__init__(parent)
        self.setupUi(self)
        self._color_stream_timer: QTimer = None
        self._depth_stream_timer: QTimer = None
        self._dev: Sensor = None
        self.skeleton_tracked: bool = False
        self.stream_live: bool = True
        self.plainTextEdit.setCursorWidth(0)
        self.tiltUpButton.clicked.connect(self.on_click_up)
        self.tiltDownButton.clicked.connect(self.on_click_down)
        self.resetButton.clicked.connect(self.on_click_reset)
        self.pauseButton.clicked.connect(self.on_click_pause)
        self.resumeButton.clicked.connect(self.on_click_resume)
        self.resumeButton.setDisabled(True)
        self.actionExit.setShortcut('Esc')
        self.actionExit.setStatusTip('Close the program')
        self.actionExit.triggered.connect(self.close)
        self.actionInformation.setStatusTip('Show additional information')
        self.actionInformation.triggered.connect(self.run_about_dialog)
        self.actionSettings.setStatusTip('Set parameters')
        self.actionSettings.triggered.connect(self.run_settings_dialog)

    def start_streaming(self, sensor: Sensor):
        self._dev = sensor
        try:
            sensor.initialize_device()
        except Exception as e:
            mb = QMessageBox()
            mb.setIcon(QMessageBox.Critical)
            mb.setWindowTitle('Sensor Error')
            mb.setText('Error opening the sensor: ' + str(e) +
                       '. Please ensure it is running and connected properly, and click on the Reset button.')
            mb.setStandardButtons(QMessageBox.Ok)
            mb.show()
            mb.exec_()
        if self._dev.is_opened:
            self._color_stream_timer = QTimer()
            self._color_stream_timer.setTimerType(Qt.PreciseTimer)
            self._color_stream_timer.start(34)
            self._color_stream_timer.timeout.connect(self.play_color_stream)
            self._depth_stream_timer = QTimer()
            self._depth_stream_timer.setTimerType(Qt.PreciseTimer)
            self._depth_stream_timer.start(50)
            self._depth_stream_timer.timeout.connect(self.play_depth_stream)
            self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Stream started')
        else:
            self.pauseButton.setEnabled(False)
            self.tiltUpButton.setEnabled(False)
            self.tiltDownButton.setEnabled(False)

    def closeEvent(self, event):
        mb = QMessageBox()
        mb.setIcon(QMessageBox.Question)
        mb.setWindowTitle('Exit Question')
        mb.setText('Are you sure you want to quit?')
        mb.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        mb.show()
        answer = mb.exec_()
        if answer == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    @pyqtSlot(name='play_color_stream')
    def play_color_stream(self):
        if self._dev.is_opened and self.stream_live:
            ret1, color_frame = self._dev.read_color_frame()
            ret2 = self._dev.skeleton_available
            if ret1:
                self.rgbLabel.setPixmap(self.convert_to_qpixmap(color_frame, QImage.Format_RGB888))
            else:
                self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Failed to read a color frame')
            if ret2 and not self.skeleton_tracked:
                self.skeleton_tracked = True
                self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Tracking _skeletons...')
            elif not ret2 and self.skeleton_tracked:
                self.skeleton_tracked = False
                self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Some joints not tracked or '
                                                                                  'not confident')

    @pyqtSlot(name='play_depth_stream')
    def play_depth_stream(self):
        if self._dev.is_opened and self.stream_live:
            ret, depth_frame = self._dev.read_depth_frame()
            if ret:
                self.depthLabel.setPixmap(self.convert_to_qpixmap(depth_frame, QImage.Format_Indexed8))
            else:
                self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Failed to read a depth frame')

    @pyqtSlot(name='on_click_up')
    def on_click_up(self):
        self._dev.tilt_up()
        self.print_tilt_angle()

    @pyqtSlot(name='on_click_down')
    def on_click_down(self):
        self._dev.tilt_down()
        self.print_tilt_angle()

    def print_tilt_angle(self):
        self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Tilt angle changed to '
                                           + str(self._dev.get_tilt_angle()) + ' deg')

    @pyqtSlot(name='on_click_reset')
    def on_click_reset(self):
        self.resetButton.setDisabled(True)  # Avoid interrupting reset
        self.pauseButton.setDisabled(True)
        self.tiltUpButton.setDisabled(True)
        self.tiltDownButton.setDisabled(True)
        try:
            self._dev.reset()
            if self._dev.is_opened:
                self.stream_live = True
                self.resumeButton.setDisabled(True)
                self.pauseButton.setDisabled(False)
                self.tiltUpButton.setDisabled(False)
                self.tiltDownButton.setDisabled(False)
                self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Stream restarted')
        except Exception as e:
            mb = QMessageBox()
            mb.setIcon(QMessageBox.Critical)
            mb.setWindowTitle('Sensor Error')
            mb.setText('Error opening the sensor: ' + str(e) +
                       '. Please ensure it is running and connected properly, and click on the Reset button again.')
            mb.setStandardButtons(QMessageBox.Ok)
            mb.show()
            mb.exec_()
        finally:
            self.resetButton.setDisabled(False)

    @staticmethod
    def convert_to_qpixmap(array_frame, format_flag: QImage.Format):
        return QPixmap.fromImage(QImage(array_frame, FRAME_WIDTH, FRAME_HEIGHT, format_flag))

    @pyqtSlot(name='on_click_pause')
    def on_click_pause(self):
        self.stream_live = False
        self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Stream paused')
        self.pauseButton.setDisabled(True)
        self.resumeButton.setDisabled(False)

    @pyqtSlot(name='on_click_resume')
    def on_click_resume(self):
        self.stream_live = True
        self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Stream resumed')
        self.resumeButton.setDisabled(True)
        self.pauseButton.setDisabled(False)

    @pyqtSlot(name='run_about_dialog')
    def run_about_dialog(self):
        dialog = QDialog()
        dialog.ui = Ui_AboutDialog()
        dialog.ui.setupUi(dialog)
        dialog.show()
        dialog.exec_()

    @pyqtSlot(name='run_settings_dialog')
    def run_settings_dialog(self):
        dialog = QDialog()
        dialog.ui = Ui_settingsDialog()
        dialog.ui.setupUi(dialog)
        dialog.show()
        ret = dialog.exec_()
