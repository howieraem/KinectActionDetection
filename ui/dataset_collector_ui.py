import time
import datetime
import sys
import numpy as np
from cv2 import circle
from os import makedirs
# from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDialog, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QIcon
from ._datasetCollector import Ui_PyQtKinect
from ._aboutDialog import Ui_AboutDialog
from ._settingsDialog import Ui_settingsDialog
from global_configs import *
from sensor.abstract import Sensor
from utils.processing import world_coords_to_image


__all__ = ['UiMainWindow']


class SensorWorker(QThread):
    """A thread to handle sensor-related tasks such as reading frames."""
    ready = pyqtSignal()

    def __init__(self, device: Sensor):
        super(SensorWorker, self).__init__()
        self._running = False
        self._dev = device
        self._frame = None

    @property
    def is_running(self):
        return self._running

    @property
    def frame(self):
        return self._frame

    def stop(self):
        self._running = False

    def __del__(self):
        self.quit()
        self.wait()

    def run(self):
        raise NotImplementedError('This method is only implemented by subclass')


class ColorStreamWorker(SensorWorker):
    """A thread to play the RGB stream."""
    def __init__(self, device: Sensor):
        super(ColorStreamWorker, self).__init__(device)

    def run(self):
        self._running = True
        while self._running:
            time.sleep(1 / (FRAME_RATE * 0.9))
            ret, self._frame = self._dev.read_color_frame()
            if ret:
                self.ready.emit()
            else:
                sys.stderr.write(str(datetime.datetime.now()) + ': Failed to read a color frame\n')


class DepthStreamWorker(SensorWorker):
    """A thread to play the depth stream."""
    def __init__(self, device: Sensor):
        super(DepthStreamWorker, self).__init__(device)

    def run(self):
        self._running = True
        while self._running:
            time.sleep(1 / (FRAME_RATE * 0.9))
            ret, self._frame = self._dev.read_depth_frame()
            if ret:
                self.ready.emit()
            else:
                sys.stderr.write(str(datetime.datetime.now()) + ': Failed to read a depth frame\n')


class SkeletonRecordWorker(SensorWorker):
    """A thread to store skeleton frames from the running skeleton stream."""
    def __init__(self, device: Sensor):
        super(SkeletonRecordWorker, self).__init__(device)
        self._skeleton_sequence = np.array([], dtype=np.float32).reshape(0, self._dev.joints_per_person * 3)

    @property
    def skeleton_sequence(self):
        return self._skeleton_sequence

    def run(self):
        self._running = True
        self._skeleton_sequence = np.array([], dtype=np.float32).reshape(0, self._dev.joints_per_person * 3)
        last_frame = np.zeros((self._dev.joints_per_person, 3))
        while self._running:
            time.sleep(1 / (FRAME_RATE * 0.8))
            frame = self._dev.get_skeleton1()[:, :3]
            if np.array_equal(frame, last_frame):   # new frame not yet ready
                continue
            last_frame = frame
            self._skeleton_sequence = np.vstack((self._skeleton_sequence, frame.flatten()))


class MotorWorker(SensorWorker):
    """A thread to control the sensor's vertical tilt angle."""
    def __init__(self, device: Sensor, is_up: bool = True):
        super(MotorWorker, self).__init__(device)
        self._is_up = is_up

    def set_upwards(self, is_up: bool = True):
        self._is_up = is_up

    def run(self):
        self._running = True
        if self._is_up:
            self._dev.tilt_up()
        else:
            self._dev.tilt_down()
        self.ready.emit()
        self._running = False


class UiMainWindow(QMainWindow, Ui_PyQtKinect):  # uic.loadUiType('pyQtKinect.ui') returns QMainWindow, Ui_PyQtKinect
    """Dataset Collector GUI."""
    def __init__(self, parent=None):
        super(UiMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.plainTextEdit.setCursorWidth(0)
        self.tiltUpButton.clicked.connect(self.on_click_up)
        self.tiltDownButton.clicked.connect(self.on_click_down)
        self.resetButton.clicked.connect(self.on_click_reset)
        self.pauseButton.clicked.connect(self.on_click_pause)
        self.resumeButton.clicked.connect(self.on_click_resume)
        self.browseButton.clicked.connect(self.on_click_browse)
        self.saveButton.clicked.connect(self.on_click_save)
        self.discardButton.clicked.connect(self.on_click_discard)
        self.resumeButton.setEnabled(False)
        self.saveButton.setEnabled(False)
        self.discardButton.setEnabled(False)
        self.startRecordButton.setEnabled(False)
        self.stopRecordButton.setEnabled(False)
        self.actionExit.setShortcut('Esc')
        self.actionExit.setStatusTip('Close the program')
        self.actionExit.triggered.connect(self.close)
        self.actionInformation.setStatusTip('Show additional information')
        self.actionInformation.triggered.connect(self.run_about_dialog)
        self.actionSettings.setStatusTip('Set parameters')
        self.actionSettings.triggered.connect(self.run_settings_dialog)
        self.data_path = DEFAULT_DATA_PATH
        self.pathLineEdit.setText(self.data_path)
        self.pathLineEdit.setReadOnly(True)
        self.centralWidget.setObjectName('c')
        self.centralWidget.setStyleSheet('QWidget#c { background-color: white }')
        self.setWindowIcon(QIcon('app.ico'))

        self._color_stream_worker: ColorStreamWorker = None
        self._depth_stream_worker: DepthStreamWorker = None
        self._skeleton_record_worker: SkeletonRecordWorker = None
        self._motor_worker: MotorWorker = None
        self._dev: Sensor = None
        self.skeleton1_tracked: bool = False
        self.skeleton2_tracked: bool = False
        self._prev_skeleton_array = None
        self._prev_skeleton_array2 = None
        self._action_label_idx = self.actionClassComboBox.currentIndex()
        self._age_label_idx = self.ageClassComboBox.currentIndex()
        self._person_label_idx = self.personClassComboBox.currentIndex()
        self._gender_label_idx = self.genderClassComboBox.currentIndex()
        self._recorded_data: np.ndarray = None
        self.actionClassComboBox.activated.connect(self.save_action_label)
        self.ageClassComboBox.activated.connect(self.save_age_label)
        self.personClassComboBox.activated.connect(self.save_person_label)
        self.genderClassComboBox.activated.connect(self.save_gender_label)
        self.startRecordButton.clicked.connect(self.on_click_start_record)
        self.stopRecordButton.clicked.connect(self.on_click_stop_record)
        self._data_count = 0

    def start_streaming(self, sensor: Sensor):
        """Hooks up the sensor to GUI."""
        self._dev = sensor
        self._prev_skeleton_array = np.zeros((self._dev.joints_per_person, 4), dtype=np.float32)
        self._prev_skeleton_array2 = np.zeros((self._dev.joints_per_person, 4), dtype=np.float32)
        self._color_stream_worker = ColorStreamWorker(self._dev)
        self._color_stream_worker.ready.connect(self.play_color_stream)
        self._depth_stream_worker = DepthStreamWorker(self._dev)
        self._depth_stream_worker.ready.connect(self.play_depth_stream)
        self._skeleton_record_worker = SkeletonRecordWorker(self._dev)
        self._motor_worker = MotorWorker(self._dev)
        self._motor_worker.ready.connect(self.print_tilt_angle)
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
            self._color_stream_worker.start()
            self._depth_stream_worker.start()
            self.startRecordButton.setEnabled(True)
            self._recorded_data = np.array([], dtype=np.float32).reshape(0, self._dev.joints_per_person * 3)
            time.sleep(0.1)
            self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Stream started')
        else:
            self.pauseButton.setEnabled(False)
            self.tiltUpButton.setEnabled(False)
            self.tiltDownButton.setEnabled(False)

    def closeEvent(self, event):
        """Defines the GUI's behaviors when closing the window."""
        mb = QMessageBox()
        mb.setIcon(QMessageBox.Question)
        mb.setWindowTitle('Exit Question')
        mb.setText('Are you sure to quit?')
        mb.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        mb.show()
        answer = mb.exec_()
        if answer == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    @pyqtSlot(name='play_color_stream')
    def play_color_stream(self):
        """Visualizes RGB as well as skeleton data if any."""
        skeleton_array = self._dev.get_skeleton1()
        ret = not np.array_equal(skeleton_array, self._prev_skeleton_array)
        skeleton_array2 = None
        if ret:
            if not self.skeleton1_tracked:
                self.skeleton1_tracked = True
                self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Tracking the 1st skeleton...')
            self._prev_skeleton_array = skeleton_array.copy()
            skeleton_array2 = self._dev.get_skeleton2()
            if np.array_equal(skeleton_array2, self._prev_skeleton_array2):
                skeleton_array2 = None
                if self.skeleton2_tracked:
                    self.skeleton2_tracked = False
                    self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': The 2nd skeleton not tracked')
            else:
                if not self.skeleton2_tracked:
                    self.skeleton2_tracked = True
                    self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) +
                                                       ': Tracking the 2nd skeleton...')
        else:
            self._dev.skeleton_available = False
            skeleton_array = None
            if self.skeleton1_tracked:
                self.skeleton1_tracked = False
                self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Skeletons not tracked')
        self.rgbLabel.setPixmap(self.convert_to_qpixmap(self._color_stream_worker.frame,
                                                        QImage.Format_RGB888,
                                                        skeleton_array=skeleton_array,
                                                        skeleton_array2=skeleton_array2))

    @pyqtSlot(name='play_depth_stream')
    def play_depth_stream(self):
        self.depthLabel.setPixmap(self.convert_to_qpixmap(self._depth_stream_worker.frame,
                                                          QImage.Format_Grayscale8))

    @pyqtSlot(name='on_click_up')
    def on_click_up(self):
        self.tiltUpButton.setDisabled(True)
        self.tiltDownButton.setDisabled(True)
        self._motor_worker.set_upwards(True)
        self._motor_worker.start()

    @pyqtSlot(name='on_click_down')
    def on_click_down(self):
        self.tiltUpButton.setDisabled(True)
        self.tiltDownButton.setDisabled(True)
        self._motor_worker.set_upwards(False)
        self._motor_worker.start()

    @pyqtSlot(name='print_tilt_angle')
    def print_tilt_angle(self):
        self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Tilt angle changed to '
                                           + str(self._dev.get_tilt_angle()) + ' deg')
        self.tiltUpButton.setDisabled(False)
        self.tiltDownButton.setDisabled(False)

    @pyqtSlot(name='on_click_reset')
    def on_click_reset(self):
        self.resetButton.setDisabled(True)  # Avoid interrupting reset
        self.pauseButton.setDisabled(True)
        self.tiltUpButton.setDisabled(True)
        self.tiltDownButton.setDisabled(True)
        self._color_stream_worker.stop()
        self._depth_stream_worker.stop()
        try:
            self._dev.reset()
            if self._dev.is_opened:
                self.resumeButton.setDisabled(True)
                self.pauseButton.setDisabled(False)
                self.tiltUpButton.setDisabled(False)
                self.tiltDownButton.setDisabled(False)
                self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Stream restarted')
                self._color_stream_worker.start()
                self._depth_stream_worker.start()
                self.startRecordButton.setDisabled(False)
                time.sleep(0.1)
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
    def convert_to_qpixmap(frame_array: np.ndarray,
                           format_flag: QImage.Format,
                           skeleton_array: np.ndarray = None,
                           min_joint_confidence: float = DEFAULT_CONFIDENCE_THRES,
                           skeleton_array2: np.ndarray = None):
        """Maps 3D skeleton joints to 2D image coordinates if any and converts numpy image array to QPixmap."""
        if skeleton_array is not None:
            for joint in skeleton_array:
                if (joint[3] < min_joint_confidence) or (joint[2] == 0):
                    continue
                image_coords = world_coords_to_image(joint[:3], COLOR_FRAME_FOCAL_LENGTH_IN_PIXELS,
                                                     FRAME_WIDTH, FRAME_HEIGHT)
                circle(frame_array, image_coords, 10, (255, 255, 0), thickness=4)
        if skeleton_array2 is not None:
            for joint in skeleton_array2:
                if (joint[3] < min_joint_confidence) or (joint[2] == 0):
                    continue
                image_coords = world_coords_to_image(joint[:3], COLOR_FRAME_FOCAL_LENGTH_IN_PIXELS,
                                                     FRAME_WIDTH, FRAME_HEIGHT)
                circle(frame_array, image_coords, 10, (0, 255, 255), thickness=4)
        return QPixmap.fromImage(QImage(frame_array, FRAME_WIDTH, FRAME_HEIGHT, format_flag))

    @pyqtSlot(name='on_click_pause')
    def on_click_pause(self):
        if self._dev.is_opened:
            self._color_stream_worker.stop()
            self._depth_stream_worker.stop()
        self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Stream paused')
        self.pauseButton.setDisabled(True)
        self.resumeButton.setDisabled(False)

    @pyqtSlot(name='on_click_resume')
    def on_click_resume(self):
        if self._dev.is_opened:
            self._color_stream_worker.start()
            self._depth_stream_worker.start()
            time.sleep(0.1)
        self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Stream resumed')
        self.resumeButton.setDisabled(True)
        self.pauseButton.setDisabled(False)

    @pyqtSlot(name='save_action_label')
    def save_action_label(self):
        self._action_label_idx = self.actionClassComboBox.currentIndex()
        self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Action class set to %s' %
                                           (self.actionClassComboBox.currentText()))

    @pyqtSlot(name='save_age_label')
    def save_age_label(self):
        self._age_label_idx = self.ageClassComboBox.currentIndex()
        self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Age class set to %s' %
                                           (self.ageClassComboBox.currentText()))

    @pyqtSlot(name='save_person_label')
    def save_person_label(self):
        self._person_label_idx = self.personClassComboBox.currentIndex()
        self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Person class set to %s' %
                                           (self.personClassComboBox.currentText()))

    @pyqtSlot(name='save_gender_label')
    def save_gender_label(self):
        self._gender_label_idx = self.genderClassComboBox.currentIndex()
        self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Gender class set to %s' %
                                           (self.genderClassComboBox.currentText()))

    @pyqtSlot(name='on_click_start_record')
    def on_click_start_record(self):
        self._skeleton_record_worker.start()
        self.startRecordButton.setDisabled(True)
        self.stopRecordButton.setDisabled(False)
        self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Recording skeleton sequence started')

    @pyqtSlot(name='on_click_stop_record')
    def on_click_stop_record(self):
        self._skeleton_record_worker.stop()
        self.stopRecordButton.setDisabled(True)
        self._recorded_data = np.copy(self._skeleton_record_worker.skeleton_sequence)
        self.saveButton.setDisabled(False)
        self.discardButton.setDisabled(False)
        self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Recording skeleton sequence stopped, '
                                           'got sequence of shape %s' % str(self._recorded_data.shape))

    @pyqtSlot(name='on_click_save')
    def on_click_save(self):
        self.saveButton.setDisabled(True)
        self.discardButton.setDisabled(True)
        subdir = self.data_path + 'S%02d/' % self._person_label_idx
        makedirs(subdir, exist_ok=True)
        filename = (subdir +
                    'S%02dA%02dAG%02dG%02d' % (self._person_label_idx,
                                               self._action_label_idx,
                                               self._age_label_idx,
                                               self._gender_label_idx)
                    + '_DID%07d' % self._data_count
                    + '.txt')
        with open(self.data_path + filename, 'w+') as f:
            f.write(str(self._action_label_idx) + '\n')
            f.write(str(self._age_label_idx) + '\n')
            f.write(str(self._person_label_idx) + '\n')
            f.write(str(self._gender_label_idx) + '\n')
            for frame in self._recorded_data:
                for val in frame:
                    f.write(str(val) + ' ')
                f.write('\n')
        self._data_count += 1
        self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Skeleton sequence saved as %s' %
                                           (self.data_path + filename))
        self.startRecordButton.setDisabled(False)

    @pyqtSlot(name='on_click_discard')
    def on_click_discard(self):
        self.saveButton.setDisabled(True)
        self.discardButton.setDisabled(True)
        self.startRecordButton.setDisabled(False)
        self._recorded_data = np.array([], dtype=np.float32).reshape(0, self._dev.joints_per_person * 3)
        self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Skeleton sequence discarded')

    @pyqtSlot(name='on_click_browse')
    def on_click_browse(self):
        directory = QFileDialog.getExistingDirectory(None, 'Select a folder:', self.data_path, QFileDialog.ShowDirsOnly)
        if len(directory) != 0:
            self.data_path = directory + '/'
            self.pathLineEdit.setText(self.data_path)
            self.plainTextEdit.appendPlainText(str(datetime.datetime.now()) + ': Data path changed to %s' %
                                               self.data_path)

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
