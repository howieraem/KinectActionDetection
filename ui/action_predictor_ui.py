import time
import datetime
import sys
import torch
import numpy as np
from cv2 import circle, line
# from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDialog, QListWidgetItem
from PyQt5.QtGui import QImage, QPixmap, QIcon
from ._actionPredictor import Ui_PyQtKinectDemo
from ._aboutDialog import Ui_AboutDialog
from global_configs import *
from sensor.abstract import Sensor
from network.jcr import *
from utils.processing import world_coords_array_to_image
from utils.misc import load_checkpoint, load_checkpoint_jcm


__all__ = ['UiMainWindow']
torch.manual_seed(0)
np.random.seed(0)


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


class SkeletonPredictWorker(SensorWorker):
    """A thread to predict classes from the running skeleton stream."""
    def __init__(self, device: Sensor):
        super(SkeletonPredictWorker, self).__init__(device)
        self._has_interaction = False
        self._model = None
        self.results = {
            'regression': None,
            'classification': (None, None)
        }
        self.is_jcm = False
        self.framerate = FRAME_RATE
        self.enable_interaction = self._has_interaction
        self.subject_result = self.age_result = None

    def set_model(self, model: torch.nn.Module, has_interaction: bool):
        self._model = model
        self.is_jcm = isinstance(self._model, JCM)
        self._has_interaction = has_interaction
        self.enable_interaction = has_interaction
        self.results = {
            'regression': None,
            'classification': (None, None)
        }
        self.subject_result = self.age_result = None

    @property
    def skeleton_sequence(self):
        return self._skeleton_sequence

    def run(self):
        assert self._model is not None
        self._running = True
        self._model.eval()
        last_frame = np.zeros((self._dev.joints_per_person, 3))
        last_frame2 = last_frame.copy()
        with torch.no_grad():
            while self._running:
                # Read data from sensor CDT
                frame = self._dev.get_skeleton1()[:, :3]
                frame_part2 = self._dev.get_skeleton2()[:, :3]

                # If data is the same for each skeleton or there are not enough joints visable,
                # skeleton tracking has probably failed, so skip this moment
                if np.array_equal(frame, last_frame):
                    joints_in_frame = 0
                else:
                    img_coords = world_coords_array_to_image(frame, COLOR_FRAME_FOCAL_LENGTH_IN_PIXELS,
                                                             FRAME_WIDTH, FRAME_HEIGHT)
                    joints_in_frame = len(np.where((img_coords[:, 0] >= 0) & (img_coords[:, 0] <= FRAME_WIDTH) &
                                                   (img_coords[:, 1] >= 0) & (img_coords[:, 1] <= FRAME_HEIGHT))[0])
                if frame_part2.any() and not np.array_equal(last_frame2, frame_part2):
                    img_coords2 = world_coords_array_to_image(frame_part2, COLOR_FRAME_FOCAL_LENGTH_IN_PIXELS,
                                                              FRAME_WIDTH, FRAME_HEIGHT)
                    joints_in_frame2 = len(np.where((img_coords2[:, 0] >= 0) & (img_coords2[:, 0] <= FRAME_WIDTH) &
                                                    (img_coords2[:, 1] >= 0) & (img_coords2[:, 1] <= FRAME_HEIGHT))[0])
                else:
                    joints_in_frame2 = 0
                if joints_in_frame < MIN_JOINTS_IN_FRAME:    # np.array_equal(frame, last_frame):
                    if joints_in_frame2 < MIN_JOINTS_IN_FRAME:
                        time.sleep(1 / FRAME_RATE)
                        continue
                    else:
                        frame = frame_part2.copy()  # skeleton 2 is valid but skeleton 1 is not
                        frame_part2 = np.zeros((self._dev.joints_per_person, 3))
                last_frame = frame.copy()
                frame -= frame[JointTypeMS1.SPINE]
                frame = frame.ravel()
                if self._has_interaction:
                    last_frame2 = frame_part2.copy()
                    if self.enable_interaction:
                        frame_part2 -= frame_part2[JointTypeMS1.SPINE]
                        frame = np.concatenate((frame, frame_part2.ravel()))
                    else:
                        frame = np.concatenate((frame, np.zeros((self._dev.joints_per_person, 3)).ravel()))
                # Unsqueeze twice such that input is 3-dimensional for the neural network
                x = torch.as_tensor(frame, dtype=torch.float32)[None, None, ...]
                if self.is_jcm:
                    _, (s_out, s_out_s, s_out_a) = self._model(torch.autograd.Variable(x))  # VA-JCM
                    self.subject_result = torch.max(s_out_s[-1], -1)
                    self.age_result = torch.max(s_out_a[-1], -1)
                else:
                    _, s_out, r_out = self._model(torch.autograd.Variable(x))   # VA-JCR
                    self.results['regression'] = torch.clamp(r_out[-1], min=0, max=1)
                self.results['classification'] = torch.max(s_out[-1], -1)
                self.ready.emit()
                time.sleep(1 / self.framerate)


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


class UiMainWindow(QMainWindow, Ui_PyQtKinectDemo):
    """Demo GUI."""
    def __init__(self, parent=None):
        super(UiMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.tiltUpButton.clicked.connect(self.on_click_up)
        self.tiltDownButton.clicked.connect(self.on_click_down)
        self.resetButton.clicked.connect(self.on_click_reset)
        self.pauseButton.clicked.connect(self.on_click_pause)
        self.resumeButton.clicked.connect(self.on_click_resume)
        self.resumeButton.setEnabled(False)
        self.startPredictButton.setEnabled(False)
        self.stopPredictButton.setEnabled(False)
        self.actionExit.setShortcut('Esc')
        self.actionExit.setStatusTip('Close the program')
        self.actionExit.triggered.connect(self.close)
        self.actionInformation.setStatusTip('Show additional information')
        self.actionInformation.triggered.connect(self.run_about_dialog)
        self.frameRateSlider.valueChanged.connect(self.set_skeleton_framerate)
        self.actionClassNameLabel.setWordWrap(True)
        self.centralWidget.setObjectName('c')
        self.centralWidget.setStyleSheet('QWidget#c { background-color: white }')
        self.setWindowIcon(QIcon('app.ico'))
        self.uniLogoLabel.setPixmap(QPixmap.fromImage(QImage('uqLogo.png')).scaled(512,
                                                                                   self.uniLogoLabel.height(),
                                                                                   Qt.KeepAspectRatio))

        self._color_stream_worker: ColorStreamWorker = None
        self._depth_stream_worker: DepthStreamWorker = None
        self._predict_worker: SkeletonPredictWorker = None
        self._motor_worker: MotorWorker = None
        self._dev: Sensor = None
        self.skeleton1_tracked: bool = False
        self.skeleton2_tracked: bool = False
        self._prev_skeleton_array = None
        self._prev_skeleton_array2 = None

        self.datasetComboBox.activated.connect(self.reload_model)
        self.startPredictButton.clicked.connect(self.on_click_start_predict)
        self.stopPredictButton.clicked.connect(self.on_click_stop_predict)
        self.interactCheckBox.stateChanged.connect(self.set_interaction)
        self._dataset_name = None
        self._model_name = 'VA-LN-SRU'
        self._model = None
        self._action_started = False
        self._is_jcm = False
        self._min_event_prob = 0.

    def start_streaming(self, sensor: Sensor):
        """Hooks up the sensor to GUI."""
        self._dev = sensor
        if sensor.joints_per_person != SensorJointNumber.KINECT_V1:
            mb = QMessageBox()
            mb.setIcon(QMessageBox.Critical)
            mb.setWindowTitle('Sensor Error')
            mb.setText('Sensor backend used is not compatible with the trained models.')
            mb.setStandardButtons(QMessageBox.Ok)
            mb.show()
            mb.exec_()
            sys.exit(1)
        self._color_stream_worker = ColorStreamWorker(self._dev)
        self._color_stream_worker.ready.connect(self.play_color_stream)
        self._depth_stream_worker = DepthStreamWorker(self._dev)
        self._depth_stream_worker.ready.connect(self.play_depth_stream)
        self._predict_worker = SkeletonPredictWorker(self._dev)
        self._predict_worker.ready.connect(self.update_results)
        self._motor_worker = MotorWorker(self._dev)
        self._motor_worker.ready.connect(self.reenable_angle_tilt)
        self._prev_skeleton_array = np.zeros((self._dev.joints_per_person, 4), dtype=np.float32)
        self._prev_skeleton_array2 = np.zeros((self._dev.joints_per_person, 4), dtype=np.float32)
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
            time.sleep(0.1)
        else:
            self.pauseButton.setEnabled(False)
            self.tiltUpButton.setEnabled(False)
            self.tiltDownButton.setEnabled(False)
            self.startPredictButton.setEnabled(False)
            self.interactCheckBox.setEnabled(False)
        self.reload_model()

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

    def enable_prediction_control_panel(self):
        self.datasetComboBox.setEnabled(True)
        self.startPredictButton.setEnabled(True)

    def disable_prediction_control_panel(self):
        self.datasetComboBox.setDisabled(True)
        self.startPredictButton.setDisabled(True)

    @pyqtSlot(name='set_skeleton_framerate')
    def set_skeleton_framerate(self):
        new_framerate = self.frameRateSlider.value() / 200 * (30 - 6) + 6
        self._predict_worker.framerate = new_framerate
        self.frameRateValueLabel.setText('%.2f' % new_framerate)

    @pyqtSlot(name='reload_model')
    def reload_model(self):
        """Loads trained weights according to selection."""
        self.disable_prediction_control_panel()
        self._dataset_name = self.datasetComboBox.currentText()
        self.actionListWidget.clear()
        for action_name in DATASET_AVAIL_ACTIONS[self._dataset_name]:
            item = QListWidgetItem('●    %s' % action_name)
            item.setTextAlignment(Qt.AlignLeft)
            self.actionListWidget.addItem(item)
        has_interaction = True if 'PKU' in self._dataset_name else False
        self.interactCheckBox.setChecked(has_interaction)
        downsample_factor = DOWNSAMPLE_FACTORS[self._dataset_name]
        model_filename = './demo/%s_%s.tar' % (self._dataset_name, self._model_name)
        try:
            if 'JL' in model_filename.split('/')[-1]:
                self._model = load_checkpoint_jcm(model_filename,
                                                  num_classes=DATASET_CLASS_NUM[self._dataset_name],
                                                  input_dim=DATASET_INPUT_DIM[self._dataset_name],
                                                  device=torch.device('cpu'))[0]
                self.eventTitleLabel.setText('Identity Prediction')
                # self.actionStartTitleLabel.setText('')
                # self.actionEndTitleLabel.setText('')
                self.eventPredictLabel.setText('Not Available')
                self.actionStartConfidenceLabel.setText('Not Available')
                self.actionEndConfidenceLabel.setText('Not Available')
                self._is_jcm = True
                self._min_event_prob = 0.
            else:
                self._model = load_checkpoint(model_filename,
                                              num_classes=DATASET_CLASS_NUM[self._dataset_name],
                                              input_dim=DATASET_INPUT_DIM[self._dataset_name],
                                              device=torch.device('cpu'))[0]
                self.eventTitleLabel.setText('Event Prediction')
                self.actionStartTitleLabel.setText('Action Start Confidence')
                self.actionEndTitleLabel.setText('Action End Confidence')
                self.eventPredictLabel.setText('No Action')
                self.actionStartConfidenceLabel.setText('0.00 %')
                self.actionEndConfidenceLabel.setText('0.00 %')
                self._is_jcm = False
                self._min_event_prob = MIN_EVENT_PROBABILITIES[self._dataset_name]
            if self._dev.is_opened:
                self.startPredictButton.setEnabled(True)
                self.interactCheckBox.setEnabled(has_interaction)
            self._predict_worker.set_model(self._model, has_interaction)
            self.frameRateSlider.setValue(round((FRAME_RATE / downsample_factor - 6) / 0.12))
            self.set_skeleton_framerate()
        except FileNotFoundError:
            mb = QMessageBox()
            mb.setIcon(QMessageBox.Critical)
            mb.setWindowTitle('Model Error')
            mb.setText('Error opening the model. Please ensure the required model is located at \n'
                       '%s' % model_filename)
            mb.setStandardButtons(QMessageBox.Ok)
            mb.show()
            mb.exec_()
        self.datasetComboBox.setEnabled(True)

    @pyqtSlot(name='play_color_stream')
    def play_color_stream(self):
        """
        Visualizes RGB as well as skeleton data if any. The determination of valid skeleton data here is
        independent of the prediction thread.
        """
        skeleton_array = self._dev.get_skeleton1()
        ret = not np.array_equal(skeleton_array, self._prev_skeleton_array)
        skeleton_array2 = None
        if ret:
            if not self.skeleton1_tracked:
                self.skeleton1_tracked = True
            self._prev_skeleton_array = skeleton_array.copy()
            skeleton_array2 = self._dev.get_skeleton2()
            if np.array_equal(skeleton_array2, self._prev_skeleton_array2):
                skeleton_array2 = None
                if self.skeleton2_tracked:
                    self.skeleton2_tracked = False
            else:
                self._prev_skeleton_array2 = skeleton_array2.copy()
                if not self.skeleton2_tracked:
                    self.skeleton2_tracked = True
        else:
            self._dev.skeleton_available = False
            skeleton_array = None
            if self.skeleton1_tracked:
                self.skeleton1_tracked = False
            self.actionClassNameLabel.setText('Unknown')
            self.actionClassConfidenceLabel.setText('0.00 %')
            if self._is_jcm:
                self.eventPredictLabel.setText('Not Available')
                # self.actionStartConfidenceLabel.setText('Not Available')
                # self.actionEndConfidenceLabel.setText('0.00 %')
            else:
                self.eventPredictLabel.setText('No Action')
                self.actionEndConfidenceLabel.setText('0.00 %')
                self.actionStartConfidenceLabel.setText('0.00 %')
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

    @pyqtSlot(name='reenable_angle_tilt')
    def reenable_angle_tilt(self):
        self.tiltUpButton.setDisabled(False)
        self.tiltDownButton.setDisabled(False)

    @pyqtSlot(name='on_click_reset')
    def on_click_reset(self):
        self.on_click_stop_predict()
        self.resetButton.setDisabled(True)
        self.pauseButton.setDisabled(True)
        self.tiltUpButton.setDisabled(True)
        self.tiltDownButton.setDisabled(True)
        self._color_stream_worker.stop()
        self._depth_stream_worker.stop()
        self.reload_model()
        try:
            self._dev.reset()
            if self._dev.is_opened:
                self.resumeButton.setDisabled(True)
                self.pauseButton.setDisabled(False)
                self.tiltUpButton.setDisabled(False)
                self.tiltDownButton.setDisabled(False)
                self._color_stream_worker.start()
                self._depth_stream_worker.start()
                self.startPredictButton.setDisabled(False)
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
                           skeleton_array2: np.ndarray = None):
        """Maps 3D skeleton joints to 2D image coordinates if any and converts numpy image array to QPixmap."""
        if skeleton_array is not None:
            img_coords = world_coords_array_to_image(skeleton_array[:, :3], COLOR_FRAME_FOCAL_LENGTH_IN_PIXELS,
                                                     FRAME_WIDTH, FRAME_HEIGHT)
            for img_coord in img_coords:
                circle(frame_array, tuple(img_coord), 4, (255, 255, 0), thickness=2)
            for joint_pair in SKELETON_EDGES:
                idx1, idx2 = joint_pair
                line(frame_array, tuple(img_coords[idx1]), tuple(img_coords[idx2]), (255, 255, 0), thickness=1)
        if skeleton_array2 is not None:
            img_coords2 = world_coords_array_to_image(skeleton_array2[:, :3], COLOR_FRAME_FOCAL_LENGTH_IN_PIXELS,
                                                      FRAME_WIDTH, FRAME_HEIGHT)
            for img_coord2 in img_coords2:
                circle(frame_array, tuple(img_coord2), 4, (0, 255, 255), thickness=2)
            for joint_pair in SKELETON_EDGES:
                idx1, idx2 = joint_pair
                line(frame_array, tuple(img_coords2[idx1]), tuple(img_coords2[idx2]), (0, 255, 255), thickness=1)
        return QPixmap.fromImage(QImage(frame_array, FRAME_WIDTH, FRAME_HEIGHT, format_flag))

    @pyqtSlot(name='on_click_pause')
    def on_click_pause(self):
        self.on_click_stop_predict()
        if self._dev.is_opened:
            self._color_stream_worker.stop()
            self._depth_stream_worker.stop()
        self.pauseButton.setDisabled(True)
        self.resumeButton.setDisabled(False)

    @pyqtSlot(name='on_click_resume')
    def on_click_resume(self):
        if self._dev.is_opened:
            self._color_stream_worker.start()
            self._depth_stream_worker.start()
            time.sleep(0.1)
        self.resumeButton.setDisabled(True)
        self.pauseButton.setDisabled(False)

    @pyqtSlot(name='on_click_start_predict')
    def on_click_start_predict(self):
        self._predict_worker.start()
        self.stopPredictButton.setEnabled(True)
        self.disable_prediction_control_panel()

    @pyqtSlot(name='on_click_stop_predict')
    def on_click_stop_predict(self):
        self._predict_worker.stop()
        self.stopPredictButton.setEnabled(False)
        self.enable_prediction_control_panel()

    @pyqtSlot(name='run_about_dialog')
    def run_about_dialog(self):
        dialog = QDialog()
        dialog.ui = Ui_AboutDialog()
        dialog.ui.setupUi(dialog)
        dialog.show()
        dialog.exec_()

    @pyqtSlot(name='update_results')
    def update_results(self):
        if self.skeleton1_tracked:
            action_class_probability, action_class_index = self._predict_worker.results['classification']
            label_name = DATASET_LABEL_STRINGS[self._dataset_name][action_class_index].capitalize()
            regress_results = self._predict_worker.results['regression']
            action_confidence = action_class_probability.item()
            if regress_results is None:     # JCM
                _, subject_class_index = self._predict_worker.subject_result
                # age_class_prob, age_class_index = self._predict_worker.age_result
                self.eventPredictLabel.setText(str(subject_class_index.item()))
                # self.actionStartConfidenceLabel.setText(AGE_LABEL_STRING[age_class_index.item()])
                # self.actionEndConfidenceLabel.setText('%.2f %%' % (age_class_prob.item() * 100))
                if action_confidence < 0.8:
                    label_name = 'Unknown'
            else:
                start_confidences, end_confidences = regress_results.transpose(0, 1)
                start_confidence, start_class = torch.max(start_confidences, 0)
                end_confidence, end_class = torch.max(end_confidences, 0)
                self.actionStartConfidenceLabel.setText('%.2f %%' % (start_confidence.item() * 100))
                self.actionEndConfidenceLabel.setText('%.2f %%' % (end_confidence.item() * 100))
                if start_confidence >= self._min_event_prob and start_confidence > end_confidence:
                    self.eventPredictLabel.setText('Action Starting')
                    if label_name == 'Unknown':
                        label_name = DATASET_LABEL_STRINGS[self._dataset_name][start_class.item()].capitalize()
                elif end_confidence >= self._min_event_prob and end_confidence > start_confidence:
                    self.eventPredictLabel.setText('Action Ending')
                    if label_name == 'Unknown':
                        label_name = DATASET_LABEL_STRINGS[self._dataset_name][end_class.item()].capitalize()
                elif label_name != 'Unknown':
                    self.eventPredictLabel.setText('Action Ongoing')
                elif label_name == 'Unknown':
                    self.eventPredictLabel.setText('No Action')
            self.actionClassNameLabel.setText(label_name)
            self.actionClassConfidenceLabel.setText('%.2f %%' % (action_confidence * 100))

    @pyqtSlot(name='set_interaction')
    def set_interaction(self):
        self._predict_worker.enable_interaction = self.interactCheckBox.isChecked()
