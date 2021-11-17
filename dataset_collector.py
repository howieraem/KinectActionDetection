"""Runs the skeleton dataset collector GUI."""
import sys
from PyQt5 import QtWidgets
from ui.dataset_collector_ui import UiMainWindow
from sensor.kinect import KinectMS1


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = UiMainWindow(None)
    ui.show()
    camera = KinectMS1()
    ui.start_streaming(camera)
    ret = app.exec_()
    camera.close()
    sys.exit(ret)
