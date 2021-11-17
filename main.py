from PyQt5 import QtWidgets
from ui.main_window import UiMainWindow
from sensor.kinect import KinectNI
import sys


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = UiMainWindow(None)
    ui.show()
    camera = KinectNI()
    ui.start_streaming(camera)
    ret = app.exec_()
    camera.close()
    sys.exit(ret)
