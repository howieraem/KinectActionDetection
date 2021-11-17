from pykinect import nui
from openni import openni2, nite2, utils
from openni import _openni2 as c_api
import numpy as np
import cv2
import _thread
from .abstract import Sensor
from global_configs import *


__all__ = ['KinectMS1', 'KinectNI']

np.set_printoptions(suppress=True)


class KinectMS1(Sensor):
    """Kinect CDT with Microsoft backends."""
    def __init__(self):
        super(KinectMS1, self).__init__()
        self._screen_lock = _thread.allocate()
        self.joints_per_person = SensorJointNumber.KINECT_V1

    def initialize_device(self):
        self._device = nui.Runtime()
        self._motor = self._device
        self._user_tracker = self._device.skeleton_engine
        self._user_tracker.enabled = True
        self._color_stream = self._device.video_stream
        self._depth_stream = self._device.depth_stream
        self.set_color_stream()
        self.set_depth_stream()
        self.handle_color_frame()
        self.handle_depth_frame()
        self.handle_skeleton_frame()
        self.is_opened = True

    def set_color_stream(self):
        self._color_stream.open(nui.ImageStreamType.Video, 2,
                                nui.ImageResolution.Resolution640x480,
                                nui.ImageType.Color)

    def set_depth_stream(self):
        self._depth_stream.open(nui.ImageStreamType.Depth, 2,
                                nui.ImageResolution.Resolution320x240,
                                nui.ImageType.Depth)

    def handle_color_frame(self):
        self._device.video_frame_ready += self.update_color_frame

    def handle_depth_frame(self):
        self._device.depth_frame_ready += self.update_depth_frame

    def handle_skeleton_frame(self):
        self._device.skeleton_frame_ready += self.update_skeletons

    def update_color_frame(self, raw_frame):
        color_frame = np.empty((FRAME_HEIGHT, FRAME_WIDTH, 4), np.uint8)
        raw_frame.image.copy_bits(color_frame.ctypes.data)
        color_frame = cv2.resize(color_frame, dsize=(FRAME_WIDTH, FRAME_HEIGHT))
        self._color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2RGB)

    def update_depth_frame(self, raw_frame):
        depth_frame_holder = np.empty((FRAME_HEIGHT // 2, FRAME_WIDTH // 2, 1), np.uint8)
        depth_frame = (depth_frame_holder >> 3) & 4095
        depth_frame >>= 4
        raw_frame.image.copy_bits(depth_frame.ctypes.data)
        depth_frame = cv2.resize(depth_frame.squeeze(), dsize=(FRAME_WIDTH, FRAME_HEIGHT)) / 256
        self._depth_frame = depth_frame.astype(np.uint8)

    def update_skeletons(self, frame):
        skeletons = []
        for skeleton in frame.SkeletonData:
            if skeleton.eTrackingState == nui.SkeletonTrackingState.TRACKED:
                self.skeleton_available = True
                skeletons.append(skeleton)
        num_skeletons = len(skeletons)
        if not num_skeletons:
            self.skeleton_available = False
            return
        self._skeleton = skeletons[0]
        if num_skeletons > 1:
            self._skeleton2 = skeletons[1]

    def read_color_frame(self):
        return True, self._color_frame

    def read_depth_frame(self):
        return True, self._depth_frame

    def tilt_up(self):
        with self._screen_lock:
            current_angle = self.get_tilt_angle()
            self._motor.camera.elevation_angle = current_angle + 2 if current_angle < MAXIMUM_ELEVATION \
                else current_angle

    def tilt_down(self):
        with self._screen_lock:
            current_angle = self.get_tilt_angle()
            self._motor.camera.elevation_angle = current_angle - 2 if current_angle > MINIMUM_ELEVATION \
                else current_angle

    def get_tilt_angle(self):
        return self._motor.camera.get_elevation_angle()

    def get_skeleton1(self) -> np.ndarray:
        ret = np.zeros((self.joints_per_person, 4), dtype=np.float32)
        if self._skeleton is not None:
            for idx, joint in enumerate(self._skeleton.SkeletonPositions):
                x = joint.x * 1000
                y = joint.y * 1000
                z = joint.z * 1000
                w = joint.w
                ret[idx] = (x, y, z, w)
        return ret

    def get_skeleton2(self) -> np.ndarray:
        ret = np.zeros((self.joints_per_person, 4), dtype=np.float32)
        if self._skeleton2 is not None:
            for idx, joint in enumerate(self._skeleton2.SkeletonPositions):
                x = joint.x * 1000
                y = joint.y * 1000
                z = joint.z * 1000
                w = joint.w
                ret[idx] = (x, y, z, w)
        return ret

    def close(self):
        if self._device is not None:
            self._device = None
            self.is_opened = False


# deprecated
class KinectNI(Sensor):
    """Kinect CDT with PrimeSense backends. Motor uses Microsoft backend."""
    def __init__(self):
        super(KinectNI, self).__init__()
        self._screen_lock = _thread.allocate()
        self.confidence_thres = DEFAULT_CONFIDENCE_THRES
        self.joints_per_person = SensorJointNumber.OPENNI

    def initialize_device(self):
        openni2.initialize()
        nite2.initialize()
        self._device = openni2.Device.open_any()
        self._motor = nui.Runtime()
        self.set_depth_stream()
        self.set_color_stream()
        self._user_tracker = nite2.UserTracker(self._device)
        self.is_opened = True

    def set_depth_stream(self):
        self._depth_stream = self._device.create_depth_stream()
        self._depth_stream.set_video_mode(
            c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                               resolutionX=FRAME_WIDTH, resolutionY=FRAME_HEIGHT, fps=FRAME_RATE))
        self._depth_stream.start()

    def set_color_stream(self):
        self._color_stream = self._device.create_color_stream()
        self._color_stream.set_video_mode(
            c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                               resolutionX=FRAME_WIDTH, resolutionY=FRAME_HEIGHT, fps=FRAME_RATE))
        self._color_stream.start()

    def read_color_frame(self):
        if self._color_stream is not None:
            try:
                raw_frame = self._color_stream.read_frame()
                self._color_frame = np.array(raw_frame.get_buffer_as_triplet()).reshape([FRAME_HEIGHT, FRAME_WIDTH, 3])
                return True, self._color_frame
            except (utils.OpenNIError, utils.NiteError, OSError):
                return False, None

    def read_depth_frame(self):
        if self._depth_stream is not None:
            try:
                raw_data = self._depth_stream.read_frame().get_buffer_as_uint16()
                raw_frame = np.ndarray((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint16, buffer=raw_data)
                self._depth_frame = ((raw_frame + 1024) / 256).astype('uint8')
                self.skeleton_available = self.update_skeletons()
                return True, self._depth_frame
            except (utils.OpenNIError, utils.NiteError, OSError):
                return False, None

    def update_skeletons(self) -> bool:
        skeleton_frame = self._user_tracker.read_frame()
        if skeleton_frame.users:
            user = skeleton_frame.users[0]
            if user.is_new():
                self._user_tracker.start_skeleton_tracking(user.id)
            elif user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED and user.is_visible():
                self._skeleton = user.skeleton
                return True
        return False

    def tilt_up(self):
        with self._screen_lock:
            current_angle = self.get_tilt_angle()
            self._motor.camera.elevation_angle = current_angle + 2 if current_angle < MAXIMUM_ELEVATION \
                else current_angle

    def tilt_down(self):
        with self._screen_lock:
            current_angle = self.get_tilt_angle()
            self._motor.camera.elevation_angle = current_angle - 2 if current_angle > MINIMUM_ELEVATION \
                else current_angle

    def get_tilt_angle(self):
        return self._motor.camera.get_elevation_angle()

    def close(self):
        if self._device is not None:
            if self._depth_stream is not None:
                self._depth_stream.stop()
            if self._color_stream is not None:
                self._color_stream.stop()
            if self._user_tracker is not None:
                self._user_tracker.close()
        if self._motor is not None:
            self._motor.close()
        nite2.unload()
        openni2.unload()
        self.is_opened = False

    def set_confidence_thres(self, thres: float):
        self.confidence_thres = thres

    def get_skeleton1(self) -> np.ndarray:
        ret = np.zeros((self.joints_per_person, 4), dtype=np.float32)
        if self._skeleton is not None:
            for idx, joint in enumerate(self._skeleton.joints):
                ret[idx] = joint.position.x, joint.position.y, joint.position.z, joint.positionConfidence
        return ret

    def get_skeleton2(self):
        ret = np.zeros((self.joints_per_person, 4), dtype=np.float32)
        return ret
