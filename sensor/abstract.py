import gc
from abc import abstractmethod


class Sensor(object):
    """Parent ADT (interface) of vision-based action recognition sensor."""
    __slots__ = ['_device', '_color_frame', '_depth_frame', '_skeleton', '_skeleton2', '_color_stream',
                 '_depth_stream', '_user_tracker', '_motor', 'skeleton_available', 'is_opened', 'joints_per_person']

    def __init__(self):
        """Constructor."""
        self._device = None
        self._color_frame = None
        self._depth_frame = None
        self._skeleton = None       # skeleton of 1st person tracked
        self._skeleton2 = None      # skeleton of 2nd person tracked
        self._color_stream = None
        self._depth_stream = None
        self._user_tracker = None
        self._motor = None
        self.skeleton_available = False
        self.is_opened = False
        self.joints_per_person = 0

    def clear_skeleton1(self):
        """Clears the skeleton data of the 1st tracked person."""
        self._skeleton = None

    def clear_skeleton2(self):
        """Clears the skeleton data of the 2nd tracked person."""
        self._skeleton2 = None

    @abstractmethod
    def initialize_device(self):
        """Sets parameters such as resolution, and opens streams."""
        raise NotImplementedError('This method is only implemented by subclass')

    @abstractmethod
    def read_color_frame(self):
        """Returns a RGB frame."""
        raise NotImplementedError('This method is only implemented by subclass')

    @abstractmethod
    def read_depth_frame(self):
        """Returns a depth frame."""
        raise NotImplementedError('This method is only implemented by subclass')

    @abstractmethod
    def tilt_up(self):
        """Increases the sensor's vertical tilt angle."""
        raise NotImplementedError('This method is only implemented by subclass')

    @abstractmethod
    def tilt_down(self):
        """Decreases the sensor's vertical tilt angle."""
        raise NotImplementedError('This method is only implemented by subclass')

    @abstractmethod
    def get_tilt_angle(self):
        """Returns the sensor's currect vertical tilt angle."""
        raise NotImplementedError('This method is only implemented by subclass')

    @abstractmethod
    def close(self):
        """Closes streams and turns off the sensor."""
        raise NotImplementedError('This method is only implemented by subclass')

    @abstractmethod
    def get_skeleton1(self):
        """Returns the skeleton data of the 1st tracked person."""
        raise NotImplementedError('This method is only implemented by subclass')

    @abstractmethod
    def get_skeleton2(self):
        """Returns the skeleton data of the 2nd tracked person."""
        raise NotImplementedError('This method is only implemented by subclass')

    def reset(self):
        """Attempts to reset the sensor."""
        self.close()
        self._motor = None
        self._device = None
        self._color_frame = None
        self._depth_frame = None
        self._skeleton = None
        self._skeleton2 = None
        self._color_stream = None
        self._depth_stream = None
        self._user_tracker = None
        self.skeleton_available = False
        gc.collect()
        self.initialize_device()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
