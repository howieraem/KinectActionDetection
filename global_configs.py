"""Constants and default settings."""
import enum
from pykinect import nui as _nui


# --------------------------------------------------------------------------------------------------------------------
# Sensor parameters
class JointTypeMS1(enum.IntEnum):
    HIP_CENTER = 0
    SPINE = 1
    SHOULDER_CENTER = 2
    HEAD = 3
    SHOULDER_LEFT = 4
    ELBOW_LEFT = 5
    WRIST_LEFT = 6
    HAND_LEFT = 7
    SHOULDER_RIGHT = 8
    ELBOW_RIGHT = 9
    WRIST_RIGHT = 10
    HAND_RIGHT = 11
    HIP_LEFT = 12
    KNEE_LEFT = 13
    ANKLE_LEFT = 14
    FOOT_LEFT = 15
    HIP_RIGHT = 16
    KNEE_RIGHT = 17
    ANKLE_RIGHT = 18
    FOOT_RIGHT = 19


class JointTypeNI(enum.IntEnum):
    HEAD = 0
    NECK = 1
    LEFT_SHOULDER = 2
    RIGHT_SHOULDER = 3
    LEFT_ELBOW = 4
    RIGHT_ELBOW = 5
    LEFT_HAND = 6
    RIGHT_HAND = 7
    TORSO = 8
    LEFT_HIP = 9
    RIGHT_HIP = 10
    LEFT_KNEE = 11
    RIGHT_KNEE = 12
    LEFT_FOOT = 13
    RIGHT_FOOT = 14


class JointTypeCAD(enum.IntEnum):
    HEAD = 0
    NECK = 1
    TORSO = 2
    LEFT_SHOULDER = 3
    LEFT_ELBOW = 4
    RIGHT_SHOULDER = 5
    RIGHT_ELBOW = 6
    LEFT_HIP = 7
    LEFT_KNEE = 8
    RIGHT_HIP = 9
    RIGHT_KNEE = 10
    LEFT_HAND = 11
    RIGHT_HAND = 12
    LEFT_FOOT = 13
    RIGHT_FOOT = 14


class JointTypeMS2(enum.IntEnum):
    HIP_CENTER = 0
    SPINE = 1
    SHOULDER_CENTER = 2
    HEAD = 3
    SHOULDER_LEFT = 4
    ELBOW_LEFT = 5
    WRIST_LEFT = 6
    HAND_LEFT = 7
    SHOULDER_RIGHT = 8
    ELBOW_RIGHT = 9
    WRIST_RIGHT = 10
    HAND_RIGHT = 11
    HIP_LEFT = 12
    KNEE_LEFT = 13
    ANKLE_LEFT = 14
    FOOT_LEFT = 15
    HIP_RIGHT = 16
    KNEE_RIGHT = 17
    ANKLE_RIGHT = 18
    FOOT_RIGHT = 19
    SPINE_SHOULDER = 20
    HAND_TIP_LEFT = 21
    THUMB_LEFT = 22
    HAND_TIP_RIGHT = 23
    THUMB_RIGHT = 24


class SensorJointNumber(enum.IntEnum):
    OPENNI = 15
    KINECT_V1 = 20
    KINECT_V2 = 25


NI_2_MS1_JOINT_IDX_CONVERT = {
    JointTypeNI.HEAD: JointTypeMS1.HEAD,
    JointTypeNI.NECK: JointTypeMS1.SHOULDER_CENTER,
    JointTypeNI.LEFT_SHOULDER: JointTypeMS1.SHOULDER_LEFT,
    JointTypeNI.RIGHT_SHOULDER: JointTypeMS1.SHOULDER_RIGHT,
    JointTypeNI.LEFT_ELBOW: JointTypeMS1.ELBOW_LEFT,
    JointTypeNI.RIGHT_ELBOW: JointTypeMS1.ELBOW_RIGHT,
    JointTypeNI.LEFT_HAND: JointTypeMS1.HAND_LEFT,
    JointTypeNI.RIGHT_HAND: JointTypeMS1.HAND_RIGHT,
    JointTypeNI.TORSO: JointTypeMS1.SPINE,
    JointTypeNI.LEFT_HIP: JointTypeMS1.HIP_LEFT,
    JointTypeNI.RIGHT_HIP: JointTypeMS1.HAND_RIGHT,
    JointTypeNI.LEFT_KNEE: JointTypeMS1.KNEE_LEFT,
    JointTypeNI.RIGHT_KNEE: JointTypeMS1.KNEE_RIGHT,
    JointTypeNI.LEFT_FOOT: JointTypeMS1.FOOT_LEFT,
    JointTypeNI.RIGHT_FOOT: JointTypeMS1.FOOT_RIGHT
}

CAD_2_MS1_JOINT_IDX_CONVERT = {
    JointTypeCAD.HEAD: JointTypeMS1.HEAD,
    JointTypeCAD.NECK: JointTypeMS1.SHOULDER_CENTER,
    JointTypeCAD.LEFT_SHOULDER: JointTypeMS1.SHOULDER_LEFT,
    JointTypeCAD.RIGHT_SHOULDER: JointTypeMS1.SHOULDER_RIGHT,
    JointTypeCAD.LEFT_ELBOW: JointTypeMS1.ELBOW_LEFT,
    JointTypeCAD.RIGHT_ELBOW: JointTypeMS1.ELBOW_RIGHT,
    JointTypeCAD.LEFT_HAND: JointTypeMS1.HAND_LEFT,
    JointTypeCAD.RIGHT_HAND: JointTypeMS1.HAND_RIGHT,
    JointTypeCAD.TORSO: JointTypeMS1.SPINE,
    JointTypeCAD.LEFT_HIP: JointTypeMS1.HIP_LEFT,
    JointTypeCAD.RIGHT_HIP: JointTypeMS1.HAND_RIGHT,
    JointTypeCAD.LEFT_KNEE: JointTypeMS1.KNEE_LEFT,
    JointTypeCAD.RIGHT_KNEE: JointTypeMS1.KNEE_RIGHT,
    JointTypeCAD.LEFT_FOOT: JointTypeMS1.FOOT_LEFT,
    JointTypeCAD.RIGHT_FOOT: JointTypeMS1.FOOT_RIGHT
}

TREE_TRAVERSAL_IDX = (JointTypeMS1.HEAD,
                      JointTypeMS1.SHOULDER_CENTER,
                      JointTypeMS1.SHOULDER_RIGHT,
                      JointTypeMS1.ELBOW_RIGHT,
                      JointTypeMS1.WRIST_RIGHT,
                      JointTypeMS1.HAND_RIGHT,
                      JointTypeMS1.WRIST_RIGHT,
                      JointTypeMS1.ELBOW_RIGHT,
                      JointTypeMS1.SHOULDER_RIGHT,
                      JointTypeMS1.SHOULDER_CENTER,
                      JointTypeMS1.SPINE,
                      JointTypeMS1.HIP_CENTER,
                      JointTypeMS1.HIP_RIGHT,
                      JointTypeMS1.KNEE_RIGHT,
                      JointTypeMS1.ANKLE_RIGHT,
                      JointTypeMS1.FOOT_RIGHT,
                      JointTypeMS1.ANKLE_RIGHT,
                      JointTypeMS1.KNEE_RIGHT,
                      JointTypeMS1.HIP_RIGHT,
                      JointTypeMS1.HIP_CENTER,
                      JointTypeMS1.HIP_LEFT,
                      JointTypeMS1.KNEE_LEFT,
                      JointTypeMS1.ANKLE_LEFT,
                      JointTypeMS1.FOOT_LEFT,
                      JointTypeMS1.ANKLE_LEFT,
                      JointTypeMS1.KNEE_LEFT,
                      JointTypeMS1.HIP_LEFT,
                      JointTypeMS1.HIP_CENTER,
                      JointTypeMS1.SPINE,
                      JointTypeMS1.SHOULDER_CENTER,
                      JointTypeMS1.SHOULDER_LEFT,
                      JointTypeMS1.ELBOW_LEFT,
                      JointTypeMS1.WRIST_LEFT,
                      JointTypeMS1.HAND_LEFT,
                      JointTypeMS1.WRIST_LEFT,
                      JointTypeMS1.ELBOW_LEFT,
                      JointTypeMS1.SHOULDER_LEFT,
                      JointTypeMS1.SHOULDER_CENTER)


SKELETON_EDGES = (
    [JointTypeMS1.HEAD, JointTypeMS1.SHOULDER_CENTER],
    [JointTypeMS1.SHOULDER_CENTER, JointTypeMS1.SHOULDER_RIGHT],
    [JointTypeMS1.SHOULDER_RIGHT, JointTypeMS1.ELBOW_RIGHT],
    [JointTypeMS1.ELBOW_RIGHT, JointTypeMS1.WRIST_RIGHT],
    [JointTypeMS1.WRIST_RIGHT, JointTypeMS1.HAND_RIGHT],
    [JointTypeMS1.SHOULDER_CENTER, JointTypeMS1.SHOULDER_LEFT],
    [JointTypeMS1.SHOULDER_LEFT, JointTypeMS1.ELBOW_LEFT],
    [JointTypeMS1.ELBOW_LEFT, JointTypeMS1.WRIST_LEFT],
    [JointTypeMS1.WRIST_LEFT, JointTypeMS1.HAND_LEFT],
    [JointTypeMS1.SHOULDER_CENTER, JointTypeMS1.SPINE],
    [JointTypeMS1.SPINE, JointTypeMS1.HIP_CENTER],
    [JointTypeMS1.HIP_CENTER, JointTypeMS1.HIP_RIGHT],
    [JointTypeMS1.HIP_RIGHT, JointTypeMS1.KNEE_RIGHT],
    [JointTypeMS1.KNEE_RIGHT, JointTypeMS1.ANKLE_RIGHT],
    [JointTypeMS1.ANKLE_RIGHT, JointTypeMS1.FOOT_RIGHT],
    [JointTypeMS1.HIP_CENTER, JointTypeMS1.HIP_LEFT],
    [JointTypeMS1.HIP_LEFT, JointTypeMS1.KNEE_LEFT],
    [JointTypeMS1.KNEE_LEFT, JointTypeMS1.ANKLE_LEFT],
    [JointTypeMS1.ANKLE_LEFT, JointTypeMS1.FOOT_LEFT]
)


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 30
DEFAULT_CONFIDENCE_THRES = 0.3
COLOR_FRAME_FOCAL_LENGTH_IN_PIXELS = _nui._NUI_CAMERA_COLOR_NOMINAL_FOCAL_LENGTH_IN_PIXELS
DEPTH_FRAME_FOCAL_LENGTH_IN_PIXELS = _nui._NUI_CAMERA_DEPTH_NOMINAL_FOCAL_LENGTH_IN_PIXELS
MAXIMUM_ELEVATION = _nui.Camera.ElevationMaximum
MINIMUM_ELEVATION = _nui.Camera.ElevationMinimum
DEFAULT_DATA_PATH = './dataset/data/'


# --------------------------------------------------------------------------------------------------------------------
# Dataset Collector GUI parameters (temp)
class TrialActionLabel(enum.IntEnum):
    WALK = 0
    WAVE = 1
    CLAP = 2
    DRINK = 3
    SIT_DOWN = 4
    STAND_UP = 5
    THROW = 6
    CROUCH = 7
    ONE_LEG_BALANCE = 8
    JUMP = 9
    COUNT = 10


class TrialAgeLabel(enum.IntEnum):
    TEN_THIRTY = 0
    THIRTY_FIFTY = 1
    FIFTY_SEVENTY = 2
    COUNT = 3


# --------------------------------------------------------------------------------------------------------------------
# Dataset and training parameters
MAX_X_TRANSLATION = 1000
MAX_Y_TRANSLATION = 1000
MAX_Z_TRANSLATION = 3000
MAX_SCALE = 1.3
MIN_SCALE = 0.7
UPSAMPLE_THRES = 1.3
DOWNSAMPLE_THRES = 0.8
MAX_PITCH_YAW_ANGLE = 0.2617993877991494    # in rad, i.e. 15 deg
IMG_SIDE_LEN = 224


class DatasetProtocol(enum.IntEnum):
    CROSS_SAMPLE = 0
    CROSS_SUBJECT = 1
    CROSS_VIEW = 2
    CROSS_AGE = 3
    CROSS_GENDER = 4


class HyperParamType(enum.IntEnum):
    USE_SGD = 0
    USE_LSTM = 1
    ENABLE_AUGMENT = 2
    ENABLE_VA = 3
    BATCH_SIZE = 4
    DROPOUTS = 5
    RNN_HIDDEN_DIMS = 6
    FC_HIDDEN_DIMS = 7
    REGRESS_LAMBDA = 8
    INIT_LR = 9
    LR_DECAY = 10
    LR_DECAY_PATIENCE = 11
    RNN_TYPE = 12
    ACTIVATION_TYPE = 13
    TRUNCATED_LENGTH = 14
    USE_FOCAL_LOSS = 15
    USE_LAYER_NORM = 16


class RNNType(enum.IntEnum):
    LSTM = 0
    GRU = 1
    SRU = 2
    OTHER = 3


RNN_NAME = {
    RNNType.LSTM: 'LSTM',
    RNNType.GRU: 'GRU',
    RNNType.SRU: 'SRU',
    RNNType.OTHER: 'OTHER'
}


class ActivationType(enum.IntEnum):
    ReLU = 0
    ELU = 1
    SELU = 2


ACTIVATION_NAME = {
    ActivationType.ReLU: 'ReLU',
    ActivationType.ELU: 'ELU',
    ActivationType.SELU: 'SELU',
}


# --------------------------------------------------------------------------------------------------------------------
# Demo GUI parameters
DATASET_CLASS_NUM = {
    'PKU-MMD-CS': 52,
    'PKU-MMD-CV': 52,
    'OAD': 11,
    'G3D': 23,
    'JL': (11, 20, 2)
}


DATASET_INPUT_DIM = {
    'PKU-MMD-CS': 120,
    'PKU-MMD-CV': 120,
    'OAD': 60,
    'G3D': 60,
    'JL': 60
}


DATASET_AVAIL_ACTIONS = {
    'PKU-MMD-CS': ('bow', 'check time (from watch)', 'cheer up',
                   'cross hands in front (say stop)', 'drop', 'eat meal/snack', 'falling',
                   'giving something to other person', 'hand waving', 'handshaking', 'hopping (one foot jumping)',
                   'hugging other person', 'jump up', 'kicking other person', 'kicking something',
                   'make a phone call/answer phone', 'pat on back of other person', 'pickup',
                   'playing with phone/tablet', 'point finger at the other person', 'pointing to something with finger',
                   'punching/slapping other person', 'pushing other person', 'put on a hat/cap',
                   'put something inside pocket', 'reading (seated)', 'salute', 'sitting down',
                   'standing up', 'take off a hat/cap', 'take off glasses', 'take off jacket',
                   'take out something from pocket', 'taking a selfie', 'throw',
                   'touch back (backache)', 'touch head (headache)',
                   'touch neck (neckache)', 'typing on a keyboard',
                   'wear jacket', 'wear on glasses', 'wipe face', 'unknown'),
    'G3D': ('PunchRight', 'PunchLeft', 'KickRight', 'KickLeft', 'Defend',
            'GolfSwing', 'TennisSwingForehand', 'TennisSwingBackhand',
            'TennisServe', 'ThrowBowlingBall', 'AimAndFireGun', 'Walk',
            'Run', 'Jump', 'Climb', 'Crouch', 'SteerCentre', 'SteerRight',
            'SteerLeft', 'Wave', 'Flap', 'Clap', 'unknown'),
    'JL': ('Walk', 'Wave', 'Clap', 'Drink', 'Sit Down', 'Stand Up',
           'Right-arm Throw', 'Crouch', 'One-leg Balance', 'Jump', 'Unknown'),
}


DATASET_LABEL_STRINGS = {
    'PKU-MMD-CS': ('bow', 'brushing hair', 'brushing teeth', 'check time (from watch)', 'cheer up', 'clapping',
                   'cross hands in front (say stop)', 'drink water', 'drop', 'eat meal/snack', 'falling',
                   'giving something to other person', 'hand waving', 'handshaking', 'hopping (one foot jumping)',
                   'hugging other person', 'jump up', 'kicking other person', 'kicking something',
                   'make a phone call/answer phone', 'pat on back of other person', 'pickup',
                   'playing with phone/tablet', 'point finger at the other person', 'pointing to something with finger',
                   'punching/slapping other person', 'pushing other person', 'put on a hat/cap',
                   'put something inside pocket', 'reading', 'rub two hands together', 'salute', 'sitting down',
                   'standing up', 'take off a hat/cap', 'take off glasses', 'take off jacket',
                   'take out something from pocket', 'taking a selfie', 'tear up paper', 'throw',
                   'touch back (backache)', 'touch chest (stomachache/heart pain)', 'touch head (headache)',
                   'touch neck (neckache)', 'typing on a keyboard', 'use a fan (with hand or paper)/feeling warm',
                   'wear jacket', 'wear on glasses', 'wipe face', 'writing', 'unknown'),
    'PKU-MMD-CV': ('bow', 'brushing hair', 'brushing teeth', 'check time (from watch)', 'cheer up', 'clapping',
                   'cross hands in front (say stop)', 'drink water', 'drop', 'eat meal/snack', 'falling',
                   'giving something to other person', 'hand waving', 'handshaking', 'hopping (one foot jumping)',
                   'hugging other person', 'jump up', 'kicking other person', 'kicking something',
                   'make a phone call/answer phone', 'pat on back of other person', 'pickup',
                   'playing with phone/tablet', 'point finger at the other person', 'pointing to something with finger',
                   'punching/slapping other person', 'pushing other person', 'put on a hat/cap',
                   'put something inside pocket', 'reading', 'rub two hands together', 'salute', 'sitting down',
                   'standing up', 'take off a hat/cap', 'take off glasses', 'take off jacket',
                   'take out something from pocket', 'taking a selfie', 'tear up paper', 'throw',
                   'touch back (backache)', 'touch chest (stomachache/heart pain)', 'touch head (headache)',
                   'touch neck (neckache)', 'typing on a keyboard', 'use a fan (with hand or paper)/feeling warm',
                   'wear jacket', 'wear on glasses', 'wipe face', 'writing', 'unknown'),
    'OAD': ('drinking', 'eating', 'writing', 'opening cupboard', 'washing hands',
            'opening microwave oven', 'sweeping', 'gargling', 'throwing trash', 'wiping', 'unknown'),
    'G3D': ('PunchRight', 'PunchLeft', 'KickRight', 'KickLeft', 'Defend',
            'GolfSwing', 'TennisSwingForehand', 'TennisSwingBackhand',
            'TennisServe', 'ThrowBowlingBall', 'AimAndFireGun', 'Walk',
            'Run', 'Jump', 'Climb', 'Crouch', 'SteerCentre', 'SteerRight',
            'SteerLeft', 'Wave', 'Flap', 'Clap', 'unknown'),
    'JL': ('Walk', 'Wave', 'Clap', 'Drink', 'Sit Down', 'Stand Up',
           'Right-arm Throw', 'Crouch', 'One-leg Balance', 'Jump', 'Unknown'),
}


DOWNSAMPLE_FACTORS = {
    # with respect to 30 fps
    'PKU-MMD-CS': 2.02,
    'PKU-MMD-CV': 4,
    'OAD': 3.75,
    'G3D': 2.14,
    'JL': 2.14
}


AGE_LABEL_STRING = ['< 55', '≥ 55']


MIN_EVENT_PROBABILITIES = {
    'G3D': 0.5,
    'PKU-MMD-CS': 0.15
}


MIN_EVENT_PROBABILITY = 0.2
MIN_JOINTS_IN_FRAME = 12
