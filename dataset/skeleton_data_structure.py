from pykinect.nui import JointId
from openni.nite2 import JointType


MS_LEFT_ARM = (JointId.ShoulderCenter,
               JointId.ShoulderLeft,
               JointId.ElbowLeft,
               JointId.WristLeft,
               JointId.HandLeft)
MS_RIGHT_ARM = (JointId.ShoulderCenter,
                JointId.ShoulderRight,
                JointId.ElbowRight,
                JointId.WristRight,
                JointId.HandRight)
MS_LEFT_LEG = (JointId.HipCenter,
               JointId.HipLeft,
               JointId.KneeLeft,
               JointId.AnkleLeft,
               JointId.FootLeft)
MS_RIGHT_LEG = (JointId.HipCenter,
                JointId.HipRight,
                JointId.KneeRight,
                JointId.AnkleRight,
                JointId.FootRight)
MS_SPINE = (JointId.HipCenter,
            JointId.Spine,
            JointId.ShoulderCenter,
            JointId.Head)
NI_LEFT_ARM = (JointType.NITE_JOINT_LEFT_SHOULDER,
               JointType.NITE_JOINT_LEFT_ELBOW,
               JointType.NITE_JOINT_LEFT_HAND)
NI_RIGHT_ARM = (JointType.NITE_JOINT_RIGHT_SHOULDER,
                JointType.NITE_JOINT_RIGHT_ELBOW,
                JointType.NITE_JOINT_RIGHT_HAND)
NI_LEFT_LEG = (JointType.NITE_JOINT_LEFT_HIP,
               JointType.NITE_JOINT_LEFT_KNEE,
               JointType.NITE_JOINT_LEFT_FOOT)
NI_RIGHT_LEG = (JointType.NITE_JOINT_RIGHT_HIP,
                JointType.NITE_JOINT_RIGHT_KNEE,
                JointType.NITE_JOINT_RIGHT_FOOT)
NI_SPINE = (JointType.NITE_JOINT_TORSO,
            JointType.NITE_JOINT_NECK,
            JointType.NITE_JOINT_HEAD)


class Skeleton(object):
    def __init__(self):
        pass
