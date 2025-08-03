from enum import IntEnum
import numpy as np

# table height when conducting wire-gripper experiments.
TABLE_HEIGHT = 95.7
# table height when conducting plugging && unplugging net wire.

TABLE_HEIGHT = 81.3

# Total number of joints, including passive ones
MAX_JOINT = 19
# Number of actively controlled joints
MAX_ACTIVE_JOINT = 12
# Number of actuators
ACTUATOR_COUNT = 12

class JointID(IntEnum):
    JThumbRoll = 0
    JThumbAbad = 1
    JThumbMCP = 2
    JThumbPIP = 3
    JThumbDIP = 4
    JIndexAbad = 5
    JIndexMCP = 6
    JIndexPIP = 7
    JIndexDIP = 8
    JMiddleAbad = 9
    JMiddleMCP = 10
    JMiddlePIP = 11
    JMiddleDIP = 12
    JRingMCP = 13
    JRingPIP = 14
    JRingDIP = 15
    JPinkyMCP = 16
    JPinkyPIP = 17
    JPinkyDIP = 18
    MaxJoint = 19

LeftJointNames = [
    "L_thumb_roll_joint", "L_thumb_abad_joint", "L_thumb_mcp_joint", "L_thumb_pip_joint", "L_thumb_dip_joint",  
    "L_index_abad_joint", "L_index_mcp_joint", "L_index_pip_joint", "L_index_dip_joint",
    "L_middle_abad_joint", "L_middle_mcp_joint", "L_middle_pip_joint", "L_middle_dip_joint",
    "L_ring_mcp_joint", "L_ring_pip_joint", "L_ring_dip_joint",
    "L_pinky_mcp_joint", "L_pinky_pip_joint", "L_pinky_dip_joint"
]
RightJointNames = [
    "R_thumb_roll_joint", "R_thumb_abad_joint", "R_thumb_mcp_joint", "R_thumb_pip_joint", "R_thumb_dip_joint",
    "R_index_abad_joint", "R_index_mcp_joint", "R_index_pip_joint", "R_index_dip_joint",
    "R_middle_abad_joint", "R_middle_mcp_joint", "R_middle_pip_joint", "R_middle_dip_joint",
    "R_ring_mcp_joint", "R_ring_pip_joint", "R_ring_dip_joint",
    "R_pinky_mcp_joint", "R_pinky_pip_joint", "R_pinky_dip_joint"
]
class ActiveJointID(IntEnum):
    ActiveJointThumbRoll = 0
    ActiveJointThumbAbAd = 1
    ActiveJointThumbMCP = 2
    ActiveJointThumbPIP = 3
    ActiveJointIndexAbAd = 4
    ActiveJointIndexMCP = 5
    ActiveJointIndexPIP = 6
    ActiveJointMiddleABAD = 7
    ActiveJointMiddleMCP = 8
    ActiveJointMiddlePIP = 9
    ActiveJointRingMCP = 10
    ActiveJointPinkyMCP = 11
    MaxActiveJoint = 12

class ManushandJointID(IntEnum):
    """
    Manushand joint IDs for the 21 joints in the Manus hand.
    Refer to: https://docs.manus-meta.com/3.0.0/Software/Skeletons/#raw-skeletons for details. This defintion is subject to ros2 brdige with Manus SDK conventions.
    """
    Wrist = 0
    FingerThumbMetacarpal = 1
    FingerThumbProximal = 2
    FingerThumbDistal = 3
    FingerThumbTip = 4
    FingerIndexMetacarpal = 5
    FingerIndexProximal = 6
    FingerIndexDistal = 7
    FingerIndexTip = 8
    FingerMiddleMetacarpal = 9
    FingerMiddleProximal = 10
    FingerMiddleDistal = 11
    FingerMiddleTip = 12
    FingerRingMetacarpal = 13
    FingerRingProximal = 14
    FingerRingDistal = 15
    FingerRingTip = 16
    FingerPinkyMetacarpal = 17
    FingerPinkyProximal = 18
    FingerPinkyDistal = 19
    FingerPinkyTip = 20
    FingerActuatorCount = 21

class O12handProActuator(IntEnum):
    """
    Constants for the O12 hand prosthetic actuators. Each actuator corresponds to a PD-motor.
    """
    ActuatorIndex1 = 0   # Closer to thumb
    ActuatorIndex2 = 1   # Further from thumb
    ActuatorMiddle1 = 2
    ActuatorMiddle2 = 3
    ActuatorThumbABAD = 4
    ActuatorThumbRoll = 5
    ActuatorIndex3 = 6
    ActuatorMiddle3 = 7
    ActuatorRing = 8
    ActuatorPinky = 9
    ActuatorThumbPIP = 10
    ActuatorThumbMCP = 11
    ActuatorCount = 12

class GestureID(IntEnum):
    """
    Note: These gestures are based on the C++ source code implementation,
    which may differ from some versions of the README.
    """
    HOME = 0
    PAPER = 1
    FIST = 2
    OK = 3
    ONE_HANDED_FINGER_HEART = 4
    THUMB2MIDDLE = 5
    THUMB2RING = 6
    THUMB2PINKY = 7
    NUM1 = 8
    INDEX_CIRCULAR1 = 9
    INDEX_CIRCULAR2 = 10
    INDEX_CIRCULAR3 = 11
    PEACE1 = 12
    PEACE2 = 13
    SCISSORS_PRE = 14
    SCISSORS_GRASP = 15
    SCISSORS_OPEN = 16
    SCISSORS_CUT = 17
    CONTROL_GRASP = 18
    CONTROL_PRESS_PRE = 19
    CONTROL_PRESS = 20
    SCREWDRIVER_GRASP = 21
    SCREWDRIVER_PRESS_PRE = 22
    SCREWDRIVER_PRESS = 23
    GRASP_PRE = 24
    GRASP = 25
    GRASP_END = 26
    UNPLUG_PRE = 27
    UNPLUG = 28
    UNPLUG_END = 29
    POINT = 30
    THUMB2INDEX = 31
    POKER_PRE = 32
    POKER_GRASP = 33
    POKER_OPEN = 34
    WIRE_STRIPPER_PRE = 35
    WIRE_STRIPPER_GRASP = 36
    WIRE_STRIPPER_END = 37
    MAXGESTUREID = 38

ActiveJointPosGestureMap = np.zeros((GestureID.MAXGESTUREID, MAX_ACTIVE_JOINT))

# Set active joint positions for each gesture
ActiveJointPosGestureMap[GestureID.HOME] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ActiveJointPosGestureMap[GestureID.PAPER] = np.array([0.34, 0.0, -0.33, -0.375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ActiveJointPosGestureMap[GestureID.FIST] = np.array([0.5, -0.2, 0.0, -1.2, 0.0, 1.413, 1.570, 0.0, 1.490, 1.570, 1.091, 1.091])
ActiveJointPosGestureMap[GestureID.OK] = np.array([0.3, -0.531, 0.0, -0.95, 0.0, 0.5, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0])
ActiveJointPosGestureMap[GestureID.ONE_HANDED_FINGER_HEART] = np.array([0.232, 0.0, -0.553, 0.0, -0.26, 1.06, 0.652, 0.0, 1.501, 1.57, 1.55, 1.55])
ActiveJointPosGestureMap[GestureID.THUMB2MIDDLE] = np.array([0.45, -0.5, 0.0, -0.68, 0.0, 0.0, 0.0, 0.0, 0.8, 0.55, 0.0, 0.0])
ActiveJointPosGestureMap[GestureID.THUMB2RING] = np.array([1.0, -0.6, 0.0, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.0])
ActiveJointPosGestureMap[GestureID.THUMB2PINKY] = np.array([1.0, -1.1, 0.0, -0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8])
ActiveJointPosGestureMap[GestureID.NUM1] = np.array([0.275, -0.524, -0.3, -1.1, 0.0, 0.0, 0.0, 0.0, 1.5, 1.57, 1.55, 1.55])
ActiveJointPosGestureMap[GestureID.INDEX_CIRCULAR1] = np.array([0.275, -0.524, -0.3, -1.1, -0.26, 0.2, 0.0, 0.0, 1.5, 1.57, 1.55, 1.55])
ActiveJointPosGestureMap[GestureID.INDEX_CIRCULAR2] = np.array([0.275, -0.524, -0.3, -1.1, 0.0, 0.7, 0.0, 0.0, 1.5, 1.57, 1.55, 1.55])
ActiveJointPosGestureMap[GestureID.INDEX_CIRCULAR3] = np.array([0.275, -0.524, -0.3, -1.1, 0.26, 0.2, 0.0, 0.0, 1.5, 1.57, 1.55, 1.55])
ActiveJointPosGestureMap[GestureID.PEACE1] = np.array([0.275, 0.0, -0.3, -1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.55, 1.55])
ActiveJointPosGestureMap[GestureID.PEACE2] = np.array([0.25, 0.0, -0.2, -0.7, -0.26, 0.0, 0.0, 0.26, 0.0, 0.0, 1.55, 1.55])
ActiveJointPosGestureMap[GestureID.SCISSORS_PRE] = np.array([0.4, 0.0, -0.2, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.57])
ActiveJointPosGestureMap[GestureID.SCISSORS_GRASP] = np.array([0.4, 0.0, -0.2, -0.4, 0.0, 0.2, 0.5, 0.0, 0.2, 0.5, 0.5, 1.57])
ActiveJointPosGestureMap[GestureID.SCISSORS_OPEN] = np.array([0.4, 0.0, -0.2, -0.4, -0.26, 0.2, 0.5, -0.26, 0.2, 0.5, 0.5, 1.57])
ActiveJointPosGestureMap[GestureID.SCISSORS_CUT] = np.array([0.4, 0.0, -0.8, -0.4, -0.26, 0.2, 0.5, -0.26, 0.2, 0.5, 0.5, 1.57])
ActiveJointPosGestureMap[GestureID.CONTROL_GRASP] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.57, 1.57])
ActiveJointPosGestureMap[GestureID.CONTROL_PRESS_PRE] = np.array([0.1, -0.5, -0.3, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.57, 1.57])
ActiveJointPosGestureMap[GestureID.CONTROL_PRESS] = np.array([0.1, -0.5, -0.3, -1.2, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.57, 1.57])
ActiveJointPosGestureMap[GestureID.SCREWDRIVER_GRASP] = np.array([0.1, -0.7, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ActiveJointPosGestureMap[GestureID.SCREWDRIVER_PRESS_PRE] = np.array([0.1, -0.7, -0.4, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.8, 1.1, 1.0])
ActiveJointPosGestureMap[GestureID.SCREWDRIVER_PRESS] = np.array([0.1, -0.7, -0.4, -1.0, 0.0, 1.5, 0.8, 0.0, 1.0, 0.8, 1.1, 1.0])
ActiveJointPosGestureMap[GestureID.GRASP_PRE] = np.array([0.339, -0.758, -0.376, 0., -0.142, 0.3, 0.488, 0.0, 1.49, 1.57, 1.558, 1.558])
ActiveJointPosGestureMap[GestureID.GRASP] = np.array([0.339, -0.758, -0.735, 0., -0.142, 0.861, 0.488, 0.0, 1.49, 1.57, 1.558, 1.558])
ActiveJointPosGestureMap[GestureID.GRASP_END] = np.array([0.339, -0.758, -0.376, 0.,  -0.142, 0.3, 0.488, 0.0, 1.49, 1.57, 1.558, 1.558])
ActiveJointPosGestureMap[GestureID.UNPLUG_PRE] = np.array([0.339, -0.758, 0, -0.7, -0.142, 0.0, 1.0, 0.0, 1.49, 1.57, 1.558, 1.558])
ActiveJointPosGestureMap[GestureID.UNPLUG] = np.array([0.339, -0.758, -0.25, -0.965, -0.142, 0.182, 1.339, 0.0, 1.49, 1.57, 1.558, 1.558])
ActiveJointPosGestureMap[GestureID.UNPLUG_END] = np.array([0.339, -0.758, 0, -0.7,  -0.142, 0.0, 1.0, 0.0, 1.49, 1.57, 1.558, 1.558])
ActiveJointPosGestureMap[GestureID.POINT] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 1.57, 1.55, 1.55])
ActiveJointPosGestureMap[GestureID.THUMB2INDEX] = np.array([0.3, -0.531, 0.0, -0.95, 0.0, 0.5, 0.65, 0.0, 1.5, 1.57, 1.55, 1.55])
ActiveJointPosGestureMap[GestureID.POKER_PRE] = np.array([0.2, 0.0, 0.0, -1.0, 0.0, 0.3, 0.8, 0.0, 0.3, 1.0, 1.0, 0.8])
ActiveJointPosGestureMap[GestureID.POKER_GRASP] = np.array([0.2, 0.0, -0.3, -1.0, 0.0, 0.3, 0.8, 0.0, 0.3, 1.0, 1.0, 0.8])
ActiveJointPosGestureMap[GestureID.POKER_OPEN] = np.array([0.2, 0.0, -0.3, -0.6, 0.0, 0.3, 0.8, 0.0, 0.3, 1.0, 1.0, 0.8])
ActiveJointPosGestureMap[GestureID.WIRE_STRIPPER_PRE] = np.array([0.5, -0.2, 0.0, -0.6, 0.0, 1.413, 1.570, 0.0, 1.490, 1.570, 1.558, 1.558])
ActiveJointPosGestureMap[GestureID.WIRE_STRIPPER_GRASP] = np.array([0.5, -0.2, -0.831, -1.3, 0.0, 1.413, 1.570, 0.0, 1.490, 1.570, 1.558, 1.558])
ActiveJointPosGestureMap[GestureID.WIRE_STRIPPER_END] = np.array([0.5, -0.2, 0.0, -0.5, 0.0, 1.413, 1.570, 0.0, 1.490, 1.570, 1.558, 1.558])


GestureName = ["Home", "Paper", "Fist", "OK", "One Handed Finger Heart",
               "Thumb to Middle", "Thumb to Ring", "Thumb to Pinky",
               "Number 1", "Index Circular 1", "Index Circular 2",
               "Index Circular 3", "Peace 1", "Peace 2", 
               "Scissors Pre", "Scissors Grasp", "Scissors Open",
               "Scissors Cut", "Control Grasp", "Control Press Pre",
               "Control Press", "Screwdriver Grasp", "Screwdriver Press Pre",
               "Screwdriver Press", "Grasp Pre", "Grasp", "Grasp End", "Unplug pre", "Unplug", "Unplug End", "Point",
               "Thumb to Index", "Poker Pre", "Poker Grasp", "Poker Open", "Wire Stripper Pre", "Wire Stripper Grasp", "Wire Stripper End"]

# Initialize the gesture map and joint limits arrays
O12RightHandLowerLimits = np.zeros(MAX_JOINT)
O12RightHandUpperLimits = np.zeros(MAX_JOINT)

# Joint limits for the O12 hand prosthetic based on URDF
O12RightHandLowerLimits[JointID.JThumbRoll] = 0.0
O12RightHandUpperLimits[JointID.JThumbRoll] = 1.21447

O12RightHandLowerLimits[JointID.JThumbAbad] = -1.385
O12RightHandUpperLimits[JointID.JThumbAbad] = 0.0

O12RightHandLowerLimits[JointID.JThumbMCP] = -0.8312
O12RightHandUpperLimits[JointID.JThumbMCP] = 0.0

O12RightHandLowerLimits[JointID.JThumbPIP] = -1.3
O12RightHandUpperLimits[JointID.JThumbPIP] = 0.0

O12RightHandLowerLimits[JointID.JThumbDIP] = -1.395
O12RightHandUpperLimits[JointID.JThumbDIP] = 0.0

O12RightHandLowerLimits[JointID.JIndexAbad] = -0.26
O12RightHandUpperLimits[JointID.JIndexAbad] = 0.26

O12RightHandLowerLimits[JointID.JIndexMCP] = 0.0
O12RightHandUpperLimits[JointID.JIndexMCP] = 1.5

O12RightHandLowerLimits[JointID.JIndexPIP] = 0.0
O12RightHandUpperLimits[JointID.JIndexPIP] = 1.57

O12RightHandLowerLimits[JointID.JIndexDIP] = 0.0
O12RightHandUpperLimits[JointID.JIndexDIP] = 1.28

O12RightHandLowerLimits[JointID.JMiddleAbad] = -0.26
O12RightHandUpperLimits[JointID.JMiddleAbad] = 0.26

O12RightHandLowerLimits[JointID.JMiddleMCP] = 0.0
O12RightHandUpperLimits[JointID.JMiddleMCP] = 1.49

O12RightHandLowerLimits[JointID.JMiddlePIP] = 0.0
O12RightHandUpperLimits[JointID.JMiddlePIP] = 1.57

O12RightHandLowerLimits[JointID.JMiddleDIP] = 0.0
O12RightHandUpperLimits[JointID.JMiddleDIP] = 1.45

O12RightHandLowerLimits[JointID.JRingMCP] = 0.0
O12RightHandUpperLimits[JointID.JRingMCP] = 1.5583

O12RightHandLowerLimits[JointID.JRingPIP] = 0.0
O12RightHandUpperLimits[JointID.JRingPIP] = 1.37758

O12RightHandLowerLimits[JointID.JRingDIP] = 0.0
O12RightHandUpperLimits[JointID.JRingDIP] = 1.2825

O12RightHandLowerLimits[JointID.JPinkyMCP] = 0.0
O12RightHandUpperLimits[JointID.JPinkyMCP] = 1.5583

O12RightHandLowerLimits[JointID.JPinkyPIP] = 0.0
O12RightHandUpperLimits[JointID.JPinkyPIP] = 1.37758

O12RightHandLowerLimits[JointID.JPinkyDIP] = 0.0
O12RightHandUpperLimits[JointID.JPinkyDIP] = 1.2825

# Initialize the gesture map and joint limits arrays
O12LeftHandLowerLimits = np.zeros(MAX_JOINT)
O12LeftHandUpperLimits = np.zeros(MAX_JOINT)

# Joint limits for the O12 hand prosthetic based on URDF
O12LeftHandLowerLimits[JointID.JThumbRoll] = -1.21447
O12LeftHandUpperLimits[JointID.JThumbRoll] = 0.0

O12LeftHandLowerLimits[JointID.JThumbAbad] = 0.0
O12LeftHandUpperLimits[JointID.JThumbAbad] = 1.385

O12LeftHandLowerLimits[JointID.JThumbMCP] = -0.8312
O12LeftHandUpperLimits[JointID.JThumbMCP] = 0.0

O12LeftHandLowerLimits[JointID.JThumbPIP] = -1.3
O12LeftHandUpperLimits[JointID.JThumbPIP] = 0.0

O12LeftHandLowerLimits[JointID.JThumbDIP] = -1.395
O12LeftHandUpperLimits[JointID.JThumbDIP] = 0.0

O12LeftHandLowerLimits[JointID.JIndexAbad] = -0.26
O12LeftHandUpperLimits[JointID.JIndexAbad] = 0.26

O12LeftHandLowerLimits[JointID.JIndexMCP] = 0.0
O12LeftHandUpperLimits[JointID.JIndexMCP] = 1.5

O12LeftHandLowerLimits[JointID.JIndexPIP] = 0.0
O12LeftHandUpperLimits[JointID.JIndexPIP] = 1.57

O12LeftHandLowerLimits[JointID.JIndexDIP] = 0.0
O12LeftHandUpperLimits[JointID.JIndexDIP] = 1.28

O12LeftHandLowerLimits[JointID.JMiddleAbad] = -0.26
O12LeftHandUpperLimits[JointID.JMiddleAbad] = 0.26

O12LeftHandLowerLimits[JointID.JMiddleMCP] = 0.0
O12LeftHandUpperLimits[JointID.JMiddleMCP] = 1.49

O12LeftHandLowerLimits[JointID.JMiddlePIP] = 0.0
O12LeftHandUpperLimits[JointID.JMiddlePIP] = 1.57

O12LeftHandLowerLimits[JointID.JMiddleDIP] = 0.0
O12LeftHandUpperLimits[JointID.JMiddleDIP] = 1.45

O12LeftHandLowerLimits[JointID.JRingMCP] = 0.0
O12LeftHandUpperLimits[JointID.JRingMCP] = 1.5583

O12LeftHandLowerLimits[JointID.JRingPIP] = 0.0
O12LeftHandUpperLimits[JointID.JRingPIP] = 1.37758

O12LeftHandLowerLimits[JointID.JRingDIP] = 0.0
O12LeftHandUpperLimits[JointID.JRingDIP] = 1.2825

O12LeftHandLowerLimits[JointID.JPinkyMCP] = 0.0
O12LeftHandUpperLimits[JointID.JPinkyMCP] = 1.5583

O12LeftHandLowerLimits[JointID.JPinkyPIP] = 0.0
O12LeftHandUpperLimits[JointID.JPinkyPIP] = 1.37758

O12LeftHandLowerLimits[JointID.JPinkyDIP] = 0.0
O12LeftHandUpperLimits[JointID.JPinkyDIP] = 1.2825