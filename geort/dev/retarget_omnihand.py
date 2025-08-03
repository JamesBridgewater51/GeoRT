"""
Retargeting module for mapping 21‑point human hand joint positions to
OmniHand dexterous robot joint angles.

The human hand contains metacarpophalangeal (MCP) joints at the base of the
fingers.  These condyloid joints connect the metacarpal bones to the
proximal phalanges and support flexion/extension and limited
abduction/adduction【222572499196937†L164-L170】.  Beyond the MCPs, the
interphalangeal joints provide hinge‑like flexion towards the palm【792489904477277†L156-L165】.  Each finger (except the thumb) has two
interphalangeal joints—proximal (PIP) and distal (DIP)—while the
thumb has a single interphalangeal (IP) joint【792489904477277†L156-L165】.  Studies
on human finger kinematics report that the primary motions of hand joints
are flexion/extension and abduction/adduction, and that the DIP and PIP
joints of the four fingers tend to move together【655647759356349†L250-L256】.

This module implements a very simple, purely geometric baseline
retargeting strategy based on these anatomical observations.  Given
three‑dimensional positions for the 21 Manus hand joints (0= wrist, 1–4
= thumb joints, 5–8 = index joints, 9–12 = middle joints, 13–16 =
ring joints, 17–20 = pinky joints), the routine below computes human
joint flexion and abduction angles and maps them to the corresponding
OmniHand joint angles.  The mapping uses basic vector geometry:

* Flexion at each joint is computed from the angle between adjacent
  bone segments.  For example, at the MCP joint of the index
  finger, the angle between the metacarpal segment (running from the
  finger base back to the wrist) and the proximal phalanx (running
  from the MCP to the PIP) is measured.  When the finger is
  extended the segments are almost colinear (angle≈π), so the
  computed flexion is near 0.  As the finger bends the angle
  decreases, producing a larger flexion value (π − θ).

* Abduction at the thumb, index and middle fingers is calculated
  relative to the palm plane.  The palm plane is estimated from
  vectors running from the wrist to the bases of the index and
  pinky fingers.  Finger base vectors are projected onto this plane
  and a signed angle is computed with respect to the reference
  direction defined by the middle finger.  Positive and negative
  angles describe radial and ulnar abduction respectively.

* Because the human thumb has only a single interphalangeal joint
  but the robot has two (PIP and DIP), the thumb IP flexion is
  partitioned between the OmniHand PIP and DIP using the same ratio
  specified in the URDF (DIP ≈ 0.7999 × PIP).  Similarly, the
  OmniHand ring and pinky fingers implement synergies in which the
  PIP and DIP joints mimic the MCP flexion; these multipliers are
  applied when distributing human MCP flexion to the corresponding
  robot joints.  These couplings reflect the anatomical tendency for
  the PIP and DIP joints to move together【655647759356349†L250-L256】.

The returned angles are clamped to the joint limits provided in the
OmniHand URDF.  Angles are expressed in radians and follow the
conventions implied by the URDF (for example, thumb flexion angles
are negative).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple, Optional
from geort.utils.vis_utils import MANUS_SKELETON_BONES
from .ik_chain_const import R_chains
from ikpy.chain import Chain
import warnings
from enum import Enum
from cprint import cprint
# o12 hand related imports
from ..control.o12_hand.py_sdk.src.constants import ActiveJointID, MAX_ACTIVE_JOINT, MAX_JOINT, ACTUATOR_COUNT, JointID, O12RightHandLowerLimits, O12RightHandUpperLimits, O12LeftHandLowerLimits, O12LeftHandUpperLimits
# manus related imports
from ..control.o12_hand.py_sdk.src.constants import ManushandJointID

import numpy as np


PINCH_DETECTION_CANDIDATES = ['R_index_DIP', 'R_index_tip', 'R_middle_DIP', 'R_middle_tip', 'R_ring_tip', 'R_pinky_tip']

PINCH_THRESHOLD = 0.025  # Threshold distance for pinch detection
MANUS_FINGER_INDICES = None

ENABLE_PYBULLET = True
JOINTS = None
LINK_NAME_TO_INDEX_MAP = None

if ENABLE_PYBULLET:
    import pybullet as p
    import pybullet_data
    from collections import namedtuple
    from ..utils.bullet_utils import build_link_name_to_index_, get_joint_id_from_joint_name

def _setup_pybullet(urdf_path: str, is_right_hand: bool = True):
  
    global MANUS_FINGER_INDICES
    if is_right_hand:
        MANUS_FINGER_INDICES = {
            'R_base': {'R_wrist': 0},
            'R_thumb': {'R_thumb_CMC': 1, 'R_thumb_MCP': 2, 'R_thumb_DIP': 3, 'R_thumb_tip': 4},
            'R_index': {'R_index_MCP': 5, 'R_index_PIP': 6, 'R_index_DIP': 7, 'R_index_tip': 8},
            'R_middle': {'R_middle_MCP': 9, 'R_middle_PIP': 10, 'R_middle_DIP': 11, 'R_middle_tip': 12},
            'R_ring': {'R_ring_MCP': 13, 'R_ring_PIP': 14, 'R_ring_DIP': 15, 'R_ring_tip': 16},
            'R_pinky': {'R_pinky_MCP': 17, 'R_pinky_PIP': 18, 'R_pinky_DIP': 19, 'R_pinky_tip': 20}
        }  
    else:
        MANUS_FINGER_INDICES = {
            'L_base': {'L_wrist': 0},
            'L_thumb': {'L_thumb_CMC': 1, 'L_thumb_MCP': 2, 'L_thumb_DIP': 3, 'L_thumb_tip': 4},
            'L_index': {'L_index_MCP': 5, 'L_index_PIP': 6, 'L_index_DIP': 7, 'L_index_tip': 8},
            'L_middle': {'L_middle_MCP': 9, 'L_middle_PIP': 10, 'L_middle_DIP': 11, 'L_middle_tip': 12},
            'L_ring': {'L_ring_MCP': 13, 'L_ring_PIP': 14, 'L_ring_DIP': 15, 'L_ring_tip': 16},
            'L_pinky': {'L_pinky_MCP': 17, 'L_pinky_PIP': 18, 'L_pinky_DIP': 19, 'L_pinky_tip': 20}
        }  

    if ENABLE_PYBULLET:
        global LINK_NAME_TO_INDEX_MAP, JOINTS
        # Load the OmniHand URDF
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
        joint_num = p.getNumJoints(robot_id)
        jointInfo = namedtuple('jointInfo', 
                    ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','ChildLinkName','jointAxis','parentFramePos','parentFrameOrn','parentIndex','controllable'])
        JOINTS = []
        controllable_joints = []

        for joint_index in range(joint_num):
            info = p.getJointInfo(robot_id, joint_index)

            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            # NOTE: in pybullet docs, it is called LinkName, actually for current joint, this is child link's name.
            ChildLinkName = info[12].decode("utf-8")
            jointAxis = info[13]
            parentFramePos = info[14]
            parentFrameOrn = info[15]
            parentIndex = info[16] # parent link index, -1 means base link
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                controllable_joints.append(jointID)
                p.setJointMotorControl2(robot_id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,ChildLinkName,jointAxis,parentFramePos,parentFrameOrn,parentIndex,controllable)
            JOINTS.append(info)
            print(info)
        
        LINK_NAME_TO_INDEX_MAP = build_link_name_to_index_(JOINTS)

ENABLE_REAL_ROBOT = False


VIS_DBG_IK = False

MAX_ACTIVE_JOINT_MOVEMENT = np.zeros(MAX_ACTIVE_JOINT)
MAX_ACTIVE_JOINT_MOVEMENT[ActiveJointID.ActiveJointThumbRoll] = 0.3
MAX_ACTIVE_JOINT_MOVEMENT[ActiveJointID.ActiveJointThumbAbAd] = 0.3
MAX_ACTIVE_JOINT_MOVEMENT[ActiveJointID.ActiveJointThumbMCP] = 0.3
MAX_ACTIVE_JOINT_MOVEMENT[ActiveJointID.ActiveJointThumbPIP] = 0.3
MAX_ACTIVE_JOINT_MOVEMENT[ActiveJointID.ActiveJointIndexAbAd] = 0.3
MAX_ACTIVE_JOINT_MOVEMENT[ActiveJointID.ActiveJointIndexMCP] = 0.3
MAX_ACTIVE_JOINT_MOVEMENT[ActiveJointID.ActiveJointIndexPIP] = 0.3
MAX_ACTIVE_JOINT_MOVEMENT[ActiveJointID.ActiveJointMiddleABAD] = 0.3
MAX_ACTIVE_JOINT_MOVEMENT[ActiveJointID.ActiveJointMiddleMCP] = 0.3
MAX_ACTIVE_JOINT_MOVEMENT[ActiveJointID.ActiveJointMiddlePIP] = 0.3
MAX_ACTIVE_JOINT_MOVEMENT[ActiveJointID.ActiveJointRingMCP] = 0.3
MAX_ACTIVE_JOINT_MOVEMENT[ActiveJointID.ActiveJointPinkyMCP] = 0.3

# Global variables for angle visualization
VIS_BASELINE_ANGLE = True
_angle_fig = None
_angle_ax = None
_angle_bars = None

ENABLE_MANUS_VIS = True
if ENABLE_MANUS_VIS:
    # Enable interactive mode for non-blocking matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion()


class IKMethod(Enum):
    """Enumeration of IK methods."""
    PYBULLET = 'pybullet'
    IKPY = 'ikpy'

def _vectorized_vector_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute the unsigned angle between two vectors.

    Args:
        v1: First 3D vector(s) of shape (..., 3).
        v2: Second 3D vector(s) of shape (..., 3).

    Returns:
        Angle between v1 and v2 in radians (0 ≤ angle ≤ π), shape (...,).
    """
    # Normalize to avoid numerical issues; add small epsilon to avoid division by zero.
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    valid_mask = (n1 >= 1e-8) & (n2 >= 1e-8)
    
    cos_theta = np.sum(v1 * v2, axis=-1) / (n1 * n2 + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.arccos(cos_theta)
    
    # Set invalid angles to 0
    angles = np.where(valid_mask, angles, 0.0)
    return angles


def _vectorized_signed_angle(v1: np.ndarray, v2: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Compute the signed angle from v1 to v2 around the given normal.

    The result lies in (−π, π].  A positive value means that rotating v1
    towards v2 follows the right‑hand rule around ``normal``.

    Args:
        v1: First 3D vector(s) of shape (..., 3).
        v2: Second 3D vector(s) of shape (..., 3).
        normal: Axis defining the plane of rotation, shape (..., 3).

    Returns:
        Signed angle from v1 to v2 in radians, shape (...,).
    """
    # Project both vectors onto the plane orthogonal to normal.
    n = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)
    v1_proj = v1 - np.sum(v1 * n, axis=-1, keepdims=True) * n
    v2_proj = v2 - np.sum(v2 * n, axis=-1, keepdims=True) * n
    
    n1 = np.linalg.norm(v1_proj, axis=-1)
    n2 = np.linalg.norm(v2_proj, axis=-1)
    valid_mask = (n1 >= 1e-8) & (n2 >= 1e-8)
    
    v1_unit = v1_proj / (n1[..., np.newaxis] + 1e-8)
    v2_unit = v2_proj / (n2[..., np.newaxis] + 1e-8)
    
    cross = np.cross(v1_unit, v2_unit, axis=-1)
    dot = np.sum(v1_unit * v2_unit, axis=-1)
    
    cross_norm = np.linalg.norm(cross, axis=-1)
    angles = np.arctan2(cross_norm, dot)
    
    # Determine sign based on the direction of the cross product relative to normal.
    sign = np.sum(cross * n, axis=-1)
    angles = np.where(sign < 0, -angles, angles)
    
    # Set invalid angles to 0
    angles = np.where(valid_mask, angles, 0.0)
    return angles


def _vectorized_compute_flexion(prev_joint: np.ndarray, joint: np.ndarray, next_joint: np.ndarray, flexion_axis: np.ndarray) -> np.ndarray:
    """Compute flexion at a joint.

    Flexion is measured by taking the angle between the vectors from the
    joint to the neighbouring bones and subtracting it from π.  This
    yields 0 for an extended joint and a positive value as the joint
    flexes.

    Args:
        prev_joint: Position of the proximal bone anchor (closer to the
            wrist), shape (..., 3).
        joint: Position of the joint at which flexion is measured, shape (..., 3).
        next_joint: Position of the distal bone anchor (further from the
            wrist), shape (..., 3).
        flexion_axis: Axis around which flexion is measured, shape (..., 3).

    Returns:
        Flexion angle in radians (≥ 0), shape (...,).
    """
    # Vector along proximal bone (from joint towards proximal anchor).
    v_prox = prev_joint - joint
    # Vector along distal bone (from joint towards distal anchor).
    v_dist = next_joint - joint
    v_straight = -v_prox
    theta = _vectorized_signed_angle(v_straight, v_dist, flexion_axis)
    # theta = _vectorized_vector_angle(v_prox, v_dist)
    # Convert to flexion: pi minus the interior angle.
    # Clamp to non‑negative.
    return theta

def _manus_mocap_anatomy(human_joints: np.ndarray, is_right_hand: bool = True):
    """
    human_joints: (N, 21, 3) numpy array

    Manus mocap joint limits in order of JointID:
    JThumbRoll: -0.927 ~ 0.472
    JThumbAbad: -0.957 ~ 0.106
    JThumbMCP: -1.577 ~ -0.004
    JThumbPIP: -1.328 ~ -0.005
    JThumbDIP: -3.142 ~ -3.142
    JIndexAbad: -0.000 ~ 0.000
    JIndexMCP: 0.007 ~ 1.423
    JIndexPIP: 0.006 ~ 2.433
    JIndexDIP: 0.007 ~ 1.239
    JMiddleAbad: -0.349 ~ 0.685
    JMiddleMCP: 0.011 ~ 1.285
    JMiddlePIP: 0.005 ~ 2.179
    JMiddleDIP: 0.006 ~ 1.402
    JRingMCP: 0.004 ~ 1.482
    JRingPIP: 0.007 ~ 2.180
    JRingDIP: 0.007 ~ 1.530
    JPinkyMCP: 0.023 ~ 1.577
    JPinkyPIP: 0.007 ~ 2.262
    JPinkyDIP: 0.007 ~ 1.148
    """
    # Convert to NumPy array for convenience and check shape.
    arr = np.asarray(human_joints, dtype=np.float64)
    T, N, D = arr.shape

    wrist = arr[:,ManushandJointID.Wrist]
    index_base = arr[:,ManushandJointID.FingerIndexMetacarpal]
    pinky_base = arr[:,ManushandJointID.FingerPinkyMetacarpal]
    palm_vec1 = index_base - wrist
    palm_vec2 = pinky_base - wrist
    palm_normal = np.cross(palm_vec1, palm_vec2)
    assert np.linalg.norm(palm_normal) > 1e-8, "Wrist and pinky base are colinear, cannot compute palm normal."
    palm_normal = palm_normal / np.linalg.norm(palm_normal)

    middle_base = arr[:,ManushandJointID.FingerMiddleMetacarpal]
    middle_dir = middle_base - wrist

    middle_dir = middle_dir - np.einsum("mj,mj->m", middle_dir, palm_normal)[..., None] * palm_normal
    # middle_dir = middle_dir - np.dot(middle_dir, palm_normal) * palm_normal
    assert np.linalg.norm(middle_dir) > 1e-8, "Middle finger base is colinear with wrist and pinky base."
    middle_dir = middle_dir / (np.linalg.norm(middle_dir) + 1e-8)

    index_base = arr[:,ManushandJointID.FingerIndexMetacarpal]
    index_dir = index_base - wrist
    index_dir = index_dir - np.einsum("mj,mj->m", index_dir, palm_normal)[..., None] * palm_normal
    # index_dir = index_dir - np.dot(index_dir, palm_normal) * palm_normal
    assert np.linalg.norm(index_dir) > 1e-8, "Index finger base is colinear with wrist and pinky base."
    index_dir = index_dir / (np.linalg.norm(index_dir) + 1e-8)

    # This axis is perpendicular to the palm plane, pointing right for a right hand.
    # By the right-hand rule, rotation around this axis is positive for flexion.
    flexion_axis = np.cross(middle_dir, palm_normal)
    if not is_right_hand:
        flexion_axis = -flexion_axis

    # thumb
    thumb_mcp = arr[:,ManushandJointID.FingerThumbMetacarpal]
    thumb_pip = arr[:,ManushandJointID.FingerThumbProximal]
    thumb_dip = arr[:,ManushandJointID.FingerThumbDistal]
    thumb_tip = arr[:,ManushandJointID.FingerThumbTip]
    thumb_dir = thumb_dip - thumb_mcp

    thumb_mcp_to_dip = thumb_dip - thumb_mcp
    thumb_mcp_to_dip_proj_palm = thumb_mcp_to_dip - np.einsum("mj,mj->m", thumb_mcp_to_dip, palm_normal)[..., None] * palm_normal
    # thumb_metacarpal_to_dip_proj_palm = thumb_metacarpal_to_dip - np.dot(thumb_metacarpal_to_dip, palm_normal) * palm_normal
    if np.linalg.norm(thumb_mcp_to_dip_proj_palm) < 1e-8:
        thumb_roll_angle = 0.0
    else:
        thumb_roll_angle = -_vectorized_signed_angle(thumb_mcp_to_dip_proj_palm, middle_dir, palm_normal)
    if not is_right_hand:
        thumb_roll_angle = -thumb_roll_angle

    thumb_abd_angle =  -_vectorized_compute_flexion(wrist, thumb_mcp, thumb_pip, flexion_axis=flexion_axis)
    if not is_right_hand:
        thumb_abd_angle = - thumb_abd_angle
    # thumb_abd_angle = _vectorized_signed_angle(thumb_mcp_to_dip_proj_palm, middle_dir, palm_normal)
    thumb_mcp_flex =  -_vectorized_compute_flexion(thumb_mcp, thumb_pip, thumb_dip, flexion_axis=flexion_axis)

    thumb_pip_flex = - _vectorized_compute_flexion(thumb_mcp, thumb_pip, thumb_dip, flexion_axis=flexion_axis)
    thumb_dip_flex = -_vectorized_compute_flexion(thumb_mcp, thumb_pip, thumb_dip, flexion_axis=flexion_axis)

    # ---------- Index finger ----------
    index_dir = arr[:,ManushandJointID.FingerIndexProximal] - arr[:,ManushandJointID.FingerIndexMetacarpal]  # proximal phalanx direction
    index_proj_palm = index_dir - np.einsum("mj,mj->m", index_dir, palm_normal)[..., None] * palm_normal
    if np.linalg.norm(index_proj_palm) < 1e-8:
        index_abd = 0.0
    else:
        index_abd = -_vectorized_signed_angle(index_proj_palm, index_dir, palm_normal)
    # Index MCP flexion.
    index_mcp_flex = _vectorized_compute_flexion(arr[:,ManushandJointID.Wrist], arr[:,ManushandJointID.FingerIndexMetacarpal], arr[:,ManushandJointID.FingerIndexDistal], flexion_axis=flexion_axis)
    # Index PIP flexion.
    index_pip_flex = _vectorized_compute_flexion(arr[:,ManushandJointID.FingerIndexMetacarpal], arr[:,ManushandJointID.FingerIndexProximal], arr[:,ManushandJointID.FingerIndexDistal], flexion_axis=flexion_axis)
    # Index DIP flexion
    index_dip_flex = _vectorized_compute_flexion(arr[:,ManushandJointID.FingerIndexProximal], arr[:,ManushandJointID.FingerIndexDistal], arr[:,ManushandJointID.FingerIndexTip], flexion_axis=flexion_axis)

    # ---------- Middle finger ----------
    # Middle abduction around x‑axis.
    middle_dir2 = arr[:,ManushandJointID.FingerMiddleProximal] - arr[:,ManushandJointID.FingerMiddleMetacarpal]
    middle_proj_palm = middle_dir2 - np.einsum("mj,mj->m", middle_dir2, palm_normal)[..., None] * palm_normal
    if np.linalg.norm(middle_proj_palm) < 1e-8:
        middle_abd = 0.0
    else:
        middle_abd = -_vectorized_signed_angle(middle_proj_palm, middle_dir, palm_normal)
    # Middle MCP flexion.
    middle_mcp_flex = _vectorized_compute_flexion(arr[:,ManushandJointID.Wrist], arr[:,ManushandJointID.FingerMiddleMetacarpal], arr[:,ManushandJointID.FingerMiddleDistal], flexion_axis=flexion_axis)
    # Middle PIP flexion.
    middle_pip_flex = _vectorized_compute_flexion(arr[:,ManushandJointID.FingerMiddleMetacarpal], arr[:,ManushandJointID.FingerMiddleProximal], arr[:,ManushandJointID.FingerMiddleDistal], flexion_axis=flexion_axis)
    # MIddle DIP flexion.
    middle_dip_flex = _vectorized_compute_flexion(arr[:,ManushandJointID.FingerMiddleProximal], arr[:,ManushandJointID.FingerMiddleDistal], arr[:,ManushandJointID.FingerMiddleTip], flexion_axis=flexion_axis)

    # ---------- Ring finger ----------
    ring_mcp_flex = _vectorized_compute_flexion(arr[:,ManushandJointID.Wrist], arr[:,ManushandJointID.FingerRingMetacarpal], arr[:,ManushandJointID.FingerRingDistal], flexion_axis=flexion_axis)
    # Ring PIP flexion.
    ring_pip_flex = _vectorized_compute_flexion(arr[:,ManushandJointID.FingerRingMetacarpal], arr[:,ManushandJointID.FingerRingProximal], arr[:,ManushandJointID.FingerRingDistal], flexion_axis=flexion_axis)
    # Ring DIP flexion.
    ring_dip_flex = _vectorized_compute_flexion(arr[:,ManushandJointID.FingerRingProximal], arr[:,ManushandJointID.FingerRingDistal], arr[:,ManushandJointID.FingerRingTip], flexion_axis=flexion_axis)

    # ---------- Pinky finger ----------
    pinky_mcp_flex = _vectorized_compute_flexion(arr[:,ManushandJointID.Wrist], arr[:,ManushandJointID.FingerPinkyMetacarpal], arr[:,ManushandJointID.FingerPinkyDistal], flexion_axis=flexion_axis)
    pinky_pip_flex = _vectorized_compute_flexion(arr[:,ManushandJointID.FingerPinkyMetacarpal], arr[:,ManushandJointID.FingerPinkyProximal], arr[:,ManushandJointID.FingerPinkyDistal], flexion_axis=flexion_axis)
    pinky_dip_flex = _vectorized_compute_flexion(arr[:,ManushandJointID.FingerPinkyProximal], arr[:,ManushandJointID.FingerPinkyDistal], arr[:,ManushandJointID.FingerPinkyTip], flexion_axis=flexion_axis)

    human_anatomy_joint_pos = np.zeros((MAX_JOINT, T), dtype=np.float64)
    human_anatomy_joint_pos[JointID.JThumbRoll] = thumb_roll_angle
    human_anatomy_joint_pos[JointID.JThumbAbad] = thumb_abd_angle
    human_anatomy_joint_pos[JointID.JThumbMCP] = thumb_mcp_flex
    human_anatomy_joint_pos[JointID.JThumbPIP] = thumb_pip_flex
    human_anatomy_joint_pos[JointID.JThumbDIP] = thumb_dip_flex
    human_anatomy_joint_pos[JointID.JIndexAbad] = index_abd
    human_anatomy_joint_pos[JointID.JIndexMCP] = index_mcp_flex
    human_anatomy_joint_pos[JointID.JIndexPIP] = index_pip_flex
    human_anatomy_joint_pos[JointID.JIndexDIP] = index_dip_flex
    human_anatomy_joint_pos[JointID.JMiddleAbad] = middle_abd
    human_anatomy_joint_pos[JointID.JMiddleMCP] = middle_mcp_flex
    human_anatomy_joint_pos[JointID.JMiddlePIP] = middle_pip_flex
    human_anatomy_joint_pos[JointID.JMiddleDIP] = middle_dip_flex
    human_anatomy_joint_pos[JointID.JRingMCP] = ring_mcp_flex
    human_anatomy_joint_pos[JointID.JRingPIP] = ring_pip_flex
    human_anatomy_joint_pos[JointID.JRingDIP] = ring_dip_flex
    human_anatomy_joint_pos[JointID.JPinkyMCP] = pinky_mcp_flex
    human_anatomy_joint_pos[JointID.JPinkyPIP] = pinky_pip_flex
    human_anatomy_joint_pos[JointID.JPinkyDIP] = pinky_dip_flex

    return human_anatomy_joint_pos

def _scale(max_in: float, min_in: float, value_in: float, max_out: float, min_out: float) -> float:
    """Linearly scales an input value from one range to another."""
    # Clamp the input value to its valid range first
    clamped_value = max(min_in, min(max_in, value_in))
    
    # Perform scaling
    range_in = max_in - min_in
    range_out = max_out - min_out
    
    # Avoid division by zero
    if range_in == 0:
        return min_out
    
    # Calculate ratio and scale
    ratio = (clamped_value - min_in) / range_in
    result = min_out + ratio * range_out
    
    return result

def _vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute the unsigned angle between two vectors.

    Args:
        v1: First 3D vector.
        v2: Second 3D vector.

    Returns:
        Angle between v1 and v2 in radians (0 ≤ angle ≤ π).
    """
    # Normalize to avoid numerical issues; add small epsilon to avoid division by zero.
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(math.acos(cos_theta))


def _signed_angle(v1: np.ndarray, v2: np.ndarray, normal: np.ndarray) -> float:
    """Compute the signed angle from v1 to v2 around the given normal.

    The result lies in (−π, π].  A positive value means that rotating v1
    towards v2 follows the right‑hand rule around ``normal``.

    Args:
        v1: First 3D vector (reference direction).
        v2: Second 3D vector.
        normal: Axis defining the plane of rotation.

    Returns:
        Signed angle from v1 to v2 in radians.
    """
    # Project both vectors onto the plane orthogonal to normal.
    n = normal / (np.linalg.norm(normal) + 1e-8)
    v1_proj = v1 - np.dot(v1, n) * n
    v2_proj = v2 - np.dot(v2, n) * n
    n1 = np.linalg.norm(v1_proj)
    n2 = np.linalg.norm(v2_proj)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    v1_unit = v1_proj / n1
    v2_unit = v2_proj / n2
    cross = np.cross(v1_unit, v2_unit)
    dot = float(np.dot(v1_unit, v2_unit))
    angle = math.atan2(np.linalg.norm(cross), dot)
    # Determine sign based on the direction of the cross product relative to normal.
    if np.dot(cross, n) < 0:
        angle = -angle
    return angle


def _compute_flexion(prev_joint: np.ndarray, joint: np.ndarray, next_joint: np.ndarray, flexion_axis: np.ndarray) -> float:
    """Compute flexion at a joint.

    Flexion is measured by taking the angle between the vectors from the
    joint to the neighbouring bones and subtracting it from π.  This
    yields 0 for an extended joint and a positive value as the joint
    flexes.

    Args:
        prev_joint: Position of the proximal bone anchor (closer to the
            wrist).
        joint: Position of the joint at which flexion is measured.
        next_joint: Position of the distal bone anchor (further from the
            wrist).
        flexion_axis: The axis around which flexion is measured.

    Returns:
        Flexion angle in radians (≥ 0).
    """
    # Vector along proximal bone (from joint towards proximal anchor).
    v_prox = prev_joint - joint
    # Vector along distal bone (from joint towards distal anchor).
    v_dist = next_joint - joint

    v_straight = -v_prox
    # theta = _vector_angle(v_prox, v_dist)
    theta = _signed_angle(v_straight, v_dist, flexion_axis)
    # Convert to flexion: pi minus the interior angle.
    assert -np.pi  <= theta <= np.pi
    # Clamp to non‑negative.
    return theta

def baselines_omnihand(human_joints: Iterable[Iterable[float]], human_anatomy_lower_limits: np.ndarray, human_anatomy_upper_limits: np.ndarray, is_right_hand: bool) -> Dict[str, Dict[str, float]]: 
    """Convert 21 Manus hand joint positions into OmniHand joint angles.

    The input should be a sequence of 21 3‑element vectors giving the
    Cartesian positions of the Manus joints in an arbitrary coordinate
    system.  The origin and scale do not matter—only the relative
    positions are used.  The returned dictionary contains robot joint
    names mapped to joint angles in radians.

    Args:
        human_joints: Iterable of 21 (x,y,z) tuples/lists/arrays.

    Returns:
        Mapping from OmniHand joint names to joint angles in radians.
    """
    # Convert to NumPy array for convenience and check shape.
    arr = np.asarray(human_joints, dtype=np.float64)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected 21×3 array, got {arr.shape}")
    
    if is_right_hand:
        O12HandLowerLimits = O12RightHandLowerLimits
        O12HandUpperLimits = O12RightHandUpperLimits
    else:
        O12HandLowerLimits = O12LeftHandLowerLimits
        O12HandUpperLimits = O12LeftHandUpperLimits

    # Precompute palm plane normal using index and pinky metacarpals.  The
    # plane is defined by vectors from the wrist to the bases of the
    # index and pinky fingers.
    wrist = arr[ManushandJointID.Wrist]
    index_base = arr[ManushandJointID.FingerIndexMetacarpal]
    pinky_base = arr[ManushandJointID.FingerPinkyMetacarpal]
    palm_vec1 = index_base - wrist
    palm_vec2 = pinky_base - wrist
    palm_normal = np.cross(palm_vec1, palm_vec2)
    if np.linalg.norm(palm_normal) < 1e-8:
        # Degenerate hand (colinear), choose arbitrary normal.
        palm_normal = np.array([0.0, 0.0, 1.0])
    palm_normal = palm_normal / np.linalg.norm(palm_normal)

    # Define a reference direction in the palm plane.  Use the middle
    # finger direction as the nominal forward direction.  The middle
    # finger base (joint 9) gives the metacarpal orientation; the
    # vector is projected onto the palm plane.
    middle_base = arr[ManushandJointID.FingerMiddleMetacarpal]
    middle_dir = middle_base - wrist
    # Project onto palm plane.
    middle_dir = middle_dir - np.dot(middle_dir, palm_normal) * palm_normal
    assert np.linalg.norm(middle_dir) > 1e-8, "Middle finger base is colinear with wrist and pinky base."
    middle_dir = middle_dir / (np.linalg.norm(middle_dir) + 1e-8)

    index_base = arr[ManushandJointID.FingerIndexMetacarpal]
    index_dir = index_base - wrist
    index_dir = index_dir - np.dot(index_dir, palm_normal) * palm_normal
    assert np.linalg.norm(index_dir) > 1e-8, "Index finger base is colinear with wrist and pinky base."
    index_dir = index_dir / (np.linalg.norm(index_dir) + 1e-8)

    
    # This axis is perpendicular to the palm plane, pointing right for a right hand.
    # By the right-hand rule, rotation around this axis is positive for flexion
    flexion_axis = np.cross(middle_dir, palm_normal)
    if not is_right_hand:
        flexion_axis = - flexion_axis

    # Container for computed manus metagloves joint angles and retargeted joint angles.
    hand_side_prefix = "L" if not is_right_hand else "R"
    joints: Dict[str, Dict[str, float]] = {
        f"{hand_side_prefix}_thumb": {},
        f"{hand_side_prefix}_index": {},
        f"{hand_side_prefix}_middle": {},
        f"{hand_side_prefix}_ring": {},
        f"{hand_side_prefix}_pinky": {}
    }
    manus_joint_angles: Dict[str, Dict[str, float]] = {
        f"{hand_side_prefix}_thumb": {},
        f"{hand_side_prefix}_index": {},
        f"{hand_side_prefix}_middle": {},
        f"{hand_side_prefix}_ring": {},
        f"{hand_side_prefix}_pinky": {}
    }

    ## NOTE: thumb is most complex.

    # ---------- Thumb joints ----------
    # Compute thumb base vectors for abduction and roll.  Use the
    # orientation of the proximal thumb bone (metacarpal→proximal) as
    # direction.

    thumb_mcp = arr[ManushandJointID.FingerThumbMetacarpal]
    thumb_pip = arr[ManushandJointID.FingerThumbProximal]
    thumb_dip = arr[ManushandJointID.FingerThumbDistal]
    thumb_tip = arr[ManushandJointID.FingerThumbTip]
    thumb_dir = thumb_pip - thumb_mcp

    # Roll: measure how much the thumb lifts out of the palm plane.  Use
    # the ratio of the component along the palm normal.  Positive roll
    # rotates the thumb pad upwards.  The URDF limits roll from 0 to
    # ~69.6° (1.21447 rad).
    thumb_mcp_to_dip = thumb_dip - thumb_mcp
    thumb_mcp_to_dip_proj_palm = thumb_mcp_to_dip - np.dot(thumb_mcp_to_dip, palm_normal) * palm_normal
    if np.linalg.norm(thumb_mcp_to_dip_proj_palm) < 1e-8:
        thumb_roll_angle = 0.0
    else:
        thumb_roll_angle = -_signed_angle(thumb_mcp_to_dip_proj_palm, middle_dir, palm_normal)
    if not is_right_hand:
        thumb_roll_angle = -thumb_roll_angle

    manus_joint_angles[f"{hand_side_prefix}_thumb"][f"{hand_side_prefix}_thumb_roll_joint"] = thumb_roll_angle
    joints[f"{hand_side_prefix}_thumb"][f"{hand_side_prefix}_thumb_roll_joint"] = float(_scale(max_in=human_anatomy_upper_limits[JointID.JThumbRoll], min_in=human_anatomy_lower_limits[JointID.JThumbRoll], value_in=thumb_roll_angle, max_out=O12HandUpperLimits[JointID.JThumbRoll], min_out=O12HandLowerLimits[JointID.JThumbRoll]))

    thumb_abd_angle = - _compute_flexion(wrist, thumb_mcp,thumb_pip, flexion_axis=flexion_axis)
    if not is_right_hand:
        thumb_abd_angle = -thumb_abd_angle
    # thumb_abd_angle = _signed_angle(thumb_mcp_to_dip_proj_palm, middle_dir, palm_normal)
    manus_joint_angles[f"{hand_side_prefix}_thumb"][f"{hand_side_prefix}_thumb_abad_joint"] = thumb_abd_angle
    joints[f"{hand_side_prefix}_thumb"][f"{hand_side_prefix}_thumb_abad_joint"] = float(_scale(max_in=human_anatomy_upper_limits[JointID.JThumbAbad], min_in=human_anatomy_lower_limits[JointID.JThumbAbad], value_in=thumb_abd_angle, max_out=O12HandUpperLimits[JointID.JThumbAbad], min_out=O12HandLowerLimits[JointID.JThumbAbad]))

    # Thumb MCP flexion: angle at joint 2 between the proximal and distal
    # segments.  Flexion for the thumb in the URDF is negative.
    thumb_mcp_flex =  -_compute_flexion(thumb_mcp, thumb_pip, thumb_dip, flexion_axis=flexion_axis)
    manus_joint_angles[f"{hand_side_prefix}_thumb"][f"{hand_side_prefix}_thumb_mcp_joint"] = thumb_mcp_flex
    joints[f"{hand_side_prefix}_thumb"][f"{hand_side_prefix}_thumb_mcp_joint"] = float(_scale(max_in=human_anatomy_upper_limits[JointID.JThumbMCP], min_in=human_anatomy_lower_limits[JointID.JThumbMCP], value_in=thumb_mcp_flex, max_out=O12HandUpperLimits[JointID.JThumbMCP], min_out=O12HandLowerLimits[JointID.JThumbMCP]))

    # Thumb IP flexion (joint 3).  Distribute this between the
    # robot's PIP and DIP using the mimic ratio (DIP ≈ 0.7999 × PIP).
    thumb_pip_flex = -_compute_flexion(thumb_mcp, thumb_pip, thumb_dip, flexion_axis=flexion_axis)
    manus_joint_angles[f"{hand_side_prefix}_thumb"][f"{hand_side_prefix}_thumb_pip_joint"] = thumb_pip_flex
    manus_joint_angles[f"{hand_side_prefix}_thumb"][f"{hand_side_prefix}_thumb_dip_joint"] = 0.799 * manus_joint_angles[f"{hand_side_prefix}_thumb"][f"{hand_side_prefix}_thumb_pip_joint"]
    pip_angle = float(_scale(max_in=human_anatomy_upper_limits[JointID.JThumbPIP], min_in=human_anatomy_lower_limits[JointID.JThumbPIP], value_in=thumb_pip_flex, max_out=O12HandUpperLimits[JointID.JThumbPIP], min_out=O12HandLowerLimits[JointID.JThumbPIP]))
    dip_angle = 0.799 * pip_angle
    joints[f"{hand_side_prefix}_thumb"][f"{hand_side_prefix}_thumb_pip_joint"] = pip_angle
    joints[f"{hand_side_prefix}_thumb"][f"{hand_side_prefix}_thumb_dip_joint"] = dip_angle

    # ---------- Index finger ----------
    # Index abduction around x‑axis.  Use signed angle of the base
    # direction relative to middle_dir.

    index_dir = arr[ManushandJointID.FingerIndexProximal] - arr[ManushandJointID.FingerIndexMetacarpal]  # proximal phalanx direction
    index_proj_palm = index_dir - np.dot(index_dir, palm_normal) * palm_normal
    if np.linalg.norm(index_proj_palm) < 1e-8:
        index_abd = 0.0
    else:
        index_abd = -_signed_angle(index_proj_palm, index_dir, palm_normal)
    manus_joint_angles[f"{hand_side_prefix}_index"][f"{hand_side_prefix}_index_abad_joint"] = index_abd
    joints[f"{hand_side_prefix}_index"][f"{hand_side_prefix}_index_abad_joint"] = float(_scale(max_in=human_anatomy_upper_limits[JointID.JIndexAbad], min_in=human_anatomy_lower_limits[JointID.JIndexAbad], value_in=index_abd, max_out=0.01, min_out=-0.01))

    # Index MCP flexion.
    index_mcp_flex = _compute_flexion(arr[ManushandJointID.Wrist], arr[ManushandJointID.FingerIndexMetacarpal], arr[ManushandJointID.FingerIndexDistal], flexion_axis=flexion_axis)
    manus_joint_angles[f"{hand_side_prefix}_index"][f"{hand_side_prefix}_index_mcp_joint"] = index_mcp_flex
    joints[f"{hand_side_prefix}_index"][f"{hand_side_prefix}_index_mcp_joint"] = float(_scale(max_in=human_anatomy_upper_limits[JointID.JIndexMCP], min_in=human_anatomy_lower_limits[JointID.JIndexMCP], value_in=index_mcp_flex, max_out=O12HandUpperLimits[JointID.JIndexMCP], min_out=O12HandLowerLimits[JointID.JIndexMCP]))
    # Index PIP flexion.
    index_pip_flex = _compute_flexion(arr[ManushandJointID.FingerIndexMetacarpal], arr[ManushandJointID.FingerIndexProximal], arr[ManushandJointID.FingerIndexDistal], flexion_axis=flexion_axis)
    manus_joint_angles[f"{hand_side_prefix}_index"][f"{hand_side_prefix}_index_pip_joint"] = index_pip_flex
    joints[f"{hand_side_prefix}_index"][f"{hand_side_prefix}_index_pip_joint"] = float(_scale(max_in=human_anatomy_upper_limits[JointID.JIndexPIP], min_in=human_anatomy_lower_limits[JointID.JIndexPIP], value_in=index_pip_flex, max_out=O12HandUpperLimits[JointID.JIndexPIP], min_out=O12HandLowerLimits[JointID.JIndexPIP]))

    # Index DIP flexion.
    index_dip_flex = 1.1162 * joints[f"{hand_side_prefix}_index"][f"{hand_side_prefix}_index_pip_joint"]
    manus_joint_angles[f"{hand_side_prefix}_index"][f"{hand_side_prefix}_index_dip_joint"] = 1.1162 * manus_joint_angles[f"{hand_side_prefix}_index"][f"{hand_side_prefix}_index_pip_joint"]
    joints[f"{hand_side_prefix}_index"][f"{hand_side_prefix}_index_dip_joint"] = float(np.clip(index_dip_flex, O12HandLowerLimits[JointID.JIndexDIP], O12HandUpperLimits[JointID.JIndexDIP]))

    # ---------- Middle finger ----------
    # Middle abduction around x‑axis.
    middle_dir2 = arr[ManushandJointID.FingerMiddleProximal] - arr[ManushandJointID.FingerMiddleMetacarpal]
    middle_proj_palm = middle_dir2 - np.dot(middle_dir2, palm_normal) * palm_normal
    if np.linalg.norm(middle_proj_palm) < 1e-8:
        middle_abd = 0.0
    else:
        middle_abd = -_signed_angle(middle_proj_palm, middle_dir, palm_normal)
    manus_joint_angles[f"{hand_side_prefix}_middle"][f"{hand_side_prefix}_middle_abad_joint"] = middle_abd
    joints[f"{hand_side_prefix}_middle"][f"{hand_side_prefix}_middle_abad_joint"] = float(_scale(max_in=human_anatomy_upper_limits[JointID.JMiddleAbad], min_in=human_anatomy_lower_limits[JointID.JMiddleAbad], value_in=middle_abd, max_out=0.08, min_out=-0.08))
    # Middle MCP flexion.
    middle_mcp_flex = _compute_flexion(arr[ManushandJointID.Wrist], arr[ManushandJointID.FingerMiddleMetacarpal], arr[ManushandJointID.FingerMiddleDistal], flexion_axis=flexion_axis)
    manus_joint_angles[f"{hand_side_prefix}_middle"][f"{hand_side_prefix}_middle_mcp_joint"] = middle_mcp_flex
    joints[f"{hand_side_prefix}_middle"][f"{hand_side_prefix}_middle_mcp_joint"] = float(_scale(max_in=human_anatomy_upper_limits[JointID.JMiddleMCP], min_in=human_anatomy_lower_limits[JointID.JMiddleMCP], value_in=middle_mcp_flex, max_out=O12HandUpperLimits[JointID.JMiddleMCP], min_out=O12HandLowerLimits[JointID.JMiddleMCP]))
    # Middle PIP flexion.
    middle_pip_flex = _compute_flexion(arr[ManushandJointID.FingerMiddleMetacarpal], arr[ManushandJointID.FingerMiddleProximal], arr[ManushandJointID.FingerMiddleDistal], flexion_axis=flexion_axis)
    manus_joint_angles[f"{hand_side_prefix}_middle"][f"{hand_side_prefix}_middle_pip_joint"] = middle_pip_flex
    joints[f"{hand_side_prefix}_middle"][f"{hand_side_prefix}_middle_pip_joint"] = float(_scale(max_in=human_anatomy_upper_limits[JointID.JMiddlePIP], min_in=human_anatomy_lower_limits[JointID.JMiddlePIP], value_in=middle_pip_flex, max_out=O12HandUpperLimits[JointID.JMiddlePIP], min_out=O12HandLowerLimits[JointID.JMiddlePIP]))
    # Middle DIP flexion.
    manus_joint_angles[f"{hand_side_prefix}_middle"][f"{hand_side_prefix}_middle_dip_joint"] = 1.1415 * manus_joint_angles[f"{hand_side_prefix}_middle"][f"{hand_side_prefix}_middle_pip_joint"]
    joints[f"{hand_side_prefix}_middle"][f"{hand_side_prefix}_middle_dip_joint"] = float(np.clip(1.1415 * joints[f"{hand_side_prefix}_middle"][f"{hand_side_prefix}_middle_pip_joint"], O12HandLowerLimits[JointID.JMiddleDIP], O12HandUpperLimits[JointID.JMiddleDIP]))

    # ---------- Ring finger ----------
    ring_mcp_flex = _compute_flexion(arr[ManushandJointID.Wrist], arr[ManushandJointID.FingerRingMetacarpal], arr[ManushandJointID.FingerRingDistal], flexion_axis=flexion_axis)
    manus_joint_angles[f"{hand_side_prefix}_ring"][f"{hand_side_prefix}_ring_mcp_joint"] = ring_mcp_flex
    joints[f"{hand_side_prefix}_ring"][f"{hand_side_prefix}_ring_mcp_joint"] = float(_scale(max_in=human_anatomy_upper_limits[JointID.JRingMCP], min_in=human_anatomy_lower_limits[JointID.JRingMCP], value_in=ring_mcp_flex, max_out=O12HandUpperLimits[JointID.JRingMCP], min_out=O12HandLowerLimits[JointID.JRingMCP]))
    # Apply mimic multipliers.
    ring_pip_mult = 0.8766
    ring_dip_mult = 0.9695
    manus_joint_angles[f"{hand_side_prefix}_ring"][f"{hand_side_prefix}_ring_pip_joint"] = ring_pip_mult * manus_joint_angles[f"{hand_side_prefix}_ring"][f"{hand_side_prefix}_ring_mcp_joint"]
    manus_joint_angles[f"{hand_side_prefix}_ring"][f"{hand_side_prefix}_ring_dip_joint"] = ring_dip_mult * manus_joint_angles[f"{hand_side_prefix}_ring"][f"{hand_side_prefix}_ring_pip_joint"]
    joints[f"{hand_side_prefix}_ring"][f"{hand_side_prefix}_ring_pip_joint"] = float(np.clip(ring_pip_mult * joints[f"{hand_side_prefix}_ring"][f"{hand_side_prefix}_ring_mcp_joint"], O12HandLowerLimits[JointID.JRingPIP], O12HandUpperLimits[JointID.JRingPIP]))
    joints[f"{hand_side_prefix}_ring"][f"{hand_side_prefix}_ring_dip_joint"] = float(np.clip(ring_dip_mult * joints[f"{hand_side_prefix}_ring"][f"{hand_side_prefix}_ring_mcp_joint"], O12HandLowerLimits[JointID.JRingDIP], O12HandUpperLimits[JointID.JRingDIP]))

    # ---------- Pinky finger ----------
    pinky_mcp_flex = _compute_flexion(arr[ManushandJointID.Wrist], arr[ManushandJointID.FingerPinkyMetacarpal], arr[ManushandJointID.FingerPinkyDistal], flexion_axis=flexion_axis)
    manus_joint_angles[f"{hand_side_prefix}_pinky"][f"{hand_side_prefix}_pinky_mcp_joint"] = pinky_mcp_flex
    joints[f"{hand_side_prefix}_pinky"][f"{hand_side_prefix}_pinky_mcp_joint"] = float(_scale(max_in=human_anatomy_upper_limits[JointID.JPinkyMCP], min_in=human_anatomy_lower_limits[JointID.JPinkyMCP], value_in=pinky_mcp_flex, max_out=O12HandUpperLimits[JointID.JPinkyMCP], min_out=O12HandLowerLimits[JointID.JPinkyMCP]))
    pinky_pip_mult = 0.8766
    pinky_dip_mult = 0.9695
    manus_joint_angles[f"{hand_side_prefix}_pinky"][f"{hand_side_prefix}_pinky_pip_joint"] = pinky_pip_mult * manus_joint_angles[f"{hand_side_prefix}_pinky"][f"{hand_side_prefix}_pinky_mcp_joint"]
    manus_joint_angles[f"{hand_side_prefix}_pinky"][f"{hand_side_prefix}_pinky_dip_joint"] = pinky_dip_mult * manus_joint_angles[f"{hand_side_prefix}_pinky"][f"{hand_side_prefix}_pinky_pip_joint"]
    joints[f"{hand_side_prefix}_pinky"][f"{hand_side_prefix}_pinky_pip_joint"] = float(np.clip(pinky_mcp_flex * pinky_pip_mult, O12HandLowerLimits[JointID.JPinkyPIP], O12HandUpperLimits[JointID.JPinkyPIP]))
    joints[f"{hand_side_prefix}_pinky"][f"{hand_side_prefix}_pinky_dip_joint"] = float(np.clip(pinky_mcp_flex * pinky_dip_mult, O12HandLowerLimits[JointID.JPinkyDIP], O12HandUpperLimits[JointID.JPinkyDIP]))

    def _init_angle_visualization():
        """Initialize the dual angle visualization plot."""
        global _angle_fig, _angle_ax, _angle_bars
        if _angle_fig is None:
            import matplotlib.pyplot as plt
            plt.ion()  # Enable interactive mode
            
            # Create figure with two subplots
            _angle_fig, _angle_ax = plt.subplots(2, 1, figsize=(12, 10))
            
            # Configure top subplot for Manus joint angles
            _angle_ax[0].set_title('Manus Joint Angles (Raw Human Data)')
            _angle_ax[0].set_ylabel('Angle (degrees)')
            _angle_ax[0].grid(True, alpha=0.3)
            
            # Configure bottom subplot for OmniHand joint angles
            _angle_ax[1].set_title('OmniHand Joint Angles (Retargeted)')
            _angle_ax[1].set_ylabel('Angle (radians)')
            _angle_ax[1].set_xlabel('Joint Names')
            _angle_ax[1].grid(True, alpha=0.3)
            
            # Get all joint names in order
            joint_names = []
            for finger in [f'{hand_side_prefix}_thumb', f'{hand_side_prefix}_index', f'{hand_side_prefix}_middle', f'{hand_side_prefix}_ring', f'{hand_side_prefix}_pinky']:
                if finger == f'{hand_side_prefix}_thumb':
                    joint_names.extend([f'{hand_side_prefix}_thumb_roll_joint', f'{hand_side_prefix}_thumb_abad_joint', f'{hand_side_prefix}_thumb_mcp_joint', f'{hand_side_prefix}_thumb_pip_joint', f'{hand_side_prefix}_thumb_dip_joint'])
                elif finger == f'{hand_side_prefix}_index':
                    joint_names.extend([f'{hand_side_prefix}_index_abad_joint', f'{hand_side_prefix}_index_mcp_joint', f'{hand_side_prefix}_index_pip_joint', f'{hand_side_prefix}_index_dip_joint'])
                elif finger == f'{hand_side_prefix}_middle':
                    joint_names.extend([f'{hand_side_prefix}_middle_abad_joint', f'{hand_side_prefix}_middle_mcp_joint', f'{hand_side_prefix}_middle_pip_joint', f'{hand_side_prefix}_middle_dip_joint'])
                elif finger == f'{hand_side_prefix}_ring':
                    joint_names.extend([f'{hand_side_prefix}_ring_mcp_joint', f'{hand_side_prefix}_ring_pip_joint', f'{hand_side_prefix}_ring_dip_joint'])
                else:  # pinky
                    joint_names.extend([f'{hand_side_prefix}_pinky_mcp_joint', f'{hand_side_prefix}_pinky_pip_joint', f'{hand_side_prefix}_pinky_dip_joint'])
            
            # Store as dictionary with manus and omnihand bars
            _angle_bars = {}
            
            # Create bar charts for both subplots
            x_pos = range(len(joint_names))
            
            # Manus angles bar chart (top subplot) - in degrees
            _angle_bars['manus'] = _angle_ax[0].bar(x_pos, [0] * len(joint_names), alpha=0.7, color='green')
            _angle_ax[0].set_xticks(x_pos)
            _angle_ax[0].set_xticklabels(joint_names, rotation=45, ha='right', fontsize=8)
            _angle_ax[0].set_ylim(-180, 180)  # Range for Manus data in degrees
            _angle_ax[0].set_yticks(range(-180, 181, 30))  # More compact grid: every 30 degrees
            
            # OmniHand angles bar chart (bottom subplot) - in radians
            _angle_bars['omnihand'] = _angle_ax[1].bar(x_pos, [0] * len(joint_names), alpha=0.7, color='blue')
            _angle_ax[1].set_xticks(x_pos)
            _angle_ax[1].set_xticklabels(joint_names, rotation=45, ha='right', fontsize=8)
            _angle_ax[1].set_ylim(-1.5, 1.5)  # Standard range for retargeted data in radians
            _angle_ax[1].set_yticks(np.arange(-1.5, 1.6, 0.3))  # More compact grid: every 0.3 radians
            
            plt.tight_layout()

    def _add_anatomy_limit_lines(human_anatomy_lower_limits, human_anatomy_upper_limits):
        """Add horizontal lines showing human anatomy limits for each joint."""
        global _angle_ax
        
        # Mapping from joint names to JointID indices
        joint_name_to_id = {
            f'{hand_side_prefix}_thumb_roll_joint': JointID.JThumbRoll,
            f'{hand_side_prefix}_thumb_abad_joint': JointID.JThumbAbad, 
            f'{hand_side_prefix}_thumb_mcp_joint': JointID.JThumbMCP,
            f'{hand_side_prefix}_thumb_pip_joint': JointID.JThumbPIP,
            f'{hand_side_prefix}_thumb_dip_joint': JointID.JThumbDIP,
            f'{hand_side_prefix}_index_abad_joint': JointID.JIndexAbad,
            f'{hand_side_prefix}_index_mcp_joint': JointID.JIndexMCP,
            f'{hand_side_prefix}_index_pip_joint': JointID.JIndexPIP,
            f'{hand_side_prefix}_index_dip_joint': JointID.JIndexDIP,
            f'{hand_side_prefix}_middle_abad_joint': JointID.JMiddleAbad,
            f'{hand_side_prefix}_middle_mcp_joint': JointID.JMiddleMCP,
            f'{hand_side_prefix}_middle_pip_joint': JointID.JMiddlePIP,
            f'{hand_side_prefix}_middle_dip_joint': JointID.JMiddleDIP,
            f'{hand_side_prefix}_ring_mcp_joint': JointID.JRingMCP,
            f'{hand_side_prefix}_ring_pip_joint': JointID.JRingPIP,
            f'{hand_side_prefix}_ring_dip_joint': JointID.JRingDIP,
            f'{hand_side_prefix}_pinky_mcp_joint': JointID.JPinkyMCP,
            f'{hand_side_prefix}_pinky_pip_joint': JointID.JPinkyPIP,
            f'{hand_side_prefix}_pinky_dip_joint': JointID.JPinkyDIP,
        }
        
        # Get joint names in the same order as the bars
        joint_names = []
        for finger in [f'{hand_side_prefix}_thumb', f'{hand_side_prefix}_index', f'{hand_side_prefix}_middle', f'{hand_side_prefix}_ring', f'{hand_side_prefix}_pinky']:
            if finger == f'{hand_side_prefix}_thumb':
                joint_names.extend([f'{hand_side_prefix}_thumb_roll_joint', f'{hand_side_prefix}_thumb_abad_joint', f'{hand_side_prefix}_thumb_mcp_joint', f'{hand_side_prefix}_thumb_pip_joint', f'{hand_side_prefix}_thumb_dip_joint'])
            elif finger == f'{hand_side_prefix}_index':
                joint_names.extend([f'{hand_side_prefix}_index_abad_joint', f'{hand_side_prefix}_index_mcp_joint', f'{hand_side_prefix}_index_pip_joint', f'{hand_side_prefix}_index_dip_joint'])
            elif finger == f'{hand_side_prefix}_middle':
                joint_names.extend([f'{hand_side_prefix}_middle_abad_joint', f'{hand_side_prefix}_middle_mcp_joint', f'{hand_side_prefix}_middle_pip_joint', f'{hand_side_prefix}_middle_dip_joint'])
            elif finger == f'{hand_side_prefix}_ring':
                joint_names.extend([f'{hand_side_prefix}_ring_mcp_joint', f'{hand_side_prefix}_ring_pip_joint', f'{hand_side_prefix}_ring_dip_joint'])
            else:  # pinky
                joint_names.extend([f'{hand_side_prefix}_pinky_mcp_joint', f'{hand_side_prefix}_pinky_pip_joint', f'{hand_side_prefix}_pinky_dip_joint'])
        
        # Add limit lines for each joint
        bar_width = 0.8  # Default matplotlib bar width
        for i, joint_name in enumerate(joint_names):
            if joint_name in joint_name_to_id:
                joint_id = joint_name_to_id[joint_name]
                
                # Get limits in radians and convert to degrees
                lower_limit_deg = human_anatomy_lower_limits[joint_id] * 180.0 / np.pi
                upper_limit_deg = human_anatomy_upper_limits[joint_id] * 180.0 / np.pi
                
                # Draw horizontal lines at the limits for this specific bar position
                x_start = i - bar_width/2
                x_end = i + bar_width/2
                
                # Lower limit line (red)
                _angle_ax[0].plot([x_start, x_end], [lower_limit_deg, lower_limit_deg], 
                                'r-', linewidth=2, alpha=0.8, label='Lower Limit' if i == 0 else "")
                
                # Upper limit line (orange)  
                _angle_ax[0].plot([x_start, x_end], [upper_limit_deg, upper_limit_deg], 
                                'orange', linewidth=2, alpha=0.8, label='Upper Limit' if i == 0 else "")
        
        # Add legend only once
        if len(_angle_ax[0].get_legend_handles_labels()[0]) > 0:
            _angle_ax[0].legend(loc='upper right', fontsize=8)

    def _update_angle_visualization(joints_dict, manus_joints_dict, human_anatomy_lower_limits=None, human_anatomy_upper_limits=None):
        """Update both angle visualizations with current joint angles."""
        global _angle_fig, _angle_ax, _angle_bars
            
        _init_angle_visualization()
        
        # Add anatomy limit lines if provided
        if human_anatomy_lower_limits is not None and human_anatomy_upper_limits is not None:
            _add_anatomy_limit_lines(human_anatomy_lower_limits, human_anatomy_upper_limits)
        
        # Flatten both joints dicts to get all angles in order
        def flatten_joints(joint_dict):
            angles = []
            joint_names = []
            for finger in [f'{hand_side_prefix}_thumb', f'{hand_side_prefix}_index', f'{hand_side_prefix}_middle', f'{hand_side_prefix}_ring', f'{hand_side_prefix}_pinky']:
                if finger in joint_dict:
                    for joint_name, angle in joint_dict[finger].items():
                        angles.append(angle)
                        joint_names.append(joint_name)
            return angles, joint_names
        
        # Get flattened data for both datasets
        manus_angles, _ = flatten_joints(manus_joints_dict)
        omnihand_angles, _ = flatten_joints(joints_dict)
        
        # Convert manus angles from radians to degrees
        manus_angles_degrees = [angle * 180.0 / np.pi for angle in manus_angles]
        
        # Update Manus angles (top subplot) - in degrees
        for bar, angle in zip(_angle_bars['manus'], manus_angles_degrees):
            bar.set_height(angle)
            # Color code: positive angles in green, negative in red
            bar.set_color('darkgreen' if angle >= 0 else 'darkred')
        
        # Update OmniHand angles (bottom subplot) - in radians
        for bar, angle in zip(_angle_bars['omnihand'], omnihand_angles):
            bar.set_height(angle)
            # Color code: positive angles in blue, negative in red
            bar.set_color('blue' if angle >= 0 else 'red')

        # Refresh the plot
        _angle_fig.canvas.draw()
        _angle_fig.canvas.flush_events()

    # Add visualization call at the end of baselines_omnihand function
    if VIS_BASELINE_ANGLE:
        _update_angle_visualization(joints, manus_joint_angles, human_anatomy_lower_limits, human_anatomy_upper_limits)

    return joints

def detect_pinch(manus_xyz) -> Tuple[bool, float, str]:
    """
    manus_xyz: (21, 3) numpy array of Manus hand joint positions.
    """
    thumb_tip_pos = manus_xyz[4]
    pinch_detection_candidates_indices = {}
    # iterate through MANUS_FINGER_INDICES to get PINCH_DETECTION_CANDIDATES's indices.
    for group_name, group in MANUS_FINGER_INDICES.items():
        for finger_name, joint_idx in group.items():
            if finger_name in PINCH_DETECTION_CANDIDATES:
                pinch_detection_candidates_indices[finger_name] = joint_idx

    best_dist, best_finger_key = np.inf, None
    for joint_name, joint_idx in pinch_detection_candidates_indices.items():
        d = np.linalg.norm(thumb_tip_pos - manus_xyz[joint_idx])
        if d < best_dist:
            best_dist, best_finger_key = d, joint_name
    return (best_dist < PINCH_THRESHOLD), best_dist, best_finger_key

def retarget_omnihand(human_joints: Iterable[Iterable[float]], robot_id, human_manus_anatomy: np.ndarray, is_right_hand: bool, prev_angles: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, float]:
    """
        manus_xyz: dict of 21 joint positions, each a length-3 numpy array
        returns: dict of robot joint names -> angles (radians)
    """
    human_anatomy_lower_limits = human_manus_anatomy.min(axis=1)
    human_anatomy_upper_limits = human_manus_anatomy.max(axis=1)

    q_baselines = baselines_omnihand(human_joints, human_anatomy_lower_limits, human_anatomy_upper_limits, is_right_hand)

    # 1) detect pinch pose.
    is_pinch, _, pinch_finger_key = detect_pinch(np.asarray(human_joints).astype(np.float64))

    # FIXME
    is_pinch = False

    if not is_pinch:
        current_angles = q_baselines
    else:
        # 2) pinch: refine thumb via IK with waypoint fallback.
        thumb_chain = R_chains.R_thumb.value
        pinch_finger_group_name = [x for x in MANUS_FINGER_INDICES if pinch_finger_key in MANUS_FINGER_INDICES[x]][0]
        pinch_finger_chain: Chain = getattr(R_chains, pinch_finger_group_name).value

        # Perform IK to refine the thumb and pinch finger positions
        # NOTE: the `Chain.forward_kinematics` method's param joints should be all joint positions, even if inactive.
        # Left Pad 0.0 for OriginLink and Right Pad 0.0 for the fingertip joint.
        pinch_finger_chain_qpos = [0.0, * q_baselines[pinch_finger_group_name].values(), 0.0]

        T_pinch_target = pinch_finger_chain.forward_kinematics(joints=pinch_finger_chain_qpos, full_kinematics=False)
        t_pinch_target = T_pinch_target[:3, 3]  # Extract position from the transformation matrix

        init_thumb_chain_qpos = [0.0, *q_baselines['R_thumb'].values(), 0.0]
        
        success, final_thumb_qpos, final_pinch_finger_chain_qpos = _try_waypoint_ik(
            thumb_chain=thumb_chain,
            pinch_finger_chain=pinch_finger_chain,
            init_thumb_qpos=init_thumb_chain_qpos,
            init_pinch_finger_qpos=pinch_finger_chain_qpos,
            target_position=t_pinch_target,
            robot_id=robot_id,
            num_waypoints=200,
            tolerance=0.01
        )
        
        if not success:
            # Fallback to original direct IK
            cprint.warn(f"Waypoint IK failed, falling back to direct IK, pinch_finger_key: {pinch_finger_key}")
            thumb_chain_full_qpos = thumb_chain.inverse_kinematics(
                target_position=t_pinch_target, 
                optimizer='scalar', 
                target_orientation=None, 
                orientation_mode=None, 
                initial_position=init_thumb_chain_qpos
            )
            final_thumb_qpos = thumb_chain_full_qpos

        if VIS_DBG_IK:
            import matplotlib.pyplot
            from mpl_toolkits.mplot3d import Axes3D
            ax = matplotlib.pyplot.figure().add_subplot(111, projection='3d')
            R_chains.R_thumb.value.plot(joints=final_thumb_qpos, ax = ax)
            R_chains.R_index.value.plot(joints=[0.0, *list(q_baselines['R_index'].values()), 0.0], ax = ax)
            R_chains.R_middle.value.plot(joints=[0.0, *list(q_baselines['R_middle'].values()), 0.0], ax = ax)
            R_chains.R_ring.value.plot(joints=[0.0, *list(q_baselines['R_ring'].values()), 0.0], ax = ax)
            R_chains.R_pinky.value.plot(joints=[0.0, *list(q_baselines['R_pinky'].values()), 0.0], ax = ax)
            matplotlib.pyplot.show()

        # Update the joint angles based on the IK results
        assert len(final_thumb_qpos) == q_baselines['R_thumb'].__len__() + 2, "IK result length mismatch with thumb joint angles."
        for joint_name, joint_angle in zip(q_baselines['R_thumb'].keys(), final_thumb_qpos.tolist()[1:-1]):
            q_baselines['R_thumb'][joint_name] = joint_angle
        for joint_name, joint_angle in zip(q_baselines[pinch_finger_group_name].keys(), final_pinch_finger_chain_qpos.tolist()[1:-1]):
            q_baselines[pinch_finger_group_name][joint_name] = joint_angle
    
        current_angles = q_baselines

    # 3) Check if we need gradual interpolation
    if prev_angles is not None:
        # Flatten both angle dictionaries to compare
        prev_flat = {joint_name: angle for finger_joints in prev_angles.values() for joint_name, angle in finger_joints.items()}
        current_flat = {joint_name: angle for finger_joints in current_angles.values() for joint_name, angle in finger_joints.items()}
        
        # Check if any joint exceeds the movement threshold
        dist = (np.array(list(current_flat.values())) - np.array(list(prev_flat.values())))**2
        dist = dist[[JointID.JThumbRoll, JointID.JThumbAbad, JointID.JThumbMCP, JointID.JThumbPIP, JointID.JIndexAbad, JointID.JIndexMCP, JointID.JIndexPIP, JointID.JMiddleAbad, JointID.JMiddleMCP, JointID.JMiddlePIP, JointID.JRingMCP, JointID.JPinkyMCP]]
        needs_interpolation = np.any(dist > MAX_ACTIVE_JOINT_MOVEMENT)   # Large overall jump

        if needs_interpolation:
            # Interpolate gradually over max 10 steps
            max_steps = 5
            for step in range(1, max_steps + 1):
                alpha = step / max_steps
                interpolated_angles = {}
                
                # Interpolate each finger group
                for finger_name in current_angles.keys():
                    interpolated_angles[finger_name] = {}
                    for joint_name in current_angles[finger_name].keys():
                        prev_val = prev_angles[finger_name][joint_name]
                        current_val = current_angles[finger_name][joint_name]
                        interpolated_val = prev_val + alpha * (current_val - prev_val)
                        interpolated_angles[finger_name][joint_name] = interpolated_val
                
                # breakpoint()
                yield interpolated_angles
            
        else:
            yield current_angles
    else:
        yield current_angles

def _generate_waypoints(start_pos: np.ndarray, end_pos: np.ndarray, num_waypoints: int) -> List[np.ndarray]:
    """Generate waypoints between start and end positions using ellipsoidal sampling for better IK success.
    
    Instead of linear interpolation, this uses an ellipsoid-based approach that samples points
    """
    waypoints = [] 
    
    # Calculate the direction and distance
    direction = end_pos - start_pos
    distance = np.linalg.norm(direction)
    
    if distance < 1e-6:
        # Start and end are the same, return intermediate points with small perturbations
        for i in range(num_waypoints):
            noise = np.random.normal(0, 0.001, 3)  # Small random perturbation
            waypoints.append(start_pos + noise)
        return waypoints
    
    # Create orthonormal basis
    unit_direction = direction / distance
    
    # Find two orthogonal vectors to create the ellipsoid cross-section
    if abs(unit_direction[2]) < 0.9:
        orthogonal1 = np.cross(unit_direction, np.array([0, 0, 1]))
    else:
        orthogonal1 = np.cross(unit_direction, np.array([1, 0, 0]))
    orthogonal1 = orthogonal1 / np.linalg.norm(orthogonal1)
    
    orthogonal2 = np.cross(unit_direction, orthogonal1)
    orthogonal2 = orthogonal2 / np.linalg.norm(orthogonal2)
    
    # Generate waypoints using ellipsoidal sampling
    for i in range(num_waypoints):
        # Linear parameter along the main axis
        t = (i + 1) / (num_waypoints + 1)
        
        # Current position along the straight line
        linear_point = start_pos + t * direction
        
        # Ellipsoid parameters: wider in the middle, narrower at the ends
        # Maximum radius at the middle (t=0.5)
        max_radius = distance * 0.2  # 20% of the total distance
        current_radius = max_radius * np.sin(np.pi * t)  # Sinusoidal profile
        
        # Generate random point on a circle in the orthogonal plane
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        
        # Sample radius using square root for uniform distribution in circle
        r = current_radius * np.sqrt(np.random.uniform(0, 1))
        
        # Convert to Cartesian coordinates in the orthogonal plane
        offset = r * (np.cos(theta) * orthogonal1 + np.sin(theta) * orthogonal2)
        
        # Add some variation along the main direction as well
        direction_noise = np.random.normal(0, distance * 0.05) * unit_direction
        
        waypoint = linear_point + offset + direction_noise
        waypoints.append(waypoint)
    
    # Add some waypoints using different sampling strategies for robustness
    extra_waypoints = []
    
    # Strategy 1: Bezier curve with random control points
    num_bezier = max(2, num_waypoints // 3)
    for i in range(num_bezier):
        t = (i + 1) / (num_bezier + 1)
        
        # Random control point offset from the straight line
        control_offset = np.random.normal(0, distance * 0.15, 3)
        control_point = start_pos + 0.5 * direction + control_offset
        
        # Quadratic Bezier curve: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
        bezier_point = ((1-t)**2 * start_pos + 
                       2*(1-t)*t * control_point + 
                       t**2 * end_pos)
        extra_waypoints.append(bezier_point)
    
    # Strategy 2: Spiral sampling around the straight line
    num_spiral = max(2, num_waypoints // 4)
    for i in range(num_spiral):
        t = (i + 1) / (num_spiral + 1)
        
        # Spiral parameters
        spiral_radius = distance * 0.1 * (1 - abs(2*t - 1))  # Max radius at middle
        spiral_angle = 4 * np.pi * t  # Two full rotations
        
        # Point on the straight line
        linear_point = start_pos + t * direction
        
        # Spiral offset
        spiral_offset = spiral_radius * (np.cos(spiral_angle) * orthogonal1 + 
                                       np.sin(spiral_angle) * orthogonal2)
        
        spiral_point = linear_point + spiral_offset
        extra_waypoints.append(spiral_point)
    
    # Combine all waypoints and shuffle for diversity
    all_waypoints = waypoints + extra_waypoints
    np.random.shuffle(all_waypoints)
    
    # Return the requested number of waypoints
    return all_waypoints[:num_waypoints]


def _compute_ik_ikpy(chain: Chain, target_pos: np.ndarray, initial_pos: List[float]) -> np.ndarray:
    """Compute IK using IKPy library."""
    return chain.inverse_kinematics(
        target_position=target_pos,
        optimizer='scalar',
        target_orientation=None,
        orientation_mode=None,
        initial_position=initial_pos
    )


def _compute_ik_pybullet(robot_id: int, end_effector_link_idx: int, target_pos: np.ndarray, 
                        joint_indices: List[int]) -> List[float]:
    """Compute IK using PyBullet library."""
    if not ENABLE_PYBULLET:
        raise RuntimeError("PyBullet is not enabled")
    
    ik_result = p.calculateInverseKinematics2(
        robot_id, 
        [end_effector_link_idx], 
        [target_pos]
    )
    return [ik_result[i] for i in joint_indices]


def _validate_ik_solution(chain: Chain, joint_angles: np.ndarray, target_pos: np.ndarray, 
                         tolerance: float) -> bool:
    """Validate IK solution by checking forward kinematics error."""
    fk_result = chain.forward_kinematics(joints=joint_angles, full_kinematics=False)
    actual_pos = fk_result[:3, 3]
    error = np.linalg.norm(actual_pos - target_pos)
    return error <= tolerance


class IKSolver:
    """Abstract base class for IK solvers."""
    
    def solve(self, chain: Chain, target_pos: np.ndarray, initial_pos: List[float], 
              robot_id: int = None, **kwargs) -> np.ndarray:
        raise NotImplementedError


class IKPySolver(IKSolver):
    """IK solver using IKPy library."""
    
    def solve(self, chain: Chain, target_pos: np.ndarray, initial_pos: List[float], 
              robot_id: int = None, **kwargs) -> np.ndarray:
        return _compute_ik_ikpy(chain, target_pos, initial_pos)


class PyBulletSolver(IKSolver):
    """IK solver using PyBullet library."""
    
    def __init__(self, end_effector_link_name: str):
        self.end_effector_link_name = end_effector_link_name
    
    def solve(self, chain: Chain, target_pos: np.ndarray, initial_pos: List[float], 
              robot_id: int = None, **kwargs) -> np.ndarray:
        if robot_id is None:
            raise ValueError("robot_id is required for PyBullet solver")
        
        end_effector_link_idx = LINK_NAME_TO_INDEX_MAP[self.end_effector_link_name]
        
        # For thumb chain, we need joints 0-4 (excluding origin and tip)
        if 'thumb' in self.end_effector_link_name.lower():
            joint_indices = list(range(5))
        else:
            # For other fingers, adjust indices as needed
            joint_indices = list(range(len(initial_pos) - 2))
        
        ik_result = _compute_ik_pybullet(robot_id, end_effector_link_idx, target_pos, joint_indices)
        # Pad with origin and tip joints for compatibility
        return np.asarray([0.0] + ik_result + [0.0])

def add_non_physical_primitive(shape_type, position, size, color=[1, 1, 1, 1], orientation=None):
    """
    Add a non-physical geometric primitive to the scene.
    
    Args:
        shape_type: p.GEOM_SPHERE, p.GEOM_BOX, p.GEOM_CYLINDER, p.GEOM_CAPSULE, etc.
        position: [x, y, z] position in world coordinates
        size: dimensions of the shape (depends on shape type)
        color: [r, g, b, a] color with transparency
        orientation: quaternion [x, y, z, w] (default: no rotation)
    """
    if orientation is None:
        orientation = [0, 0, 0, 1]
    
    # Create visual shape only (no collision shape)
    visual_shape_id = p.createVisualShape(
        shapeType=shape_type,
        rgbaColor=color,
        visualFramePosition=[0, 0, 0],
        visualFrameOrientation=[0, 0, 0, 1],
        **get_size_parameter(shape_type, size)
    )
    
    # Create multi-body with mass=0 to make it non-physical
    obj_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=position,
        baseOrientation=orientation
    )
    
    return obj_id

def get_size_parameter(shape_type, size):
    """Helper to get the correct size parameter for each shape type"""
    if shape_type == p.GEOM_SPHERE:
        return {'radius': size}
    elif shape_type == p.GEOM_BOX:
        return {'halfExtents': [size/2, size/2, size/2]}
    elif shape_type == p.GEOM_CYLINDER:
        return {'radius': size[0], 'length': size[1]}
    elif shape_type == p.GEOM_CAPSULE:
        return {'radius': size[0], 'length': size[1]}
    else:
        return {'radius': size}  # default

def _try_waypoint_ik(thumb_chain: Chain, pinch_finger_chain: Chain, 
                    init_thumb_qpos: List[float], init_pinch_finger_qpos: List[float],
                    target_position: np.ndarray, robot_id: int,
                    num_waypoints: int = 200, tolerance: float = 0.01) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Try waypoint-based IK approach for pinch motion.
    
    Returns:
        Tuple of (success, final_thumb_joint_angles, final_pinch_finger_joint_angles)
    """
    # Get current thumb tip position
    current_thumb_fk = thumb_chain.forward_kinematics(joints=init_thumb_qpos, full_kinematics=False)
    current_thumb_tip = current_thumb_fk[:3, 3]
    
    # Generate waypoints from current thumb tip to target
    # breakpoint()
    waypoints = _generate_waypoints(current_thumb_tip, target_position, num_waypoints)

    # Create solvers
    solvers = [IKPySolver()]
    if ENABLE_PYBULLET:
        solvers.append(PyBulletSolver('R_thumb_distal_vertices_virtual_coordinates'))
    
    # Test waypoints from furthest to closest
    for waypoint in reversed(waypoints):
        for solver in solvers:
            try:
                # Try IK for thumb to reach waypoint
                thumb_ik_result = solver.solve(
                    chain=thumb_chain,
                    target_pos=waypoint,
                    initial_pos=init_thumb_qpos,
                    robot_id=robot_id
                )
                
                # Validate thumb IK solution
                if not _validate_ik_solution(thumb_chain, thumb_ik_result, waypoint, tolerance):
                    continue

                # Try IK for the pinch finger to reach the waypoint
                pinch_finger_ik_result = solver.solve(
                    chain=pinch_finger_chain,
                    target_pos=waypoint,
                    initial_pos=init_pinch_finger_qpos,
                    robot_id=robot_id
                )
                
                # Validate pinch finger IK solution
                if not _validate_ik_solution(pinch_finger_chain, pinch_finger_ik_result, waypoint, tolerance):
                    continue

                # If both IKs are valid, return success
                cprint.info(f"Waypoint IK succeeded for waypoint {waypoint} with solver {solver.__class__.__name__}")
                return True, thumb_ik_result, pinch_finger_ik_result
            
            except Exception as e:
                # IK failed for this solver/waypoint combination
                continue
    
    return False, np.array(init_thumb_qpos), np.array(init_pinch_finger_qpos)

def main(is_right_hand: bool) -> None:
    """Example usage.

    When executed as a script this function generates random joint
    positions and prints the resulting robot joint angles.  In real
    applications you would replace the random data with actual
    measurements from a Manus glove.
    """
 
    human_arr = np.load("data/Aug_5_2025/manus_mocap_test_long_fps_10.npy")
    human_manus_anatomy = _manus_mocap_anatomy(human_joints=human_arr)
    import time
    
    if ENABLE_MANUS_VIS:
        # Set up matplotlib figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate global bounds for consistent axis limits across all frames
        all_frames_data = human_arr.reshape(-1, 3)
        min_x, max_x = np.min(all_frames_data[:, 0]), np.max(all_frames_data[:, 0])
        min_y, max_y = np.min(all_frames_data[:, 1]), np.max(all_frames_data[:, 1])
        min_z, max_z = np.min(all_frames_data[:, 2]), np.max(all_frames_data[:, 2])
        
        # Make the plot cubic with consistent bounds
        max_range = np.array([max_x-min_x, max_y-min_y, max_z-min_z]).max() / 2.0
        mid_x = (max_x+min_x) * 0.5
        mid_y = (max_y+min_y) * 0.5
        mid_z = (max_z+min_z) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Initialize empty plot elements that we'll update
        joints_scatter = ax.scatter([], [], [], c='red', marker='o')
        bone_lines = []
        for _ in MANUS_SKELETON_BONES:
            line, = ax.plot([], [], [], 'b-')
            bone_lines.append(line)
        
    prev_angles = None
    try:
        for frame_idx in range(len(human_arr)):
            # FIXME: skip first 120 frames to directly go into pinch pose.
            # if frame_idx < 440:
            #     continue
            skeleton_data = human_arr[frame_idx].reshape(21, 3)

            if ENABLE_MANUS_VIS:
                # Update matplotlib visualization
                ax.set_title(f'Manus Hand Skeleton - Frame {frame_idx + 1}/{len(human_arr)}')
                # Update scatter plot
                joints_scatter._offsets3d = (skeleton_data[:, 0], skeleton_data[:, 1], skeleton_data[:, 2])
                # Update bone lines
                for i, bone in enumerate(MANUS_SKELETON_BONES):
                    start_joint = skeleton_data[bone[0]]
                    end_joint = skeleton_data[bone[1]]
                    bone_lines[i].set_data_3d([start_joint[0], end_joint[0]], 
                                            [start_joint[1], end_joint[1]], 
                                            [start_joint[2], end_joint[2]])
                
                # Update matplotlib display
                fig.canvas.draw()
                fig.canvas.flush_events()

            # Retarget current frame to robot joint angles
            for angles in retarget_omnihand(skeleton_data.tolist(), robot_id, JOINTS, human_manus_anatomy, is_right_hand, prev_angles):
                prev_angles = angles
                angles = {
                    joint_name: angle 
                    for finger_joints in angles.values() 
                    for joint_name, angle in finger_joints.items()
                }

                # Print current frame info
                # print(f"Frame {frame_idx + 1}/{len(human_arr)}:")
                # for name, value in angles.items():
                #     print(f"  {name}: {value:.3f} rad")

                if ENABLE_PYBULLET:
                    for joint_name, qpos in angles.items():
                        joint_id = get_joint_id_from_joint_name(JOINTS, joint_name)

                        joint_info = JOINTS[joint_id]
                        p.setJointMotorControl2(
                            robot_id,
                            joint_id,
                            p.POSITION_CONTROL,
                            targetPosition=qpos,
                            force=joint_info.maxForce if joint_info.maxForce else 10.0,
                            maxVelocity=joint_info.maxVelocity if joint_info.maxVelocity else 10.0
                        )

                    for i in range(100):
                        p.stepSimulation()
                    # time.sleep(1 / 240)
                
                # real robot control
                if ENABLE_REAL_ROBOT:
                    motor_ticks = driver.kin_ctrl.convert_joint_to_actuator(joint_pos=np.asarray(list(angles.values())), is_active_only=False)
                    driver.set_target_positions(motor_ticks=motor_ticks)

    except KeyboardInterrupt:
        print("\nAnimation stopped by user")
    finally:
        if ENABLE_MANUS_VIS:
            # Keep the final frame displayed
            plt.ioff()  # Turn off interactive mode
            plt.show()  # Show final state
        if ENABLE_PYBULLET:
            p.disconnect()
        if ENABLE_REAL_ROBOT:
            driver.close()


def manus_teleop_omnihand(robot_id: int, human_manus_anatomy: np.ndarray, is_right_hand: bool) -> Dict[str, float]:
    """
    Teleoperation function for OmniHand using Manus hand data.
    
    Args:
        robot_id: ID of the robot in the simulation.
        human_joints: Iterable of human joint positions.
        human_manus_anatomy: Numpy array of Manus hand anatomy limits.
    
    Returns:
        Dictionary of robot joint names and their angles.
    """
    from ..mocap.manus_mocap import ManusMocap
    mocap = ManusMocap()
    data = []
    
    if ENABLE_MANUS_VIS:
        # Set up matplotlib figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate global bounds for consistent axis limits across all frames
        human_arr = np.load(OFFLINE_MANUS_SKELETON_PATH)
        all_frames_data = human_arr.reshape(-1, 3)
        min_x, max_x = np.min(all_frames_data[:, 0]), np.max(all_frames_data[:, 0])
        min_y, max_y = np.min(all_frames_data[:, 1]), np.max(all_frames_data[:, 1])
        min_z, max_z = np.min(all_frames_data[:, 2]), np.max(all_frames_data[:, 2])
        
        # Make the plot cubic with consistent bounds
        max_range = np.array([max_x-min_x, max_y-min_y, max_z-min_z]).max() / 2.0
        mid_x = (max_x+min_x) * 0.5
        mid_y = (max_y+min_y) * 0.5
        mid_z = (max_z+min_z) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Initialize empty plot elements that we'll update
        joints_scatter = ax.scatter([], [], [], c='red', marker='o')
        bone_lines = []
        for _ in MANUS_SKELETON_BONES:
            line, = ax.plot([], [], [], 'b-')
            bone_lines.append(line)
        
    # assert ENABLE_REAL_ROBOT, "Real robot control is not enabled. Set ENABLE_REAL_ROBOT to True to control the robot."
    prev_angles = None
    for frame_idx in range(80000000):
        result = mocap.get()
        if result["result"] is not None:
            data.append(result["result"])
            skeleton_data = result["result"]

            if ENABLE_MANUS_VIS:
                # Update matplotlib visualization
                ax.set_title(f'Manus Hand Skeleton - Frame {frame_idx + 1}')
                
                # Update scatter plot
                joints_scatter._offsets3d = (skeleton_data[:, 0], skeleton_data[:, 1], skeleton_data[:, 2])
                
                # Update bone lines
                for i, bone in enumerate(MANUS_SKELETON_BONES):
                    start_joint = skeleton_data[bone[0]]
                    end_joint = skeleton_data[bone[1]]
                    bone_lines[i].set_data_3d([start_joint[0], end_joint[0]], 
                                            [start_joint[1], end_joint[1]], 
                                            [start_joint[2], end_joint[2]])
                
                # Update matplotlib display
                fig.canvas.draw()
                fig.canvas.flush_events()

            # Retarget current frame to robot joint angles
            for _angles in retarget_omnihand(skeleton_data.tolist(), robot_id, human_manus_anatomy, is_right_hand, prev_angles):
                angles = {
                    joint_name: angle
                    for finger_joints in _angles.values()
                    for joint_name, angle in finger_joints.items()
                }

                if ENABLE_PYBULLET:
                    for joint_name, qpos in angles.items():
                        joint_id = get_joint_id_from_joint_name(JOINTS, joint_name)

                        joint_info = JOINTS[joint_id]
                        p.setJointMotorControl2(
                            robot_id,
                            joint_id,
                            p.POSITION_CONTROL,
                            targetPosition=qpos,
                            force=joint_info.maxForce if joint_info.maxForce else 10.0,
                            maxVelocity=joint_info.maxVelocity if joint_info.maxVelocity else 10.0
                        )

                    for i in range(100):
                        p.stepSimulation()
                
                # real robot control
                if ENABLE_REAL_ROBOT:
                    motor_ticks = driver.kin_ctrl.convert_joint_to_actuator(joint_pos=np.asarray(list(angles.values())), is_active_only=False)
                    driver.set_target_positions(motor_ticks=motor_ticks)

            prev_angles = _angles

    driver.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--right_hand", action="store_true")
    parser.add_argument("--urdf_path", type=str, default="assets/o12_hand_description-main/urdf/o12_t1_right.urdf")
    args = parser.parse_args()
    if ENABLE_REAL_ROBOT:
        from ..control.o12_hand.can_py.can_controller import OmniHandDriver
        from ..control.o12_hand.py_sdk.src.constants import GestureID
        driver = OmniHandDriver(is_right_hand=args.right_hand, render=False, urdf_path=args.urdf_path, render_freq=60, mode='monitor')
 
    OFFLINE_MANUS_SKELETON_PATH = "data/Aug_5_2025/manus_mocap_test_long_fps_10_left_hand.npy"
    OFFLINE_MANUS_SKELETON_PATH = "data/Aug_19_2025/manus_mocap_test_long_fps_10_left_hand.npy"
    _setup_pybullet(urdf_path = args.urdf_path, is_right_hand = args.right_hand)
    from ..utils.ipdb_safety_net import ipdb_safety_net
    human_manus_anatomy = _manus_mocap_anatomy(human_joints=np.load(OFFLINE_MANUS_SKELETON_PATH), is_right_hand=args.right_hand)
    ipdb_safety_net()
    manus_teleop_omnihand(robot_id=0, human_manus_anatomy=human_manus_anatomy, is_right_hand=args.right_hand)
    # main(is_right_hand=args.right_hand)