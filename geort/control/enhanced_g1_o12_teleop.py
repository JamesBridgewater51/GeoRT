#!/usr/bin/env python3
"""
Enhanced Unified Teleoperation Script for Unitree G1 and O12 Dexterous Hand

This script provides a high-level interface to control a Unitree G1 robot
equipped with an O12 dexterous hand in a PyBullet simulation with proper
joint/link mapping and constraint creation.

Features:
- Comprehensive joint/link information collection using getJointInfo
- Proper parent/child link index mapping for constraint creation
- Debug and validation capabilities
- Multiple control modes (slider, exoskeleton, IK)

Author: GitHub Copilot (Enhanced)
Date: 2025-08-14
"""
import pybullet as p
import pybullet_data
import numpy as np
import time
import argparse
from enum import Enum
from typing import Dict, List, Tuple, Optional
from cprint import cprint
from geort.utils.bullet_utils import axiscreator
from geort.utils.ipdb_safety_net import ipdb_safety_net

# Import the comprehensive joint/link mapper
from geort.control.joint_link_mapper import JointLinkMapper, MultiBodyJointLinkMapper

# --- Import from O12 Hand Controller ---
# Ensure the GeoRT project is in your PYTHONPATH
from geort.control.o12_hand.can_py.can_controller import OmniHandDriver
from geort.control.o12_hand.py_sdk.src.constants import GestureID, RightJointNames, LeftJointNames
from geort.control.o12_hand.py_sdk.src.ctrl import O12HandCtrl

import os
import sys
sys.path.append(os.path.abspath("third_party/OpenWBT-ExoSkel"))

# --- Import from G1 Controller / Exoskeleton ---
try:
    from deploy.exo_skel.exo_skel import ExoSkeletonConfig, ExoSkeletonReader
except ImportError:
    cprint.warn("WARNING: Exoskeleton modules not available")
    ExoSkeletonConfig = None
    ExoSkeletonReader = None

# --- Helper Enum for Control Modes ---
class ControlMode(Enum):
    PYBULLET_SLIDER = 0
    EXOSKELETON = 1
    IK = 2

    def __str__(self):
        return self.name.replace('_', ' ').title()

ENABLE_DEBUG = False

class EnhancedUnifiedTeleop:
    """
    Enhanced unified controller for teleoperating the Unitree G1 robot with O12 hands.
    
    This version properly handles PyBullet joint/link mapping and constraint creation
    using comprehensive getJointInfo analysis.
    """
    
    def __init__(self, g1_urdf_path, lh_urdf_path=None, rh_urdf_path=None, 
                 exo_config_path=None, real_robot=False):
        self.real_robot = real_robot
        self.lh_urdf_path = lh_urdf_path
        self.rh_urdf_path = rh_urdf_path
        self.exo_config_path = exo_config_path
        
        # PyBullet body IDs
        self.g1_id = None
        self.lh_id = None  # Left hand
        self.rh_id = None  # Right hand
        
        # Multi-body joint/link mapper
        self.mapper = MultiBodyJointLinkMapper()
        
        # Control mode and interfaces
        self.control_mode = ControlMode.PYBULLET_SLIDER
        self.exo_reader = None
        self.real_hand_ctrl = None
        
        # Constraint IDs for tracking
        self.constraint_ids = []
        
        # UI elements
        self.sliders = []
        
        # G1 specific joint configurations
        self.g1_end_effector_link_name = "right_wrist_yaw_link"  # For IK
        self.ik_target_link_id = None
        
        # Initialize the simulation
        self._setup_simulation(g1_urdf_path)
        self._setup_joint_link_mappers()
        self._setup_multibody_dynamics()
        self._setup_collisions()
        self._setup_debug_axes()
        self._mount_hands()
        self._setup_control_interfaces()
        
    def _setup_simulation(self, g1_urdf_path):
        """Setup PyBullet simulation environment."""
        cprint.info("INFO: Setting up PyBullet simulation...")
        
        # Connect to PyBullet
        self.physics_client = p.connect(p.GUI, options="--title='Enhanced G1 + O12 Hand Teleop'")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        # Load robot bodies
        cprint.info(f"INFO: Loading G1 robot from: {g1_urdf_path}")
        self.g1_id = p.loadURDF(g1_urdf_path, [0, 0, 0.95], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        
        hand_flags = (p.URDF_USE_SELF_COLLISION |
            p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT |
            p.URDF_MERGE_FIXED_LINKS)
        if self.lh_urdf_path:
            cprint.info(f"INFO: Loading left hand from: {self.lh_urdf_path}")
            self.lh_id = p.loadURDF(self.lh_urdf_path,  useFixedBase=False, flags=hand_flags)

        if self.rh_urdf_path:
            cprint.info(f"INFO: Loading right hand from: {self.rh_urdf_path}")
            self.rh_id = p.loadURDF(self.rh_urdf_path,  useFixedBase=False, flags = hand_flags)

        cprint.ok("INFO: PyBullet simulation setup complete.")
        
    def _setup_joint_link_mappers(self):
        """Setup comprehensive joint/link mappers for all bodies."""
        cprint.info("INFO: Setting up joint/link mappers...")
        
        # Add all bodies to the multi-body mapper
        if self.g1_id is not None:
            self.mapper.add_body(self.g1_id, "G1")
            cprint.info(f"INFO: Added G1 robot (body ID: {self.g1_id}) to mapper")
            
        if self.lh_id is not None:
            self.mapper.add_body(self.lh_id, "L")
            cprint.info(f"INFO: Added left hand (body ID: {self.lh_id}) to mapper")
            
        if self.rh_id is not None:
            self.mapper.add_body(self.rh_id, "R")
            cprint.info(f"INFO: Added right hand (body ID: {self.rh_id}) to mapper")
            
        # Validate all mappings
        if ENABLE_DEBUG:
            cprint.info("INFO: Validating joint/link mappings...")
            self.mapper.print_all_debug_info()
        
        # Setup IK target if specified
        if self.g1_id is not None:
            g1_mapper = self.mapper.get_mapper(self.g1_id)
            if g1_mapper is not None:
                link_idx = g1_mapper.get_link_index_from_link_name(self.g1_end_effector_link_name)
                if link_idx is not None:
                    self.ik_target_link_id = link_idx
                    cprint.info(f"INFO: IK target link '{self.g1_end_effector_link_name}' -> index {link_idx}")
                else:
                    cprint.warn(f"WARNING: IK target link '{self.g1_end_effector_link_name}' not found")
            else:
                cprint.warn("WARNING: G1 mapper not available")

    def _setup_multibody_dynamics(self):
        """Scale down the mass and inertia of the multibody hands for better simulation stability."""
        if self.g1_id is not None:
            for link_name, link_index in self.mapper.get_mapper(self.g1_id).link_name_to_index.items():
                dyminfo_raw = p.getDynamicsInfo(self.g1_id, link_index)
                mass = dyminfo_raw[0]
                ix, iy, iz = dyminfo_raw[2]
                _coeff = 50
                p.changeDynamics(self.g1_id, link_index, mass=mass*_coeff, localInertiaDiagonal=[ix*_coeff, iy*_coeff, iz*_coeff])
        if self.lh_id is not None:
            for link_name, link_index in self.mapper.get_mapper(self.lh_id).link_name_to_index.items():
                # breakpoint()
                dyminfo_raw = p.getDynamicsInfo(self.lh_id, link_index)
                mass = dyminfo_raw[0]
                _coeff = 0.01
                ix, iy, iz = dyminfo_raw[2] # local inertia diagonal.
                p.changeDynamics(self.lh_id, link_index, mass=mass*_coeff, localInertiaDiagonal=[ix*_coeff, iy*_coeff, iz*_coeff])
                p.changeDynamics(self.lh_id, link_index, jointDamping=0.05, lateralFriction=1.0, rollingFriction=0.001, spinningFriction=0.001)
        if self.rh_id is not None:
            for link_name, link_index in self.mapper.get_mapper(self.rh_id).link_name_to_index.items():
                dyminfo_raw = p.getDynamicsInfo(self.lh_id, link_index)
                mass = dyminfo_raw[0]
                ix, iy, iz = dyminfo_raw[2] # local inertia diagonal.
                _coeff = 0.01
                p.changeDynamics(self.rh_id, link_index, mass=mass*_coeff, localInertiaDiagonal=[ix*_coeff, iy*_coeff, iz*_coeff])
                p.changeDynamics(self.lh_id, link_index, jointDamping=0.05, lateralFriction=1.0, rollingFriction=0.001, spinningFriction=0.001)

    def _setup_collisions(self):
        """Setup collision properties for all loaded bodies."""
        if self.lh_id is not None:
            g1_lwrist_link_idx = self.mapper.get_mapper(self.g1_id).get_link_index_from_link_name("left_wrist_yaw_link")
            p.setCollisionFilterPair(self.lh_id, self.g1_id, -1, -1, enableCollision=0)
        
        # Example: Disable collisions between left and right hands
        if self.rh_id is not None:
            g1_rwrist_link_idx = self.mapper.get_mapper(self.g1_id).get_link_index_from_link_name("right_wrist_yaw_link")
            p.setCollisionFilterPair(self.rh_id, self.g1_id, -1, -1, enableCollision=0)
            cprint.info("INFO: Disabled collisions between left and right hands")
        
        # Additional collision setup can be added here as needed
    def _setup_debug_axes(self):
        """Setup debug axes for all loaded bodies."""
        cprint.info("INFO: Setting up debug axes for all bodies...")
        
        if self.g1_id is not None:
            g1_mapper = self.mapper.get_mapper(self.g1_id)
            if g1_mapper is not None and hasattr(g1_mapper, 'link_name_to_index'):
                for link_name, link_idx in g1_mapper.link_name_to_index.items():
                    if link_name in ['left_wrist_yaw_link', 'right_wrist_yaw_link', 'left_wrist_pitch_link', 'right_wrist_pitch_link']:
                        axiscreator(bodyId=self.g1_id, linkId=link_idx)

    def _mount_hands(self):
        """Mount hands to G1 end-effectors using proper constraint creation."""
        if self.g1_id is None:
            cprint.warn("WARNING: G1 robot not loaded, skipping hand mounting")
            return
            
        cprint.info("INFO: Mounting hands to G1 end-effectors...")
        
        # Mount left hand to left wrist
        if self.lh_id is not None:
            # first rotate along x-axis by -90 degress, then rotate along z-axis by -90 degress.
            constraint_id = self._create_hand_constraint(
                wrist_joint_name="left_wrist_yaw_joint",
                hand_body_id=self.lh_id,
                hand_name="left",
                mount_offset=[0.1, 0, 0],
                mount_orientation=[-0.5, 0.5, -0.5, 0.5]
            )
            if constraint_id is not None:
                self.constraint_ids.append(constraint_id)
                
        # Mount right hand to right wrist
        if self.rh_id is not None:
            constraint_id = self._create_hand_constraint(
                wrist_joint_name="right_wrist_yaw_joint",
                hand_body_id=self.rh_id,
                hand_name="right",
                mount_offset=[0.1, 0, 0],
                mount_orientation=[-0.5, 0.5, -0.5, 0.5]
            )
            if constraint_id is not None:
                self.constraint_ids.append(constraint_id)
                
        cprint.ok(f"INFO: Hand mounting completed. Created {len(self.constraint_ids)} constraints.")
        
    def _create_hand_constraint(self, wrist_joint_name: str, hand_body_id: int, 
                              hand_name: str, mount_offset: List[float], 
                              mount_orientation: List[float]) -> Optional[int]:
        """
        Create a constraint to mount a hand to a wrist using proper link mapping.
        
        Args:
            wrist_joint_name: Name of the wrist joint on G1
            hand_body_id: PyBullet body ID of the hand
            hand_name: Name for logging (e.g., "left", "right")
            mount_offset: [x, y, z] offset from wrist link center
            mount_orientation: [roll, pitch, yaw] orientation in radians
            
        Returns:
            Constraint ID if successful, None if failed
        """
        
        if self.g1_id is None:
            cprint.err(f"ERROR: G1 robot not loaded", interrupt=False)
            return None
            
        g1_mapper = self.mapper.get_mapper(self.g1_id)
        if g1_mapper is None:
            cprint.err(f"ERROR: G1 mapper not available", interrupt=False)
            return None
        
        # Get the child link of the wrist joint (this is where we attach)
        wrist_link_idx = g1_mapper.get_link_index_from_joint_name(wrist_joint_name)
        if wrist_link_idx is None:
            cprint.err(f"ERROR: Cannot find wrist link for joint '{wrist_joint_name}'", interrupt=False)
            return None
            
        # Print debug information
        cprint(f"\nDEBUG: Creating {hand_name} hand constraint")
        # g1_mapper.print_constraint_debug_info(wrist_joint_name)
        
        # Create the constraint
        # breakpoint()
        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.g1_id,
            parentLinkIndex=wrist_link_idx,
            childBodyUniqueId=hand_body_id,
            childLinkIndex=-1,  # Base of the hand (link index -1 means base)
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=mount_offset,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=mount_orientation,
            childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        cprint.ok(f"INFO: {hand_name.capitalize()} hand mounted successfully (constraint ID: {constraint_id})")
        return constraint_id
     
    
    def _setup_control_interfaces(self):
        """Setup control interfaces based on available hardware."""
        cprint.info("INFO: Setting up control interfaces...")
        
        # Setup exoskeleton interface if config provided
        if (self.exo_config_path and os.path.exists(self.exo_config_path) and 
            ExoSkeletonReader is not None and ExoSkeletonConfig is not None):
            try:
                exo_config = ExoSkeletonConfig(self.exo_config_path)
                self.exo_reader = ExoSkeletonReader(exo_config, left_device='/dev/ttyUSB0', right_device='/dev/ttyUSB1')
                cprint.ok("INFO: Exoskeleton interface initialized")
            except Exception as e:
                cprint.warn(f"WARNING: Failed to initialize exoskeleton: {e}")
                self.exo_reader = None
        else:
            self.exo_reader = None
        
        # Setup real robot control interfaces if enabled
        if self.real_robot:
            try:
                # Initialize G1 robot controller - placeholder for actual G1 interface
                # self.g1_runner = Runner_online_real(config, args)
                cprint.info("INFO: G1 robot controller would be initialized here")
                
                # Initialize O12 hand controllers
                self.lh_hand_driver = OmniHandDriver(is_right_hand=False, render=False)
                self.rh_hand_driver = OmniHandDriver(is_right_hand=True, render=False)
                cprint.ok("INFO: O12 hand controllers initialized")
                
            except Exception as e:
                cprint.err(f"ERROR: Failed to initialize real robot interfaces: {e}", interrupt=False)
                self.lh_hand_driver = None
                self.rh_hand_driver = None
        else:
            self.lh_hand_driver = None
            self.rh_hand_driver = None
        
        # Default to PyBullet slider control
        self.control_mode = ControlMode.PYBULLET_SLIDER
        self._setup_pybullet_ui()
        
    def _setup_pybullet_ui(self):
        """Setup PyBullet GUI sliders for manual control."""
        cprint.info("INFO: Setting up PyBullet UI sliders...")
        
        # Create sliders for all controllable joints
        self.sliders = []
        
        for body_id, mapper in self.mapper.body_mappers.items():
            body_name = self.mapper.body_names[body_id]
            controllable_joints = mapper.get_controllable_joints()
            
            for joint_info in controllable_joints:
                slider_id = p.addUserDebugParameter(
                    paramName=f"{body_name}_{joint_info.joint_name}",
                    rangeMin=joint_info.joint_lower_limit,
                    rangeMax=joint_info.joint_upper_limit,
                    startValue=(joint_info.joint_lower_limit + joint_info.joint_upper_limit) / 2
                )
                
                self.sliders.append({
                    'id': slider_id,
                    'body_id': body_id,
                    'joint_index': joint_info.joint_index,
                    'joint_name': joint_info.joint_name,
                    'limits': (joint_info.joint_lower_limit, joint_info.joint_upper_limit)
                })
        
        cprint.ok(f"INFO: Created {len(self.sliders)} control sliders")
        
        # Add control mode selector, as a button
        self.mode_selector = p.addUserDebugParameter(
            paramName="Control_Mode",
            rangeMin=1,
            rangeMax=0,
            startValue=0
        )
    
    def get_joint_targets_from_sliders(self) -> Dict[int, Dict[int, float]]:
        """Get joint targets from PyBullet sliders."""
        targets = {}
        
        for slider in self.sliders:
            body_id = slider['body_id']
            joint_index = slider['joint_index']
            slider_value = p.readUserDebugParameter(slider['id'])
            
            if body_id not in targets:
                targets[body_id] = {}
            targets[body_id][joint_index] = slider_value
            
        return targets
    
    def apply_joint_targets(self, targets: Dict[int, Dict[int, float]]):
        """Apply joint targets to the simulation with body-specific control parameters."""
        for body_id, joint_targets in targets.items():
            # Determine control parameters based on body type
            if body_id == self.g1_id:
                # G1 arm joints: higher gains and forces for larger joints
                kp, kd, max_force = 1.0, 1.5, 1000.0
                max_velocity = 2.0
            elif body_id == self.lh_id or body_id == self.rh_id:
                # O12 hand finger joints: lower gains and forces for delicate control
                kp, kd, max_force = 0.25, 1.0, 100.0
                max_velocity = 0.1
            else:
                # Default fallback
                kp, kd, max_force = 0.2, 1.0, 50.0
                max_velocity = 1.0
                
            for joint_index, target_position in joint_targets.items():
                p.setJointMotorControl2(
                    bodyUniqueId=body_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_position,
                    maxVelocity=max_velocity,
                    force=max_force,
                    positionGain=kp,
                    velocityGain=kd
                )
    
    def get_joint_targets_from_exoskeleton(self) -> Optional[Dict[int, Dict[int, float]]]:
        """Get joint targets from exoskeleton input."""
        if self.exo_reader is None:
            return None
            
        try:
            # Read exoskeleton angles (14 values: 7 left arm + 7 right arm)
            exo_angles = self.exo_reader.read()
            if exo_angles is None or len(exo_angles) < 14:
                return None
                
            targets = {}
            
            # Map exoskeleton data to G1 arm joints
            if self.g1_id is not None:
                g1_mapper = self.mapper.get_mapper(self.g1_id)
                if g1_mapper is not None:
                    targets[self.g1_id] = {}
                    
                    # Define the mapping from exoskeleton to G1 joint names
                    # Based on exo_skel.yaml configuration
                    exo_to_g1_joint_mapping = [
                        # Left arm (indices 0-6 in exo_angles)
                        ("left_shoulder_pitch_joint", 0),
                        ("left_shoulder_roll_joint", 1), 
                        ("left_shoulder_yaw_joint", 2),
                        ("left_elbow_joint", 3),
                        ("left_wrist_roll_joint", 4),
                        ("left_wrist_pitch_joint", 5),
                        ("left_wrist_yaw_joint", 6),
                        # Right arm (indices 7-13 in exo_angles)
                        ("right_shoulder_pitch_joint", 7),
                        ("right_shoulder_roll_joint", 8),
                        ("right_shoulder_yaw_joint", 9), 
                        ("right_elbow_joint", 10),
                        ("right_wrist_roll_joint", 11),
                        ("right_wrist_pitch_joint", 12),
                        ("right_wrist_yaw_joint", 13)
                    ]
                    
                    # Apply exoskeleton readings to corresponding G1 joints
                    for joint_name, exo_index in exo_to_g1_joint_mapping:
                        if exo_index < len(exo_angles):
                            joint_info = g1_mapper.get_joint_info(joint_name)
                            if joint_info:
                                # Apply joint limits and smoothing
                                target_angle = np.clip(
                                    exo_angles[exo_index],
                                    joint_info.joint_lower_limit,
                                    joint_info.joint_upper_limit
                                )
                                targets[self.g1_id][joint_info.joint_index] = target_angle
                                
                    cprint.ok(f"INFO: Mapped {len(targets[self.g1_id])} exoskeleton joints to G1")
            
            return targets
            
        except Exception as e:
            cprint.warn(f"WARNING: Error reading exoskeleton data: {e}")
            return None
    
    def send_to_real_robot(self, targets: Dict[int, Dict[int, float]]):
        """Send commands to real robot hardware."""
        if not self.real_robot:
            return
            
        try:
            # Send G1 robot arm commands - placeholder for actual G1 interface
            if self.g1_id in targets:
                g1_targets = targets[self.g1_id]
                cprint.info(f"INFO: Would send {len(g1_targets)} joint commands to G1 robot")
                # TODO: Implement actual G1 robot control interface here
                # This would use the Runner_online_real class or similar
            
            # Send left hand commands
            if self.lh_id in targets and self.lh_hand_driver is not None:
                left_hand_targets = targets[self.lh_id]
                
                # Convert PyBullet joint targets to O12 hand format
                left_joint_angles = self._extract_hand_joint_angles(left_hand_targets, is_right_hand=False)
                if left_joint_angles is not None:
                    # Convert joint angles to motor commands
                    motor_ticks = self.lh_hand_driver.kin_ctrl.convert_joint_to_actuator(
                        left_joint_angles, is_active_only=True
                    )
                    # Send motor commands to physical hand
                    self.lh_hand_driver.set_target_positions(motor_ticks.tolist())
                    cprint.ok(f"INFO: Sent commands to left O12 hand")
                    
            # Send right hand commands  
            if self.rh_id in targets and self.rh_hand_driver is not None:
                right_hand_targets = targets[self.rh_id]
                
                # Convert PyBullet joint targets to O12 hand format
                right_joint_angles = self._extract_hand_joint_angles(right_hand_targets, is_right_hand=True)
                if right_joint_angles is not None:
                    # Convert joint angles to motor commands
                    motor_ticks = self.rh_hand_driver.kin_ctrl.convert_joint_to_actuator(
                        right_joint_angles, is_active_only=True
                    )
                    # Send motor commands to physical hand
                    self.rh_hand_driver.set_target_positions(motor_ticks.tolist())
                    cprint.ok(f"INFO: Sent commands to right O12 hand")
                    
        except Exception as e:
            cprint.err(f"ERROR: Failed to send commands to real robot: {e}", interrupt=False)
    
    def _extract_hand_joint_angles(self, hand_targets: Dict[int, float], is_right_hand: bool) -> Optional[np.ndarray]:
        """Extract and order hand joint angles from PyBullet targets."""
        try:
            hand_id = self.rh_id if is_right_hand else self.lh_id
            if hand_id is None:
                return None
                
            hand_mapper = self.mapper.get_mapper(hand_id)
            if hand_mapper is None:
                return None
                
            # Get expected joint names for O12 hand
            expected_joint_names = RightJointNames if is_right_hand else LeftJointNames
            joint_angles = np.zeros(len(expected_joint_names))
            
            # Map PyBullet joint indices to ordered joint angles
            for joint_index, target_angle in hand_targets.items():
                # Find joint info by searching through all joints
                for joint_info in hand_mapper.get_controllable_joints():
                    if joint_info.joint_index == joint_index and joint_info.joint_name in expected_joint_names:
                        joint_name_index = expected_joint_names.index(joint_info.joint_name)
                        joint_angles[joint_name_index] = target_angle
                        break
                    
            return joint_angles
            
        except Exception as e:
            cprint.warn(f"WARNING: Error extracting hand joint angles: {e}")
            return None
                
       
    def run(self):
        """Main control loop."""
        cprint.info("INFO: Starting enhanced unified teleoperation...")
        cprint.info("INFO: Available control modes:")
        for mode in ControlMode:
            cprint(f"  {mode.value}: {mode}")
        
        last_mode = self.control_mode
        
        try:
            while True:
                # Check for mode changes
                current_mode_value = int(p.readUserDebugParameter(self.mode_selector)) % len(ControlMode)
                current_mode = list(ControlMode)[current_mode_value]
                
                if current_mode != last_mode:
                    cprint.info(f"INFO: Switching to {current_mode} mode")
                    self.control_mode = current_mode
                    last_mode = current_mode
                
                # Get joint targets based on current mode
                targets = None
                
                if self.control_mode == ControlMode.PYBULLET_SLIDER:
                    targets = self.get_joint_targets_from_sliders()
                    
                elif self.control_mode == ControlMode.EXOSKELETON:
                    targets = self.get_joint_targets_from_exoskeleton()
                    
                elif self.control_mode == ControlMode.IK:
                    # IK mode - placeholder for inverse kinematics control
                    targets = self.get_joint_targets_from_sliders()  # Fallback to sliders
                
                # Apply targets to simulation
                if targets:
                    cprint.ok(f"INFO: Applying targets in {self.control_mode} mode")
                    self.apply_joint_targets(targets)
                    
                    # Send to real robot if enabled
                    if self.real_robot:
                        self.send_to_real_robot(targets)
                
                # Step simulation
                for i in range(50):
                    p.stepSimulation()
                time.sleep(1.0/240.0)  # 240 Hz simulation
                
        except KeyboardInterrupt:
            cprint.info("\nINFO: Teleoperation stopped by user")
        except Exception as e:
            cprint.err(f"ERROR: Unexpected error in main loop: {e}", interrupt=False)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        cprint.info("INFO: Cleaning up...")
        
        # Remove constraints
        for constraint_id in self.constraint_ids:
            try:
                p.removeConstraint(constraint_id)
            except:
                pass
        
        # Close exoskeleton interface
        if self.exo_reader:
            self.exo_reader.close()

        
        # Close real robot interface
        if self.real_hand_ctrl:
            if self.lh_hand_driver:
                self.lh_hand_driver.close()
            if self.rh_hand_driver:
                self.rh_hand_driver.close()
        
        # Disconnect PyBullet
        try:
            p.disconnect()
        except:
            pass
        
        cprint.ok("INFO: Cleanup complete")
    
    def print_debug_info(self):
        """Print comprehensive debug information."""
        cprint("\n" + "="*80)
        cprint("ENHANCED UNIFIED TELEOP DEBUG INFORMATION")
        cprint("="*80)
        
        cprint(f"G1 Robot Body ID: {self.g1_id}")
        cprint(f"Left Hand Body ID: {self.lh_id}")
        cprint(f"Right Hand Body ID: {self.rh_id}")
        cprint(f"Active Constraints: {len(self.constraint_ids)}")
        cprint(f"Control Mode: {self.control_mode}")
        cprint(f"Real Robot Mode: {self.real_robot}")
        
        # Print mapper information
        self.mapper.print_all_debug_info()
        
        cprint("="*80)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Enhanced Unified G1 + O12 Hand Teleoperation')
    
    parser.add_argument('--g1_urdf', type=str, required=True,
                       help='Path to G1 robot URDF file')
    parser.add_argument('--left_hand_urdf', type=str, 
                       help='Path to left O12 hand URDF file')
    parser.add_argument('--right_hand_urdf', type=str,
                       help='Path to right O12 hand URDF file')
    parser.add_argument('--exo_config', type=str,
                       help='Path to exoskeleton configuration file')
    parser.add_argument('--real_robot', action='store_true',
                       help='Enable real robot control')
    parser.add_argument('--debug', action='store_true',
                       help='Print debug information and exit')
    
    args = parser.parse_args()
    
    # Validate required files
    if not os.path.exists(args.g1_urdf):
        cprint.err(f"ERROR: G1 URDF file not found: {args.g1_urdf}", interrupt=False)
        return 1
    
    if args.left_hand_urdf and not os.path.exists(args.left_hand_urdf):
        cprint.err(f"ERROR: Left hand URDF file not found: {args.left_hand_urdf}", interrupt=False)
        return 1
        
    if args.right_hand_urdf and not os.path.exists(args.right_hand_urdf):
        cprint.err(f"ERROR: Right hand URDF file not found: {args.right_hand_urdf}", interrupt=False)
        return 1
    
    # Create and run the teleoperation system
 
    teleop = EnhancedUnifiedTeleop(
        g1_urdf_path=args.g1_urdf,
        lh_urdf_path=args.left_hand_urdf,
        rh_urdf_path=args.right_hand_urdf,
        exo_config_path=args.exo_config,
        real_robot=args.real_robot
    )
    
    if args.debug:
        teleop.print_debug_info()
        return 0
    else:
        teleop.run()
        return 0

if __name__ == "__main__":
    ipdb_safety_net()
    exit(main())
