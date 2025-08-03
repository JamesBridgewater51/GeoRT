#!/usr/bin/env python3
"""
Example Usage of the Enhanced Joint/Link Mapper for PyBullet Constraint Creation

This script demonstrates how to properly use the JointLinkMapper and 
MultiBodyJointLinkMapper classes to create constraints between PyBullet bodies
with correct parent/child link index resolution.

Author: GitHub Copilot
Date: 2025-08-14
"""

import pybullet as p
import pybullet_data
import numpy as np
import os
import sys

# Add the project path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from geort.control.joint_link_mapper import JointLinkMapper, MultiBodyJointLinkMapper


def example_basic_usage():
    """Example of basic JointLinkMapper usage with a single body."""
    print("="*60)
    print("EXAMPLE 1: Basic JointLinkMapper Usage")
    print("="*60)
    
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # Load a simple robot (e.g., KUKA arm)
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
    
    # Create mapper for the robot
    mapper = JointLinkMapper(robot_id, "KUKA_IIWA")
    
    # Print comprehensive information
    print("\n--- Joint Information ---")
    mapper.print_joint_info()
    
    print("\n--- Link Information ---") 
    mapper.print_link_info()
    
    # Validate mappings
    mapper.validate_mappings()
    
    # Example: Query specific joint/link information
    print("\n--- Specific Queries ---")
    joint_names = mapper.get_joint_names(include_fixed=False)
    print(f"Controllable joints: {joint_names}")
    
    if joint_names:
        example_joint = joint_names[0]
        print(f"\nExample constraint debug for joint '{example_joint}':")
        mapper.print_constraint_debug_info(example_joint)
        
        # Get parent and child link indices for constraint creation
        parent_idx, child_idx = mapper.get_constraint_link_indices(example_joint)
        print(f"For createConstraint: parent_link={parent_idx}, child_link={child_idx}")
    
    p.disconnect()
    

def example_multi_body_constraint():
    """Example of creating constraints between multiple bodies."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Body Constraint Creation")
    print("="*60)
    
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    # Load multiple bodies
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
    box_id = p.loadURDF("cube.urdf", [0.5, 0, 0.5], useFixedBase=False)
    
    # Create multi-body mapper
    multi_mapper = MultiBodyJointLinkMapper()
    multi_mapper.add_body(robot_id, "KUKA_Robot")
    multi_mapper.add_body(box_id, "Box")
    
    # Print debug information for all bodies
    multi_mapper.print_all_debug_info()
    
    # Example constraint creation (attach box to robot end-effector)
    robot_mapper = multi_mapper.get_mapper(robot_id)
    if robot_mapper is not None:
        joint_names = robot_mapper.get_joint_names(include_fixed=False)
        if joint_names:
            # Use the last joint as end-effector
            ee_joint_name = joint_names[-1]
            print(f"\nAttempting to attach box to robot end-effector joint: '{ee_joint_name}'")
            
            # This would attach the box to the child link of the end-effector joint
            constraint_id = multi_mapper.create_constraint_with_validation(
                parent_body_id=robot_id,
                parent_joint_name=ee_joint_name,
                child_body_id=box_id,
                child_joint_name="base_link",  # Box base (special case for fixed base)
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0.1],  # 10cm offset
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0])
            )
            
            if constraint_id is not None:
                print(f"Constraint created successfully with ID: {constraint_id}")
                
                # Demonstrate constraint by moving robot joints
                print("\nMoving robot to demonstrate constraint...")
                for i in range(240):  # 1 second at 240Hz
                    # Simple sinusoidal motion
                    for joint_idx in range(len(joint_names)):
                        target_pos = 0.5 * np.sin(i * 0.02 + joint_idx)
                        p.setJointMotorControl2(
                            bodyUniqueId=robot_id,
                            jointIndex=joint_idx,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_pos,
                            maxVelocity=1.0,
                            force=500
                        )
                    
                    p.stepSimulation()
                    
                print("Constraint demonstration complete!")
            else:
                print("Failed to create constraint")
    
    # Keep simulation running for inspection
    print("\nSimulation running - close PyBullet window to exit")
    try:
        while True:
            p.stepSimulation()
    except KeyboardInterrupt:
        pass
    
    p.disconnect()


def example_o12_hand_simulation():
    """Example specifically for O12 hand constraint creation."""
    print("\n" + "="*60)
    print("EXAMPLE 3: O12 Hand Simulation (if URDF available)")
    print("="*60)
    
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    # Try to load O12 hand URDF (adjust path as needed)
    o12_urdf_path = "/home/minghao/src/robotflow/GeoRT/assets/o12_hand_description-main/urdf/o12_t1_right.urdf"
    
    if os.path.exists(o12_urdf_path):
        print(f"Loading O12 hand from: {o12_urdf_path}")
        hand_id = p.loadURDF(o12_urdf_path, [0, 0, 0.2], useFixedBase=False)
        
        # Create mapper for O12 hand
        hand_mapper = JointLinkMapper(hand_id, "O12_Right_Hand")
        
        # Print detailed information about the hand
        print("\n--- O12 Hand Joint Information ---")
        hand_mapper.print_joint_info()
        
        print("\n--- O12 Hand Link Information ---")
        hand_mapper.print_link_info()
        
        # Validate mappings
        hand_mapper.validate_mappings()
        
        # Example: Get constraint information for specific joints
        joint_names = hand_mapper.get_joint_names(include_fixed=False)
        print(f"\nControllable O12 joints: {len(joint_names)}")
        for joint_name in joint_names[:5]:  # Show first 5 joints
            print(f"\nConstraint debug for '{joint_name}':")
            hand_mapper.print_constraint_debug_info(joint_name)
        
        # Demonstrate joint control
        print("\nDemonstrating O12 hand joint control...")
        for i in range(240):  # 1 second at 240Hz
            for joint_idx, joint_name in enumerate(joint_names):
                joint_info = hand_mapper.get_joint_info(joint_name)
                if joint_info:
                    # Simple motion within joint limits
                    mid_range = (joint_info.joint_lower_limit + joint_info.joint_upper_limit) / 2
                    amplitude = (joint_info.joint_upper_limit - joint_info.joint_lower_limit) / 4
                    target_pos = mid_range + amplitude * np.sin(i * 0.02 + joint_idx)
                    
                    p.setJointMotorControl2(
                        bodyUniqueId=hand_id,
                        jointIndex=joint_info.joint_index,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        maxVelocity=1.0,
                        force=100
                    )
            
            p.stepSimulation()
        
        print("O12 hand demonstration complete!")
        
    else:
        print(f"O12 hand URDF not found at: {o12_urdf_path}")
        print("Skipping O12-specific example")
    
    # Keep simulation running for inspection
    print("\nSimulation running - close PyBullet window to exit")
    try:
        while True:
            p.stepSimulation()
    except KeyboardInterrupt:
        pass
    
    p.disconnect()


def main():
    """Run all examples."""
    print("Joint/Link Mapper Examples for PyBullet Constraint Creation")
    print("=" * 70)
    
    try:
        # Run basic example
        example_basic_usage()
        
        # Run multi-body constraint example
        example_multi_body_constraint()
        
        # Run O12 hand specific example
        example_o12_hand_simulation()
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
