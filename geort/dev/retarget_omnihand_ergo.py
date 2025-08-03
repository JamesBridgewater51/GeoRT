# dev/retarget_ergonomics.py

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Assuming the constants file is accessible
from ..control.o12_hand.py_sdk.src.constants import JointID, O12RightHandLowerLimits, O12RightHandUpperLimits, O12LeftHandLowerLimits, O12LeftHandUpperLimits, RightJointNames, LeftJointNames, MAX_JOINT
from cprint import cprint
import sys
from third_party.manus_core_sdk_ros2_bridge.manus_ros2.client_scripts.manus_data_viz import EnhancedManusSubscriber, _global_subscriber

# From the Manus SDK documentation/headers - Ergonomics data indices
class ErgonomicsDataType(Enum):
    """
    Manus MetaGloves Pro ergonomics data indices.
    Data is expressed as joint angles (flexion/extension, abduction/adduction) in degrees.
    
    Finger flex: angle between consecutive joint segments
    Finger spread: splay of MCP joint relative to forward hand vector
    Thumb flex: extension/flexion angle  
    Thumb spread: palmar abduction angle
    """
    # Thumb
    ThumbCMCSpread = 0      # Thumb spread (palmar abduction)
    ThumbCMCFlex = 1        # Thumb CMC flexion
    ThumbPIPFlex = 2        # Thumb PIP flexion  
    ThumbDIPFlex = 3        # Thumb DIP flexion
    
    # Index finger
    IndexMCPSpread = 4      # Index finger spread
    IndexMCPFlex = 5        # Index MCP flexion
    IndexPIPFlex = 6        # Index PIP flexion
    IndexDIPFlex = 7        # Index DIP flexion
    
    # Middle finger
    MiddleMCPSpread = 8     # Middle finger spread
    MiddleMCPFlex = 9       # Middle MCP flexion
    MiddlePIPFlex = 10      # Middle PIP flexion
    MiddleDIPFlex = 11      # Middle DIP flexion
    
    # Ring finger
    RingMCPSpread = 12      # Ring finger spread
    RingMCPFlex = 13        # Ring MCP flexion
    RingPIPFlex = 14        # Ring PIP flexion
    RingDIPFlex = 15        # Ring DIP flexion
    
    # Pinky finger
    PinkyMCPSpread = 16     # Pinky finger spread
    PinkyMCPFlex = 17       # Pinky MCP flexion
    PinkyPIPFlex = 18       # Pinky PIP flexion
    PinkyDIPFlex = 19       # Pinky DIP flexion

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


def _degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * (math.pi / 180.0)


def _compute_geometry_aware_joint_angle(ergonomics_data: List[float], 
                                      ergo_indices: List[int],
                                      weights: List[float],
                                      human_min_limits: np.ndarray,
                                      human_max_limits: np.ndarray,
                                      robot_joint_id: int,
                                      robot_min_limit: float,
                                      robot_max_limit: float,
                                      multiplier: float = 1.0) -> float:
    """
    Compute a robot joint angle using geometry-aware calculations.
    
    Args:
        ergonomics_data: Raw ergonomics data in degrees
        ergo_indices: List of ergonomic data indices to use
        weights: Weights for combining multiple ergonomic inputs
        human_min_limits: Minimum observed human anatomy limits
        human_max_limits: Maximum observed human anatomy limits
        robot_joint_id: Robot joint ID for limits lookup
        robot_min_limit: Robot joint minimum limit
        robot_max_limit: Robot joint maximum limit
        multiplier: Sign/scaling multiplier
        
    Returns:
        Computed robot joint angle in radians
    """
    if len(ergo_indices) != len(weights):
        raise ValueError("ergo_indices and weights must have same length")
    
    # Combine multiple ergonomic inputs with weights
    combined_value = 0.0
    total_weight = 0.0
    
    for idx, weight in zip(ergo_indices, weights):
        if 0 <= idx < len(ergonomics_data):
            # Scale individual input from human range to normalized [0,1]
            ergo_val = ergonomics_data[idx]
            human_min = human_min_limits[idx] if idx < len(human_min_limits) else -90.0
            human_max = human_max_limits[idx] if idx < len(human_max_limits) else 90.0
            
            normalized_val = _scale(human_max, human_min, ergo_val, 1.0, 0.0)
            combined_value += weight * normalized_val
            total_weight += abs(weight)
    
    if total_weight > 0:
        combined_value /= total_weight
    
    # Convert normalized combined value to robot joint range
    robot_angle_rad = _scale(1.0, 0.0, combined_value, robot_max_limit, robot_min_limit)
    
    # Apply multiplier and clamp to final limits
    final_angle = robot_angle_rad * multiplier
    final_angle = np.clip(final_angle, robot_min_limit, robot_max_limit)
    
    return float(final_angle)


class HandRetargeter:
    """
    Retargets Manus MetaGloves Pro ergonomics data to an O12 dexterous hand.
    
    This implementation uses geometry-aware calculations similar to the skeletal
    retargeting approach but adapted for ergonomic angle data.
    """
    
    def __init__(self, is_right_hand: bool = True, smoothing_alpha: float = 0.4):
        self.is_right_hand = is_right_hand
        self.alpha = smoothing_alpha
        self.calibrated = False
        
        self.human_min_limits: Optional[np.ndarray] = None
        self.human_max_limits: Optional[np.ndarray] = None
        self.last_angles: Optional[np.ndarray] = None
        
        if is_right_hand:
            self.joint_names = RightJointNames
            self.robot_lower_limits = O12RightHandLowerLimits
            self.robot_upper_limits = O12RightHandUpperLimits
        else:
            self.joint_names = LeftJointNames  
            self.robot_lower_limits = O12LeftHandLowerLimits
            self.robot_upper_limits = O12LeftHandUpperLimits

    def calibrate(self, temporal_data: np.ndarray):
        """
        Calibrates the retargeter using temporal ergonomics data.
        
        Args:
            temporal_data (np.ndarray): Array of shape (N, 20) where N is the number of timesteps
                                       and 20 is the ergonomics data dimension.
        """
        if temporal_data.shape[1] != 20:
            raise ValueError(f"Expected 20 ergonomics dimensions, got {temporal_data.shape[1]}")
        
        print(f"Calibrating user's range of motion from {temporal_data.shape[0]} timesteps...")
        
        # Find min and max values across the temporal dimension
        # Add a small buffer to prevent division by zero and allow full range
        buffer = 1e-5
        self.human_min_limits = np.min(temporal_data, axis=0) - buffer
        self.human_max_limits = np.max(temporal_data, axis=0) + buffer
        self.calibrated = True
        
        print("Calibration complete.")
        print(f"Human anatomy ranges (degrees):")
        if self.human_min_limits is not None and self.human_max_limits is not None:
            for i, (min_val, max_val) in enumerate(zip(self.human_min_limits, self.human_max_limits)):
                print(f"  Index {i}: [{min_val:.1f}, {max_val:.1f}]")

    def _baselines_omnihand_ergo(self, ergonomics_data: List[float]) -> np.ndarray:
        """
        Convert 20-element Manus ergonomics data into OmniHand joint angles.
        
        This method is inspired by the geometry-aware approach from retarget_omnihand.py
        but adapted for ergonomic angle data instead of 3D joint positions.
        
        Args:
            ergonomics_data: List of 20 ergonomic angle values in degrees
            
        Returns:
            numpy array of OmniHand joint angles in radians
        """
        if len(ergonomics_data) != 20:
            raise ValueError(f"Expected 20 ergonomics values, got {len(ergonomics_data)}")
        
        # Use calibrated limits if available, otherwise use reasonable defaults
        if self.calibrated and self.human_min_limits is not None and self.human_max_limits is not None:
            human_min = self.human_min_limits
            human_max = self.human_max_limits
        else:
            human_min = np.full(20, -90.0)
            human_max = np.full(20, 90.0)
        
        # Initialize target angles
        target_angles = np.zeros(MAX_JOINT)
        
        # ========== THUMB JOINTS ==========
        # Thumb Roll: Use thumb spread data with some neutral offset
        thumb_spread_deg = ergonomics_data[ErgonomicsDataType.ThumbCMCSpread.value] 
        # Map thumb spread to roll - positive spread should increase roll
        thumb_roll = _compute_geometry_aware_joint_angle(
            ergonomics_data,
            [ErgonomicsDataType.ThumbCMCSpread.value],
            [1.0],
            human_min, human_max,
            JointID.JThumbRoll,
            self.robot_lower_limits[JointID.JThumbRoll],
            self.robot_upper_limits[JointID.JThumbRoll],
            multiplier=1.0 if self.is_right_hand else -1.0  # Mirror for left hand
        )
        target_angles[JointID.JThumbRoll] = thumb_roll
        
        # Thumb Abduction: Directly from thumb spread
        thumb_abad = _compute_geometry_aware_joint_angle(
            ergonomics_data,
            [ErgonomicsDataType.ThumbCMCSpread.value],
            [1.0],
            human_min, human_max,
            JointID.JThumbAbad,
            self.robot_lower_limits[JointID.JThumbAbad],
            self.robot_upper_limits[JointID.JThumbAbad],
            multiplier=-1.0 if self.is_right_hand else 1.0  # Negative for right hand URDF
        )
        target_angles[JointID.JThumbAbad] = thumb_abad
        
        # Thumb MCP: From CMC flex data
        thumb_mcp = _compute_geometry_aware_joint_angle(
            ergonomics_data,
            [ErgonomicsDataType.ThumbCMCFlex.value],
            [1.0],
            human_min, human_max,
            JointID.JThumbMCP,
            self.robot_lower_limits[JointID.JThumbMCP],
            self.robot_upper_limits[JointID.JThumbMCP],
            multiplier=-1.0 if self.is_right_hand else 1.0  # Negative flexion for right hand
        )
        target_angles[JointID.JThumbMCP] = thumb_mcp
        
        # Thumb PIP: From PIP flex data
        thumb_pip = _compute_geometry_aware_joint_angle(
            ergonomics_data,
            [ErgonomicsDataType.ThumbPIPFlex.value],
            [1.0],
            human_min, human_max,
            JointID.JThumbPIP,
            self.robot_lower_limits[JointID.JThumbPIP],
            self.robot_upper_limits[JointID.JThumbPIP],
            multiplier=-1.0 if self.is_right_hand else 1.0  # Negative flexion for right hand
        )
        target_angles[JointID.JThumbPIP] = thumb_pip
        
        # ========== INDEX FINGER ==========
        # Index Abduction: From spread data, but constrained to small range
        index_abad = _compute_geometry_aware_joint_angle(
            ergonomics_data,
            [ErgonomicsDataType.IndexMCPSpread.value],
            [1.0],
            human_min, human_max,
            JointID.JIndexAbad,
            -0.05, 0.05,  # Constrained range for dexterous manipulation
            multiplier=1.0
        )
        target_angles[JointID.JIndexAbad] = index_abad
        
        # Index MCP: Directly from MCP flex
        index_mcp = _compute_geometry_aware_joint_angle(
            ergonomics_data,
            [ErgonomicsDataType.IndexMCPFlex.value],
            [1.0],
            human_min, human_max,
            JointID.JIndexMCP,
            self.robot_lower_limits[JointID.JIndexMCP],
            self.robot_upper_limits[JointID.JIndexMCP],
            multiplier=1.0  # Positive flexion
        )
        target_angles[JointID.JIndexMCP] = index_mcp
        
        # Index PIP: From PIP flex
        index_pip = _compute_geometry_aware_joint_angle(
            ergonomics_data,
            [ErgonomicsDataType.IndexPIPFlex.value],
            [1.0],
            human_min, human_max,
            JointID.JIndexPIP,
            self.robot_lower_limits[JointID.JIndexPIP],
            self.robot_upper_limits[JointID.JIndexPIP],
            multiplier=1.0  # Positive flexion
        )
        target_angles[JointID.JIndexPIP] = index_pip
        
        # ========== MIDDLE FINGER ==========
        # Middle Abduction: From spread data, constrained range
        middle_abad = _compute_geometry_aware_joint_angle(
            ergonomics_data,
            [ErgonomicsDataType.MiddleMCPSpread.value],
            [1.0],
            human_min, human_max,
            JointID.JMiddleAbad,
            -0.08, 0.08,  # Slightly larger range than index
            multiplier=1.0
        )
        target_angles[JointID.JMiddleAbad] = middle_abad
        
        # Middle MCP: Directly from MCP flex
        middle_mcp = _compute_geometry_aware_joint_angle(
            ergonomics_data,
            [ErgonomicsDataType.MiddleMCPFlex.value],
            [1.0],
            human_min, human_max,
            JointID.JMiddleMCP,
            self.robot_lower_limits[JointID.JMiddleMCP],
            self.robot_upper_limits[JointID.JMiddleMCP],
            multiplier=1.0  # Positive flexion
        )
        target_angles[JointID.JMiddleMCP] = middle_mcp
        
        # Middle PIP: From PIP flex
        middle_pip = _compute_geometry_aware_joint_angle(
            ergonomics_data,
            [ErgonomicsDataType.MiddlePIPFlex.value],
            [1.0],
            human_min, human_max,
            JointID.JMiddlePIP,
            self.robot_lower_limits[JointID.JMiddlePIP],
            self.robot_upper_limits[JointID.JMiddlePIP],
            multiplier=1.0  # Positive flexion
        )
        target_angles[JointID.JMiddlePIP] = middle_pip
        
        # ========== RING FINGER ==========
        # Ring MCP: Combine MCP flex and PIP flex for better control (compound input)
        ring_mcp = _compute_geometry_aware_joint_angle(
            ergonomics_data,
            [ErgonomicsDataType.RingMCPFlex.value, ErgonomicsDataType.RingPIPFlex.value],
            [0.7, 0.3],  # Weight MCP more than PIP
            human_min, human_max,
            JointID.JRingMCP,
            self.robot_lower_limits[JointID.JRingMCP],
            self.robot_upper_limits[JointID.JRingMCP],
            multiplier=1.0  # Positive flexion
        )
        target_angles[JointID.JRingMCP] = ring_mcp
        
        # ========== PINKY FINGER ==========
        # Pinky MCP: Combine MCP flex and PIP flex for better control (compound input)
        pinky_mcp = _compute_geometry_aware_joint_angle(
            ergonomics_data,
            [ErgonomicsDataType.PinkyMCPFlex.value, ErgonomicsDataType.PinkyPIPFlex.value],
            [0.7, 0.3],  # Weight MCP more than PIP
            human_min, human_max,
            JointID.JPinkyMCP,
            self.robot_lower_limits[JointID.JPinkyMCP],
            self.robot_upper_limits[JointID.JPinkyMCP],
            multiplier=1.0  # Positive flexion
        )
        target_angles[JointID.JPinkyMCP] = pinky_mcp
        
        # ========== MIMIC JOINTS ==========
        # Apply mimic joint logic based on URDF specifications
        target_angles[JointID.JThumbDIP] = target_angles[JointID.JThumbPIP] * 0.7999
        target_angles[JointID.JIndexDIP] = target_angles[JointID.JIndexPIP] * 1.1162  
        target_angles[JointID.JMiddleDIP] = target_angles[JointID.JMiddlePIP] * 1.1415
        target_angles[JointID.JRingPIP] = target_angles[JointID.JRingMCP] * 0.8766
        target_angles[JointID.JRingDIP] = target_angles[JointID.JRingMCP] * 0.9695
        target_angles[JointID.JPinkyPIP] = target_angles[JointID.JPinkyMCP] * 0.8766
        target_angles[JointID.JPinkyDIP] = target_angles[JointID.JPinkyMCP] * 0.9695
        
        # Final clamping to hardware limits
        final_angles = np.clip(target_angles, self.robot_lower_limits, self.robot_upper_limits)
        
        return final_angles

    def retarget(self, ergonomics_data: List[float]) -> Optional[Dict[str, float]]:
        """
        Takes a list of 20 ergonomics values and returns a dictionary of O12 joint angles.
        
        Args:
            ergonomics_data (List[float]): The raw ergonomics data (in degrees).
            
        Returns:
            A dictionary mapping O12 joint names to angles in radians, or None if input is invalid.
        """
        if len(ergonomics_data) != 20:
            print("Error: Expected 20 ergonomics values.")
            return None
        
        # Use the improved baseline retargeting method
        target_angles = self._baselines_omnihand_ergo(ergonomics_data)
        
        # Apply smoothing if we have previous angles
        if self.last_angles is None:
            self.last_angles = target_angles.copy()
        
        smoothed_angles = self.alpha * target_angles + (1.0 - self.alpha) * self.last_angles
        self.last_angles = smoothed_angles.copy()
        
        # Format for output
        output_dict = {name: float(angle) for name, angle in zip(self.joint_names, smoothed_angles)}
        
        return output_dict


# Example usage and utility functions

def create_sample_ergonomics_data() -> List[float]:
    """Create sample ergonomics data for testing."""
    # Sample data representing a relaxed hand pose with slight flexion
    return [
        10.0,   # Thumb CMC Spread
        15.0,   # Thumb CMC Flex
        20.0,   # Thumb PIP Flex
        15.0,   # Thumb DIP Flex
        -5.0,   # Index MCP Spread  
        25.0,   # Index MCP Flex
        30.0,   # Index PIP Flex
        20.0,   # Index DIP Flex
        0.0,    # Middle MCP Spread
        30.0,   # Middle MCP Flex
        35.0,   # Middle PIP Flex
        25.0,   # Middle DIP Flex
        5.0,    # Ring MCP Spread
        35.0,   # Ring MCP Flex
        40.0,   # Ring PIP Flex
        30.0,   # Ring DIP Flex
        8.0,    # Pinky MCP Spread
        40.0,   # Pinky MCP Flex
        45.0,   # Pinky PIP Flex
        35.0    # Pinky DIP Flex
    ]


def parse_manus_csv_data(csv_file_path: str) -> np.ndarray:
    """
    Parse Manus CSV export data into ergonomics array.
    
    Args:
        csv_file_path: Path to exported Manus CSV file
        
    Returns:
        Array of shape (N, 20) with N timesteps of ergonomics data
    """
    import pandas as pd
    
    # Read CSV file  
    df = pd.read_csv(csv_file_path)
    
    # Extract ergonomics columns (assuming Manus CSV format)
    ergo_columns = [
        'Thumb_CMC_Spread', 'Thumb_CMC_Flex', 'Thumb_PIP_Flex', 'Thumb_DIP_Flex',
        'Index_MCP_Spread', 'Index_MCP_Flex', 'Index_PIP_Flex', 'Index_DIP_Flex',
        'Middle_MCP_Spread', 'Middle_MCP_Flex', 'Middle_PIP_Flex', 'Middle_DIP_Flex',
        'Ring_MCP_Spread', 'Ring_MCP_Flex', 'Ring_PIP_Flex', 'Ring_DIP_Flex',
        'Pinky_MCP_Spread', 'Pinky_MCP_Flex', 'Pinky_PIP_Flex', 'Pinky_DIP_Flex'
    ]
    
    # Extract data
    ergo_data = df[ergo_columns].values
    return ergo_data.astype(np.float64)


def main():
    """Example usage of the improved retargeting system."""
    print("=== Improved Manus to OmniHand Retargeting ===")
    import rclpy
    from rclpy.node import Node
    global _global_subscriber
    
    rclpy.init(args=sys.argv)
    
    cprint("🚀 Starting Enhanced MANUS Data Visualization", "green")
    cprint("📋 Features:", "cyan")
    cprint("  • Dynamic topic discovery", "white")
    cprint("  • Comprehensive data visualization", "white") 
    cprint("  • Colored terminal output", "white")
    cprint("  • Real-time 3D skeleton rendering", "white")
    cprint("  • Sensor orientation frames", "white")
    cprint("  • Ergonomics data monitoring", "white")
    cprint("  • Online/Offline data collection pipelines", "white")
    
    _global_subscriber = EnhancedManusSubscriber()
    
    # Spin the node
    rclpy.spin(_global_subscriber)
    _global_subscriber.start
    
    # Cleanup
    _global_subscriber.destroy_node()
    rclpy.shutdown()
    # Create retargeter for right hand
    retargeter = HandRetargeter(is_right_hand=True, smoothing_alpha=0.3)
    
    # Create sample calibration data (simulate recording session)
    print("\n1. Creating sample calibration data...")
    sample_data = []
    for i in range(100):  # 100 timesteps
        # Simulate varying hand poses during calibration
        base_pose = create_sample_ergonomics_data()
        # Add some variation
        varied_pose = [val + np.random.normal(0, 10) for val in base_pose]
        sample_data.append(varied_pose)
    
    calibration_data = np.array(sample_data)
    
    # Calibrate the system
    print("2. Calibrating retargeter...")
    retargeter.calibrate(calibration_data)
    
    # Test retargeting with sample data
    print("\n3. Testing retargeting...")
    test_ergo_data = create_sample_ergonomics_data()
    print(f"Input ergonomics data: {test_ergo_data[:5]}... (showing first 5 values)")
    
    result = retargeter.retarget(test_ergo_data)
    if result:
        print(f"Output joint angles (radians):")
        for joint_name, angle in result.items():
            print(f"  {joint_name}: {angle:.3f}")
    else:
        print("Error in retargeting!")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()