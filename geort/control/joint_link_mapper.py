# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pybullet as p
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from cprint import cprint


@dataclass
class JointInfo:
    """Comprehensive joint information from PyBullet's getJointInfo."""
    joint_index: int
    joint_name: str
    joint_type: int
    q_index: int
    u_index: int
    flags: int
    joint_damping: float
    joint_friction: float
    joint_lower_limit: float
    joint_upper_limit: float
    joint_max_force: float
    joint_max_velocity: float
    link_name: str
    joint_axis: Tuple[float, float, float]
    parent_frame_pos: Tuple[float, float, float]
    parent_frame_orn: Tuple[float, float, float, float]
    parent_index: int  # This is the parent link index (-1 for base)


@dataclass
class LinkInfo:
    """Link information derived from joint information."""
    link_index: int
    link_name: str
    parent_joint_index: Optional[int]  # Joint that connects to this link
    child_joint_indices: List[int]     # Joints that this link is parent to
    

class JointLinkMapper:
    """
    Comprehensive mapper for PyBullet joint and link information.
    
    This class properly handles the mapping between:
    - Joint names/indices and joint information
    - Joint indices and their corresponding child link indices
    - Link names/indices and link information
    - Proper parent-child relationships for constraint creation
    """
    
    def __init__(self, body_id: int, body_name: str = ""):
        """
        Initialize the mapper for a specific PyBullet body.
        
        Args:
            body_id: The PyBullet body unique ID
            body_name: Optional name for debugging purposes
        """
        self.body_id = body_id
        self.body_name = body_name
        
        # Core mappings
        self.joint_infos: Dict[int, JointInfo] = {}           # joint_index -> JointInfo
        self.joint_name_to_index: Dict[str, int] = {}         # joint_name -> joint_index
        self.link_infos: Dict[int, LinkInfo] = {}             # link_index -> LinkInfo
        self.link_name_to_index: Dict[str, int] = {}          # link_name -> link_index
        
        # Convenience mappings for constraint creation
        self.joint_to_child_link: Dict[int, int] = {}         # joint_index -> child_link_index
        self.link_to_parent_joint: Dict[int, int] = {}        # link_index -> parent_joint_index
        
        # Debug information storage
        self.debug_info: Dict[str, Any] = {}
        
        self._collect_joint_link_information()
        
    def _collect_joint_link_information(self):
        """Collect and organize all joint and link information."""
        num_joints = p.getNumJoints(self.body_id)
        
        # First pass: collect all joint information
        for joint_idx in range(num_joints):
            joint_info_raw = p.getJointInfo(self.body_id, joint_idx)
            
            # Parse the raw PyBullet joint info
            joint_info = JointInfo(
                joint_index=joint_info_raw[0],
                joint_name=joint_info_raw[1].decode('utf-8'),
                joint_type=joint_info_raw[2],
                q_index=joint_info_raw[3],
                u_index=joint_info_raw[4],
                flags=joint_info_raw[5],
                joint_damping=joint_info_raw[6],
                joint_friction=joint_info_raw[7],
                joint_lower_limit=joint_info_raw[8],
                joint_upper_limit=joint_info_raw[9],
                joint_max_force=joint_info_raw[10],
                joint_max_velocity=joint_info_raw[11],
                link_name=joint_info_raw[12].decode('utf-8'),
                joint_axis=joint_info_raw[13],
                parent_frame_pos=joint_info_raw[14],
                parent_frame_orn=joint_info_raw[15],
                parent_index=joint_info_raw[16]
            )
            # cprint.ok(f"Joint {joint_idx}, joint_name: {joint_info.joint_name}, child_link_name:{joint_info.link_name}, parent_index: {joint_info.parent_index}")
            
            self.joint_infos[joint_idx] = joint_info
            self.joint_name_to_index[joint_info.joint_name] = joint_idx
            
            # The child link of this joint has index = joint_index + 1 (PyBullet convention)
            child_link_idx = joint_idx
            self.joint_to_child_link[joint_idx] = child_link_idx
            self.link_to_parent_joint[child_link_idx] = joint_idx
            
        # Second pass: create link information
        # Base link (index 0) has no parent joint
        base_link = LinkInfo(
            link_index=0,
            link_name="base_link",  # Base link typically doesn't have a specific name from joint info
            parent_joint_index=None,
            child_joint_indices=[]
        )
        self.link_infos[0] = base_link
        self.link_name_to_index["base_link"] = -1
        
        # Create link info for all other links
        for joint_idx, joint_info in self.joint_infos.items():
            link_idx = joint_idx
            
            # Find child joints of this link
            child_joints = []
            for other_joint_idx, other_joint_info in self.joint_infos.items():
                if other_joint_info.parent_index == link_idx:
                    child_joints.append(other_joint_idx)
            
            link_info = LinkInfo(
                link_index=link_idx,
                link_name=joint_info.link_name,
                parent_joint_index=joint_idx,
                child_joint_indices=child_joints
            )
            
            self.link_infos[link_idx] = link_info
            self.link_name_to_index[joint_info.link_name] = link_idx
            
        # Update base link's child joints
        for joint_idx, joint_info in self.joint_infos.items():
            if joint_info.parent_index == -1:  # Parent is base link
                self.link_infos[0].child_joint_indices.append(joint_idx)
                
        # Store debug information
        self._store_debug_info()
    
    def _store_debug_info(self):
        """Store comprehensive debug information."""
        self.debug_info = {
            "body_id": self.body_id,
            "body_name": self.body_name,
            "num_joints": len(self.joint_infos),
            "num_links": len(self.link_infos),
            "joint_types": {idx: self._get_joint_type_name(info.joint_type) 
                           for idx, info in self.joint_infos.items()},
            "joint_limits": {idx: (info.joint_lower_limit, info.joint_upper_limit)
                           for idx, info in self.joint_infos.items()},
            "link_hierarchy": self._build_link_hierarchy()
        }
    
    def _get_joint_type_name(self, joint_type: int) -> str:
        """Convert joint type integer to readable name."""
        type_map = {
            p.JOINT_REVOLUTE: "REVOLUTE",
            p.JOINT_PRISMATIC: "PRISMATIC", 
            p.JOINT_SPHERICAL: "SPHERICAL",
            p.JOINT_PLANAR: "PLANAR",
            p.JOINT_FIXED: "FIXED"
        }
        return type_map.get(joint_type, f"UNKNOWN({joint_type})")
    
    def _build_link_hierarchy(self) -> Dict[int, List[int]]:
        """Build a hierarchy showing parent-child relationships between links."""
        hierarchy = {}
        for link_idx, link_info in self.link_infos.items():
            parent_link_idx = -1
            if link_info.parent_joint_index is not None:
                parent_joint = self.joint_infos[link_info.parent_joint_index]
                parent_link_idx = parent_joint.parent_index
            hierarchy[link_idx] = parent_link_idx
        return hierarchy
    
    # Core query methods for constraint creation
    def get_link_index_from_joint_name(self, joint_name: str) -> Optional[int]:
        """Get the child link index for a given joint name."""
        if joint_name not in self.joint_name_to_index:
            return None
        joint_idx = self.joint_name_to_index[joint_name]
        return self.joint_to_child_link.get(joint_idx)
    
    def get_link_index_from_link_name(self, link_name: str) -> Optional[int]:
        """Get link index from link name."""
        return self.link_name_to_index.get(link_name)
    
    def get_parent_link_index(self, joint_name: str) -> Optional[int]:
        """Get the parent link index for a given joint name."""
        if joint_name not in self.joint_name_to_index:
            return None
        joint_idx = self.joint_name_to_index[joint_name]
        joint_info = self.joint_infos[joint_idx]
        return joint_info.parent_index if joint_info.parent_index != -1 else 0  # Base link
    
    def get_constraint_link_indices(self, joint_name: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Get both parent and child link indices for a joint - useful for constraints.
        
        Returns:
            Tuple of (parent_link_index, child_link_index)
        """
        parent_idx = self.get_parent_link_index(joint_name)
        child_idx = self.get_link_index_from_joint_name(joint_name)
        return parent_idx, child_idx
    
    # Information retrieval methods
    def get_joint_info(self, joint_identifier: Optional[Union[int, str]] = None) -> Optional[JointInfo]:
        """Get joint information by index or name."""
        if joint_identifier is None:
            return None
        
        if isinstance(joint_identifier, str):
            joint_idx = self.joint_name_to_index.get(joint_identifier)
            if joint_idx is None:
                return None
        else:
            joint_idx = joint_identifier
            
        return self.joint_infos.get(joint_idx)

    def get_link_info(self, link_identifier: Optional[Union[int, str]] = None) -> Optional[LinkInfo]:
        """Get link information by index or name."""
        if link_identifier is None:
            return None
            
        if isinstance(link_identifier, str):
            link_idx = self.link_name_to_index.get(link_identifier)
            if link_idx is None:
                return None
        else:
            link_idx = link_identifier
            
        return self.link_infos.get(link_idx)
    
    def get_controllable_joints(self) -> List[JointInfo]:
        """Get all non-fixed joints."""
        return [info for info in self.joint_infos.values() 
                if info.joint_type != p.JOINT_FIXED]
    
    def get_joint_names(self, include_fixed: bool = False) -> List[str]:
        """Get all joint names."""
        if include_fixed:
            return list(self.joint_name_to_index.keys())
        else:
            return [info.joint_name for info in self.joint_infos.values()
                   if info.joint_type != p.JOINT_FIXED]
    
    def get_link_names(self) -> List[str]:
        """Get all link names."""
        return list(self.link_name_to_index.keys())
    
    # Debug and validation methods
    def print_joint_info(self, joint_identifier: Optional[Union[int, str]] = None):
        """Print detailed joint information."""
        if joint_identifier is None:
            print(f"\n=== Joint Information for Body {self.body_id} ({self.body_name}) ===")
            for idx, info in self.joint_infos.items():
                self._print_single_joint_info(info)
        else:
            info = self.get_joint_info(joint_identifier)
            if info:
                self._print_single_joint_info(info)
            else:
                print(f"Joint '{joint_identifier}' not found")
    
    def _print_single_joint_info(self, info: JointInfo):
        """Print information for a single joint."""
        print(f"Joint {info.joint_index}: '{info.joint_name}'")
        print(f"  Type: {self._get_joint_type_name(info.joint_type)}")
        print(f"  Child Link: '{info.link_name}' (index: {info.joint_index + 1})")
        print(f"  Parent Link Index: {info.parent_index}")
        print(f"  Limits: [{info.joint_lower_limit:.3f}, {info.joint_upper_limit:.3f}]")
        print(f"  Max Force: {info.joint_max_force}")
        print(f"  Damping: {info.joint_damping}, Friction: {info.joint_friction}")
        print(f"  Axis: {info.joint_axis}")
        print()

    def print_link_info(self, link_identifier: Optional[Union[int, str]] = None):
        """Print detailed link information."""
        if link_identifier is None:
            print(f"\n=== Link Information for Body {self.body_id} ({self.body_name}) ===")
            for idx, info in self.link_infos.items():
                self._print_single_link_info(info)
        else:
            info = self.get_link_info(link_identifier)
            if info:
                self._print_single_link_info(info)
            else:
                print(f"Link '{link_identifier}' not found")
    
    def _print_single_link_info(self, info: LinkInfo):
        """Print information for a single link."""
        print(f"Link {info.link_index}: '{info.link_name}'")
        print(f"  Parent Joint: {info.parent_joint_index}")
        print(f"  Child Joints: {info.child_joint_indices}")
        print()
    
    def print_constraint_debug_info(self, joint_name: str):
        """Print debug information specifically for constraint creation."""
        print(f"\n=== Constraint Debug Info for Joint '{joint_name}' ===")
        
        joint_info = self.get_joint_info(joint_name)
        if joint_info is None:
            print(f"ERROR: Joint '{joint_name}' not found!")
            return
        
        parent_idx, child_idx = self.get_constraint_link_indices(joint_name)
        
        print(f"Joint Index: {joint_info.joint_index}")
        print(f"Joint Type: {self._get_joint_type_name(joint_info.joint_type)}")
        print(f"Parent Link Index: {parent_idx}")
        print(f"Child Link Index: {child_idx}")
        print(f"Child Link Name: '{joint_info.link_name}'")
        
        if parent_idx is not None and parent_idx in self.link_infos:
            parent_link = self.link_infos[parent_idx]
            print(f"Parent Link Name: '{parent_link.link_name}'")
        
        print(f"\nFor createConstraint:")
        print(f"  parentLinkIndex = {parent_idx}")
        print(f"  childLinkIndex = {child_idx}")
    
    def validate_mappings(self) -> bool:
        """Validate the internal mappings for consistency."""
        print(f"\n=== Validating Mappings for Body {self.body_id} ===")
        
        errors = []
        
        # Check joint-to-link mappings
        for joint_idx, child_link_idx in self.joint_to_child_link.items():
            if child_link_idx not in self.link_infos:
                errors.append(f"Joint {joint_idx} maps to non-existent link {child_link_idx}")
        
        # Check link-to-joint mappings
        for link_idx, parent_joint_idx in self.link_to_parent_joint.items():
            if parent_joint_idx not in self.joint_infos:
                errors.append(f"Link {link_idx} maps to non-existent joint {parent_joint_idx}")
        
        # Check name-to-index mappings
        for name, idx in self.joint_name_to_index.items():
            if idx not in self.joint_infos:
                errors.append(f"Joint name '{name}' maps to non-existent index {idx}")
        
        for name, idx in self.link_name_to_index.items():
            if idx not in self.link_infos:
                errors.append(f"Link name '{name}' maps to non-existent index {idx}")
        
        if errors:
            print("VALIDATION ERRORS:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("All mappings are valid!")
            return True


class MultiBodyJointLinkMapper:
    """
    Manager for multiple PyBullet bodies with comprehensive joint/link mapping.
    
    This is useful for complex scenarios with multiple robots, hands, etc.
    """
    
    def __init__(self):
        self.body_mappers: Dict[int, JointLinkMapper] = {}
        self.body_names: Dict[int, str] = {}
    
    def add_body(self, body_id: int, body_name: str = ""):
        """Add a body to be managed."""
        self.body_mappers[body_id] = JointLinkMapper(body_id, body_name)
        self.body_names[body_id] = body_name
    
    def get_mapper(self, body_id: int) -> Optional[JointLinkMapper]:
        """Get the mapper for a specific body."""
        return self.body_mappers.get(body_id)
    
    def create_constraint_with_validation(self, 
                                        parent_body_id: int, 
                                        parent_joint_name: str,
                                        child_body_id: int, 
                                        child_joint_name: str,
                                        **constraint_kwargs) -> Optional[int]:
        """
        Create a PyBullet constraint with proper link index resolution and validation.
        
        Args:
            parent_body_id: PyBullet body ID of the parent
            parent_joint_name: Name of the joint whose child link will be the parent
            child_body_id: PyBullet body ID of the child  
            child_joint_name: Name of the joint whose child link will be the child
            **constraint_kwargs: Additional arguments passed to p.createConstraint
            
        Returns:
            Constraint ID if successful, None if failed
        """
        
        parent_mapper = self.get_mapper(parent_body_id)
        child_mapper = self.get_mapper(child_body_id)
        
        if parent_mapper is None:
            print(f"ERROR: No mapper found for parent body {parent_body_id}")
            return None
        
        if child_mapper is None:
            print(f"ERROR: No mapper found for child body {child_body_id}")
            return None
        
        # Get parent link index
        parent_link_idx = parent_mapper.get_link_index_from_joint_name(parent_joint_name)
        if parent_link_idx is None:
            print(f"ERROR: Cannot find parent link for joint '{parent_joint_name}' in body {parent_body_id}")
            return None
        
        # Get child link index  
        child_link_idx = child_mapper.get_link_index_from_joint_name(child_joint_name)
        if child_link_idx is None:
            print(f"ERROR: Cannot find child link for joint '{child_joint_name}' in body {child_body_id}")
            return None
        
        # Debug information
        print(f"\nCreating constraint:")
        print(f"  Parent: Body {parent_body_id} ({self.body_names.get(parent_body_id, 'unnamed')})")
        print(f"    Joint: '{parent_joint_name}' -> Link Index: {parent_link_idx}")
        print(f"  Child: Body {child_body_id} ({self.body_names.get(child_body_id, 'unnamed')})")
        print(f"    Joint: '{child_joint_name}' -> Link Index: {child_link_idx}")
        
        # Create the constraint
        try:
            constraint_id = p.createConstraint(
                parentBodyUniqueId=parent_body_id,
                parentLinkIndex=parent_link_idx,
                childBodyUniqueId=child_body_id,
                childLinkIndex=child_link_idx,
                **constraint_kwargs
            )
            print(f"  Constraint created successfully with ID: {constraint_id}")
            return constraint_id
            
        except Exception as e:
            print(f"ERROR: Failed to create constraint: {e}")
            return None
    
    def print_all_debug_info(self):
        """Print debug information for all managed bodies."""
        for body_id, mapper in self.body_mappers.items():
            mapper.print_joint_info()
            mapper.print_link_info()
            mapper.validate_mappings()
