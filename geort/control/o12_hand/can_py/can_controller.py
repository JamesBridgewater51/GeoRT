from ctypes import *
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any
import argparse
from queue import Queue
import queue
from geort.control.o12_hand.py_sdk.src.ctrl import O12HandCtrl
from ..py_sdk.src.constants import GestureID

import threading
import pybullet as p
import pybullet_data
import numpy as np
import warnings
from cprint import cprint
from ..py_sdk.src.constants import ActiveJointPosGestureMap, MAX_JOINT, MAX_ACTIVE_JOINT, GestureID, GestureName, LeftJointNames, RightJointNames, ActiveJointID
from ..py_sdk.src.constants import O12handProActuator
import os
import sys

# CAN FD Library Setup (from original script)
STATUS_OK = 0

DEFAULT_DEV = int(os.getenv("CAN_DEV", "0"))
DEFAULT_CH = int(os.getenv("CAN_CH", "0"))  # 仅在 LINUX_EXTRA_CHANNEL=True 时使用

class Can_Config(Structure):  
    _fields_ = [("baudrate", c_uint),
                ("Pres", c_ushort),
                ("Tseg1", c_ubyte),
                ("Tseg2", c_ubyte),
                ("SJW", c_ubyte),
                ("config", c_ubyte),
                ("Model", c_ubyte),
                ("Reserved", c_ubyte)]
    
class CanFD_Config(Structure):  
    _fields_ = [("NomBaud", c_uint),
                ("DatBaud", c_uint),
                ("NomPres", c_ushort),
                ("NomTseg1", c_char),
                ("NomTseg2", c_char),
                ("NomSJW", c_char),
                ("DatPres", c_char),
                ("DatTseg1", c_char),
                ("DatTseg2", c_char),
                ("DatSJW", c_char),
                ("Config", c_char),
                ("Model", c_char),
                ("Cantype", c_char)]
    
class Can_Msg(Structure):  
    _fields_ = [("ID", c_uint),
                ("TimeStamp", c_uint),
                ("FrameType", c_ubyte),
                ("DataLen", c_ubyte),
                ("ExternFlag", c_ubyte),                
                ("RemoteFlag", c_ubyte),
                ("BusSatus", c_ubyte),
                ("ErrSatus", c_ubyte),
                ("TECounter", c_ubyte), 
                ("RECounter", c_ubyte),
                ("Data", c_ubyte*8)]
    
class CanFD_Msg(Structure):  
    _fields_ = [("ID", c_uint),
                ("TimeStamp", c_uint),
                ("FrameType", c_ubyte),
                ("DLC", c_ubyte),
                ("ExternFlag", c_ubyte),                
                ("RemoteFlag", c_ubyte),
                ("BusSatus", c_ubyte),
                ("ErrSatus", c_ubyte),
                ("TECounter", c_ubyte), 
                ("RECounter", c_ubyte),
                ("Data", c_ubyte*64)]
    
class CanFD_Msg_ARRAY(Structure):
    _fields_ = [('SIZE', c_uint16), ('STRUCT_ARRAY', POINTER(CanFD_Msg))]

    def __init__(self, num_of_structs):
        self.STRUCT_ARRAY = cast((CanFD_Msg * num_of_structs)(), POINTER(CanFD_Msg))
        self.SIZE = num_of_structs
        self.ADDR = self.STRUCT_ARRAY[0]

# Load CAN library
CDLL("/usr/local/lib/libusb-1.0.so", RTLD_GLOBAL)
libcan = cdll.LoadLibrary("/usr/local/lib/libcanbus.so")

# DLC to length mapping
dlc2len = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 12, 10: 16, 11: 20, 12: 24, 13: 32, 14: 48, 15: 64}

# Length to DLC mapping
len2dlc = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 12: 9, 16: 10, 20: 11, 24: 12, 32: 13, 48: 14, 64: 15}
# --- PyBullet Visualization Class ---
class HandVisualizer:
    """
    Manages a PyBullet simulation for visualizing or controlling a hand pose.
    Supports two modes:
    - 'monitor': Passively displays poses received via the `update_pose` method.
    - 'control': Creates joint sliders and sends their values to a callback.
    """

    def __init__(self, title: str, urdf_path: str, mode: str, is_right_hand: bool, slider_callback: Optional[Callable[[np.ndarray], None]] = None):
        if mode == 'control' and slider_callback is None:
            raise ValueError("A slider_callback must be provided for 'control' mode")

        self.title = title
        self.urdf_path = urdf_path
        self.mode = mode
        self.slider_callback = slider_callback
        
        self.pose_queue = Queue()
        self._stop_event = threading.Event()
        
        self.physics_client = None
        self.hand_id = None
        self.joint_indices = []
        self.slider_ids = []
        self.joint_names = RightJointNames if is_right_hand else LeftJointNames
        self.joint_info_map = {}
        self.joint_pos_gesture_map = {}
        self.gesture_button_ids = {}
        self.is_right_hand = is_right_hand
        
        # Track last slider positions and button states
        self.last_slider_positions = []
        self.last_button_counts = {}

    def _setup_simulation(self):
        """Connects to PyBullet and sets up the scene in its own thread."""
        self.physics_client = p.connect(p.GUI, options=f"--title='{self.title} - {self.mode.upper()} Mode'")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        self.hand_id = p.loadURDF(self.urdf_path, useFixedBase=True)
        
        self.kin_ctrl = O12HandCtrl(is_right_hand=self.is_right_hand)
        all_joints = {p.getJointInfo(self.hand_id, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(self.hand_id))}
        
        for idx, joint_name in enumerate(self.joint_names):
            if joint_name not in all_joints:
                warnings.warn(f"Joint '{joint_name}' not found in URDF. Skipping.")
                continue
            joint_index = all_joints[joint_name]
            info = p.getJointInfo(self.hand_id, joint_index)
            if info[2] == p.JOINT_FIXED:
                continue
            self.joint_indices.append(joint_index)
            self.joint_info_map[joint_index] = {
                'name': info[1].decode('UTF-8'),
                'lower_limit': info[8],
                'upper_limit': info[9]
            }

        if self.mode == 'control':
            active_joint_pos = ActiveJointPosGestureMap[GestureID.HOME].copy()
            if not self.is_right_hand:
                active_joint_pos[ActiveJointID.ActiveJointThumbRoll] *= -1  # Flip thumb roll for right hand.
                active_joint_pos[ActiveJointID.ActiveJointThumbAbAd] *= -1  # Flip thumb ab/ad for right hand.
            # FIXME: this is a temporary fix to set the initial slider positions to zeros.
            # active_joint_pos = np.zeros_like(active_joint_pos)

            self._create_sliders(qpos=self.kin_ctrl.get_all_joint_pos(active_joint_pos))
            self._create_buttons()
            # Start simulation loop in a separate thread for control mode
            self.simulation_thread = threading.Thread(target=self._run_simulation_loop, daemon=True)
            self.simulation_thread.start()

    def _create_sliders(self, qpos: Optional[list] = None):
        """Creates a debug slider for each controllable joint."""
        # cprint.ok("Creating joint control sliders...")
        for idx, (joint_index, info) in enumerate(self.joint_info_map.items()):
            slider_id = p.addUserDebugParameter(
                paramName=info['name'],
                rangeMin=info['lower_limit'],
                rangeMax=info['upper_limit'],
                startValue=0.0 if qpos is None else qpos[idx]
            )
            self.slider_ids.append(slider_id)
            self.last_slider_positions.append(0.0 if qpos is None else qpos[idx])
    
    def _remove_sliders(self):
        """Removes all debug sliders."""
        # cprint.ok("Removing joint control sliders...")
        for slider_id in self.slider_ids:
            p.removeUserDebugItem(slider_id)
        self.slider_ids.clear()
        self.last_slider_positions.clear()

    def _update_sliders(self, qpos: list):
        # Since pybullet doesn't provide APIs like `resetUserDebugParameter`, updating sliders can only be achieved by removing and recreating.
        self._remove_sliders()
        self._create_sliders(qpos=qpos)

    def _create_buttons(self):
        """Creates a debug button for each controllable joint."""
        cprint.ok("Creating gesture control buttons...")
        for gid, active_joint_pos in enumerate(ActiveJointPosGestureMap):
            # if gid not in [GestureID.HOME, GestureID.WIRE_STRIPPER_PRE, GestureID.WIRE_STRIPPER_GRASP, GestureID.WIRE_STRIPPER_END, GestureID.GRASP_PRE, GestureID.GRASP, GestureID.GRASP_END, GestureID.UNPLUG_PRE, GestureID.UNPLUG, GestureID.UNPLUG_END]:
            #     continue
            if not self.is_right_hand:
                # Left urdf's ThumbRollJoint and ThumbAbAdJoint are flipped compared to right hand.
                active_joint_pos[ActiveJointID.ActiveJointThumbRoll] *= -1  # Flip thumb roll for right hand.
                active_joint_pos[ActiveJointID.ActiveJointThumbAbAd] *= -1  # Flip thumb ab/ad for right hand.
            self.joint_pos_gesture_map[gid] = self.kin_ctrl.get_all_joint_pos(active_joint_pos)
            btn_id = p.addUserDebugParameter(f"Gesture: {GestureName[gid]}", 1, 0, 1)  # button (min>max)
            self.gesture_button_ids[gid] = btn_id
            self.last_button_counts[gid] = p.readUserDebugParameter(btn_id)

    def run(self):
        """The main loop for the visualization thread."""
        self._setup_simulation()
        while not self._stop_event.is_set():
            try:
                if self.mode == 'monitor':
                    self._run_monitor_mode()
                else: # control mode
                    self._run_control_mode()
            except p.error as e:
                # This can happen if the user closes the window.
                cprint.err(f"PyBullet error in visualizer thread: {e}. Shutting down.")
                self._stop_event.set()
                break
        
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        print(f"Visualizer '{self.title}' stopped.")

    def _run_monitor_mode(self):
        """Loop for passively displaying hand states."""
        try:
            joint_angles = self.pose_queue.get(timeout=0.1)
            if joint_angles is None: # Poison pill
                self._stop_event.set()
                return
            
            for i, angle in enumerate(joint_angles):
                if i < len(self.joint_indices):
                    p.resetJointState(self.hand_id, self.joint_indices[i], targetValue=angle)
        except queue.Empty:
            pass # Normal on timeout

    def _run_control_mode(self):
        """Loop for reading sliders and controlling the hand."""
        update_simulation = False
        target_positions = None

        # Check buttons first (counter increases by 1 per press)
        button_pressed = False
        for gid, btn_id in self.gesture_button_ids.items():
            count_now = int(p.readUserDebugParameter(btn_id))
            count_prev = int(self.last_button_counts[gid])
            if count_now > count_prev:
                # Button pressed, apply the corresponding gesture
                cprint.ok(f"[DexterousHandController] Applying gesture: {GestureName[gid]}, count_now: {count_now}, count_prev: {count_prev}, qpos: {self.joint_pos_gesture_map[gid].tolist()}")
                target_positions = self.joint_pos_gesture_map[gid].tolist()
                button_pressed = True
                update_simulation = True
                self.last_button_counts[gid] = count_now
                # self._update_sliders(target_positions)
                break

        # If no button pressed, check sliders for changes
        if not button_pressed:
            current_slider_positions = []
            for slider_id in self.slider_ids:
                current_slider_positions.append(p.readUserDebugParameter(slider_id))
            
            # Check if slider positions have changed
            if current_slider_positions != self.last_slider_positions:
                target_positions = current_slider_positions
                self.last_slider_positions = current_slider_positions[:]
                update_simulation = True

        # Only update if there were changes
        if update_simulation and target_positions is not None:
            # Put target positions in queue for simulation thread
            try:
                self.pose_queue.put_nowait(target_positions)
            except queue.Full:
                # If queue is full, skip this update to prevent blocking
                pass
            
            # Send the joint angles to the driver via the callback (in current thread)
            if self.slider_callback:
                self.slider_callback(np.array(target_positions))

    def _run_simulation_loop(self):
        """Background thread for PyBullet simulation updates."""
        while not self._stop_event.is_set():
            try:
                # Get target positions from queue (blocking with timeout)
                target_positions = self.pose_queue.get(timeout=0.1)
                
                # Update the PyBullet model
                for i, pos in enumerate(target_positions):
                    if i < len(self.joint_indices):
                        joint_index = self.joint_indices[i]
                        p.setJointMotorControl2(
                            self.hand_id,
                            joint_index,
                            p.POSITION_CONTROL,
                            targetPosition=pos,
                            force=60.0 # Apply some force to reach the position
                        )
                
                # Step simulation multiple times for smoother movement
                for i in range(30):
                    p.stepSimulation()
                    
            except queue.Empty:
                # No new positions, continue simulation with current targets
                p.stepSimulation()
                time.sleep(1./240.)
            except Exception as e:
                if not self._stop_event.is_set():
                    cprint.err(f"Error in simulation loop: {e}", interrupt=False)

    def update_pose(self, joint_angles: np.ndarray):
        """Public method for the main thread to update the hand's pose (in 'monitor' mode)."""
        if self.mode == 'monitor' and not self._stop_event.is_set():
            self.pose_queue.put(joint_angles)

    def stop(self):
        """Signals the visualization thread to stop."""
        self._stop_event.set()
        if self.mode == 'monitor':
            self.pose_queue.put(None) # Unblock the queue.get() call

@dataclass
class RegisterDefinition:
    """Class to hold register definition metadata"""
    reg_addr: int
    reg_name: str
    sub_reg_addr: int
    sub_reg_name: str
    description: str
    length: int
    access: str
    data_type: str

class DexterousHandController:
    def __init__(self, device_id=0x01, product_id=0x01, active_dof=12, auto_clear_errors=True):
        self.device_id = device_id
        self.product_id = product_id
        self.active_dof = active_dof  # Number of active degrees of freedom
        self.auto_clear_errors = auto_clear_errors  # Whether to automatically clear motor errors
        self.initialize_can()
        self.register_map = self.build_register_map()
        
    def initialize_can(self):
        """Initialize CAN FD interface"""
        ret = libcan.CAN_ScanDevice()
        cprint.info(f'ScanDevice found {ret} channels')

        ret = libcan.CAN_OpenDevice(DEFAULT_DEV, DEFAULT_CH)
        if ret != STATUS_OK:
            # raise RuntimeError('Failed to open CAN device')
            cprint.err("[DexterousHandController] Failed to open CAN device ...", interrupt=False)
        else:
            cprint.info("[DexterousHandController] CAN device opened successfully ...")
            
        # Configure CAN FD (1Mbps arbitration, 5Mbps data)
        can_initconfig = CanFD_Config(
            1000000, 5000000, 0x0, 0x0, 0x0, 0x0, 
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1)

        ret = libcan.CANFD_Init(DEFAULT_DEV, DEFAULT_CH, byref(can_initconfig))
        if ret != STATUS_OK:
            # raise RuntimeError('Failed to initialize CAN FD')
            cprint.err("[DexterousHandController] Failed to initialize CAN FD ...", interrupt=False)
        else:
            cprint.info("[DexterousHandController] CAN FD initialized successfully ...")

    def build_register_map(self) -> Dict[Tuple[int, int], RegisterDefinition]:
        """Build a map of (reg_addr, sub_reg_addr) to RegisterDefinition"""
        registers = {}
        
        # Manufacturer Info (Pn1)
        registers[(0x01, 0x00)] = RegisterDefinition(0x01, "Manufacturer Info", 0x00, 
            "Sub-register Operation", "Read all sub-registers", 48, "RO", "raw")
        registers[(0x01, 0x01)] = RegisterDefinition(0x01, "Manufacturer Info", 0x01, 
            "Product Model", "ASCII string", 16, "RO", "ascii")
        registers[(0x01, 0x02)] = RegisterDefinition(0x01, "Manufacturer Info", 0x02, 
            "Serial Number", "Formatted string", 14, "RO", "serial")
        registers[(0x01, 0x03)] = RegisterDefinition(0x01, "Manufacturer Info", 0x03, 
            "Hardware Version", "Version numbers", 4, "RO", "version")
        registers[(0x01, 0x04)] = RegisterDefinition(0x01, "Manufacturer Info", 0x04, 
            "Software Version", "Version numbers", 4, "RO", "version")
        registers[(0x01, 0x05)] = RegisterDefinition(0x01, "Manufacturer Info", 0x05, 
            "Power Voltage", "Voltage in mV", 2, "RO", "uint16")
        registers[(0x01, 0x06)] = RegisterDefinition(0x01, "Manufacturer Info", 0x06, 
            "Active DOF", "Number of degrees of freedom", 1, "RO", "uint8")
            
        # Device Info (Pn2)
        registers[(0x02, 0x00)] = RegisterDefinition(0x02, "Device Info", 0x00, 
            "Sub-register Operation", "Read all sub-registers", 5, "RW", "raw")
        registers[(0x02, 0x01)] = RegisterDefinition(0x02, "Device Info", 0x01, 
            "Device ID", "Device identifier", 1, "RW", "uint8")
        registers[(0x02, 0x02)] = RegisterDefinition(0x02, "Device Info", 0x02, 
            "Comm Config", "CAN FD parameters", 4, "RW", "comm_config")
            
        # Current Thresholds (Pn3)
        registers[(0x03, 0x00)] = RegisterDefinition(0x03, "Current Thresholds", 0x00,
            "Sub-register Operation", "Read/write all current thresholds", 
            2 * self.active_dof, "RW", "current_thresholds")
        
        for i in range(1, self.active_dof + 1):
            registers[(0x03, i)] = RegisterDefinition(0x03, "Current Thresholds", i,
                f"Motor #{i} Current Threshold", "Current threshold in mA", 
                2, "RW", "uint16")

        # Temperature Thresholds (Pn4)
        registers[(0x04, 0x00)] = RegisterDefinition(0x04, "Temperature Thresholds", 0x00,
            "Sub-register Operation", "Read/write all temperature thresholds", 
            2 * self.active_dof, "RW", "temp_thresholds")
        
        for i in range(1, self.active_dof + 1):
            registers[(0x04, i)] = RegisterDefinition(0x04, "Temperature Thresholds", i,
                f"Motor #{i} Temp Thresholds", "Warm-up and over-temp thresholds", 
                2, "RW", "temp_threshold_pair")

        # Control Modes (Pn16)
        registers[(0x10, 0x00)] = RegisterDefinition(0x10, "Control Modes", 0x00,
            "Sub-register Operation", "Read/write all control modes", 
            self.active_dof, "RW", "control_modes")
        
        for i in range(1, self.active_dof + 1):
            registers[(0x10, i)] = RegisterDefinition(0x10, "Control Modes", i,
                f"Motor #{i} Control Mode", "Control mode setting", 
                1, "RW", "control_mode")

        # Torque Control (Pn17)
        registers[(0x11, 0x00)] = RegisterDefinition(0x11, "Torque Control", 0x00,
            "Sub-register Operation", "Read/write all torque values", 
            2 * self.active_dof, "RW", "torque_values")
        
        for i in range(1, self.active_dof + 1):
            registers[(0x11, i)] = RegisterDefinition(0x11, "Torque Control", i,
                f"Motor #{i} Torque", "Target/actual torque in g", 
                2, "RW", "int16")

        # Speed Control (Pn18)
        registers[(0x12, 0x00)] = RegisterDefinition(0x12, "Speed Control", 0x00,
            "Sub-register Operation", "Read/write all speed values", 
            2 * self.active_dof, "RW", "speed_values")
        
        for i in range(1, self.active_dof + 1):
            registers[(0x12, i)] = RegisterDefinition(0x12, "Speed Control", i,
                f"Motor #{i} Speed", "Target/actual speed (1/65535 max)", 
                2, "RW", "int16")

        # Position Control (Pn19)
        registers[(0x13, 0x00)] = RegisterDefinition(0x13, "Position Control", 0x00,
            "Sub-register Operation", "Read/write all position values", 
            2 * self.active_dof, "RW", "position_values")
        
        for i in range(1, self.active_dof + 1):
            registers[(0x13, i)] = RegisterDefinition(0x13, "Position Control", i,
                f"Motor #{i} Position", "Target/actual position (1/65535 max)", 
                2, "RW", "int16")

        # Gesture Control (Pn21)
        registers[(0x15, 0x00)] = RegisterDefinition(0x15, "Gesture Control", 0x00,
            "Digit 1", "Predefined gesture 1", 64, "RW", "gesture_data")
        # Add other predefined gestures (0x01-0x09)
        for i in range(1, 10):
            registers[(0x15, i)] = RegisterDefinition(0x15, "Gesture Control", i,
                f"Digit {i+1}", f"Predefined gesture {i+1}", 64, "RW", "gesture_data")
        
        # Add custom gestures (0x0A-0x1F)
        for i in range(0x0A, 0x20):
            registers[(0x15, i)] = RegisterDefinition(0x15, "Gesture Control", i,
                f"Custom Gesture {i-0x09}", "User-defined gesture", 64, "RW", "gesture_data")

        # Error Reporting (Pn32)
        registers[(0x20, 0x00)] = RegisterDefinition(0x20, "Error Reporting", 0x00,
            "Sub-register Operation", "Read all error states", 
            2 * self.active_dof, "RO", "error_states")
        
        for i in range(1, self.active_dof + 1):
            registers[(0x20, i)] = RegisterDefinition(0x20, "Error Reporting", i,
                f"Motor #{i} Errors", "Motor error states", 
                2, "RW", "error_bits")

        # Temperature Reporting (Pn33)
        registers[(0x21, 0x00)] = RegisterDefinition(0x21, "Temperature Reporting", 0x00,
            "Sub-register Operation", "Configure temp reporting", 
            2 * self.active_dof, "RW", "temp_reporting")
        
        for i in range(1, self.active_dof + 1):
            registers[(0x21, i)] = RegisterDefinition(0x21, "Temperature Reporting", i,
                f"Motor #{i} Temp", "Configure/read temperature", 
                2, "RW", "temp_report")

        # Current Reporting (Pn34)
        registers[(0x22, 0x00)] = RegisterDefinition(0x22, "Current Reporting", 0x00,
            "Sub-register Operation", "Configure current reporting", 
            2 * self.active_dof, "RW", "current_reporting")
        
        for i in range(1, self.active_dof + 1):
            registers[(0x22, i)] = RegisterDefinition(0x22, "Current Reporting", i,
                f"Motor #{i} Current", "Configure/read current", 
                2, "RW", "current_report")

        return registers


    def build_can_id(self, reg_addr: int, sub_reg_addr: int, write: bool = False) -> int:
        """
        设备ID: Bit0-Bit6. 定义为：广播地址 0x00, 默认地址 0x01, 设备地址 0x01~0x7F.
        Bit7 是读写标志位, 0表示读, 1表示写.
        """
        can_id = 0
        can_id |= (self.device_id & 0x7F)           # Bits 0-6: Device ID
        can_id |= (0x80 if write else 0x00)          # Bit 7: R/W flag
        can_id |= (self.product_id & 0x7F) << 8      # Bits 8-14: Product ID
        can_id |= (reg_addr & 0xFF) << 16            # Bits 16-23: Register address
        can_id |= (sub_reg_addr & 0x1F) << 24        # Bits 24-28: Sub-register address
        return can_id

    def send_query(self, reg_addr: int, sub_reg_addr: int, 
                  data: Optional[bytes] = None, timeout_ms: int = 100, wait_for_response: bool = True) -> Optional[CanFD_Msg]:
        """Send a query and wait for response"""
        is_write = data is not None
        can_id = self.build_can_id(reg_addr, sub_reg_addr, is_write)
        
        # Prepare message
        # CAN_FD 接口默认使用 1Mbps 80% + 5Mbps 80% 扩展标识符(29bit) 数据帧格式, 禁用标准标识符和远程帧格式.
        msg = CanFD_Msg()
        msg.ID = can_id
        msg.TimeStamp = 0
        msg.FrameType = 0xC  # CAN FD frame
        msg.ExternFlag = 1   # 启用扩展标识符格式
        msg.RemoteFlag = 0   # 禁用远程帧格式

        if is_write:
            data_len = min(len(data), 64)
            msg.DLC = len2dlc[data_len]
            for i in range(data_len):
                msg.Data[i] = data[i]
        else:
            msg.DLC = 0
            
        # Send message
        ret = libcan.CANFD_Transmit(DEFAULT_DEV, DEFAULT_CH, byref(msg), 1, timeout_ms)
        if ret != 1:
            # cprint.warn(f"[DexterousHandController/CAN] Failed to send message to reg {reg_addr:08X} sub {sub_reg_addr:08X}")
            return None
        
        if not wait_for_response:
            return None
            
        # Wait for response
        receive_buffer = CanFD_Msg_ARRAY(500)

        ret = libcan.CANFD_Receive(DEFAULT_DEV, DEFAULT_CH, byref(receive_buffer.ADDR), 500, timeout_ms)

        if ret > 0:
            # Check all received messages for both expected response and error reports
            expected_response = None
            for i in range(ret):
                msg = receive_buffer.STRUCT_ARRAY[i]
                
                # Check if this is the expected response
                # NOTE: We ignore the R/W flag (bit 7) when matching IDs since paired requests/responses may have different R/W flags.
                if (msg.ID & ~0x80) == (can_id & ~0x80):
                    expected_response = msg
                
                # Check if this is an error report (Pn32, register 0x20)
                self._check_and_handle_error_report(msg)
            
            if expected_response:
                return expected_response
            else:
                cprint.warn(f"[DexterousHandController/CAN] Response ID mismatch for reg {reg_addr:08X} sub {sub_reg_addr:05X}. Expected {can_id:08X}, got {receive_buffer.STRUCT_ARRAY[0].ID:08X}")
        else:
            cprint.warn(f"[DexterousHandController/CAN] No response received for reg {reg_addr:08X} sub {sub_reg_addr:05X}")
            return None

    def parse_response(self, reg_addr: int, sub_reg_addr: int, response: CanFD_Msg) -> Any:
        """Parse response data according to register definition"""
        if (reg_addr, sub_reg_addr) not in self.register_map:
            return bytes(response.Data[:dlc2len[response.DLC]])
            
        reg_def = self.register_map[(reg_addr, sub_reg_addr)]
        data = bytes(response.Data[:dlc2len[response.DLC]])
        
        # Existing parsers remain the same...
        
        # Add new parsers for motor control registers
        if reg_def.data_type == "uint16":
            return int.from_bytes(data[:2], byteorder='little')
        elif reg_def.data_type == "int16":
            val = int.from_bytes(data[:2], byteorder='little', signed=True)
            return val
        elif reg_def.data_type == "temp_threshold_pair":
            return {
                "warmup_threshold": data[0],
                "overtemp_threshold": data[1]
            }
        elif reg_def.data_type == "control_mode":
            modes = {
                0: "Position",
                1: "Velocity",
                2: "Torque",
                3: "Position-Torque",
                4: "Velocity-Torque",
                5: "Position-Velocity-Torque"
            }
            return modes.get(data[0] & 0x07, "Unknown")
        elif reg_def.data_type == "error_bits":
            errors = []
            error_flags = [
                "Stall Protection",
                "Overtemp Protection",
                "Overcurrent Protection",
                "Motor Error",
                "Comm Error"
            ]
            error_word = int.from_bytes(data[:2], byteorder='little')
            for i in range(5):
                if error_word & (1 << i):
                    errors.append(error_flags[i])
            return errors or "No errors"
        elif reg_def.data_type == "temp_report":
            if len(data) >= 2:
                return {
                    "report_interval": int.from_bytes(data[:2], byteorder='little'),
                    "temperature": int.from_bytes(data[:2], byteorder='little', signed=True)
                }
        elif reg_def.data_type == "current_report":
            if len(data) >= 2:
                return {
                    "report_interval": int.from_bytes(data[:2], byteorder='little'),
                    "current": int.from_bytes(data[:2], byteorder='little', signed=True)
                }
        elif reg_def.data_type == "gesture_data":
            # Parse gesture data (array of joint positions)
            positions = []
            for i in range(0, min(len(data), 64), 2):
                pos = int.from_bytes(data[i:i+2], byteorder='little', signed=True)
                positions.append(pos)
            return positions
            
        return data

    def _check_and_handle_error_report(self, msg: CanFD_Msg) -> None:
        """Check if a received message is an error report and handle it"""
        # Extract components from CAN ID
        device_id = msg.ID & 0x7F
        rw_flag = (msg.ID >> 7) & 0x01
        product_id = (msg.ID >> 8) & 0x7F
        reg_addr = (msg.ID >> 16) & 0xFF
        sub_reg_addr = (msg.ID >> 24) & 0x1F
        
        # Check if this is an error report (Pn32, reg_addr 0x20)
        if reg_addr == 0x20 and device_id == self.device_id and product_id == self.product_id:
            if sub_reg_addr == 0x00:
                # All motor errors
                self._handle_all_motor_errors(msg)
            elif 1 <= sub_reg_addr <= self.active_dof:
                # Individual motor error
                self._handle_individual_motor_error(sub_reg_addr, msg)

    def _handle_individual_motor_error(self, motor_id: int, msg: CanFD_Msg) -> None:
        """Handle error report for an individual motor"""
        if dlc2len[msg.DLC] < 2:
            return
            
        data = bytes(msg.Data[:dlc2len[msg.DLC]])
        error_word = int.from_bytes(data[:2], byteorder='little')
        
        if error_word == 0:
            return  # No errors
            
        error_flags = [
            "Stall Protection",
            "Overtemp Protection", 
            "Overcurrent Protection",
            "Motor Error",
            "Comm Error"
        ]
        
        detected_errors = []
        for i in range(5):
            if error_word & (1 << i):
                detected_errors.append(error_flags[i])
        
        if detected_errors:
            error_msg = f"[ERROR] Motor #{motor_id} errors detected: {', '.join(detected_errors)}"
            cprint.err(error_msg, interrupt=False)
            
            # Auto-clear errors if enabled
            if self.auto_clear_errors:
                cprint.info(f"Auto-clearing errors for Motor #{motor_id}")
                self.clear_errors(motor_id)
            else:
                cprint.info(f"Motor #{motor_id} has errors. Call clear_errors({motor_id}) to clear them manually.")

    def _handle_all_motor_errors(self, msg: CanFD_Msg) -> None:
        """Handle error reports for all motors"""
        data_len = dlc2len[msg.DLC]
        if data_len < 2:
            return
            
        data = bytes(msg.Data[:data_len])
        
        # Process error data for each motor (2 bytes per motor)
        for motor_id in range(1, min(self.active_dof + 1, data_len // 2 + 1)):
            offset = (motor_id - 1) * 2
            if offset + 1 < len(data):
                error_word = int.from_bytes(data[offset:offset+2], byteorder='little')
                if error_word != 0:
                    # Create a mock message for individual handling
                    mock_msg = CanFD_Msg()
                    mock_msg.DLC = 1  # len2dlc[2]
                    mock_msg.Data[0] = data[offset]
                    mock_msg.Data[1] = data[offset + 1]
                    self._handle_individual_motor_error(motor_id, mock_msg)

    def set_motor_current_threshold(self, motor_id: int, threshold_ma: int) -> bool:
        """Set current threshold for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
        if not 0 <= threshold_ma <= 65535:
            raise ValueError("Threshold must be between 0-65535 mA")
            
        data = threshold_ma.to_bytes(2, byteorder='little')
        self.send_query(0x03, motor_id, data, wait_for_response=False)
        return True

    def get_motor_current_threshold(self, motor_id: int) -> Optional[int]:
        """Get current threshold for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
            
        response = self.send_query(0x03, motor_id)
        if response is None:
            return None
        return self.parse_response(0x03, motor_id, response)

    def get_all_current_thresholds(self) -> Optional[List[int]]:
        """Get current thresholds for all motors"""
        response = self.send_query(0x03, 0x00)
        if response is None:
            return None
        data = bytes(response.Data[:dlc2len[response.DLC]])
        thresholds = []
        for i in range(0, len(data), 2):
            threshold = int.from_bytes(data[i:i+2], byteorder='little')
            thresholds.append(threshold)
        return thresholds

    def set_motor_temperature_thresholds(self, motor_id: int, warmup_temp: int, overtemp: int) -> bool:
        """Set temperature thresholds for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
        if not (50 <= warmup_temp <= overtemp - 5):
            raise ValueError("Invalid temperature thresholds")
            
        data = bytes([warmup_temp, overtemp])
        self.send_query(0x04, motor_id, data, wait_for_response=False)
        return True

    def get_motor_temperature_thresholds(self, motor_id: int) -> Optional[Dict[str, int]]:
        """Get temperature thresholds for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
            
        response = self.send_query(0x04, motor_id)
        if response is None:
            return None
        return self.parse_response(0x04, motor_id, response)

    def get_all_temperature_thresholds(self) -> Optional[List[Dict[str, int]]]:
        """Get temperature thresholds for all motors"""
        response = self.send_query(0x04, 0x00)
        if response is None:
            return None
        data = bytes(response.Data[:dlc2len[response.DLC]])
        thresholds = []
        for i in range(0, len(data), 2):
            warmup = data[i]
            overtemp = data[i+1]
            thresholds.append({"warmup_threshold": warmup, "overtemp_threshold": overtemp})
        return thresholds

    def set_control_mode(self, motor_id: int, mode: int) -> bool:
        """Set control mode for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
        if not 0 <= mode <= 5:
            raise ValueError("Mode must be 0-5")
            
        data = bytes([mode])
        self.send_query(0x10, motor_id, data, wait_for_response=False)
        return True

    def get_control_mode(self, motor_id: int) -> Optional[str]:
        """Get control mode for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
            
        response = self.send_query(0x10, motor_id)
        if response is None:
            return None
        return self.parse_response(0x10, motor_id, response)

    def get_all_control_modes(self) -> Optional[List[str]]:
        """Get control modes for all motors"""
        response = self.send_query(0x10, 0x00)
        if response is None:
            return None
        data = bytes(response.Data[:dlc2len[response.DLC]])
        modes = {0: "Position", 1: "Velocity", 2: "Torque", 3: "Position-Torque", 4: "Velocity-Torque", 5: "Position-Velocity-Torque"}
        return [modes.get(data[i] & 0x07, "Unknown") for i in range(len(data))]

    def set_target_torque(self, motor_id: int, torque_g: int) -> bool:
        """Set target torque for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
            
        data = torque_g.to_bytes(2, byteorder='little', signed=True)
        self.send_query(0x11, motor_id, data, wait_for_response=False)
        return True

    def get_actual_torque(self, motor_id: int) -> Optional[int]:
        """Get actual torque for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
            
        response = self.send_query(0x11, motor_id)
        if response is None:
            return None
        return self.parse_response(0x11, motor_id, response)

    def get_all_actual_torques(self) -> Optional[List[int]]:
        """Get actual torques for all motors"""
        response = self.send_query(0x11, 0x00)
        if response is None:
            return None
        data = bytes(response.Data[:dlc2len[response.DLC]])
        torques = []
        for i in range(0, len(data), 2):
            torque = int.from_bytes(data[i:i+2], byteorder='little', signed=True)
            torques.append(torque)
        return torques

    def set_target_speed(self, motor_id: int, speed: int) -> bool:
        """Set target speed for a specific motor (0-1000 = 0-100% of max speed)"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
        if not 0 <= speed <= 1000:
            raise ValueError("Speed must be 0-1000")
            
        data = speed.to_bytes(2, byteorder='little', signed=True)
        self.send_query(0x12, motor_id, data, wait_for_response=False)
        return True

    def get_actual_speed(self, motor_id: int) -> Optional[int]:
        """Get actual speed for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
            
        response = self.send_query(0x12, motor_id)
        if response is None:
            return None
        return self.parse_response(0x12, motor_id, response)

    def get_all_actual_speeds(self) -> Optional[List[int]]:
        """Get actual speeds for all motors"""
        response = self.send_query(0x12, 0x00)
        if response is None:
            return None
        data = bytes(response.Data[:dlc2len[response.DLC]])
        speeds = []
        for i in range(0, len(data), 2):
            speed = int.from_bytes(data[i:i+2], byteorder='little', signed=True)
            speeds.append(speed)
        return speeds

    def set_target_position(self, motor_id: int, position: int) -> bool:
        """Set target position for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
            
        data = position.to_bytes(2, byteorder='little', signed=True)
        self.send_query(0x13, motor_id, data, wait_for_response=False)
        return True

    def get_actual_position(self, motor_id: int) -> Optional[int]:
        """Get actual position for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
            
        response = self.send_query(0x13, motor_id)
        if response is None:
            return None
        return self.parse_response(0x13, motor_id, response)

    def get_all_actual_positions(self) -> Optional[List[int]]:
        """Get actual positions for all motors"""
        response = self.send_query(0x13, 0x00)
        if response is None:
            return None
        data = bytes(response.Data[:dlc2len[response.DLC]])
        positions = []
        for i in range(0, len(data), 2):
            position = int.from_bytes(data[i:i+2], byteorder='little', signed=True)
            positions.append(position)
        return positions

    def execute_gesture(self, gesture_id: int) -> bool:
        """Execute a predefined or custom gesture"""
        if not 0 <= gesture_id <= 0x1F:
            raise ValueError("Gesture ID must be 0x00-0x1F")
            
        # For gestures, a read operation executes the gesture
        response = self.send_query(0x15, gesture_id, wait_for_response=False)
        return response is not None

    def configure_gesture(self, gesture_id: int, positions: List[int]) -> bool:
        """Configure a custom gesture (IDs 0x0A-0x1F)"""
        if not 0x0A <= gesture_id <= 0x1F:
            raise ValueError("Custom gesture ID must be 0x0A-0x1F")
        if len(positions) != self.active_dof:
            raise ValueError(f"Must provide positions for all {self.active_dof} motors")
            
        data = bytearray()
        for pos in positions:
            data.extend(pos.to_bytes(2, byteorder='little', signed=True))
        # Pad to 64 bytes if needed
        data.extend(bytes(64 - len(data)))
        
        self.send_query(0x15, gesture_id, data, wait_for_response=False)
        return True

    def get_motor_errors(self, motor_id: int) -> Optional[List[str]]:
        """Get error states for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
            
        response = self.send_query(0x20, motor_id)
        if response is None:
            return None
        return self.parse_response(0x20, motor_id, response)

    def clear_errors(self, motor_id: int) -> bool:
        """Clear error states for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
            
        data = bytes([0, 0])  # Write 0x0000 to clear errors
        self.send_query(0x20, motor_id, data, wait_for_response=False)
        return True

    def configure_temperature_reporting(self, motor_id: int, interval_ms: int) -> bool:
        """Configure temperature reporting interval for a motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
            
        data = interval_ms.to_bytes(2, byteorder='little')
        self.send_query(0x21, motor_id, data, wait_for_response=False)
        return True

    def get_motor_temperature(self, motor_id: int) -> Optional[int]:
        """Get current temperature for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
            
        response = self.send_query(0x21, motor_id)
        if response is None:
            return None
        data = bytes(response.Data[:dlc2len[response.DLC]])
        if len(data) >= 2:
            return int.from_bytes(data[:2], byteorder='little', signed=True)
        return None

    def configure_current_reporting(self, motor_id: int, interval_ms: int) -> bool:
        """Configure current reporting interval for a motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
            
        data = interval_ms.to_bytes(2, byteorder='little')
        self.send_query(0x22, motor_id, data, wait_for_response=False)
        return True

    def get_motor_current(self, motor_id: int) -> Optional[int]:
        """Get current current for a specific motor"""
        if not 1 <= motor_id <= self.active_dof:
            raise ValueError(f"Motor ID must be between 1 and {self.active_dof}")
            
        response = self.send_query(0x22, motor_id)
        if response is None:
            return None
        data = bytes(response.Data[:dlc2len[response.DLC]])
        if len(data) >= 2:
            return int.from_bytes(data[:2], byteorder='little', signed=True)
        return None


    def parse_serial_number(self, data: bytes) -> str:
        # FIXME: the serial number docs description is not clear, you should figure it out.
        """Parse the serial number format"""
        if len(data) < 14:
            return "Invalid serial number"
            
        # Decode according to specification
        ss1 = chr(ord('A') + (data[0] >> 4))
        ss2 = str(data[0] & 0x0F)
        c = data[1]
        mm = f"{data[2]:02d}"
        vv = f"{data[3]:02d}"
        y = chr(ord('A') + data[4])
        m = chr(ord('1') + data[5] - 1) if data[5] <= 12 else '?'
        xxxxx = f"{int.from_bytes(data[6:11], 'little'):05d}"
        
        return f"{ss1}{ss2}{c}{mm}{vv}{y}{m}{xxxxx}"

    def parse_comm_config(self, data: bytes) -> Dict[str, Any]:
        """Parse communication configuration"""
        if len(data) < 4:
            return {}
            
        arb_baud = {0: "125Kbps", 1: "500Kbps", 2: "1Mbps"}.get(data[0], "Unknown")
        arb_sample = {0: "75.0%", 1: "80.0%", 2: "87.5%"}.get(data[1], "Unknown")
        data_baud = {0: "125Kbps", 1: "500Kbps", 2: "1Mbps", 3: "5Mbps"}.get(data[2], "Unknown")
        data_sample = {0: "75.0%", 1: "80.0%", 2: "87.5%"}.get(data[3], "Unknown")
        
        return {
            "arbitration_baudrate": arb_baud,
            "arbitration_sample_point": arb_sample,
            "data_baudrate": data_baud,
            "data_sample_point": data_sample
        }

    def query_register(self, reg_addr: int, sub_reg_addr: int) -> Optional[Any]:
        """Query a specific register and return parsed data"""
        response = self.send_query(reg_addr, sub_reg_addr)
        if response is None:
            return None
            
        return self.parse_response(reg_addr, sub_reg_addr, response)

    def dump_all_registers(self):
        """Query and display all readable registers"""
        print("\n=== Dexterous Hand Register Dump ===")
        
        # Group registers by their main register address
        registers_by_group = {}
        for (reg_addr, sub_reg_addr), reg_def in self.register_map.items():
            if reg_addr not in registers_by_group:
                registers_by_group[reg_addr] = []
            registers_by_group[reg_addr].append(reg_def)
        
        # Query each group
        for reg_addr, reg_defs in sorted(registers_by_group.items()):
            print(f"\nRegister Group 0x{reg_addr:02X}: {reg_defs[0].reg_name}")
            
            for reg_def in sorted(reg_defs, key=lambda x: x.sub_reg_addr):
                if reg_def.access not in ["RO", "RW"]:
                    continue
                    
                result = self.query_register(reg_def.reg_addr, reg_def.sub_reg_addr)
                if result is None:
                    print(f"  {reg_def.sub_reg_name} (0x{reg_def.sub_reg_addr:02X}): Failed to read")
                    continue
                    
                if isinstance(result, dict):
                    print(f"  {reg_def.sub_reg_name} (0x{reg_def.sub_reg_addr:02X}):")
                    for k, v in result.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {reg_def.sub_reg_name} (0x{reg_def.sub_reg_addr:02X}): {result}")

    def read_status(self) -> Dict[str, Any]:
        """Read and parse all status information from the OmniHand"""
        status = {}
        
        # Device and Product Information
        print("Reading device and product information...")
        
        # Manufacturer Info (Pn1)
        product_model = self.query_register(0x01, 0x01)
        if product_model:
            if isinstance(product_model, bytes):
                status['product_model'] = product_model.decode('ascii', errors='ignore').strip('\x00')
            else:
                status['product_model'] = str(product_model)
        
        serial_number = self.query_register(0x01, 0x02)
        if serial_number:
            if isinstance(serial_number, bytes):
                status['serial_number'] = self.parse_serial_number(serial_number)
            else:
                status['serial_number'] = str(serial_number)
        
        hw_version = self.query_register(0x01, 0x03)
        if hw_version:
            if isinstance(hw_version, bytes) and len(hw_version) >= 4:
                status['hardware_version'] = f"{hw_version[0]}.{hw_version[1]}.{hw_version[2]}.{hw_version[3]}"
            else:
                status['hardware_version'] = str(hw_version)
        
        sw_version = self.query_register(0x01, 0x04)
        if sw_version:
            if isinstance(sw_version, bytes) and len(sw_version) >= 4:
                status['software_version'] = f"{sw_version[0]}.{sw_version[1]}.{sw_version[2]}.{sw_version[3]}"
            else:
                status['software_version'] = str(sw_version)
        
        power_voltage = self.query_register(0x01, 0x05)
        if power_voltage is not None:
            status['power_voltage_mv'] = power_voltage
        
        active_dof = self.query_register(0x01, 0x06)
        if active_dof is not None:
            if isinstance(active_dof, bytes):
                status['active_dof'] = active_dof[0] if len(active_dof) > 0 else 0
            else:
                status['active_dof'] = active_dof
        
        # Device Info (Pn2)
        device_id = self.query_register(0x02, 0x01)
        if device_id is not None:
            if isinstance(device_id, bytes):
                status['device_id'] = device_id[0] if len(device_id) > 0 else 0
            else:
                status['device_id'] = device_id
        
        comm_config = self.query_register(0x02, 0x02)
        if comm_config:
            if isinstance(comm_config, bytes):
                status['comm_config'] = self.parse_comm_config(comm_config)
            else:
                status['comm_config'] = comm_config
        
        # Current Thresholds (Pn3)
        print("Reading current thresholds...")
        current_thresholds = self.get_all_current_thresholds()
        if current_thresholds:
            status['current_thresholds_ma'] = current_thresholds
        
        # Temperature Thresholds (Pn4)
        print("Reading temperature thresholds...")
        temp_thresholds = self.get_all_temperature_thresholds()
        if temp_thresholds:
            status['temperature_thresholds'] = temp_thresholds
        
        # Control Modes (Pn16)
        print("Reading control modes...")
        control_modes = self.get_all_control_modes()
        if control_modes:
            status['control_modes'] = control_modes
        
        # Current Positions, Speeds, Torques
        print("Reading current motor states...")
        actual_positions = self.get_all_actual_positions()
        if actual_positions:
            status['actual_positions'] = actual_positions
        
        actual_speeds = self.get_all_actual_speeds()
        if actual_speeds:
            status['actual_speeds'] = actual_speeds
        
        actual_torques = self.get_all_actual_torques()
        if actual_torques:
            status['actual_torques_g'] = actual_torques
        
        # Error Reporting (Pn32)
        print("Reading error states...")
        motor_errors = {}
        for motor_id in range(1, self.active_dof + 1):
            errors = self.get_motor_errors(motor_id)
            if errors:
                motor_errors[f'motor_{motor_id}'] = errors
        if motor_errors:
            status['motor_errors'] = motor_errors
        
        # Temperature Reporting (Pn33)
        print("Reading motor temperatures...")
        motor_temperatures = {}
        for motor_id in range(1, self.active_dof + 1):
            temp = self.get_motor_temperature(motor_id)
            if temp is not None:
                motor_temperatures[f'motor_{motor_id}'] = temp
        if motor_temperatures:
            status['motor_temperatures_c'] = motor_temperatures
        
        # Current Reporting (Pn34)
        print("Reading motor currents...")
        motor_currents = {}
        for motor_id in range(1, self.active_dof + 1):
            current = self.get_motor_current(motor_id)
            if current is not None:
                motor_currents[f'motor_{motor_id}'] = current
        if motor_currents:
            status['motor_currents_ma'] = motor_currents
        
        return status

    def print_status(self):
        """Read and print all status information in a formatted way"""
        print("\n" + "="*80)
        print("                    OMNI HAND STATUS REPORT")
        print("="*80)
        
        status = self.read_status()
        
        # Device Information
        print("\n📋 DEVICE INFORMATION")
        print("-" * 40)
        if 'product_model' in status:
            print(f"Product Model    : {status['product_model']}")
        if 'serial_number' in status:
            print(f"Serial Number    : {status['serial_number']}")
        if 'hardware_version' in status:
            print(f"Hardware Version : {status['hardware_version']}")
        if 'software_version' in status:
            print(f"Software Version : {status['software_version']}")
        if 'device_id' in status:
            print(f"Device ID        : 0x{status['device_id']:02X}")
        if 'active_dof' in status:
            print(f"Active DOF       : {status['active_dof']}")
        if 'power_voltage_mv' in status:
            print(f"Power Voltage    : {status['power_voltage_mv']/1000:.2f}V")
        
        # Communication Configuration
        if 'comm_config' in status:
            print(f"\n📡 COMMUNICATION CONFIG")
            print("-" * 40)
            config = status['comm_config']
            for key, value in config.items():
                print(f"{key:20} : {value}")
        
        # Motor Configuration
        print(f"\n⚙️  MOTOR CONFIGURATION")
        print("-" * 40)
        
        if 'control_modes' in status:
            print("Control Modes:")
            for i, mode in enumerate(status['control_modes'], 1):
                print(f"  Motor {i:2d}: {mode}")
        
        if 'current_thresholds_ma' in status:
            print("\nCurrent Thresholds (mA):")
            for i, threshold in enumerate(status['current_thresholds_ma'], 1):
                print(f"  Motor {i:2d}: {threshold:4d} mA")
        
        if 'temperature_thresholds' in status:
            print("\nTemperature Thresholds (°C):")
            for i, threshold in enumerate(status['temperature_thresholds'], 1):
                print(f"  Motor {i:2d}: Warmup={threshold['warmup_threshold']:2d}°C, "
                        f"Overtemp={threshold['overtemp_threshold']:2d}°C")
        
        # Current Motor States
        print(f"\n🔧 CURRENT MOTOR STATES")
        print("-" * 40)
        
        if 'actual_positions' in status:
            print("Positions (ticks):")
            for i, pos in enumerate(status['actual_positions'], 1):
                print(f"  Motor {i:2d}: {pos:6d}")
        
        if 'actual_speeds' in status:
            print("\nSpeeds (0-1000 scale):")
            for i, speed in enumerate(status['actual_speeds'], 1):
                print(f"  Motor {i:2d}: {speed:4d}")
        
        if 'actual_torques_g' in status:
            print("\nTorques (g):")
            for i, torque in enumerate(status['actual_torques_g'], 1):
                print(f"  Motor {i:2d}: {torque:6d}g")
        
        # Temperature Status
        if 'motor_temperatures_c' in status:
            print(f"\n🌡️  MOTOR TEMPERATURES")
            print("-" * 40)
            for motor, temp in status['motor_temperatures_c'].items():
                motor_num = motor.split('_')[1]
                print(f"Motor {motor_num:2s}: {temp:3d}°C")
        
        # Current Status
        if 'motor_currents_ma' in status:
            print(f"\n⚡ MOTOR CURRENTS")
            print("-" * 40)
            for motor, current in status['motor_currents_ma'].items():
                motor_num = motor.split('_')[1]
                print(f"Motor {motor_num:2s}: {current:4d} mA")
        
        # Error Status
        if 'motor_errors' in status:
            print(f"\n🚨 ERROR STATUS")
            print("-" * 40)
            has_errors = False
            for motor, errors in status['motor_errors'].items():
                motor_num = motor.split('_')[1]
                if isinstance(errors, list) and errors:
                    has_errors = True
                    print(f"Motor {motor_num:2s}: {', '.join(errors)}")
                elif errors != "No errors":
                    has_errors = True
                    print(f"Motor {motor_num:2s}: {errors}")
            
            if not has_errors:
                print("✅ No motor errors detected")
        else:
            print(f"\n✅ ERROR STATUS: All motors OK")
        
        print("\n" + "="*80)
        print("                    END OF STATUS REPORT")
        print("="*80)

# --- High-Level Driver Class ---
class OmniHandDriver:
    """A high-level driver that combines kinematics, CAN control, and visualization."""
    def __init__(self, device_id=0x01, product_id=0x01, is_right_hand=True, 
                 render=False, render_freq=100, urdf_path="o12_hand.urdf", mode='monitor', 
                 auto_clear_errors=True):
        self.kin_ctrl = O12HandCtrl(is_right_hand=is_right_hand)
        self.can_ctrl = DexterousHandController(device_id, product_id, auto_clear_errors=auto_clear_errors)
        self.mode = mode
        self.visualizer = None
        self.last_sent_ticks = None
        
        self.visualizer = None
        if render:
            cprint.info("On-screen rendering enabled.")

            slider_callback = self._on_slider_update if self.mode == 'control' else None

            self.visualizer = HandVisualizer(
                title='O12-Hand', 
                urdf_path=urdf_path, 
                mode=self.mode,
                is_right_hand=is_right_hand,
                slider_callback=slider_callback
            )
            self.visualizer_thread = threading.Thread(target=self.visualizer.run, daemon=True)
            self.visualizer_thread.start()

            if self.mode == 'monitor':
                self.poll_thread = threading.Thread(target=self._poll_actual_state, args=(render_freq,), daemon=True)
                self.poll_stop_event = threading.Event()
                self.poll_thread.start()

    def set_target_positions(self, motor_ticks: List[int]):
        """Sets target positions for all motors and updates desired visualization."""
        if len(motor_ticks) != self.can_ctrl.active_dof:
            raise ValueError(f"Expected {self.can_ctrl.active_dof} motor positions.")

        for i, ticks in enumerate(motor_ticks):
            # NOTE: Single call of set_target_position is 102ms +- 59.1 us. (mean +- std.dev. of 7 runs)
            self.can_ctrl.set_target_position(motor_id=i + 1, position=int(ticks))

        if self.visualizer:
            # Convert motor ticks back to full joint angles for visualization
            active_joints = self.kin_ctrl.convert_actuator_to_joint(np.array(motor_ticks))
            all_joint_angles = self.kin_ctrl.get_all_joint_pos(active_joints)
            # FIXME: setting target positions should not update visualizer's pose directly, we only visualize actual positions.
            # self.visualizer.update_pose(all_joint_angles)

    def set_gesture(self, gesture_id):
        """Calculates and sets a predefined gesture."""
        print(f"Setting gesture: {gesture_id.name}")
        motor_ticks = self.kin_ctrl.set_hand_gesture(gesture_id)
        self.set_target_positions(motor_ticks.tolist())

    def get_all_joint_angles(self) -> Optional[np.ndarray]:
        """Reads all motor positions and converts them to full joint angles."""
        motor_ticks = self.can_ctrl.get_all_actual_positions()
        if motor_ticks is None:
            return None
        
        active_joints = self.kin_ctrl.convert_actuator_to_joint(np.array(motor_ticks))
        all_joint_angles = self.kin_ctrl.get_all_joint_pos(active_joints)
        return all_joint_angles

    def _poll_actual_state(self, render_freq=1000):
        """Background thread to continuously read and visualize the actual hand state."""
        while not self.poll_stop_event.is_set():
            actual_angles = self.get_all_joint_angles()
            if actual_angles is not None and self.visualizer:
                self.visualizer.update_pose(actual_angles)
            time.sleep(1/render_freq) # Poll at ~1000Hz

    def _on_slider_update(self, joint_angles: np.ndarray):
        """
        Callback function passed to the visualizer for 'control' mode.
        Converts joint angles from sliders to motor ticks and sends them to the hand.
        """
        try:
            motor_ticks = self.kin_ctrl.convert_joint_to_actuator(joint_angles, is_active_only=False)
            
            # Optimization: Only send commands if the ticks have changed to avoid flooding the CAN bus.
            if self.last_sent_ticks is None or not np.array_equal(motor_ticks, self.last_sent_ticks):
                self.set_target_positions(motor_ticks.tolist())
                self.last_sent_ticks = motor_ticks

        except AttributeError:
            print("ERROR: The kinematics controller (`O12HandCtrl`) is missing the required "
                  "`convert_joint_to_actuator` method. Cannot send commands.")
        except Exception as e:
            print(f"Error during slider update: {e}")

    def close(self):
        """Gracefully closes all resources."""
        print("Closing OmniHand Driver...")
        if hasattr(self, 'poll_stop_event'):
            self.poll_stop_event.set()
            self.poll_thread.join()
        if self.visualizer:
            self.visualizer.stop()
            self.visualizer_thread.join()
        if libcan:
            libcan.CAN_CloseDevice(DEFAULT_DEV, DEFAULT_CH)
        print("Driver closed.")


def main():
    parser = argparse.ArgumentParser(description='O12 Dexterous Hand Controller with Visualization')
    parser.add_argument('--render', action='store_true', help='Enable on-screen PyBullet rendering.')
    parser.add_argument('--right_hand', action='store_true', help='Enable right hand control.')
    parser.add_argument('--render_freq', type=int, default=1000, help='Rendering frequency in Hz.')
    parser.add_argument('--urdf', type=str, default='o12_hand.urdf', help='Path to the O12 hand URDF file.')
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['monitor', 'control', 'debug'], 
        default='monitor', 
        help='Operation mode: "monitor" (visualize real hand state) or "control" (control real hand with sliders).'
    )
    args = parser.parse_args()

    driver = None
    try:
        # Initialize the main driver
        driver = OmniHandDriver(is_right_hand=args.right_hand, render=args.render, urdf_path=args.urdf, render_freq=args.render_freq, mode=args.mode)

        if args.mode == 'monitor':
            print("\n--- Starting Demo Sequence ---")
            for gesture_id in GestureID:
                driver.set_gesture(gesture_id)
                print(f"{gesture_id.name} gesture made. Pausing for 3 seconds...")
                # breakpoint()
                time.sleep(3)

            # --- Sequence 4: HOME Gesture ---
            driver.set_gesture(GestureID.HOME)
            print("HOME gesture made. Pausing for 3 seconds...")
        
        elif args.mode == 'control': # 'control' mode
            print("\n--- Starting CONTROL mode ---")
            print("Drag the sliders in the PyBullet window to control the real hand.")
            print("Press Ctrl+C in this terminal to exit.")
        
        elif args.mode == 'debug': # 'debug' mode
            print("\n--- Starting DEBUG mode ---")
            print("Debugging information will be printed to the console.")
            driver.can_ctrl.print_status()
            sys.exit(0)

        # Keep the script alive
        while True:
            # In control mode, the visualizer thread handles everything.
            # In monitor mode, the polling thread handles updates.
            # We just need to check if the visualizer window was closed.
            if driver.visualizer and driver.visualizer._stop_event.is_set():
                break
            time.sleep(1)  

    except KeyboardInterrupt:
        print("\nExit signal received.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if driver:
            driver.close()
            
def test():
    # Create controller (assuming 12 DOF)
    from ..py_sdk.src.ctrl import O12HandCtrl
    controller = DexterousHandController(device_id=0x01, active_dof=12)

    # Set motor 1 to torque control mode
    # for i in range(O12handProActuator.ActuatorCount):
    #     controller.set_control_mode(i + 1, 3)  # 3 = position-torque model.

    controller.print_status()

    target_qpos = [0.275, -0.524, -0.3, -1.1, 0.0, 0.0, 0.0, 0.0, 1.5, 1.57, 1.55, 1.55]
    o12_ctrl = O12HandCtrl(is_right_hand=True)
    motor_ticks = o12_ctrl.convert_joint_to_actuator(np.zeros_like(np.array(target_qpos)), is_active_only=True)
    # Target positions for all motors
    print(f"Target motor ticks: {motor_ticks}")
    for i in range(len(motor_ticks)):
        controller.set_target_position(i+1, int(motor_ticks[i]))

    for i in range(controller.active_dof):
        err = controller.get_motor_errors(i + 1)  # Get errors for each motor
        print(f"Motor {i+1} Errors: {err}")

    positions = None
    while positions is None:
        positions = controller.get_all_actual_positions()
    print(f"Actual positions: {positions}")
    qpos = o12_ctrl.convert_actuator_to_joint(np.array(positions))
    print(f"Actual qpos: {qpos}")

    # Set torque target for motor 1 (5700g)
    controller.set_target_torque(1, 5700)

    # Configure temperature thresholds for motor 2
    controller.set_motor_temperature_thresholds(2, 60, 80)

    # Execute predefined "OK" gesture (sub-register 0x09)
    controller.execute_gesture(0x09)

    # Configure custom gesture
    positions = [0, 0, 0, 0, 1854, 326, 0, 0, 175, 175, 1864, 0]  # Positions for all 12 motors
    controller.configure_gesture(0x0A, positions)  # Configure custom gesture 1

    # Clear errors on motor 3
    controller.clear_errors(3)


    # Configure temperature reporting for motor 4 (1000ms interval)
    # controller.configure_temperature_reporting(4, 1000)

    # try:
    #     controller = DexterousHandController(device_id=args.device_id, product_id=args.product_id)
        
    #     if args.reg:
    #         # Query specific register
    #         try:
    #             reg_parts = args.reg.split(':')
    #             reg_addr = int(reg_parts[0], 16)
    #             sub_reg_addr = int(reg_parts[1], 16) if len(reg_parts) > 1 else 0x00
                
    #             result = controller.query_register(reg_addr, sub_reg_addr)
    #             if result is not None:
    #                 print(f"Register 0x{reg_addr:02X}:0x{sub_reg_addr:02X} = {result}")
    #             else:
    #                 print("Failed to read register")
    #         except ValueError:
    #             print("Invalid register format. Use format like '0x01:0x02'")
    #     else:
    #         # Dump all registers
    #         controller.dump_all_registers()
            
    # except Exception as e:
    #     print(f"Error: {e}")
    # finally:
    #     libcan.CAN_CloseDevice(0, 0)
    libcan.CAN_CloseDevice(DEFAULT_DEV, DEFAULT_CH)

if __name__ == "__main__":
    # test()
    main()
