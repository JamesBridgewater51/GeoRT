from ctypes import *
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any
import argparse
from queue import Queue
import queue
from geort.control.o12_hand.py_sdk.src.ctrl import O12HandCtrl

import threading
import pybullet as p
import pybullet_data
import numpy as np
import warnings
from cprint import cprint
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
    def __init__(self, device_id=0x08, active_dof=10, auto_clear_errors=True):
        self.device_id = device_id            # Node ID (0..0x7FF)
        self.active_dof = active_dof          # 10 per spec (axes 1..10)
        self.auto_clear_errors = auto_clear_errors
        self._last_rx = None
        self.initialize_can()

    # ---------- CAN-FD bring-up ----------
    def initialize_can(self):
        ret = libcan.CAN_ScanDevice()
        cprint.info(f"[CAN] ScanDevice found {ret} channels")

        ret = libcan.CAN_OpenDevice(DEFAULT_DEV, DEFAULT_CH)
        if ret != STATUS_OK:
            cprint.err("[CAN] Failed to open CAN device", interrupt=False)
        else:
            cprint.info("[CAN] CAN device opened successfully")

        # Per spec: arb=1M (80%), data=5M (75%), FD only; classic CAN not supported.
        canfd = CanFD_Config(
            c_uint(1_000_000), c_uint(5_000_000),
            c_ushort(0), c_char(0), c_char(0), c_char(13), # 常规波特同同步跳转宽度, normal SJW.
            c_char(0), c_char(0), c_char(0), c_char(13), # 数据波特率同步跳转宽度, data SJW.
            c_char(7), c_char(0), c_char(1) # 分别是 config配置信息, 0x01: 接通内部电阻, 0x02: 离线唤醒， 0x04：自动; model工作模式：0为自动模式，can_type: 1为ISO-CANFD.
        )
        ret = libcan.CANFD_Init(DEFAULT_DEV, DEFAULT_CH, byref(canfd))
        if ret != STATUS_OK:
            cprint.err("[CAN] Failed to initialize CAN-FD", interrupt=False)
        else:
            cprint.info("[CAN] CAN-FD initialized (arb=1M, data=5M)")

    # ---------- DLC helpers ----------
    def _len_to_dlc(self, n: int) -> int:
        if n in len2dlc:
            return len2dlc[n]
        # CAN-FD legal payloads only; clamp to nearest supported
        if n <= 8:  return len2dlc[n]
        if n <= 12: return 9
        if n <= 16: return 10
        if n <= 20: return 11
        if n <= 24: return 12
        if n <= 32: return 13
        if n <= 48: return 14
        return 15

    def _dlc_len(self, dlc: int) -> int:
        return dlc2len.get(dlc, 0)

    # ---------- Core send/receive ----------
    def send_cmd(self, cmd: int, payload: bytes = b"", *, expect_reply: bool = True,
                 timeout_ms: int = 50, to_id: Optional[int] = None,
                 accept_from_any: bool = False) -> Optional[CanFD_Msg]:
        """
        Build and send a CAN-FD **standard-ID** frame:
        Data[0] = CMD, Data[1:] = payload. Read reply with same CMD.
        """
        node_id = (self.device_id if to_id is None else to_id) & 0x7FF

        msg = CanFD_Msg()
        msg.ID = node_id
        msg.TimeStamp = 0
        msg.FrameType = 0x04            # FD frame (matches vendor lib semantics)
        msg.ExternFlag = 0              # standard ID (per protocol)
        msg.RemoteFlag = 0              # data frame
        total_len = 1 + len(payload)
        msg.DLC = self._len_to_dlc(total_len) + 2

        # Fill data: first byte CMD, then payload
        msg.Data[0] = cmd & 0xFF
        for i in range(min(len(payload), 63)):   # 64 total, 1 already used by CMD
            msg.Data[1 + i] = payload[i]

        sent = libcan.CANFD_Transmit(DEFAULT_DEV, DEFAULT_CH, byref(msg), 1, timeout_ms)
        if sent != 1:
            cprint.warn(f"[CAN] TX failed, CMD=0x{cmd:02X}, ID=0x{node_id:03X}")
            return None
        if not expect_reply:
            return None

        # Read and find matching reply (same CMD; from our device unless accept_from_any=True)
        rx_array = CanFD_Msg_ARRAY(256)
        got = libcan.CANFD_Receive(DEFAULT_DEV, DEFAULT_CH, byref(rx_array.ADDR), rx_array.SIZE, timeout_ms)
        if got <= 0:
            cprint.warn(f"[CAN] RX timeout, CMD=0x{cmd:02X}, ID=0x{node_id:03X}")
            return None

        for i in range(got):
            rx = rx_array.STRUCT_ARRAY[i]
            # Must be FD, standard, and contain at least the CMD byte
            if self._dlc_len(rx.DLC) < 1 or rx.RemoteFlag != 0 or rx.ExternFlag != 0:
                continue
            if (rx.Data[0] & 0xFF) != (cmd & 0xFF):
                continue
            if (not accept_from_any) and (rx.ID != node_id):
                continue
            self._last_rx = rx
            return rx

        cprint.warn(f"[CAN] No matching reply for CMD=0x{cmd:02X}")
        return None

    # ---------- Common parsers ----------
    def _parse_u16_array(self, data: bytes, n: int) -> List[int]:
        out = []
        for i in range(0, min(len(data), 2*n), 2):
            out.append(int.from_bytes(data[i:i+2], 'little', signed=False))
        return out

    def _parse_i16_array(self, data: bytes, n: int) -> List[int]:
        out = []
        for i in range(0, min(len(data), 2*n), 2):
            out.append(int.from_bytes(data[i:i+2], 'little', signed=True))
        return out

    def _parse_status60(self, data: bytes) -> Dict[str, List[int]]:
        # 0..19 pos (u16), 20..39 speed (u16), 40..49 torque (u16), 50..59 fault (u8)
        if len(data) < 60:
            return {}
        pos = self._parse_u16_array(data[0:20], 10)
        spd = self._parse_u16_array(data[20:40], 10)
        tq  = self._parse_u16_array(data[40:50], 5) + self._parse_u16_array(data[50:60], 0)  # safeguard
        # For faults, treat as 10 bytes (each axis one byte)
        faults = list(data[50:60])
        return {"positions": pos, "speeds": spd, "torques": tq, "faults": faults}

    # ---------- Discovery & ID ----------
    def discover_node_ids(self, timeout_ms: int = 50) -> List[int]:
        """Broadcast a benign query (CMD 0x02: status) to paging ID 0x7FF and collect responders."""
        rx_ids = set()
        _ = self.send_cmd(0x02, b"", expect_reply=False, timeout_ms=timeout_ms, to_id=0x7FF, accept_from_any=True)
        # Drain any extra replies within the same window
        rx_array = CanFD_Msg_ARRAY(256)
        got = libcan.CANFD_Receive(DEFAULT_DEV, DEFAULT_CH, byref(rx_array.ADDR), rx_array.SIZE, timeout_ms)
        for i in range(max(0, got)):
            rx = rx_array.STRUCT_ARRAY[i]
            if self._dlc_len(rx.DLC) >= 1 and rx.Data[0] == 0x02:
                rx_ids.add(rx.ID & 0x7FF)
        ids = sorted(rx_ids)
        cprint.info(f"[CAN] Discovered node IDs: {ids}")
        return ids

    def set_node_id(self, new_id: int) -> bool:
        """CMD 0x04: 2 bytes new ID (1..0x7FF)."""
        if not (1 <= new_id < 0x800):
            cprint.warn("[CAN] new_id must be 1..0x7FE")
            return False
        payload = new_id.to_bytes(2, 'little', signed=False)
        rx = self.send_cmd(0x04, payload, expect_reply=True)
        ok = (rx is not None and self._dlc_len(rx.DLC) >= 2 and rx.Data[1] == 1)
        if ok:
            self.device_id = new_id & 0x7FF
        cprint.info(f"[CAN] Set node ID -> 0x{self.device_id:03X}: {'OK' if ok else 'FAIL'}")
        return ok

    # ---------- Enable / status ----------
    def set_enable(self, mode: int) -> bool:
        """CMD 0x01: 0x00 disable, 0x01 enable, 0x02 calibrate."""
        payload = bytes([mode & 0xFF])
        rx = self.send_cmd(0x01, payload)
        ok = (rx is not None and self._dlc_len(rx.DLC) >= 2 and rx.Data[1] == 1)
        cprint.info(f"[CAN] enable(mode={mode}): {'OK' if ok else 'FAIL'}")
        return ok

    def get_enable_status(self) -> Optional[int]:
        """CMD 0x02 → 1 byte: 0x00 disable, 0x01 enable, 0x02 calibrate."""
        rx = self.send_cmd(0x02, b"")
        if rx and self._dlc_len(rx.DLC) >= 2:
            return rx.Data[1]
        cprint.warn("[CAN] get_enable_status: no reply")
        return None

    # ---------- Positions ----------
    def set_current_position(self, axis: int, pos: int) -> bool:
        """CMD 0x03: d0=axis(1..10), d1-2=pos u16 0..4096; reply 1B success."""
        cprint.warn("[CAN] this function currently would always return a FAIL status.")
        payload = bytes([axis & 0xFF]) + int(pos).to_bytes(2, 'little', signed=False)
        rx = self.send_cmd(0x03, payload)
        ok = (rx is not None and self._dlc_len(rx.DLC) >= 2 and rx.Data[1] == 1)
        return ok

    def set_single_axis_position(self, axis: int, pos: int) -> Optional[Tuple[int, int]]:
        """CMD 0x06: send (axis,pos) → reply (axis, current_pos)."""
        payload = bytes([axis & 0xFF]) + int(pos).to_bytes(2, 'little', signed=False)
        rx = self.send_cmd(0x06, payload)
        if rx and self._dlc_len(rx.DLC) >= 4:
            axis_r = rx.Data[1]
            cur = int.from_bytes(bytes([rx.Data[2], rx.Data[3]]), 'little')
            return (axis_r, cur)
        cprint.warn("[CAN] set_single_axis_position: no/short reply")
        return None

    def read_single_axis_position(self, axis: int) -> Optional[int]:
        """CMD 0x07: send (axis) → reply like 0x06."""
        payload = bytes([axis & 0xFF])
        rx = self.send_cmd(0x07, payload)
        if rx and self._dlc_len(rx.DLC) >= 4:
            return int.from_bytes(bytes([rx.Data[2], rx.Data[3]]), 'little')
        cprint.warn("[CAN] read_single_axis_position: no/short reply")
        return None

    def set_all_positions(self, positions_10: List[int]) -> Optional[bool]:
        """CMD 0x08: 20B (10*u16). Reply 60B (pos, speed, torque, faults)."""
        if len(positions_10) != 10:
            raise ValueError("Need 10 positions")
        payload = b''.join(int(p).to_bytes(2, 'little', signed=False) for p in positions_10) + b'\x00\x00\x00\x00'
        # FIXME: append another 7 \x00 placeholders to receive message. In practice, sending payload less than 24 bytes won't receive any replies.
        payload = payload + b'\x00\x00\x00\x00\x00\x00\x00'
        rx = self.send_cmd(0x08, payload)
        # if rx and self._dlc_len(rx.DLC) >= 61:  # 1B CMD + 60B data
        #     data = bytes(rx.Data[1:61])
        #     return self._parse_status60(data)
        # FIXME: according to docs, would return 60-bytes data contaiing current qpos, speeds, torques, faults, but in practice, it would return only a 32-bytes containing sent qpos.
        if rx and self._dlc_len(rx.DLC) == 32:
            return True
        cprint.warn("[CAN] set_all_positions: no/short reply")
        return None

    def get_all_positions(self) -> Optional[List[int]]:
        """CMD 0x09 → 20B u16 positions."""
        rx = self.send_cmd(0x09, b"")
        if rx and self._dlc_len(rx.DLC) >= 21:
            return self._parse_u16_array(bytes(rx.Data[1:21]), 10)
        cprint.warn("[CAN] get_all_positions: no/short reply")
        return None

    # ---------- States: current / speed / temp / load ----------
    def get_all_currents(self) -> Optional[List[int]]:
        """CMD 0x0A: 20B; unit 0.01A."""
        rx = self.send_cmd(0x0A, b"")
        if rx and self._dlc_len(rx.DLC) >= 21:
            return self._parse_u16_array(bytes(rx.Data[1:21]), 10)
        return None

    def get_all_speeds(self) -> Optional[List[int]]:
        """CMD 0x0B: 20B speeds (u16)."""
        rx = self.send_cmd(0x0B, b"")
        if rx and self._dlc_len(rx.DLC) >= 21:
            return self._parse_u16_array(bytes(rx.Data[1:21]), 10)
        return None

    def get_all_temperatures(self) -> Optional[List[int]]:
        """CMD 0x0C: 10B int8 °C."""
        rx = self.send_cmd(0x0C, b"")
        if rx and self._dlc_len(rx.DLC) >= 11:
            raw = bytes(rx.Data[1:11])
            return [int.from_bytes(raw[i:i+1], 'little', signed=True) for i in range(10)]
        return None

    def get_all_loads(self) -> Optional[List[int]]:
        """CMD 0x1A: 20B load (0.1% duty)."""
        rx = self.send_cmd(0x1A, b"")
        if rx and self._dlc_len(rx.DLC) >= 21:
            return self._parse_u16_array(bytes(rx.Data[1:21]), 10)
        return None

    # ---------- Errors ----------
    def read_error_code(self) -> Optional[int]:
        """CMD 0x0D: 2B error code (0x0000=no error; see ranges in PDF)."""
        rx = self.send_cmd(0x0D, b"")
        if rx and self._dlc_len(rx.DLC) >= 3:
            return int.from_bytes(bytes([rx.Data[1], rx.Data[2]]), 'little', signed=False)
        cprint.warn("[CAN] read_error_code: no/short reply")
        return None

    def clear_error(self) -> bool:
        """CMD 0x0E: no payload; reply 1B success."""
        rx = self.send_cmd(0x0E, b"")
        ok = (rx is not None and self._dlc_len(rx.DLC) >= 2 and rx.Data[1] == 1)
        cprint.info(f"[CAN] clear_error: {'OK' if ok else 'FAIL'}")
        return ok

    # ---------- Ranges / tactile / plot ----------
    def read_position_ranges(self) -> Optional[List[int]]:
        """CMD 0x10: 20B (0.1° units)."""
        rx = self.send_cmd(0x10, b"")
        if rx and self._dlc_len(rx.DLC) >= 21:
            return self._parse_u16_array(bytes(rx.Data[1:21]), 10)
        return None

    def read_one_tactile(self, sensor_idx: int) -> Optional[bytes]:
        """CMD 0x11: idx → reply varies (17B finger / 26B palm/back)."""
        rx = self.send_cmd(0x11, bytes([sensor_idx & 0xFF]))
        if rx and self._dlc_len(rx.DLC) > 1:
            return bytes(rx.Data[1:self._dlc_len(rx.DLC)])
        return None

    def read_tactile_block1(self) -> Optional[bytes]:
        """CMD 0x12: thumb/index/middle (48B)."""
        rx = self.send_cmd(0x12, b"")
        if rx and self._dlc_len(rx.DLC) >= 49:
            return bytes(rx.Data[1:49])
        return None

    def read_tactile_block2(self) -> Optional[bytes]:
        """CMD 0x13: ring/little (32B)."""
        rx = self.send_cmd(0x13, b"")
        if rx and self._dlc_len(rx.DLC) >= 33:
            return bytes(rx.Data[1:33])
        return None

    def read_tactile_block3(self) -> Optional[bytes]:
        """CMD 0x14: palm/back (50B)."""
        rx = self.send_cmd(0x14, b"")
        if rx and self._dlc_len(rx.DLC) >= 51:
            return bytes(rx.Data[1:51])
        return None

    def set_run_mode(self, motor_id: int, mode: int) -> bool:
        """CMD 0x15: (id, mode) → 1B success."""
        rx = self.send_cmd(0x15, bytes([motor_id & 0xFF, mode & 0xFF]))
        return (rx is not None and self._dlc_len(rx.DLC) >= 2 and rx.Data[1] == 1)

    def set_plot_interval_ms(self, interval_ms: int) -> bool:
        """CMD 0x28: 2B ms → 1B success."""
        rx = self.send_cmd(0x28, int(interval_ms).to_bytes(2, 'little'))
        return (rx is not None and self._dlc_len(rx.DLC) >= 2 and rx.Data[1] == 1)

    def get_plot_data(self) -> Optional[List[Tuple[int,int,int]]]:
        """CMD 0x29: 60B; per axis: pos(2), speed(2), current(2)."""
        rx = self.send_cmd(0x29, b"")
        if rx and self._dlc_len(rx.DLC) >= 61:
            block = bytes(rx.Data[1:61])
            triples = []
            for k in range(10):
                off = k*6
                pos = int.from_bytes(block[off:off+2],  'little')
                spd = int.from_bytes(block[off+2:off+4],'little')
                cur = int.from_bytes(block[off+4:off+6],'little')
                triples.append((pos, spd, cur))
            return triples
        return None

    # ---------- Speed/torque limits ----------
    def set_all_speeds(self, speeds_10: List[int]) -> bool:
        """CMD 0x20: 20B int16 speeds (-4096..4096)."""
        if len(speeds_10) != 10:
            raise ValueError("Need 10 speeds")
        payload = b''.join(int(s).to_bytes(2, 'little', signed=True) for s in speeds_10)
        rx = self.send_cmd(0x20, payload)
        return (rx is not None and self._dlc_len(rx.DLC) >= 2 and rx.Data[1] == 1)

    def set_overload_torque(self, motor_id: int, torque_0p1pct: int) -> bool:
        """CMD 0x21: (id, torque 0..1000 → 0..100.0%). Motors 4/7/9."""
        rx = self.send_cmd(0x21, bytes([motor_id & 0xFF, torque_0p1pct & 0xFF]))
        return (rx is not None and self._dlc_len(rx.DLC) >= 2 and rx.Data[1] == 1)

    def set_overload_time(self, motor_id: int, time_0p01s: int) -> bool:
        """CMD 0x22: (id, time in 0.01s)."""
        rx = self.send_cmd(0x22, bytes([motor_id & 0xFF, time_0p01s & 0xFF]))
        return (rx is not None and self._dlc_len(rx.DLC) >= 2 and rx.Data[1] == 1)

    def set_protection_torque(self, motor_id: int, pct_1: int) -> bool:
        """CMD 0x23: (id, torque in 1% units 0..100)."""
        rx = self.send_cmd(0x23, bytes([motor_id & 0xFF, pct_1 & 0xFF]))
        return (rx is not None and self._dlc_len(rx.DLC) >= 2 and rx.Data[1] == 1)

    def set_min_start_force(self, axis: int, value: int, *, scs2304: bool = False) -> bool:
        """CMD 0x24: HLS3606 d1=u8; SCS2304 d1-2=u16."""
        if scs2304:
            payload = bytes([axis & 0xFF]) + int(value).to_bytes(2, 'little', signed=False)
        else:
            payload = bytes([axis & 0xFF, value & 0xFF])
        rx = self.send_cmd(0x24, payload)
        return (rx is not None and self._dlc_len(rx.DLC) >= 2 and rx.Data[1] == 1)

    def set_max_torque(self, axis: int, value_0p1pct: int) -> bool:
        """CMD 0x25: (axis, u16 0..1000 → 0..100.0%)."""
        payload = bytes([axis & 0xFF]) + int(value_0p1pct).to_bytes(2, 'little')
        rx = self.send_cmd(0x25, payload)
        return (rx is not None and self._dlc_len(rx.DLC) >= 2 and rx.Data[1] == 1)

    # ---------- Misc ----------
    def read_sensor_ids(self) -> Optional[List[int]]:
        """CMD 0x27: 7B int8 sensor IDs."""
        rx = self.send_cmd(0x27, b"")
        if rx and self._dlc_len(rx.DLC) >= 8:
            raw = bytes(rx.Data[1:8])
            return [int.from_bytes(raw[i:i+1], 'little', signed=True) for i in range(7)]
        return None

    def set_hand_lr(self, left: bool) -> bool:
        """CMD 0x31: 0 right (default), 1 left."""
        rx = self.send_cmd(0x31, bytes([1 if left else 0]))
        return (rx is not None and self._dlc_len(rx.DLC) >= 2 and rx.Data[1] == 1)

    def set_control_source(self, src: int) -> Optional[int]:
        """CMD 0x80: 0 robot (default), 1 HMI. Echoes payload on success."""
        rx = self.send_cmd(0x80, bytes([src & 0xFF]))
        if rx and self._dlc_len(rx.DLC) >= 2:
            return rx.Data[1]
        return None

    def get_control_source(self) -> Optional[int]:
        """CMD 0x81: return source (0 robot, 1 HMI)."""
        rx = self.send_cmd(0x81, b"")
        if rx and self._dlc_len(rx.DLC) >= 2:
            return rx.Data[1]
        return None

    def set_product_serial(self, raw19: bytes) -> bool:
        """CMD 0xC1: 19B serial fields."""
        if len(raw19) != 19:
            raise ValueError("Need 19 bytes")
        rx = self.send_cmd(0xC1, raw19)
        return (rx is not None and self._dlc_len(rx.DLC) >= 2 and rx.Data[1] == 1)

    def read_product_serial(self) -> Optional[bytes]:
        """CMD 0xC2: 19B serial fields."""
        rx = self.send_cmd(0xC2, b"")
        if rx and self._dlc_len(rx.DLC) >= 20:
            return bytes(rx.Data[1:20])
        return None

    def query_model_versions(self) -> Optional[bytes]:
        """CMD 0xCD: returns model & SW/HW versions (size per PDF)."""
        rx = self.send_cmd(0xCD, b"")
        if rx and self._dlc_len(rx.DLC) > 1:
            return bytes(rx.Data[1:self._dlc_len(rx.DLC)])
        return None

    # ---------- Close ----------
    def close(self):
        try:
            libcan.CAN_CloseDevice(DEFAULT_DEV, DEFAULT_CH)
        except Exception:
            pass

def main():
    # Create controller
    controller = DexterousHandController(device_id=0x08, active_dof=10)

    # Test discovery and ID management
    controller.discover_node_ids()
    # controller.set_node_id(0x02)
    
    # Test enable/status functions
    controller.set_enable(1)  # Enable
    status = controller.get_enable_status()
    print(f"Enable status: {status}")
    
    # Test position functions
    controller.set_current_position(5, 1024)
    result = controller.set_single_axis_position(1, 4096)
    print(f"Single axis result: {result}")
    
    pos = controller.read_single_axis_position(1)
    print(f"Single axis position: {pos}")
    
    # Test all positions
    positions = [0] * 10
    result = controller.set_all_positions(positions)
    print(f"Set all positions result: {result}")
    
    all_pos = controller.get_all_positions()
    print(f"All positions: {all_pos}")
    
    # Test state readings
    currents = controller.get_all_currents()
    print(f"All currents: {currents}")
    
    speeds = controller.get_all_speeds()
    print(f"All speeds: {speeds}")
    
    temperatures = controller.get_all_temperatures()
    print(f"All temperatures: {temperatures}")
    
    loads = controller.get_all_loads()
    print(f"All loads: {loads}")
    
    # Test error functions
    error_code = controller.read_error_code()
    print(f"Error code: {error_code}")
    
    controller.clear_error()
    
    # Test range and tactile functions
    ranges = controller.read_position_ranges()
    print(f"Position ranges: {ranges}")
    
    tactile = controller.read_one_tactile(0)
    print(f"Tactile sensor 0: {tactile}")
    
    tactile_block1 = controller.read_tactile_block1()
    print(f"Tactile block 1 length: {len(tactile_block1) if tactile_block1 else 0}")
    
    tactile_block2 = controller.read_tactile_block2()
    print(f"Tactile block 2 length: {len(tactile_block2) if tactile_block2 else 0}")
    
    tactile_block3 = controller.read_tactile_block3()
    print(f"Tactile block 3 length: {len(tactile_block3) if tactile_block3 else 0}")
    
    # Test run mode and plot functions
    controller.set_run_mode(1, 0)
    controller.set_plot_interval_ms(100)
    
    plot_data = controller.get_plot_data()
    print(f"Plot data length: {len(plot_data) if plot_data else 0}")
    
    # Test speed and torque functions
    speeds = [100, -100, 200, -200, 0, 0, 0, 0, 0, 0]
    controller.set_all_speeds(speeds)
    
    controller.set_overload_torque(4, 500)  # 50% torque
    controller.set_overload_time(4, 100)    # 1 second
    controller.set_protection_torque(4, 80) # 80% protection
    controller.set_min_start_force(1, 50)
    controller.set_max_torque(1, 800)       # 80% max torque
    
    # Test misc functions
    sensor_ids = controller.read_sensor_ids()
    print(f"Sensor IDs: {sensor_ids}")
    
    controller.set_hand_lr(False)  # Right hand
    
    controller.set_control_source(0)  # Robot control
    control_src = controller.get_control_source()
    print(f"Control source: {control_src}")
    
    # Test serial and version functions
    serial_data = controller.read_product_serial()
    print(f"Product serial length: {len(serial_data) if serial_data else 0}")
    
    versions = controller.query_model_versions()
    print(f"Model versions length: {len(versions) if versions else 0}")
    
    # Close the controller
    controller.close()
    libcan.CAN_CloseDevice(DEFAULT_DEV, DEFAULT_CH)

if __name__ == "__main__":
    main()
