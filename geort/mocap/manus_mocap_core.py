# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Python ROS2-to-ZMQ Bridge (manus_mocap_core.py)

Purpose: This script acts as a connector and processor. It decouples the main application from needing a full ROS2 environment.

Functionality:

    It runs as a ROS2 node (class Manus(Node)) and subscribes to the topics published by the C++ client.

    Forward Kinematics (FK): It contains a ManusForwardKinematicsSolver and a hardcoded model of the human hand's bone vectors (self.pos). It uses the incoming joint quaternions and these vectors to calculate the 3D Cartesian positions of the 21 keypoints.

    Canonicalization: Just like the MediaPipe processor, it has a hand_to_canonical function that transforms the calculated 3D keypoints into a wrist-local coordinate frame.

    Broadcast: After processing, it broadcasts the final (21, 3) NumPy array over a lightweight, high-performance ZMQ PUB socket on localhost.
"""

# Meta Internal Manus. 
import math
import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import zmq 
from geort.control.o12_hand.py_sdk.src.constants import ManushandJointID


def hand_to_canonical(hand_point, is_right_hand = True):
    z_axis = hand_point[ManushandJointID.FingerMiddleMetacarpal] - hand_point[ManushandJointID.Wrist]
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis_aux = hand_point[ManushandJointID.FingerIndexMetacarpal] - hand_point[ManushandJointID.FingerRingMetacarpal]
    y_axis_aux = y_axis_aux / np.linalg.norm(y_axis_aux)
    if not is_right_hand:
        y_axis_aux = -y_axis_aux

    x_axis = np.cross(y_axis_aux, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    rotation_base = np.array([x_axis, y_axis, z_axis]).transpose()
    tranlation_base = hand_point[ManushandJointID.Wrist]

    transform = np.eye(4)
    transform[:3, :3] = rotation_base
    transform[:3, 3] = tranlation_base
    
    transform_inv = np.linalg.inv(transform)
    hand_point = np.array(hand_point)
    hand_point = np.concatenate((np.array(hand_point), np.ones((21, 1))), axis=-1)
    hand_point = hand_point @ transform_inv.transpose()
    return hand_point[:, :3]



class ManusForwardKinematicsSolver:
    def __init__(self):
        return 

    def make_transformation_matrix(self, pos, quat):
        from scipy.spatial.transform import Rotation as R
        out = np.eye(4)
        out[:3, 3] = pos
        out[:3, :3] = R.from_quat(quat).as_matrix()
        return out

    def solve_keypoints(self, positions, orientation):
        thumb_chain = [0, 1, 2, 3, 4]
        index_chain = [0, 5, 6, 7, 8]
        middle_chain = [0, 9, 10, 11, 12]
        ring_chain = [0, 13, 14, 15, 16]
        pinky_chain = [0, 17, 18, 19, 20]
        all_chains = [thumb_chain, index_chain, middle_chain, ring_chain, pinky_chain]

        all_keypoints = {}
        
        for chain in all_chains:
            current_transformation_to_world = np.eye(4)

            for idx in chain:
                pos = np.array(positions[idx])
                transformation = self.make_transformation_matrix(pos, orientation[idx])

                last_position = np.array(current_transformation_to_world[:3, 3])
                current_transformation_to_world = current_transformation_to_world @ transformation
                position = current_transformation_to_world[:3, 3]

                if idx not in all_keypoints:
                    all_keypoints[idx] = position
        return all_keypoints

class Manus(Node):
    def __init__(self, is_right_hand = True):
        super().__init__("manus_visualizer")
        self.x_axis = []
        self.y_axis = []
        self.z_axis = []
        self.pos = None
        self.quat = None

        # Damn, this part is manually measured human finger link vectors.
        self.is_right_hand = is_right_hand
        if self.is_right_hand:
            # NOTE: this is the old relative-to-parent offset in SDKClient.cpp from ManusCoreSdk3.0.1
            # self.pos = np.array([
            #     [0.0, 0.0, 0.0],  # Wrist
            #     [0.024950, 0.000000, 0.025320],  # Thumb CMC joint
            #     [0.000000, 0.000000, 0.032742],  # Thumb MCP joint
            #     [0.000000, 0.000000, 0.028739],  # Thumb IP joint
            #     [0.000000, 0.000000, 0.028739],  # Thumb Tip joint

            #     [0.011181, 0.000000, 0.052904],  # Index MCP joint
            #     [0.000000, 0.000000, 0.038257],  # Index PIP joint
            #     [0.000000, 0.000000, 0.020884],  # Index DIP joint
            #     [0.000000, 0.000000, 0.018759],  # Index Tip joint

            #     [0.000000, 0.000000, 0.051287],  # Middle MCP joint
            #     [0.000000, 0.000000, 0.041861],  # Middle PIP joint
            #     [0.000000, 0.000000, 0.024766],  # Middle DIP joint
            #     [0.000000, 0.000000, 0.019683],  # Middle Tip joint

            #     [-0.011274, 0.000000, 0.049802],  # Ring MCP joint
            #     [0.000000, 0.000000, 0.039736],  # Ring PIP joint
            #     [0.000000, 0.000000, 0.023564],  # Ring DIP joint
            #     [0.000000, 0.000000, 0.019868],  # Ring Tip joint

            #     [-0.020145, 0.000000, 0.047309],  # Pinky MCP joint
            #     [0.000000, 0.000000, 0.033175],  # Pinky PIP joint
            #     [0.000000, 0.000000, 0.018020],  # Pinky DIP joint
            #     [0.000000, 0.000000, 0.019129],  # Pinky Tip joint
            # ])
            self.pos = np.array([
                [0.0, 0.0, 0.0],  # Wrist
                [0.025000, 0.000000, 0.005000],  # Thumb CMC joint
                [0.000000, 0.000000, 0.039000],  # Thumb MCP joint
                [0.000000, 0.000000, 0.033000],  # Thumb IP joint
                [0.000000, 0.000000, 0.021000],  # Thumb Tip joint

                [0.017000, 0.000000, 0.087000],  # Index MCP joint
                [0.000000, 0.000000, 0.026000],  # Index PIP joint
                [0.000000, 0.000000, 0.022000],  # Index DIP joint
                [0.000000, 0.000000, 0.020000],  # Index Tip joint

                [0.000000, 0.000000, 0.092000],  # Middle MCP joint
                [0.000000, 0.000000, 0.026000],  # Middle PIP joint
                [0.000000, 0.000000, 0.026000],  # Middle DIP joint
                [0.000000, 0.000000, 0.022000],  # Middle Tip joint

                [-0.017000, 0.000000, 0.084000],  # Ring MCP joint
                [0.000000, 0.000000, 0.021000],  # Ring PIP joint
                [0.000000, 0.000000, 0.021000],  # Ring DIP joint
                [0.000000, 0.000000, 0.020000],  # Ring Tip joint

                [-0.034000, 0.000000, 0.072000],  # Pinky MCP joint
                [0.000000, 0.000000, 0.021000],  # Pinky PIP joint
                [0.000000, 0.000000, 0.021000],  # Pinky DIP joint
                [0.000000, 0.000000, 0.020000],  # Pinky Tip joint
            ])
        else:
            # NOTE: this is the old relative-to-parent offset in SDKClient.cpp from ManusCoreSdk3.0.1
            # self.pos = np.array([
            #     [0.0, 0.0, 0.0],  # Wrist
            #     [-0.024950, 0.000000, 0.025320],  # Thumb CMC joint
            #     [0.000000, 0.000000, 0.032742],  # Thumb MCP joint
            #     [0.000000, 0.000000, 0.028739],  # Thumb IP joint
            #     [0.000000, 0.000000, 0.028739],  # Thumb Tip joint

            #     [-0.011181, 0.000000, 0.052904],  # Index MCP joint
            #     [0.000000, 0.000000, 0.038257],  # Index PIP joint
            #     [0.000000, 0.000000, 0.020884],  # Index DIP joint
            #     [0.000000, 0.000000, 0.018759],  # Index Tip joint

            #     [0.000000, 0.000000, 0.051287],  # Middle MCP joint
            #     [0.000000, 0.000000, 0.041861],  # Middle PIP joint
            #     [0.000000, 0.000000, 0.024766],  # Middle DIP joint
            #     [0.000000, 0.000000, 0.019683],  # Middle Tip joint

            #     [0.011274, 0.000000, 0.049802],  # Ring MCP joint
            #     [0.000000, 0.000000, 0.039736],  # Ring PIP joint
            #     [0.000000, 0.000000, 0.023564],  # Ring DIP joint
            #     [0.000000, 0.000000, 0.019868],  # Ring Tip joint

            #     [0.020145, 0.000000, 0.047309],  # Pinky MCP joint
            #     [0.000000, 0.000000, 0.033175],  # Pinky PIP joint
            #     [0.000000, 0.000000, 0.018020],  # Pinky DIP joint
            #     [0.000000, 0.000000, 0.019129],  # Pinky Tip joint
            # ])
            self.pos = np.array([
                [0.0, 0.0, 0.0],  # Wrist
                [-0.025000, 0.000000, 0.005000],  # Thumb CMC joint
                [0.000000, 0.000000, 0.039000],  # Thumb MCP joint
                [0.000000, 0.000000, 0.033000],  # Thumb IP joint
                [0.000000, 0.000000, 0.021000],  # Thumb Tip joint

                [-0.017000, 0.000000, 0.087000],  # Index MCP joint
                [0.000000, 0.000000, 0.026000],  # Index PIP joint
                [0.000000, 0.000000, 0.022000],  # Index DIP joint
                [0.000000, 0.000000, 0.020000],  # Index Tip joint

                [0.000000, 0.000000, 0.092000],  # Middle MCP joint
                [0.000000, 0.000000, 0.026000],  # Middle PIP joint
                [0.000000, 0.000000, 0.026000],  # Middle DIP joint
                [0.000000, 0.000000, 0.022000],  # Middle Tip joint

                [0.017000, 0.000000, 0.084000],  # Ring MCP joint
                [0.000000, 0.000000, 0.021000],  # Ring PIP joint
                [0.000000, 0.000000, 0.021000],  # Ring DIP joint
                [0.000000, 0.000000, 0.020000],  # Ring Tip joint

                [0.034000, 0.000000, 0.072000],  # Pinky MCP joint
                [0.000000, 0.000000, 0.021000],  # Pinky PIP joint
                [0.000000, 0.000000, 0.021000],  # Pinky DIP joint
                [0.000000, 0.000000, 0.020000],  # Pinky Tip joint
            ])


        self.manus_x_subscription = self.create_subscription(
            Float32MultiArray, "/x_manus_rotations", self.listener_callback_x, 10
        )

        self.manus_y_subscription = self.create_subscription(
            Float32MultiArray, "/y_manus_rotations", self.listener_callback_y, 10
        )

        self.manus_z_subscription = self.create_subscription(
            Float32MultiArray, "/z_manus_rotations", self.listener_callback_z, 10
        )

        self.manus_quats_subscription = self.create_subscription(
            Float32MultiArray, "/manus_quats", self.listener_callback_quat, 10
        )

        # Broadcast on localhost
        self.port = 8765
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")
        self.socket.setsockopt(zmq.SNDHWM, 0)

    def listener_callback_x(self, msg):
        self.x_msg = list(msg.data)
        self.x_axis = [math.degrees(r) for r in self.x_msg]

    def listener_callback_y(self, msg):
        self.y_msg = list(msg.data)
        self.y_axis = [math.degrees(r) for r in self.y_msg]

    def listener_callback_z(self, msg):
        self.z_msg = list(msg.data)
        self.z_axis = [math.degrees(r) for r in self.z_msg]

    def listener_callback_quat(self, msg):
        self.quat = np.array(list(msg.data)).reshape(21, 4)
        # construct all identity xyzw quat
        # self.quat = np.array([[0, 0, 0, 1]] * 21)


    def run(self):
        count = 0
        kinematics_solver = ManusForwardKinematicsSolver()
        while rclpy.ok():
            count = count + 1
            if self.pos is None or self.quat is None or len(self.x_axis) == 0:
                continue

            keypoints = kinematics_solver.solve_keypoints(self.pos, self.quat)
            keypoints =  np.array([keypoints[i] for i in range(21)])
            keypoints = hand_to_canonical(keypoints, is_right_hand=self.is_right_hand).astype(np.float32)

            self.socket.send(keypoints.tobytes())
            print("Broadcasting Manus Reading", count, keypoints.shape)
     
def main(args=None):
    rclpy.init()
    manus_node = Manus(is_right_hand=args.right_hand)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(manus_node)
    run_thread = threading.Thread(target=manus_node.run)
    run_thread.start()
    executor.spin()
    run_thread.join()  


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--right_hand", action="store_true")
    args = parser.parse_args()
    main(args)
