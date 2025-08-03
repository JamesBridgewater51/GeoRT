# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from geort.mocap.replay_mocap import ReplayMocap
from geort.env.hand import HandKinematicModel
from geort.env.hand import SkeletonViewer
from geort import load_model, get_config
import argparse
from scipy.spatial.transform import Rotation as R
import numpy as np

ENABLE_REAL_ROBOT = False

if ENABLE_REAL_ROBOT:
    from ..control.o12_hand.can_py.can_controller import OmniHandDriver
    from ..control.o12_hand.py_sdk.src.constants import GestureID
    driver = OmniHandDriver(is_right_hand=True, render=False, urdf_path="assets/o12_hand_description-main/urdf/o12_t1_right.urdf", render_freq=60, mode='monitor')
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hand', type=str, default='allegro')

    parser.add_argument('-allegro_ckpt_tag', type=str, default='')

    parser.add_argument('-ckpt_tag', type=str, default='alex')
    parser.add_argument('-human_data_path', type=str, default='human')

    args = parser.parse_args()

    # GeoRT Model.
    model = load_model(args.ckpt_tag)
    
    # Motion Capture.
    mocap = ReplayMocap(args.human_data_path)
    print("[ReplayMocap] Initialized with human data from:", args.human_data_path)
    print("[ReplayMocap] Total frames:", mocap.T)
    
    # Robot Simulation.
    config = get_config(args.hand)
    hand = HandKinematicModel.build_from_config(config, trans=[0, 0, 0.35], quat=[0.695, 0, -0.718, 0], render=True)

    # Enable allegro only when the tag is provided and the hand is not allegro (so we can use allegro as a reference).
    enable_allegro = args.allegro_ckpt_tag != '' and args.hand != 'allegro'
    if enable_allegro:
        allegro_config = get_config("allegro_right")
        allegro_hand = HandKinematicModel.build_from_config(allegro_config, trans=[-0.1,-0.2,0.35], quat=[0.695, 0, -0.718, 0], render=True)
        allegro_model = load_model(args.allegro_ckpt_tag)

    skeleton_viewer = SkeletonViewer(scene=hand.scene, renderer=hand.renderer)
    viewer_env = hand.get_viewer_env()

    # Run!
    retargeted_qpos = []
    cnt = 0
    while True:
        cnt += 1
        if cnt > mocap.T:
            break
        print(f"[ReplayMocap] Frame {cnt}/{mocap.T} ...")
        viewer_env.update()
        # for i in range(3):
        #     viewer_env.update()

        result = mocap.get()

        if result['status'] == 'recording' and result["result"] is not None:
            qpos = model.forward(result["result"])

            hand.set_qpos_target(qpos)
            retargeted_qpos.append(qpos)

            if enable_allegro:
                allegro_qpos = allegro_model.forward(result["result"])
                allegro_hand.set_qpos_target(allegro_qpos)

            # NOTE: The coordinate frame in Sapien is: x(forward), y(left), z(upward)
            # Transform canonicalized hand joint positions to Sapien conventions.
            # First, rotate against y-axis by -90 degrees.
            R_ = R.from_euler('y', -90, degrees=True)
            pos = R_.apply(result['result'])
            # Then, translate the rotated joint positions to adapt to robotic hand's coordinate.
            t = hand.trans.copy()
            t[1] += 0.2
            pos += t

            skeleton_viewer.update(joint_positions=pos,)

            # real robot control
            if ENABLE_REAL_ROBOT:
                motor_ticks = driver.kin_ctrl.convert_joint_to_actuator(joint_pos=np.asarray(qpos), is_active_only=False)
                driver.set_target_positions(motor_ticks=motor_ticks)


        if result['status'] == 'quit':
            break 
    
    retargeted_qpos = np.array(retargeted_qpos)
    np.save(f"retargeted_qpos_{args.ckpt_tag}.npy", retargeted_qpos)


if __name__ == '__main__':
    main()
