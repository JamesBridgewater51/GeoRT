# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sapien
from sapien.utils import Viewer
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from geort.utils.config_utils import get_config, save_json
from geort.utils.hand_utils import get_entity_by_name, get_active_joints, get_active_joint_indices
from datetime import datetime
from tqdm import tqdm 
import os
from pathlib import Path 
import math
from scipy.spatial.transform import Rotation as R

SAPIEN_VERSION = sapien.__version__

# sapien 3.x changed tremendously its APIs.
if SAPIEN_VERSION.startswith("3."):
    SAPIEN_USE_OLD_API = False
else:
    SAPIEN_USE_OLD_API = True
    sapien = sapien.core

if not SAPIEN_USE_OLD_API:
    SCENE = sapien.Scene()

class HandKinematicModel:
    def __init__(self, 
                 scene=None, 
                 render=False, 
                 hand_urdf='', 
                 n_hand_dof=16, 
                 trans=[0, 0, 0.35],
                 quat=[0.695, 0, -0.718, 0],
                 base_link='base_link', 
                 joint_names=[],
                 # Ideally, these two guys (PD controller args) shouldn't be here. 
                 # -- There should be a controller class. I leave them here for code simplicity (maybe truth: or because I am lazy).
                 # If you see your hand model doing something weird (in the simulation viewer below), tune them.
                 kp=400.0, 
                 kd=10):
        
        self.engine = None
        if scene is None and SAPIEN_USE_OLD_API:
            engine = sapien.Engine()
            
            if render:
                renderer = sapien.VulkanRenderer()  
                engine.set_renderer(renderer)
                print("Enable Render Mode.")
            else:
                renderer = None 
            scene_config = sapien.SceneConfig()
            scene_config.default_dynamic_friction = 1.0
            scene_config.default_static_friction = 1.0
            scene_config.default_restitution = 0.00
            scene_config.contact_offset = 0.02
            scene_config.enable_pcm = False
            scene_config.solver_iterations = 25
            scene_config.solver_velocity_iterations = 1
            scene = engine.create_scene(scene_config)  
            self.engine = engine 
        elif scene is None and not SAPIEN_USE_OLD_API:
            scene = SCENE
            renderer = None

        self.scene = scene 
        self.renderer = renderer 

        loader = self.scene.create_urdf_loader()
        self.hand = loader.load(hand_urdf)
        self.hand.set_root_pose(sapien.Pose(trans, quat))

        self.pmodel = self.hand.create_pinocchio_model()
        self.trans, self.quat = trans, quat

        # Setup hand base link.
        self.base_link = get_entity_by_name(self.hand.get_links(), base_link)
        self.base_link_idx = self.hand.get_links().index(self.base_link)

        # Setup hand dofs.
        self.all_joints = get_active_joints(self.hand, joint_names)
        all_limits = [joint.get_limits() for joint in self.all_joints]

        self.joint_names = joint_names
        self.user_idx_to_sim_idx = get_active_joint_indices(self.hand, joint_names)
        print("User-to-Sim Joint", self.user_idx_to_sim_idx)
        self.sim_idx_to_user_idx = [self.user_idx_to_sim_idx.index(i) for i in range(len(self.user_idx_to_sim_idx))]
        print("Sim-to-User Joint", self.sim_idx_to_user_idx)

        self.joint_lower_limit = np.array([l[0][0] for l in all_limits])  # this is in user specified "joint_name" order
        self.joint_upper_limit = np.array([l[0][1] for l in all_limits])  # this is in user specified "joint_name" order
        print(self.joint_lower_limit, self.joint_upper_limit)

        init_qpos = self.convert_user_order_to_sim_order((self.joint_lower_limit + self.joint_upper_limit) / 2)
        self.hand.set_qpos(init_qpos)
        self.hand.set_qvel(0.0 * init_qpos)
        self.qpos_target = init_qpos

        for i, joint in enumerate(self.all_joints):
            print(i, self.joint_names[i], joint, self.joint_lower_limit[i], self.joint_upper_limit[i])
            joint.set_drive_property(kp, kd, force_limit=10)

    def __del__(self):
        del self.engine 
        del self.scene 

    def get_n_dof(self):
        '''
            number of dof.
        '''
        return len(self.joint_lower_limit)

    def get_joint_limit(self):
        '''
            Get the hand joint limit.
        '''
        return self.joint_lower_limit, self.joint_upper_limit

    def initialize_keypoint(self, keypoint_link_names, keypoint_offsets):
        '''
            Setup keypoints to track.
        '''
        keypoint_links = [get_entity_by_name(self.hand.get_links(), link) for link in keypoint_link_names]
        print(keypoint_links)

        keypoint_links_id_dict = {link_name: (self.hand.get_links().index(keypoint_links[i]), i) for i, link_name in enumerate(keypoint_link_names)}
        self.keypoint_links = keypoint_links
        self.keypoint_links_id_dict = keypoint_links_id_dict
        self.keypoint_offsets = np.array(keypoint_offsets)

    def convert_user_order_to_sim_order(self, qpos):
        return qpos[self.sim_idx_to_user_idx]

    def keypoint_from_qpos(self, qpos, ret_vec=False):
        '''
            Get keypoints from hand qpos. qpos is specified using the user order.
        '''
        qpos = self.convert_user_order_to_sim_order(qpos)
        self.pmodel.compute_forward_kinematics(qpos)
        base_pose = self.pmodel.get_link_pose(self.base_link_idx)

        result = {} 
        vec_result = []

        for m, (link_idx, i) in self.keypoint_links_id_dict.items():
            pose = self.pmodel.get_link_pose(link_idx)
            new_pose = sapien.Pose(p=pose.p + (pose.to_transformation_matrix()[:3, :3] @ self.keypoint_offsets[i].reshape(3, 1)).reshape(-1), q=pose.q)

            x = (base_pose.inv() * new_pose).p # convert to hand base frame.
            vec_result.append(x)
            result[m] = x

        if ret_vec:
            return np.array(vec_result)
        return result

    @staticmethod
    def build_from_config(config, **kwargs):
        '''
            Build a kinematic model from user config.
        '''
        render = kwargs.get("render", False)
        urdf_path = config["urdf_path"]
        n_hand_dof = len(config["joint_order"])
        base_link = config["base_link"]
        joint_order = config["joint_order"]

        model = HandKinematicModel(hand_urdf=urdf_path, render=render, n_hand_dof=n_hand_dof,base_link=base_link, joint_names=joint_order)
        return model 

    def get_viewer_env(self):
        return HandViewerEnv(self)

    def get_scene(self):
        return self.scene

    def get_renderer(self):
        return self.renderer

    def set_qpos_target(self, qpos):
        '''
            This function is only used during visualization
        '''
        qpos = np.clip(qpos, self.joint_lower_limit + 1e-3, self.joint_upper_limit - 1e-3)
        qpos = self.convert_user_order_to_sim_order(qpos)
        self.qpos_target = qpos 

        for i in range(len(qpos)):
            self.all_joints[i].set_drive_target(self.qpos_target[i])

class SkeletonViewer:
    """
    Manages the creation and live updates of a simple skeleton visualization in a Sapien scene.
    """
    def __init__(self, scene, renderer, joint_radius=0.005, bone_radius=0.003):
        self.scene = scene
        self.renderer = renderer
        assert (self.renderer is not None and SAPIEN_USE_OLD_API) or not SAPIEN_USE_OLD_API, "Renderer must be set for SkeletonViewer."
        import json
        skeleton_config = json.loads(
            """
        {
            "num_joints": 21,
        "bones": [
            [0, 1], [0, 5], [0, 9], [0, 13], [0, 17], 
            [1, 2], [2, 3], [3, 4],
            [5, 6], [6, 7], [7, 8],
            [9, 10], [10, 11], [11, 12],
            [13, 14], [14, 15], [15, 16],
            [17, 18], [18, 19], [19, 20]
        ]
        }
                                    """
                                    )
        self.config = skeleton_config
        self.joint_radius = joint_radius
        self.bone_radius = bone_radius
        
        self.joint_actors = []
        self.bone_actors = []
        
        self._create_actors()

    def _create_actors(self):
        if SAPIEN_USE_OLD_API:
            sphere_material = self.renderer.create_material()
            sphere_material.set_base_color([1, 0, 0, 1])  # Red for joints
            cylinder_material = self.renderer.create_material()
            cylinder_material.set_base_color([0, 0, 1, 1])

        """ Creates sphere and cylinder actors for joints and bones. """
        # Create an actor for each joint
        builder = self.scene.create_actor_builder()
        for i in range(self.config["num_joints"]):
            material = sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]) if not SAPIEN_USE_OLD_API else sphere_material
            builder.add_sphere_visual(radius=self.joint_radius, material=material) # Red joints
            joint_actor = builder.build_static(name=f"skel_joint_{i}")
            self.joint_actors.append(joint_actor)
        
        # Create an actor for each bone
        builder = self.scene.create_actor_builder()
        for i in range(len(self.config["bones"])):
            material = sapien.render.RenderMaterial(base_color=[0, 0, 1, 1]) if not SAPIEN_USE_OLD_API else cylinder_material
            builder.add_capsule_visual(radius=self.bone_radius, half_length=0.01, material=material) # Blue bones
            bone_actor = builder.build_static(name=f"skel_bone_{i}")
            self.bone_actors.append(bone_actor)

    def _calculate_bone_pose(self, p_start, p_end):
        """Calculates the pose for a cylinder to connect two points."""
        # Position is the midpoint
        midpoint = (p_start + p_end) / 2
        
        # Calculate orientation
        direction = p_end - p_start
        height = np.linalg.norm(direction)
        if height < 1e-6: # Avoid division by zero
            return sapien.Pose(p=midpoint), 0
            
        direction /= height
        
        # Cylinder's default axis is Y, get rotation from (0,1,0) to direction
        y_axis = np.array([1, 0, 0])
        rot_axis = np.cross(y_axis, direction)
        rot_angle = np.arccos(np.dot(y_axis, direction))
        
        if np.linalg.norm(rot_axis) < 1e-6:
             # Vectors are parallel or anti-parallel
             q = [1, 0, 0, 0] if np.allclose(direction, y_axis) else [0, 1, 0, 0]
        else:
            q = R.from_rotvec(rot_angle * (rot_axis / np.linalg.norm(rot_axis))).as_quat()
            q = np.roll(q, 1)  # Convert from (x, y, z, w) to (w, x, y, z) for Sapien

        return sapien.Pose(p=midpoint, q=q), height

    def update(self, joint_positions=None):
        """Updates the poses of all skeleton actors based on the model's current qpos."""
        # Update joint spheres
        for i, actor in enumerate(self.joint_actors):
            actor.set_pose(sapien.Pose(p=joint_positions[i]))
            
        # Update bone cylinders
        for i, bone_indices in enumerate(self.config["bones"]):
            start_idx, end_idx = bone_indices
            p_start = joint_positions[start_idx]
            p_end = joint_positions[end_idx]
            
            pose, height = self._calculate_bone_pose(p_start, p_end)
            actor = self.bone_actors[i]
            actor.set_pose(pose)

            # Update cylinder scale to match bone length
            # Cylinder's visual body is tied to its first collision shape
            # FIXME: the  `get_visual_bodies` method is deprecated since Sapien 3.x, and new version currently does not provide APIs to `set_scale`.
            # visual = actor.get_visual_bodies()[0]
            # Half-length is used for scale
            # visual.set_scale([1, height / (2 * visual.half_length), 1])

class HandViewerEnv:
    def __init__(self, model):
        scene = model.get_scene()
        scene.set_timestep(1 / 100.0) 
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        scene.add_ground(altitude=0) 

        viewer = Viewer(model.get_renderer())
        viewer.set_scene(scene) 
        if SAPIEN_USE_OLD_API:
            viewer.window.set_camera_position([0.0457491, -0.0509193, 0.455975])
            viewer.window.set_camera_rotation([0.8716827, 0.3260138, 0.12817779, 0.3427167])
        else:
            viewer.window.set_camera_pose(sapien.Pose([0.18299, -0.0560369, 0.661718], [0.253402, -0.405266, 0.117983, 0.870418]))
        viewer.window.set_camera_parameters(near=0.1, far=100, fovy=1)

        self.model = model
        self.scene = scene 
        self.viewer = viewer 

    def update(self):
        self.scene.step()
        self.scene.update_render()  
        self.viewer.render()

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand', type=str, default='allegro')

    args = parser.parse_args()

    # Load Hand Model
    config = get_config(args.hand)
    model = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = model.get_viewer_env()
   
    # Control Loop
    n_dof = model.get_n_dof()
    dof_lower, dof_upper = model.get_joint_limit()

    steps = 0
    while True:
        viewer_env.update()

        steps += 1
        if steps % 30 == 0:
            targets = np.random.uniform(0, 1, n_dof) * (dof_upper - dof_lower - 1e-7) + dof_lower + 1e-7
            model.set_qpos_target(targets)