# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import random
import os
import time
import sys

import pybullet as p
import numpy as np
import IPython

from env.tm5_gripper_hand_camera import TM5
from transforms3d.quaternions import *
import scipy.io as sio
from utils.utils import *
import json
from itertools import product
import math
from utils.utils import SlideBars
from scipy.spatial.transform import Rotation
import re


BASE_LINK = -1
MAX_DISTANCE = 0.000


def get_num_joints(body, CLIENT=None):
    return p.getNumJoints(body, physicsClientId=CLIENT)


def get_links(body, CLIENT=None):
    return list(range(get_num_joints(body, CLIENT)))


def get_all_links(body, CLIENT=None):
    return [BASE_LINK] + list(get_links(body, CLIENT))


class SimulatedYCBEnv():    
    def __init__(self,
                 renders=True,
                 blockRandom=0.5,
                 cameraRandom=0,
                 use_hand_finger_point=False,
                 data_type='RGB',
                 filter_objects=[],
                #  img_resize=(224, 224),
                 img_resize=(640, 640),
                 regularize_pc_point_count=True,
                 egl_render=False,
                #  width=224,
                #  height=224,
                 width=640,
                 height=640,
                 uniform_num_pts=2048,    # 必須使用uniform才會好train
                 change_dynamics=False,
                 initial_near=0.2,
                 initial_far=0.5,
                 disable_unnece_collision=False,
                 use_acronym=False):
        self._timeStep = 1. / 1000.
        self._observation = []
        self._renders = renders
        self._resize_img_size = img_resize

        self._p = p
        self._window_width = width
        self._window_height = height
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._use_hand_finger_point = use_hand_finger_point
        self._data_type = data_type
        self._egl_render = egl_render
        self._disable_unnece_collision = disable_unnece_collision

        self._change_dynamics = change_dynamics
        self._initial_near = initial_near
        self._initial_far = initial_far
        self._filter_objects = filter_objects
        self._regularize_pc_point_count = regularize_pc_point_count
        self._uniform_num_pts = uniform_num_pts
        self.observation_dim = (self._window_width, self._window_height, 3)
        self._use_acronym = use_acronym
        #Henry add target fixed   true代表會懸空
        self.target_fixed = False

        self.init_constant()
        self.connect()

    def init_constant(self):
        # 原本[0.8, 0.8, 0.8]
        self._shift = [-0.0, 0., -0.]  # to work without axis in DIRECT mode # traslate to the base position
        self.root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        self._standoff_dist = 0.08
        self.cam_offset = np.eye(4)
        self.cam_offset[:3, 3] = (np.array([0., 0.1186, -0.0191344123493]))
        self.cam_offset[:3, :3] = euler2mat(0, 0, np.pi)

        self.target_idx = 0
        self.objects_loaded = False
        self.connected = False
        self.stack_success = True

    def connect(self):
        """
        Connect pybullet.
        """
        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = p.connect(p.GUI)
            # pybullet.resetDebugVisualizerCamera(cameraDistance = 3,cameraYaw = 30,cameraPitch = 52,cameraTargetPosition = [0,0,0])
            p.resetDebugVisualizerCamera(1.5, 50., -17.4, [-0.08, 0.43, -0.62])


        else:
            self.cid = p.connect(p.DIRECT)

        if self._egl_render:
            import pkgutil
            egl = pkgutil.get_loader("eglRenderer")
            if egl: 
                p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self.connected = True

    def disconnect(self):
        """
        Disconnect pybullet.
        """
        p.disconnect()
        self.connected = False

    def reset(self, save=False, init_joints=None, num_object=1, if_stack=True,
              cam_random=0, reset_free=False, enforce_face_target=False):
        """
        Environment reset called at the beginning of an episode.
        """
        self.retracted = False
        if reset_free:
            return self.cache_reset(init_joints, enforce_face_target, num_object=num_object, if_stack=if_stack)

        self.disconnect()
        self.connect()

        # Set the camera  .
        look = [0.1 - self._shift[0], 0.2 - self._shift[1], 0 - self._shift[2]]
        distance = 2.5
        pitch = -56
        yaw = 245
        roll = 0.
        fov = 20.
        aspect = float(self._window_width) / self._window_height
        self.near = 0.1
        self.far = 10
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, self.near, self.far)
        self._light_position = np.array([-1.0, 0, 2.5])

        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setPhysicsEngineParameter(enableConeFriction=0)

        p.setGravity(0, 0, -9.81)
        p.stepSimulation()

        # Set table and plane
        plane_file = os.path.join(self.root_dir,  'data/objects/floor/model_normalized.urdf')  # _white
        table_file = os.path.join(self.root_dir,  'data/objects/table/models/model_normalized.urdf')

        self.obj_path = [plane_file, table_file]

        self.plane_id = p.loadURDF(plane_file, [0 - self._shift[0], 0 - self._shift[1], -.82 - self._shift[2]])
        self.table_pos = np.array([0.5 - self._shift[0], 0.0 - self._shift[1], -.82 - self._shift[2]])
        self.table_id = p.loadURDF(table_file, self.table_pos[0], self.table_pos[1], self.table_pos[2],
                                  0.707, 0., 0., 0.707)    

        # Intialize robot and objects
        if init_joints is None:
            self._panda = TM5(stepsize=self._timeStep, base_shift=self._shift)

        else:
            self._panda = TM5(stepsize=self._timeStep, init_joints=init_joints, base_shift=self._shift)
            for _ in range(1000):
                p.stepSimulation()

        if not self.objects_loaded:
            self._objectUids = self.cache_objects()

        self._randomly_place_objects_pack(self._get_random_object(num_object), scale=1, if_stack=if_stack)

        self._objectUids += [self.plane_id, self.table_id]
   
        self.collided = False
        self.collided_before = False
        self.obj_names, self.obj_poses = self.get_env_info()
        self.init_target_height = self._get_target_relative_pose()[2, 3]
        return None  # observation
    def cabinet(self, save=False, init_joints=None, num_object=1, if_stack=True,
              cam_random=0, reset_free=False, enforce_face_target=False, stable_pose = False, 
              single_release = False, place_target_matrix = None):
        
        # 自定義要讓物品隨機落下去獲取點雲或是穩定擺放在桌上 stable_pose = true表示要讓它穩定擺放
        self.stable_pose = stable_pose
        """
        Environment reset called at the beginning of an episode.
        """
        self.retracted = False
        if reset_free:
            return self.cache_reset(init_joints, enforce_face_target, num_object=num_object, if_stack=if_stack, 
            single_release = single_release, place_target_matrix=place_target_matrix)

        self.disconnect()
        self.connect()

        # Set the camera  .
        look = [0.1 - self._shift[0], 0.2 - self._shift[1], 0 - self._shift[2]]
        distance = 2.5
        pitch = -56
        yaw = 245
        roll = 0.
        fov = 20.
        aspect = float(self._window_width) / self._window_height
        self.near = 0.1
        self.far = 10
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, self.near, self.far)
        self._light_position = np.array([-1.0, 0, 2.5])

        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setPhysicsEngineParameter(enableConeFriction=0)

        p.setGravity(0, 0, -9.81)
        p.stepSimulation()

        # Set table and plane
        plane_file = os.path.join(self.root_dir,  'data/objects/floor/model_normalized.urdf')  # _white
        table_file = os.path.join(self.root_dir,  'data/objects/table/models/model_normalized.urdf')
        cabinet_file = os.path.join(self.root_dir,  'data/real_shelf.urdf')
        place_object_file1 = os.path.join(self.root_dir,  'data/objects/004_sugar_box/model_normalized.urdf')
        place_object_file2 = os.path.join(self.root_dir,  'data/objects/006_mustard_bottle/model_normalized.urdf')
        place_object_file3 = os.path.join(self.root_dir,  'data/objects/025_mug/model_normalized.urdf')

       
        # self.obj_path = [plane_file, table_file, shelf_file]
        self.obj_path = [plane_file, table_file]

        self.plane_id = p.loadURDF(plane_file, [0 - self._shift[0], 0 - self._shift[1], -.82 - self._shift[2]])
        self.table_pos = np.array([0.5 - self._shift[0], 0.0 - self._shift[1], -.82 - self._shift[2]])
        self.table_id = p.loadURDF(table_file, [self.table_pos[0], self.table_pos[1], self.table_pos[2]],
                                  [0.707, 0., 0., 0.707])
        # create the shelf  x-0.5, y+0.5
        orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
        #Henry 20240201 y從0.5改成0.7
        self.cabinet_pos = np.array([0.1-self._shift[0], 0.9 - self._shift[1], -.82 - self._shift[2]])
        self.cabinet_id = p.loadURDF(cabinet_file, [self.cabinet_pos[0], self.cabinet_pos[1], self.cabinet_pos[2]],
                                  [orientation[0], orientation[1], orientation[2], orientation[3]], useFixedBase=True)   # default 0.8

        # orientation = p.getQuaternionFromEuler([0, 0, np.pi/8])
        # self.place_object_pos = np.array([ -0.2-self._shift[0], 0.5 - self._shift[1], -.5 - self._shift[2]])
        # # self.place_object_1 = p.loadURDF(place_object_file1, [self.place_object_pos[0], self.place_object_pos[1], self.place_object_pos[2]],
        # #                           [orientation[0], orientation[1], orientation[2], orientation[3]], useFixedBase=False, globalScaling=1.) 
        # self.place_object_2 = p.loadURDF(place_object_file2, [self.place_object_pos[0] + 0.2, self.place_object_pos[1], self.place_object_pos[2]],
        #                             [orientation[0], orientation[1], orientation[2], orientation[3]], useFixedBase=False, globalScaling=1.)
        # orientation = p.getQuaternionFromEuler([0, 0, 0])
        # self.place_object_3 = p.loadURDF(place_object_file3, [self.place_object_pos[0] + 0.4, self.place_object_pos[1], self.place_object_pos[2]],
        #                             [orientation[0], orientation[1], orientation[2], orientation[3]], useFixedBase=False, globalScaling=1.)
        

        # Intialize robot and objects
        if init_joints is None:
            self._panda = TM5(stepsize=self._timeStep, base_shift=self._shift)

        else:
            self._panda = TM5(stepsize=self._timeStep, init_joints=init_joints, base_shift=self._shift)
            for _ in range(1000):
                p.stepSimulation()

        if not self.objects_loaded:
            self._objectUids = self.cache_objects()
        print("self._objectUids:", self._objectUids)

        self._randomly_place_objects_pack(self._get_random_object(num_object), scale=1, if_stack=if_stack, single_release = single_release)
        print("urdfList(object number):", len(self._get_random_object(num_object)))  #如果數量 = 1(default) _randomly_place_objects_pack就是執行 _randomly_place_objects
        self._objectUids += [self.plane_id, self.table_id]            #從高處落下我只改_randomly_place_objects
        print(self._objectUids)
        # 20221230 henry
        # self._objectUids += [self.plane_id]

        self.collided = False
        self.collided_before = False
        self.obj_names, self.obj_poses = self.get_env_info()
        self.init_target_height = self._get_target_relative_pose()[2, 3]
        return None  # observation
    

    def different_cabinet(self, save=False, init_joints=None, num_object=1, if_stack=True,
              cam_random=0, reset_free=False, enforce_face_target=False, stable_pose = False, single_release = False):
        
        # 自定義要讓物品隨機落下去獲取點雲或是穩定擺放在桌上 stable_pose = true表示要讓它穩定擺放
        self.stable_pose = stable_pose
        """
        Environment reset called at the beginning of an episode.
        """
        self.retracted = False
        if reset_free:
            return self.cache_reset(init_joints, enforce_face_target, num_object=num_object, if_stack=if_stack)

        self.disconnect()
        self.connect()

        # Set the camera  .
        look = [0.1 - self._shift[0], 0.2 - self._shift[1], 0 - self._shift[2]]
        distance = 2.5
        pitch = -56
        yaw = 245
        roll = 0.
        fov = 20.
        aspect = float(self._window_width) / self._window_height
        self.near = 0.1
        self.far = 10
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, self.near, self.far)
        self._light_position = np.array([-1.0, 0, 2.5])

        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setPhysicsEngineParameter(enableConeFriction=0)

        p.setGravity(0, 0, -9.81)
        p.stepSimulation()

        # Set table and plane
        plane_file = os.path.join(self.root_dir,  'data/objects/floor/model_normalized.urdf')  # _white
        table_file = os.path.join(self.root_dir,  'data/objects/table/models/model_normalized.urdf')
        cabinet_file = os.path.join(self.root_dir,  'data/objects/cabinet/new_model.urdf')
        place_object_file = os.path.join(self.root_dir,  'data/objects/004_sugar_box/model_normalized.urdf')


        #### paper shelf testing  (just change the cabinet_file and the orientation)
        # 2. 
        # cabinet_file = os.path.join(self.root_dir,  'data/shelf_2.urdf')
        # orientation = p.getQuaternionFromEuler([0, 0, 0])
        # self.cabinet_pos = np.array([-self._shift[0] - 0.3, 0.5 - self._shift[1], -.82 - self._shift[2]])
        # self.cabinet_id = p.loadURDF(cabinet_file, [self.cabinet_pos[0], self.cabinet_pos[1], self.cabinet_pos[2]],
        #                           [orientation[0], orientation[1], orientation[2], orientation[3]], useFixedBase=True, globalScaling=0.1) 
        # 3. 
        # cabinet_file = os.path.join(self.root_dir,  'data/shelf_4.urdf')
        # orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
        # self.cabinet_pos = np.array([-self._shift[0], 0.5 - self._shift[1], -.82 - self._shift[2]])
        # self.cabinet_id = p.loadURDF(cabinet_file, [self.cabinet_pos[0], self.cabinet_pos[1], self.cabinet_pos[2]],
        #                           [orientation[0], orientation[1], orientation[2], orientation[3]], useFixedBase=True, globalScaling=0.8) 
        ####


        # self.obj_path = [plane_file, table_file, shelf_file]
        self.obj_path = [plane_file, table_file]

        self.plane_id = p.loadURDF(plane_file, [0 - self._shift[0], 0 - self._shift[1], -.82 - self._shift[2]])
        self.table_pos = np.array([0.5 - self._shift[0], 0.0 - self._shift[1], -.82 - self._shift[2]])
        self.table_id = p.loadURDF(table_file, [self.table_pos[0], self.table_pos[1], self.table_pos[2]],
                                  [0.707, 0., 0., 0.707])
        # create the shelf  x-0.5, y+0.5
        orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
        self.cabinet_pos = np.array([-self._shift[0], 0.5 - self._shift[1], -.82 - self._shift[2]])
        self.cabinet_id = p.loadURDF(cabinet_file, [self.cabinet_pos[0], self.cabinet_pos[1], self.cabinet_pos[2]],
                                  [orientation[0], orientation[1], orientation[2], orientation[3]], useFixedBase=True, globalScaling=0.8)   # default 0.8

        orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
        self.place_object_pos = np.array([ -0.15 - self._shift[0], 0.5 - self._shift[1], -.4 - self._shift[2]])
        self.place_object_id = p.loadURDF(place_object_file, [self.place_object_pos[0], self.place_object_pos[1], self.place_object_pos[2]],
                                  [orientation[0], orientation[1], orientation[2], orientation[3]], useFixedBase=False, globalScaling=1.) 

        # Intialize robot and objects
        if init_joints is None:
            self._panda = TM5(stepsize=self._timeStep, base_shift=self._shift)

        else:
            self._panda = TM5(stepsize=self._timeStep, init_joints=init_joints, base_shift=self._shift)
            for _ in range(1000):
                p.stepSimulation()

        if not self.objects_loaded:
            self._objectUids = self.cache_objects()

        self._randomly_place_objects_pack(self._get_random_object(num_object), scale=1, if_stack=if_stack, single_release = single_release)
        print("urdfList(object number):", len(self._get_random_object(num_object)))  #如果數量 = 1(default) _randomly_place_objects_pack就是執行 _randomly_place_objects
        self._objectUids += [self.plane_id, self.table_id]            #從高處落下我只改_randomly_place_objects
        print(self._objectUids)
        # 20221230 henry
        # self._objectUids += [self.plane_id]

        self.collided = False
        self.collided_before = False
        self.obj_names, self.obj_poses = self.get_env_info()
        self.init_target_height = self._get_target_relative_pose()[2, 3]
        return None  # observation

    def slidebar_cabinet(self, save=False, init_joints=None, num_object=1, if_stack=True,
              cam_random=0, reset_free=False, enforce_face_target=False, stable_pose = False, single_release = False):
        
        # 自定義要讓物品隨機落下去獲取點雲或是穩定擺放在桌上 stable_pose = true表示要讓它穩定擺放
        self.stable_pose = stable_pose
        """
        Environment reset called at the beginning of an episode.
        """
        self.retracted = False
        if reset_free:
            return self.cache_reset(init_joints, enforce_face_target, num_object=num_object, if_stack=if_stack, single_release = single_release)

        self.disconnect()
        self.connect()

        # Set the camera  .
        look = [0.1 - self._shift[0], 0.2 - self._shift[1], 0 - self._shift[2]]
        distance = 2.5
        pitch = -56
        yaw = 245
        roll = 0.
        fov = 20.
        aspect = float(self._window_width) / self._window_height
        self.near = 0.1
        self.far = 10
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, self.near, self.far)
        self._light_position = np.array([-1.0, 0, 2.5])

        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setPhysicsEngineParameter(enableConeFriction=0)

        p.setGravity(0, 0, -9.81)
        p.stepSimulation()

        # Set table and plane
        plane_file = os.path.join(self.root_dir,  'data/objects/floor/model_normalized.urdf')  # _white
        table_file = os.path.join(self.root_dir,  'data/objects/table/models/model_normalized.urdf')
        cabinet_file = os.path.join(self.root_dir,  'data/objects/cabinet/new_model.urdf')
        place_object_file1 = os.path.join(self.root_dir,  'data/objects/004_sugar_box/model_normalized.urdf')
        place_object_file2 = os.path.join(self.root_dir,  'data/objects/006_mustard_bottle/model_normalized.urdf')
        place_object_file3 = os.path.join(self.root_dir,  'data/objects/025_mug/model_normalized.urdf')

       
        # self.obj_path = [plane_file, table_file, shelf_file]
        self.obj_path = [plane_file, table_file]

        self.plane_id = p.loadURDF(plane_file, [0 - self._shift[0], 0 - self._shift[1], -.82 - self._shift[2]])
        self.table_pos = np.array([0.5 - self._shift[0], 0.0 - self._shift[1], -.82 - self._shift[2]])
        self.table_id = p.loadURDF(table_file, [self.table_pos[0], self.table_pos[1], self.table_pos[2]],
                                  [0.707, 0., 0., 0.707])
        # create the shelf  x-0.5, y+0.5
        orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
        self.cabinet_pos = np.array([-self._shift[0], 0.7 - self._shift[1], -.82 - self._shift[2]])
        self.cabinet_id = p.loadURDF(cabinet_file, [self.cabinet_pos[0], self.cabinet_pos[1], self.cabinet_pos[2]],
                                  [orientation[0], orientation[1], orientation[2], orientation[3]], useFixedBase=True, globalScaling=0.8)   # default 0.8

    

        # Intialize robot and objects
        if init_joints is None:
            self._panda = TM5(stepsize=self._timeStep, base_shift=self._shift)

        else:
            self._panda = TM5(stepsize=self._timeStep, init_joints=init_joints, base_shift=self._shift)
            for _ in range(1000):
                p.stepSimulation()

        

        if not self.objects_loaded:
            self._objectUids = self.cache_objects()
        print("self._objectUids:", self._objectUids)

        self._randomly_place_objects_pack(self._get_random_object(num_object), scale=1, if_stack=if_stack, single_release = single_release)
        print("urdfList(object number):", len(self._get_random_object(num_object)))  #如果數量 = 1(default) _randomly_place_objects_pack就是執行 _randomly_place_objects
        self._objectUids += [self.plane_id, self.table_id]            #從高處落下我只改_randomly_place_objects
        print(self._objectUids)
        # 20221230 henry
        # self._objectUids += [self.plane_id]

        self.collided = False
        self.collided_before = False
        self.obj_names, self.obj_poses = self.get_env_info()
        self.init_target_height = self._get_target_relative_pose()[2, 3]


        #Henry slidebar
        TM_id = self._panda.pandaUid
        slide_bars=SlideBars(TM_id)
        motorIndices=slide_bars.add_slidebars()

        start_time = time.time()

        
        while True:
            p.stepSimulation()
            slide_values=slide_bars.get_slidebars_values()
            # 設定每5秒print一次
            if time.time() - start_time > 5:
                start_time = time.time()
                print("Joint:", slide_values[:6])
                print("ef in world frame:", self._get_ef_pose(mat=True))
            p.setJointMotorControlArray(TM_id,
                                        motorIndices,
                                        p.POSITION_CONTROL,
                                        targetPositions=slide_values,
                                        )
            observation = self._get_observation()
        return None  # observation
    

    def step(self, action, delta=False, obs=True, repeat=150, config=False, vis=False):
        """
        Environment step.
        """
        action = self.process_action(action, delta, config)
        self._panda.setTargetPositions(action)
        for _ in range(int(repeat)):  # repeat為numstep
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)  # 如果timestep为0.01秒，则模拟器每0.01秒计算一次物理世界的状态并更新模拟环境。
        '''
        在PyBullet中，timeStep通常是以秒为单位的浮点数，
        而numSteps通常是一个整数。numSteps的计算方法是
        通过将模拟的总时间除以时间步长来得到的。因此，
        它们之间的关系是numSteps = totalTime / timeStep。
        '''
        observation = self._get_observation(vis=vis)

        reward = self.target_lifted()

        return observation, reward, self._get_ef_pose(mat=True)
    # raw_data可以讓sagmentation顯示全部
    def _get_observation(self, pose=None, vis=False, raw_data=False):
        """
        Get observation
        """

        object_pose = self._get_target_relative_pose('ef')  # self._get_relative_ef_pose()
        ef_pose = self._get_ef_pose('mat')

        joint_pos, joint_vel = self._panda.getJointStates()
        near, far = self.near, self.far
        view_matrix, proj_matrix = self._view_matrix, self._proj_matrix
        camera_info = tuple(view_matrix) + tuple(proj_matrix)
        hand_cam_view_matrix, hand_proj_matrix, lightDistance, lightColor, lightDirection, near, far = self._get_hand_camera_view(pose)
        camera_info += tuple(hand_cam_view_matrix.flatten()) + tuple(hand_proj_matrix)
        _, _, rgba, depth, mask = p.getCameraImage(width=self._window_width,
                                                   height=self._window_height,
                                                   viewMatrix=tuple(hand_cam_view_matrix.flatten()),
                                                   projectionMatrix=hand_proj_matrix,
                                                   physicsClientId=self.cid,
                                                   renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        depth = (far * near / (far - (far - near) * depth) * 5000).astype(np.uint16)  # transform depth from NDC to actual depth
        # 20221227
        # depth = (far * near / (far - (far - near) * depth) * 5000).astype(np.int32)  # transform depth from NDC to actual depth
        intrinsic_matrix = projection_to_intrinsics(hand_proj_matrix, self._window_width, self._window_height)
        # 有虛線的從utils中匯入
        if raw_data:   # 如果要有全部ground truth分割的話就要把raw_data = TRUE
            print('target_idx = ', self.target_idx)
            print('mask = ', mask)
            mask[mask <= 2] = -1
            mask[mask > 0] -= 3
            # use np.where to choose the range of the depth
            print(depth)
            depth = np.where((depth > 0) & (depth < 5000), depth, 0)
            obs = np.concatenate([rgba[..., :3], depth[..., None], mask[..., None]], axis=-1)
            obs = self.process_image(obs[..., :3], obs[..., [3]], obs[..., [4]], tuple(self._resize_img_size), if_raw=True)
            point_state = backproject_camera_target(obs[3].T, intrinsic_matrix, None)  # obs[4].T
            point_state[1] *= -1
            # 將點雲變成(2048, 3)
            # point_state = self.process_pointcloud(point_state, vis) 

            obs = (point_state, obs)
        else:
            mask[mask >= 0] += 1  # transform mask to have target id 0
            target_idx = self.target_idx + 5

            mask[mask == target_idx] = 0
            mask[mask == -1] = 50
            mask[mask != 0] = 1

            obs = np.concatenate([rgba[..., :3], depth[..., None], mask[..., None]], axis=-1)
            obs = self.process_image(obs[..., :3], obs[..., [3]], obs[..., [4]], tuple(self._resize_img_size))
            point_state = backproject_camera_target(obs[3].T, intrinsic_matrix, obs[4].T)  # obs[4].T
            point_state[1] *= -1
            point_state = self.process_pointcloud(point_state, vis)
            obs = (point_state, obs)
        pose_info = (object_pose, ef_pose)
        return [obs, joint_pos, camera_info, pose_info, intrinsic_matrix]
    def retract(self):
        """
        Move the arm to lift the object.
        """

        cur_joint = np.array(self._panda.getJointStates()[0])
        cur_joint[-1] = 0.8  # close finger
        observations = [self.step(cur_joint, repeat=300, config=True, vis=False)[0]]
        pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]

        for i in range(10):
            pos = (pos[0], pos[1], pos[2] + 0.03)
            jointPoses = np.array(p.calculateInverseKinematics(self._panda.pandaUid,
                                                               self._panda.pandaEndEffectorIndex, pos,
                                                               maxNumIterations=500,
                                                               residualThreshold=1e-8))
            jointPoses[6] = 0.85
            jointPoses = jointPoses[:7].copy()
            obs = self.step(jointPoses, config=True)[0]

        self.retracted = True
        rew = self._reward()
        return rew
    def henry_retract(self, x_translation, y_translation, z_translation):
        """
        Move the arm to lift the object.
        """

        cur_joint = np.array(self._panda.getJointStates()[0])
        cur_joint[-1] = 0.8  # close finger
        observations = [self.step(cur_joint, repeat=300, config=True, vis=False)[0]]
        pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]
        for i in range(10):
            print(pos)
            print(orn)
            pos = (pos[0] + x_translation, pos[1] + y_translation, pos[2] + z_translation)
            jointPoses = np.array(p.calculateInverseKinematics(self._panda.pandaUid,
                                                               self._panda.pandaEndEffectorIndex, pos,
                                                               maxNumIterations=500,
                                                               residualThreshold=1e-8))
            jointPoses[6] = 0.  # 0.8是關閉夾爪
            jointPoses = jointPoses[:7].copy()
            obs = self.step(jointPoses, config=True)[0]

        self.retracted = True
        rew = self._reward()
        return rew

    def _reward(self):
        """
        Calculates the reward for the episode.
        """
        reward = 0

        if self.retracted and self.target_lifted():
            print('target {} lifted !'.format(self.target_name))
            reward = 1
        print("not lift！")
        return reward

    def cache_objects(self):
        """
        Load all YCB objects and set up
        """

        obj_path = os.path.join(self.root_dir, 'data/objects/')
        objects = self.obj_indexes
        obj_path = [obj_path + objects[i] for i in self._all_obj]

        self.target_obj_indexes = [self._all_obj.index(idx) for idx in self._target_objs]
        pose = np.zeros([len(obj_path), 3])
        pose[:, 0] = -0.5 - np.linspace(0, 8, len(obj_path))
        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        objects_paths = [p_.strip() + '/' for p_ in obj_path]
        objectUids = []
        self.object_heights = []
        self.obj_path = objects_paths + self.obj_path
        self.placed_object_poses = []
        self.object_scale = []

        for i, name in enumerate(objects_paths):
            mesh_scale = name.split('_')[-1][:-1]
            name = name.replace(f"_{mesh_scale}/", "/")
            self.object_scale.append(float(mesh_scale))
            trans = pose[i] + np.array(pos)  # fixed position
            self.placed_object_poses.append((trans.copy(), np.array(orn).copy()))
            uid = self._add_mesh(os.path.join(self.root_dir, name, 'model_normalized.urdf'), trans, orn, scale=float(mesh_scale))  # xyzw

            if self._change_dynamics:
                p.changeDynamics(uid, -1, lateralFriction=0.15, spinningFriction=0.1, rollingFriction=0.1)

            point_z = np.loadtxt(os.path.join(self.root_dir, name, 'model_normalized.extent.txt'))
            half_height = float(point_z.max()) / 2 if len(point_z) > 0 else 0.01
            self.object_heights.append(half_height)
            objectUids.append(uid)
            p.setCollisionFilterPair(uid, self.plane_id, -1, -1, 0)

            if self._disable_unnece_collision:
                for other_uid in objectUids:
                    p.setCollisionFilterPair(uid, other_uid, -1, -1, 0)
        self.objects_loaded = True
        self.placed_objects = [False] * len(self.obj_path)
        return objectUids

    def cache_reset(self, init_joints, enforce_face_target, num_object=3, if_stack=True, single_release=False, place_target_matrix = None):
        #Henry 20240117 use target matrix to place object in specific position
        """
        Hack to move the loaded objects around to avoid loading multiple times
        """

        self._panda.reset(init_joints)
        #Henry 20240117 裡面有reset
        self.place_back_objects()
        self._randomly_place_objects_pack(self._get_random_object(num_object), scale=1, if_stack=if_stack, single_release=single_release, place_target_matrix=place_target_matrix)
        self.retracted = False
        self.collided = False
        self.collided_before = False
        self.obj_names, self.obj_poses = self.get_env_info()
        self.init_target_height = self._get_target_relative_pose()[2, 3]

        observation = self.enforce_face_target() if enforce_face_target else self._get_observation()
        return observation

    def place_back_objects(self):
        for idx, obj in enumerate(self._objectUids):
            if self.placed_objects[idx]:
                p.resetBasePositionAndOrientation(obj, self.placed_object_poses[idx][0], self.placed_object_poses[idx][1])
            self.placed_objects[idx] = False

    def _add_mesh(self, obj_file, trans, quat, scale=1):
        """
        Add a mesh with URDF file.
        Henry add the target_fixed parameter useFixedBase=True（代表物體不會動）
        """
        print("target_fixed7878787878787878", self.target_fixed)
        bid = p.loadURDF(obj_file, trans, quat, globalScaling=scale, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES, useFixedBase=self.target_fixed)
        return bid

    def reset_joint(self, init_joints):
        if init_joints is not None:
            self._panda.reset(np.array(init_joints).flatten())

    def process_action(self, action, delta=False, config=False):
        """
        Process different action types
        """
        # transform to local coordinate
        if config:
            if delta:
                cur_joint = np.array(self._panda.getJointStates()[0])
                action = cur_joint + action
        else:
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]

            pose = np.eye(4)
            pose[:3, :3] = quat2mat(tf_quat(orn))
            pose[:3, 3] = pos

            pose_delta = np.eye(4)
            pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
            pose_delta[:3, 3] = action[:3]

            new_pose = pose.dot(pose_delta)
            orn = ros_quat(mat2quat(new_pose[:3, :3]))
            pos = new_pose[:3, 3]

            jointPoses = np.array(p.calculateInverseKinematics(self._panda.pandaUid,
                                  self._panda.pandaEndEffectorIndex, pos, orn,
                                  maxNumIterations=500,
                                  residualThreshold=1e-8))
            jointPoses[6] = 0.0   # 0.0為打開, 0.8為關閉
            action = jointPoses[:7]
        return action

    def _get_hand_camera_view(self, cam_pose=None):
        """
        Get hand camera view
        """
        if cam_pose is None:
            pos, orn = p.getLinkState(self._panda.pandaUid, 19)[4:6]
            cam_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        cam_pose_mat = unpack_pose(cam_pose)

        fov = 90
        aspect = float(self._window_width) / (self._window_height)
        hand_near = 0.035
        hand_far = 2
        hand_proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, hand_near, hand_far)
        hand_cam_view_matrix = se3_inverse(cam_pose_mat.dot(rotX(np.pi/2).dot(rotY(-np.pi/2)))).T  # z backward
        # print(hand_cam_view_matrix)
        lightDistance = 2.0
        #lightDirection = self.table_pos - self._light_position
        lightDirection =  self._light_position

        lightColor = np.array([1., 1., 1.])
        light_center = np.array([-1.0, 0, 2.5])
        return hand_cam_view_matrix, hand_proj_matrix, lightDistance, lightColor, lightDirection, hand_near, hand_far

    def target_lifted(self):
        """
        Check if target has been lifted
        """
        end_height = self._get_target_relative_pose()[2, 3]
        if end_height - self.init_target_height > 0.08:
            return True
        return False

    #Henry reset the placed object
    def _reset_placed_objects(self):
        pos = [-1, 0, 0]
        orn = [0, 0, 0, 1]

        p.resetBasePositionAndOrientation(self._objectUids[self.target_idx],
                                            [pos[0], pos[1], pos[2]], [orn[0], orn[1], orn[2], orn[3]]) 
        p.resetBaseVelocity(
            self._objectUids[self.target_idx], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        )
        print("------------------reset placed object!")
        for _ in range(400):
            p.stepSimulation()

    def _randomly_place_objects_pack(self, urdfList, scale, if_stack=True, single_release=False, place_target_matrix = None):
        '''
        Input:
            urdfList: File path of mesh urdf, support single and multiple object list
            scale: mesh scale
            if_stack (bool): scene setup with uniform position or stack with collision
                true: uniform position, true代表旁邊有東西會擋住掉落下來的物體
                false: stack with random position

        Func:
            For object in urdfList do:
                (1) find Uid of pybullet body by indexing object
                (1) reset position of pybullet body in urdfList
                (2) set self.placed_objects[idx] = True
        '''
        self.stack_success = True

        if len(urdfList) == 1:
            #Henry test 20230831當只有一個物體時會跑這裡而不是下面
            return self._randomly_place_objects(urdfList=urdfList, scale=scale, single_release=single_release, place_target_matrix=place_target_matrix)
        else:
            if if_stack:
                self.place_back_objects()
                for i in range(len(urdfList)):
                    #Henry test 20230831 （多物體才會跑這>1）
                    print(single_release)
                    if single_release == True:
                        print("single_release!")
                        xpos = 0.5 - self._shift[0]
                        ypos = -self._shift[0]
                        obj_path = '/'.join(urdfList[i].split('/')[:-1]) + '/'
                        self.target_idx = self.obj_path.index(os.path.join(self.root_dir, obj_path))
                        self.placed_objects[self.target_idx] = True
                        self.target_name = urdfList[i].split('/')[-2]
                        height_weight = self.object_heights[self.target_idx]
                        z_init = height_weight  # test出來的
                        orn = p.getQuaternionFromEuler([np.random.uniform(-np.pi, np.pi), 0, 0])
                        p.resetBasePositionAndOrientation(self._objectUids[self.target_idx],
                                                        [xpos, ypos,  z_init - self._shift[2]],
                                                        [orn[0], orn[1], orn[2], orn[3]])
                        p.resetBaseVelocity(
                            self._objectUids[self.target_idx], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                        )
                        for _ in range(400):
                            p.stepSimulation()
                        print('>>>> target name: {}'.format(self.target_name))
                        pos, new_orn = p.getBasePositionAndOrientation(self._objectUids[self.target_idx])  # to target
                        ang = np.arccos(2 * np.power(np.dot(tf_quat(orn), tf_quat(new_orn)), 2) - 1) * 180.0 / np.pi

                        if (self.target_name in self._filter_objects or ang > 50) and not self._use_acronym:  # self.target_name.startswith('0') and
                            self.target_name = 'noexists'
                            self.stack_success = False

                    else:
                        print("test!")
                        if i == 0:
                            xpos = 0.5 - self._shift[0]
                            ypos = -self._shift[0]
                        else:
                            spare = False
                            while not spare:
                                spare = True
                                xpos = 0.5 + 0.28 * (random.random() - 0.5) - self._shift[0]
                                ypos = 0.9 * self._blockRandom * (random.random() - 0.5) - self._shift[1]
                                for idx in range(len(self.placed_objects)):
                                    if self.placed_objects[idx]:
                                        pos, _ = p.getBasePositionAndOrientation(self._objectUids[idx])   # get world的轉移矩陣
                                        if (xpos-pos[0])**2+(ypos-pos[1])**2 < 0.0165:
                                            spare = False
                        obj_path = '/'.join(urdfList[i].split('/')[:-1]) + '/'
                        self.target_idx = self.obj_path.index(os.path.join(self.root_dir, obj_path))
                        self.placed_objects[self.target_idx] = True
                        self.target_name = urdfList[i].split('/')[-2]
                        x_rot = 0
                        if self._use_acronym:
                            object_bbox = p.getAABB(self._objectUids[self.target_idx])
                            height_weight = (object_bbox[1][2] - object_bbox[0][2]) / 2
                            z_init = -.60 + 2.5 * height_weight 
                        else:
                            height_weight = self.object_heights[self.target_idx]
                            z_init = -.65 + 1.95 * height_weight  # test出來的
                        orn = p.getQuaternionFromEuler([x_rot, 0, np.random.uniform(-np.pi, np.pi)])
                        p.resetBasePositionAndOrientation(self._objectUids[self.target_idx],
                                                        [xpos, ypos,  z_init - self._shift[2]],
                                                        [orn[0], orn[1], orn[2], orn[3]])
                        p.resetBaseVelocity(
                            self._objectUids[self.target_idx], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                        )
                        for _ in range(400):
                            p.stepSimulation()
                        print('>>>> target name: {}'.format(self.target_name))
                        pos, new_orn = p.getBasePositionAndOrientation(self._objectUids[self.target_idx])  # to target
                        ang = np.arccos(2 * np.power(np.dot(tf_quat(orn), tf_quat(new_orn)), 2) - 1) * 180.0 / np.pi

                        if (self.target_name in self._filter_objects or ang > 50) and not self._use_acronym:  # self.target_name.startswith('0') and
                            self.target_name = 'noexists'
                            self.stack_success = False

                for _ in range(2000):
                    p.stepSimulation()
            else:
                self.place_back_objects()
                wall_file = os.path.join(self.root_dir,  'data/objects/box_box000/model_normalized.urdf')
                wall_file2 = os.path.join(self.root_dir,  'data/objects/box_box001/model_normalized.urdf')
                self.wall = []
                for i in range(4):
                    if i % 2 == 0:
                        orn = p.getQuaternionFromEuler([0, 0, 0])
                    else:
                        orn = p.getQuaternionFromEuler([0, 0, 1.57])
                    x_offset = 0.26 * math.cos(i*1.57)
                    y_offset = 0.26 * math.sin(i*1.57)
                    self.wall.append(p.loadURDF(wall_file, self.table_pos[0] + x_offset, self.table_pos[1] + y_offset,
                                     self.table_pos[2] + 0.3,
                                     orn[0], orn[1], orn[2], orn[3]))
                for i in range(4):
                    x_offset = 0.26 * math.cos(i*1.57 + 0.785)
                    y_offset = 0.26 * math.sin(i*1.57 + 0.785)
                    self.wall.append(p.loadURDF(wall_file2, self.table_pos[0] + x_offset, self.table_pos[1] + y_offset,
                                     self.table_pos[2] + 0.33,
                                     orn[0], orn[1], orn[2], orn[3]))
                for i in range(8):
                    p.changeDynamics(self.wall[i], linkIndex=-1, mass=0)

                for i in range(len(urdfList)):
                    xpos = 0.5 - self._shift[0] + 0.17 * (random.random() - 0.5)
                    ypos = -self._shift[0] + 0.23 * (random.random() - 0.5)
                    obj_path = '/'.join(urdfList[i].split('/')[:-1]) + '/'
                    self.target_idx = self.obj_path.index(os.path.join(self.root_dir, obj_path))
                    self.placed_objects[self.target_idx] = True
                    self.target_name = urdfList[i].split('/')[-2]
                    z_init = -.26 + 2 * self.object_heights[self.target_idx]
                    #Henry 20240401 更改掉落角度, 比較不會碰撞
                    orn = p.getQuaternionFromEuler([0, np.random.uniform(0, np.pi/2), 0])
                    p.resetBasePositionAndOrientation(self._objectUids[self.target_idx],
                                                      [xpos, ypos,  z_init - self._shift[2]],
                                                      [orn[0], orn[1], orn[2], orn[3]])
                    p.resetBaseVelocity(
                        self._objectUids[self.target_idx], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                    )
                    for _ in range(350):
                        p.stepSimulation()
                    print('>>>> target name: {}'.format(self.target_name))

                for i in range(8):
                    p.removeBody(self.wall[i])

                for _ in range(2000):
                    p.stepSimulation()

    def _randomly_place_objects(self, urdfList, scale, target_poses=None, single_release=False, place_target_matrix=None, transparent=True):
        """
        Randomize positions of each object urdf.
        """
        if place_target_matrix is not None:
            #Henry 20240117 可以用在world的place_target_matrix將target放在指定位置
            print("urdfList: ", urdfList)   
            self.place_target_matrix = place_target_matrix

            # change the place_target_matrix to the target_poses
            obj_path = '/'.join(urdfList[0].split('/')[:-1]) + '/'

            print(self.target_idx)
            self.target_idx = self.obj_path.index(os.path.join(self.root_dir, obj_path))
            print(self.target_idx)
            self.placed_objects[self.target_idx] = True
            self.target_name = urdfList[0].split('/')[-2]

            # 假設 self._objectUids 和 self.target_idx 已經定義
            # copy new self._objectUids[self.target_idx]
            object_id = self._objectUids[self.target_idx]

            # 定義新的顏色，RGBA 格式
            if place_target_matrix is not None:
                print("place_target_matrix is not None")
                new_color = [1, 1, 1, 0.8]  # 不透明
            else:
                print("place_target_matrix is None")
                new_color = [1, 1, 1, 1]  # 原始顏色

            # 獲取物體的關節數量
            num_joints = p.getNumJoints(object_id)

            # 更改每個視覺部件的顏色，包括 base link（關節編號 -1）
            for j in range(-1, num_joints):
                p.changeVisualShape(object_id, j, rgbaColor=new_color)

            # get the 4*4 place_target_matrix xpos, ypos, zpos, xrot, yrot, zrot, wrot
            # 取得 xpos、ypos、zpos 值
            xpos, ypos, zpos = place_target_matrix[:3, 3]

            # 取得旋轉矩陣
            rotation_matrix = place_target_matrix[:3, :3]
            rotation = Rotation.from_matrix(rotation_matrix)
            x_rot, y_rot, z_rot = rotation.as_euler('xyz', degrees=False)

        

            # 使用旋轉矩陣計算旋轉的四元數
            orn = p.getQuaternionFromEuler([x_rot, y_rot, z_rot])  #依照xyz順序旋轉
            p.resetBasePositionAndOrientation(self._objectUids[self.target_idx],
                                            [xpos, ypos, zpos], [orn[0], orn[1], orn[2], orn[3]])
            p.resetBaseVelocity(  
                self._objectUids[self.target_idx], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            )
            # Henry 可以跟改simulated時間
            for _ in range(5000):
                p.stepSimulation()

            pos, new_orn = p.getBasePositionAndOrientation(self._objectUids[self.target_idx])  # to target
            ang = np.arccos(2 * np.power(np.dot(tf_quat(orn), tf_quat(new_orn)), 2) - 1) * 180.0 / np.pi
            print('>>>> target name: {}'.format(self.target_name))

            if (self.target_name in self._filter_objects or ang > 50) and not self._use_acronym:  # self.target_name.startswith('0') and
                self.target_name = 'noexists'
            return []

        #Henry use for generate object's contact plane data （can change the angle）
        if single_release==True:
            print("single_release：", single_release)
            #Henry change the target object on the table (just one object)
            xpos = 0.5-self._shift[0]
            ypos = -self._shift[0]
            obj_path = '/'.join(urdfList[0].split('/')[:-1]) + '/'

            self.target_idx = self.obj_path.index(os.path.join(self.root_dir, obj_path))
            self.placed_objects[self.target_idx] = True
            self.target_name = urdfList[0].split('/')[-2]

            # x_rot = 0  # let the object fall from this rotation
            # y_rot = np.pi*0.75# 135 degree
            # z_rot = 0
                        
            # zyx照順序轉
            x_rot = 0  # let the object fall from this rotation
            y_rot = np.random.uniform(0, np.pi/2)
            # 限制z在+-135
            z_rot = np.random.uniform(-np.pi/4*3, np.pi/4*3)


            print("x_rot: ", x_rot)
            print("y_rot: ", y_rot)
            print("z_rot: ", z_rot)
            '''
            20240311 test tomato
            x_rot = 0  # let the object fall from this rotation
            y_rot = np.pi*0.75 # 135 degree
            z_rot = 0
            '''

            height_weight = self.object_heights[self.target_idx]
            z_init = -.65 + 1.95 * height_weight
            orn = p.getQuaternionFromEuler([x_rot, y_rot, z_rot])  #依照xyz順序旋轉
            p.resetBasePositionAndOrientation(self._objectUids[self.target_idx],
                                            [xpos, ypos,  z_init - self._shift[2]], [orn[0], orn[1], orn[2], orn[3]])
            p.resetBaseVelocity(
                self._objectUids[self.target_idx], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            )
            # Henry 20240428可以是掉落物體的pose
            # time.sleep(1)
            # # get target object's pose
            # targetPose = self._get_ef_pose(mat=True)@ self._get_target_relative_pose(option = 'ef')
            # self.draw_ef_coordinate(targetPose)
            # Henry 可以跟改simulated時間
            for _ in range(5000):
                p.stepSimulation()

            pos, new_orn = p.getBasePositionAndOrientation(self._objectUids[self.target_idx])  # to target
            ang = np.arccos(2 * np.power(np.dot(tf_quat(orn), tf_quat(new_orn)), 2) - 1) * 180.0 / np.pi
            print('>>>> target name: {}'.format(self.target_name))

            if (self.target_name in self._filter_objects or ang > 50) and not self._use_acronym:  # self.target_name.startswith('0') and
                self.target_name = 'noexists'
            return []

        else:
            # random release on the table and no rotation
            print("single_release：", single_release)
            xpos = 0.5 + 0.2 * (self._blockRandom * random.random() - 0.5) - self._shift[0]
            ypos = 0.5 * self._blockRandom * (random.random() - 0.5) - self._shift[0]
            obj_path = '/'.join(urdfList[0].split('/')[:-1]) + '/'

            self.target_idx = self.obj_path.index(os.path.join(self.root_dir, obj_path))
            self.placed_objects[self.target_idx] = True
            self.target_name = urdfList[0].split('/')[-2]
            # x_rot = 3.14/4  # let the object fall from this rotation
            x_rot = 0
            if self._use_acronym:
                object_bbox = p.getAABB(self._objectUids[self.target_idx])
                height_weight = (object_bbox[1][2] - object_bbox[0][2]) / 2
                z_init = -.60 + 2.5 * height_weight
            else:
                height_weight = self.object_heights[self.target_idx]
                z_init = -.65 + 1.95 * height_weight
                # z_init = -.65 + 1.95 * height_weight + 0.5   # 物體從高處落下20230303
            orn = p.getQuaternionFromEuler([x_rot, 0, np.random.uniform(-np.pi, np.pi)])
            p.resetBasePositionAndOrientation(self._objectUids[self.target_idx],
                                            [xpos, ypos,  z_init - self._shift[2]], [orn[0], orn[1], orn[2], orn[3]])
            p.resetBaseVelocity(
                self._objectUids[self.target_idx], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            )
            for _ in range(2000):
                p.stepSimulation()

            pos, new_orn = p.getBasePositionAndOrientation(self._objectUids[self.target_idx])  # to target
            ang = np.arccos(2 * np.power(np.dot(tf_quat(orn), tf_quat(new_orn)), 2) - 1) * 180.0 / np.pi
            print('>>>> target name: {}'.format(self.target_name))

            if (self.target_name in self._filter_objects or ang > 50) and not self._use_acronym:  # self.target_name.startswith('0') and
                self.target_name = 'noexists'
            return []

    def _get_random_object(self, num_objects):
        """
        Randomly choose an object urdf from the selected objects
        """
        target_obj = np.random.choice(np.arange(0, len(self.obj_indexes)), size=num_objects, replace=False)
        selected_objects = target_obj
        selected_objects_filenames = [os.path.join('data/objects/', self.obj_indexes[int(selected_objects[i])],
                                      'model_normalized.urdf') for i in range(num_objects)]
        return selected_objects_filenames

    def _load_index_objs(self, file_dir):

        self._target_objs = range(len(file_dir))
        self._all_obj = range(len(file_dir))
        self.obj_indexes = file_dir

    def enforce_face_target(self):
        """
        Move the gripper to face the target
        """
        target_forward = self._get_target_relative_pose('ef')[:3, 3]
        target_forward = target_forward / np.linalg.norm(target_forward)
        r = a2e(target_forward)
        action = np.hstack([np.zeros(3), r])
        return self.step(action, repeat=200, vis=False)[0]

    def random_perturb(self):
        """
        Random perturb
        """
        t = np.random.uniform(-0.04, 0.04, size=(3,))
        r = np.random.uniform(-0.2, 0.2, size=(3,))
        action = np.hstack([t, r])
        return self.step(action, repeat=150, vis=False)[0]

    def get_env_info(self, scene_file=None):
        """
        Return object names and poses of the current scene
        """

        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        base_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        poses = []
        obj_dir = []

        for idx, uid in enumerate(self._objectUids):
            if self.placed_objects[idx] or idx >= len(self._objectUids) - 2:
                pos, orn = p.getBasePositionAndOrientation(uid)  # center offset of base
                obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
                poses.append(inv_relative_pose(obj_pose, base_pose))
                obj_dir.append('/'.join(self.obj_path[idx].split('/')[:-1]).strip())  # .encode("utf-8")

        return obj_dir, poses

    def process_image(self, color, depth, mask, size=None, if_raw=False):
        """
        Normalize RGBDM
        """
        if not if_raw:
            color = color.astype(np.float32) / 255.0
            mask = mask.astype(np.float32)
            depth = depth.astype(np.float32) / 5000
        if size is not None:
            color = cv2.resize(color, size)
            mask = cv2.resize(mask, size)
            depth = cv2.resize(depth, size)
        obs = np.concatenate([color, depth[..., None], mask[..., None]], axis=-1)
        obs = obs.transpose([2, 1, 0])
        return obs

    def process_pointcloud(self, point_state, vis, use_farthest_point=False):
        """
        Process point cloud input
        """
        if self._regularize_pc_point_count and point_state.shape[1] > 0:
            point_state = regularize_pc_point_count(point_state.T, self._uniform_num_pts, use_farthest_point).T

        if vis:
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(point_state.T[:, :3])
            o3d.visualization.draw_geometries([pred_pcd])

        return point_state

    def transform_pose_from_camera(self, pose, mat=False):
        """
        Input: pose with length 7 [pos_x, pos_y, pos_z, orn_w, orn_x, orn_y, orn_z]
        Transform from 'pose relative to camera' to 'pose relative to ef'
        """
        mat_camera = unpack_pose(list(pose[:3]) + list(pose[3:]))
        if mat:
            return self.cam_offset.dot(mat_camera)
        else:
            return pack_pose(self.cam_offset.dot(mat_camera))

    def _get_relative_ef_pose(self):    
        """
        Get all obejct poses with respect to the end effector
        """
        pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]
        ef_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        poses = []
        for idx, uid in enumerate(self._objectUids):
            if self.placed_objects[idx]:
                pos, orn = p.getBasePositionAndOrientation(uid)  # to target
                obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
                poses.append(inv_relative_pose(obj_pose, ef_pose))
        return poses
#########################################################
    def _get_ef_pose(self, mat=False):# 得到world translation matrix
        """
        end effector pose in world frame
        """
        if not mat:
            return p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]   # get the  worldLinkFramePosition and worldLinkFrameOrientation
        else:
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]
            return unpack_pose(list(pos) + [orn[3], orn[0], orn[1], orn[2]])
        
    def _get_world_frame_pose(self, mat=False):
        """
        world frame
        """
        if not mat:
            return p.getBasePositionAndOrientation(self._panda.pandaUid)
        else:
            pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
            return unpack_pose(list(pos) + [orn[3], orn[0], orn[1], orn[2]])
########################################################### 

    def _get_target_relative_pose(self, option='base'):
        """
        Get target obejct poses with respect to the different frame.
        """
        if option == 'base':
            pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        elif option == 'ef':
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]
        elif option == 'tcp':
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]
            rot = quat2mat(tf_quat(orn))
            tcp_offset = rot.dot(np.array([0, 0, 0.13]))
            pos = np.array(pos) + tcp_offset

        pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        uid = self._objectUids[self.target_idx]
        pos, orn = p.getBasePositionAndOrientation(uid)  # to target
        obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        return inv_relative_pose(obj_pose, pose)
    
    # get target urdf pose
    def _get_target_urdf_pose(self, option='cabinet_world', mat=False):
        if option == 'cabinet_world':
            pos, orn = p.getBasePositionAndOrientation(self.cabinet_id)
            # 往上移動0.5
            pos = np.array(pos) + np.array([0, 0, 1.02])
            # 對orn的z軸旋轉180度
            # 旋轉 180 度的四元數可以表示為 (0, 0, 1, 0)
            rotation_quaternion = p.getQuaternionFromEuler([0, 0, np.pi])

            # 結合當前方向和旋轉四元數
            orn = p.multiplyTransforms([0, 0, 0], rotation_quaternion, [0, 0, 0], orn)[1]
            
        elif option == 'table_world':
            pos, orn = p.getBasePositionAndOrientation(self.table_id)
        
        if not mat:
            return list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        else:
            return unpack_pose(list(pos) + [orn[3], orn[0], orn[1], orn[2]])

    def draw_ef_coordinate(self, robot_pos_mat, lifeTime = 0):
        # 取得ef的座標   robot_pos_mat要放世界座標lifeTime = 0為 permanent

        frame_start_postition = robot_pos_mat[:, 3][:3]

        #x axis
        x_axis = robot_pos_mat[:,0][:3]
        x_end_p = (np.array(frame_start_postition) + np.array(x_axis* 0.2)).tolist()
        x_line_id = p.addUserDebugLine(frame_start_postition,x_end_p,[1,0,0], lineWidth=5, lifeTime = lifeTime)

        # y axis
        y_axis = robot_pos_mat[:,1][:3]
        y_end_p = (np.array(frame_start_postition) + np.array(y_axis* 0.2)).tolist()
        y_line_id = p.addUserDebugLine(frame_start_postition,y_end_p,[0,1,0], lineWidth=5, lifeTime = lifeTime)

        # z axis
        z_axis = robot_pos_mat[:,2][:3]
        z_end_p = (np.array(frame_start_postition) + np.array(z_axis* 0.2)).tolist()
        z_line_id = p.addUserDebugLine(frame_start_postition,z_end_p,[0,0,1], lineWidth=5, lifeTime = lifeTime)

    def clean_debug_line(self):
        p.removeAllUserDebugItems()

    def get_pcd(self, pcd_id, file_name, raw_data=False, vis=False):

        file_name = file_name
        obs, joint_pos, camera_info, pose_info, intrinsic = self._get_observation(raw_data = raw_data)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obs[0][:3].T)
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # print the points number
        print(np.asarray(pcd.points).shape)
        if vis:
            o3d.visualization.draw_geometries([pcd]+ [axis_pcd])
        o3d.io.write_point_cloud(f"./{file_name}/{pcd_id}.pcd", pcd, write_ascii=True)
        o3d.io.write_point_cloud(f"./{file_name}/{pcd_id}.ply", pcd, write_ascii=True)

if __name__ == '__main__':
    pass
