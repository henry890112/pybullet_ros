#!/usr/bin/env python3

import pybullet as p
import numpy as np
import sys
import os
import json
import open3d as o3d
import matplotlib.pyplot as plt
import time
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env.ycb_scene import SimulatedYCBEnv
from utils.grasp_checker import ValidGraspChecker
from utils.utils import *
from utils.planner import GraspPlanner
import utils.tcp_utils as tcp_utils
import rospy
from pybullet_ros.srv import GraspGroup, GraspGroupRequest
from sensor_msgs.point_cloud2 import create_cloud_xyz32
from utils.my_utils import *
from utils.planner import GraspPlanner
from cv_bridge import CvBridge
import pyscreenshot as ImageGrab

    
class ExecutionNode:
    def __init__(self):
        self.parent_directory = os.path.join(os.path.dirname(__file__), '..')
        self.bridge = CvBridge()
        self.env = None
        self.planner = GraspPlanner()
        self.contact_client = rospy.ServiceProxy('contact_graspnet/get_grasp_result', GraspGroup)
        rospy.wait_for_service('contact_graspnet/get_grasp_result', timeout=30)
        self.execute = rospy.get_param('~execute', False)
        self.vis_draw_coordinate = rospy.get_param('~vis_draw_coordinate', True)
        self.test_num = rospy.get_param('~test_num', 100)
        self.object_record_init()
        self.placed_obj = None
        self.path_length = 20
        self.vis_pcd = False

        '多物體擺放策略'
        self.count_location = 0
        self.placing_location = [0.2, 0, -0.2]
        # stage2 是最上面的貨價
        self.placing_stage = 1


    def initial(self):
        
        '放置物體相關 data'
        self.placed_name = None  # 一個list
        self.target_placed_name = None # 一個target name
        self.target_object = None # 會在find_closet_target中被賦值
        self.poseEstimate = None
        self.target_pose_world = None
        self.result_z_rotation_angle = 0
        self.final_place_grasp_pose = None
        self.final_grasp_pose = None

        'pointcloud 相關 data'
        self.rgb_list = []
        self.depth_list = []
        self.mask_list = []
        self.pc_segments_combine = None
        self.pc_full_combine = None
        self.pc_full_combine_colors = None
        self.pc_segments_combine_noise = None
        self.pc_full_combine_noise = None
        self.pc_segments_path = None
        self.pc_segments_noise_path = None



        '場景設置相關 data'
        '''
        single release: (只在num_object = 1有用) true為以自己設定的角度放在桌上; (多object也可用)false就是pack放在桌上
        if_stack: false代表旁邊有東西會擋住掉落下來的物體

        _randomly_place_objects_pack
        single_release/if_stack = t/t: 丟物體在桌上 沒遮擋
        single_release/if_stack = f/t: pack物體在桌上 沒遮擋
        single_release/if_stack = f/f: 丟物體在桌上 有遮擋
        single_release/if_stack = t/f: 丟物體在桌上 有遮擋
        '''

        self.num_object = 1
        # self.single_release =True
        # self.if_stack = True
        self.single_release =True
        self.if_stack = False
        self.cabinet_pose_world = None
        self.place_pose_world = None

        'jointpose 相關data'
        self.success_joint_grasp_list = []
        self.success_joint_mid_list = []
        self.success_joint_place_list = []
        self.left_joint_list = []
        self.right_joint_list = []

        'ros 傳輸相關 data'
        self.contact_client = rospy.ServiceProxy('contact_graspnet/get_grasp_result', GraspGroup)
        rospy.wait_for_service('contact_graspnet/get_grasp_result')

        ### 將open3d下的點雲轉到world座標
        # self.world_frame_pose = np.array([[ 1.,    0.,    0.,   -0.05],
        #                             [ 0.,    1.,    0.,    0.  ],
        #                             [ 0.,    0.,    1.,   -0.65],
        #                             [ 0.,    0.,    0.,    1.  ]])
        self.world_frame_pose = np.array([[ 1.,    0.,    0.,   -0.0],
                                    [ 0.,    1.,    0.,    0.  ],
                                    [ 0.,    0.,    1.,   -0.],
                                    [ 0.,    0.,    0.,    1.  ]])

        self.init_ef_mat = np.array([[-1.98785608e-01,  7.23231525e-01,  6.61377686e-01,  1.56898375e-01],
                                [9.80042993e-01,  1.46612626e-01,  1.34240345e-01, -9.29623842e-02],
                                [1.20530092e-04,  6.74863616e-01, -7.37942468e-01, 0.35],
                                [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        
        self.left_view_ef_mat = np.array([[ 0.98757027,  0.02243495,  0.15556875,  0.50691898],
                                    [ 0.14573556, -0.501431,   -0.85283533,  0.36891946],
                                    [ 0.05887368,  0.86490672, -0.49846791, 0.35],
                                    [ 0.,          0.,          0.,          1.]])

        self.right_view_ef_mat = np.array([[ 0.98691477, -0.16087768,  0.010845,    0.51446365],
                                    [-0.10023915, -0.55945926,  0.82277424, -0.28816143],
                                    [-0.12629867, -0.81309514, -0.56826485, 0.35],
                                    [ 0.,          0.,          0.,          1.]])

        self.intrinsic_matrix = np.array([[320, 0, 320],
                                    [0, 320, 320],
                                    [0, 0, 1]])
        
        # rotate the matrix -90 degree around z axis
        self.init_ef_mat = rotZ(-np.pi/2).dot(self.init_ef_mat)
        self.left_view_ef_mat = rotZ(-np.pi/2).dot(self.left_view_ef_mat)
        self.right_view_ef_mat = rotZ(-np.pi/2).dot(self.right_view_ef_mat)

        self.cam_offset = np.eye(4)
        # 先轉到pybullet座標後再往上移動0.13變相機座標
        self.cam_offset[:3, 3] = (np.array([0., 0.1186, -0.0191344123493]))
        # open3d 和 pybullet中的coordinate要對z旋轉180度才會一樣
        self.cam_offset[:3, :3] = np.array([[-1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]])

        ### 轉換關係
        self.origin_camera2world = self.cam_offset@ np.linalg.inv(self.init_ef_mat)@ self.world_frame_pose
        self.left_camera2world = self.cam_offset@ np.linalg.inv(self.left_view_ef_mat)@ self.world_frame_pose
        self.right_camera2world = self.cam_offset@ np.linalg.inv(self.right_view_ef_mat)@ self.world_frame_pose

    def load_environment(self):
        file = os.path.join(self.parent_directory, "object_index", 'contact_plane_object.json')
        with open(file) as f:
            file_dir = json.load(f)
        file_dir = file_dir['005_tomato_soup_can_1.0']
        file_dir = [f[:-5] for f in file_dir]
        test_file_dir = list(set(file_dir))
        self.env = SimulatedYCBEnv()
        self.env._load_index_objs(test_file_dir)
        state = self.env.cabinet(save=False, enforce_face_target=True)

    def expert_plan(self, goal_pose, world=False, visual=False):
        if world:
            pos, orn = self.env._get_ef_pose()
            ef_pose_list = [*pos, *orn]
        else:
            ef_pose_list = [0, 0, 0, 0, 0, 0, 1]
        goal_pos = [*goal_pose[:3], *ros_quat(goal_pose[3:])]

        solver = self.planner.plan(ef_pose_list, goal_pos, path_length=self.path_length)
        if visual:
            self.path_visulization(solver)
        path = solver.getSolutionPath().getStates()
        planer_path = []
        for i in range(len(path)):
            waypoint = path[i]
            rot = waypoint.rotation()
            action = [waypoint.getX(), waypoint.getY(), waypoint.getZ(), rot.w, rot.x, rot.y, rot.z]
            planer_path.append(action)

        return planer_path
    
    def expert_plan_multiview(self, goal_pose, world=False, visual=False):
        if world:
            pos, orn = self.env._get_ef_pose()
            ef_pose_list = [*pos, *orn]
        else:
            ef_pose_list = [0, 0, 0, 0, 0, 0, 1]
        goal_pos = [*goal_pose[:3], *ros_quat(goal_pose[3:])]

        solver = self.planner.plan(ef_pose_list, goal_pos, path_length=5)
        if visual:
            self.path_visulization(solver)
        path = solver.getSolutionPath().getStates()
        planer_path = []
        for i in range(len(path)):
            waypoint = path[i]
            rot = waypoint.rotation()
            action = [waypoint.getX(), waypoint.getY(), waypoint.getZ(), rot.w, rot.x, rot.y, rot.z]
            planer_path.append(action)

        return planer_path

    def path_visulization(self, ss):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x = []
        y = []
        z = []
        for i in range(len(ss.getSolutionPath().getStates())):
            state = ss.getSolutionPath().getStates()[i]
            x.append(state.getX())
            y.append(state.getY())
            z.append(state.getZ())
        ax.plot(x, y, z, color='gray', label='Curve')

        ax.scatter(x, y, z, c=z, cmap='jet', label='Points')
        plt.show()
        
    def execute_motion_plan(self, plan, execute=False, gripper_set="close", repeat=100, mode='nothing'):
        """
        Executes a series of movements in a robot environment based on the provided plan.

        Parameters:
        - env: The current robot environment, providing inverse kinematics solving and stepping capabilities.
        - plan: A plan containing target positions, each position is a list with coordinates and orientation.
        - execute: A boolean indicating whether to execute the actions. If False, only resets the robot's joint positions without stepping through the environment.
        """

        '''
        plan中的每一個step都會做碰撞檢查, 如果有碰撞就會停止並回傳flase
        '''
        for i in range(len(plan)):
            # Set target position using world frame based coordinates
            next_pos = plan[i]
            jointPoses = self.env._panda.solveInverseKinematics(next_pos[:3], ros_quat(next_pos[3:]))
            if gripper_set == "close":
                jointPoses[6] = 0.85
            else:
                jointPoses[6] = 0.0
            jointPoses = jointPoses[:7].copy()  # Consider only the first 7 joint positions

            if execute:
                # Execute the action and obtain the observation
                obs = self.env.step(jointPoses, config=True, repeat=repeat)[0]
                if(mode == 'left'):
                        self.left_joint_list.append(jointPoses[:6])
                elif(mode == 'right'):
                    self.right_joint_list.append(jointPoses[:6])
            else:
                # Only reset the robot's joint positions
                self.env._panda.reset(joints=jointPoses)
                # 在path length中每一步都檢查是否會發生碰撞
                if(self.env._panda.check_for_collisions() == True):
                    print("Collision detected in the path")
                    # 有碰撞表示此grasp pose不行, 初始化joint list
                    if(mode == 'grasping'):
                        self.success_joint_grasp_list = []
                    elif(mode == 'placing_mid'):
                        self.success_joint_mid_list = []
                    elif(mode == 'placing'):
                        self.success_joint_place_list = []
                    return False
                else:
                    if(mode == 'grasping'):
                        self.success_joint_grasp_list.append(jointPoses[:6])
                    elif(mode == 'placing_mid'):
                        self.success_joint_mid_list.append(jointPoses[:6])
                    elif(mode == 'placing'):
                        self.success_joint_place_list.append(jointPoses[:6])
        return True

    def check_scene(self):
        self.env._panda.reset()
        target_on_table = 0
        first_time = True
        while(target_on_table < self.num_object):
            state = self.env.cabinet(save=False, reset_free=True, num_object=self.num_object, if_stack=self.if_stack, single_release=self.single_release)
            self.placed_obj = {}
            placed_idx = np.where(np.array(self.env.placed_objects))[0]
            self.placed_name = np.array(self.env.obj_indexes)[np.where(np.array(self.env.placed_objects))]
            for i in range(self.num_object):
                self.placed_obj[placed_idx[i]] = self.placed_name[i]
            print("self.placed_obj!!!!!!!!!!!!!!!", self.placed_obj)
            target_on_table = 0
            for i, index in enumerate(self.placed_obj.keys()):
                self.placed_name = self.placed_obj[index]
                self.env.target_idx = index
                self.target_pose_world = self.env._get_ef_pose(mat=True)@ self.env._get_target_relative_pose(option = 'ef')
                # get the target pose z height
                target_z_height = self.target_pose_world[2, 3]
                if target_z_height > -0.7:
                    target_on_table += 1

    def get_placed_obj(self):
        placed_idx = np.where(np.array(self.env.placed_objects))[0]
        self.placed_name = np.array(self.env.obj_indexes)[np.where(np.array(self.env.placed_objects))]
        for i in range(self.    num_object):
            self.placed_obj[placed_idx[i]] = self.placed_name[i]
        print("==============self.placed_obj!!!!!!!!!!!!!!!", self.placed_obj)
    
    def get_multiview_data(self):
        # multiview use reset or not
        execute = False
        self.env._panda.reset()
        end_points = [self.init_ef_mat, self.left_view_ef_mat, self.right_view_ef_mat]
        views_dict = {"origin": 0, "left": 1, "right": 2}
        save_path = os.path.join(self.parent_directory, "multiview_data")


        # 透過for迴圈處理不同的視角
        for view_name, multiview_index in views_dict.items():
            END_POINT = end_points[multiview_index]  # 根據索引從結構中選擇對應的終點矩陣

            if self.vis_draw_coordinate:
                self.env.draw_ef_coordinate(END_POINT, 5)
            plan = self.expert_plan_multiview(pack_pose(END_POINT), world=True, visual=False)
            self.execute_motion_plan(plan, execute=execute, gripper_set="open", repeat=200, mode=view_name)
            checker = check_pose_difference(self.env._get_ef_pose(mat=True), END_POINT, tolerance=0.04)
            print(f"{view_name} checker: {checker}")
            save_observation_images(save_path, self.env, self.placed_obj, multiview_index, visual=False)
            self.env._panda.reset()
        
        print('==================finish multiview data=====================')
        print('left_joint_list:', self.left_joint_list)
        print('right_joint_list:', self.right_joint_list)

        for i, index in enumerate(self.placed_obj.keys()):
            placed_name = self.placed_obj[index]
            print(f"Processing {placed_name}...")
            # 清空列表來儲存下一個物體的數據rgbd
            obj_rgb = []
            obj_depth = []
            obj_mask = []

            for multiview_name, multiview_index in views_dict.items():
                # load特定物體的rgbd
                rgb, depth, mask = load_data(save_path, placed_name, multiview_index)
                obj_rgb.append(rgb)
                obj_depth.append(depth)
                obj_mask.append(mask)

            # 將特定物體的rgbd存入列表
            self.rgb_list.append(obj_rgb)
            self.depth_list.append(obj_depth)
            self.mask_list.append(obj_mask)

    def find_closet_target(self):
        # check the target object 誰距離init camera最近就選誰
        min_depth = np.inf
        closest_target_index = None

        # 遍歷每個目標物體
        for i, index in enumerate(self.placed_obj.keys()):
            # camera_transform = np.eye(4)因為用init camera的座標系去比較
            target_full, target_segments = create_point_cloud_and_camera(self.rgb_list[i][0], self.depth_list[i][0], self.mask_list[i][0], K = self.intrinsic_matrix)  
            target_segments = np.array(target_segments.points)
            # 使用目標物體對應的深度資訊中的最小值來計算目標深度
            target_depth = np.min(target_segments[:, 2])
            print(f"target_closet_depth: {target_depth}")
            # 如果這個目標物體比之前記錄的更接近相機，更新最小深度和索引
            if target_depth < min_depth:
                min_depth = target_depth
                # target_object 和 closest_target_index 不同
                closest_target_index = index
                target_object = i

        print("最接近相機的目標物體索引是：", closest_target_index)
        return closest_target_index, target_object

    def get_multiview_pcd(self, target_object):

        for multiview_index in range(3):
            rgb, depth, mask = self.rgb_list[target_object][multiview_index], self.depth_list[target_object][multiview_index], self.mask_list[target_object][multiview_index]
            
            if multiview_index == 0:
                camera_transform = np.linalg.inv(self.origin_camera2world)
                color = None
                pc_full_pcd_0, pc_segments_pcd_0 = create_point_cloud_and_camera(rgb, depth, mask, self.intrinsic_matrix, camera_transform, color)
            elif multiview_index == 1:
                camera_transform = np.linalg.inv(self.left_camera2world)
                color = None
                pc_full_pcd_1, pc_segments_pcd_1 = create_point_cloud_and_camera(rgb, depth, mask, self.intrinsic_matrix, camera_transform, color)
            else:
                camera_transform = np.linalg.inv(self.right_camera2world)
                color = None
                pc_full_pcd_2, pc_segments_pcd_2 = create_point_cloud_and_camera(rgb, depth, mask, self.intrinsic_matrix, camera_transform, color)
            

        robot_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        camera0 = getCamera(np.linalg.inv(self.origin_camera2world), length=0.1, color=[1, 0, 0])
        camera1 = getCamera(np.linalg.inv(self.left_camera2world), length=0.1, color=[0, 1, 0])
        camera2 = getCamera(np.linalg.inv(self.right_camera2world), length=0.1, color=[0, 0, 1])

        camera = [*camera0, *camera1, *camera2]

        if self.vis_pcd:
            # o3d.visualization.draw_geometries_with_animation_callback([*camera, robot_base, pc_segments_pcd_0, pc_segments_pcd_1, pc_segments_pcd_2], rotate_view)
            # o3d.visualization.draw_geometries_with_animation_callback([*camera, robot_base, pc_full_pcd_0, pc_full_pcd_1, pc_full_pcd_2], rotate_view)
            o3d.visualization.draw_geometries([*camera, robot_base, pc_segments_pcd_0, pc_segments_pcd_1, pc_segments_pcd_2])
            o3d.visualization.draw_geometries([*camera, robot_base, pc_full_pcd_0, pc_full_pcd_1, pc_full_pcd_2])

        #  
        pc_segments_combine_pcd = get_combine_pcd(pc_segments_pcd_0, pc_segments_pcd_1, pc_segments_pcd_2)
        pc_full_combine_pcd = get_combine_pcd(pc_full_pcd_0, pc_full_pcd_1, pc_full_pcd_2)
        pc_segments_combine_pcd.transform(self.origin_camera2world)
        pc_full_combine_pcd.transform(self.origin_camera2world)

        # save the combine point cloud as npy
        self.pc_segments_combine = np.asarray(pc_segments_combine_pcd.points)
        self.pc_full_combine = np.asarray(pc_full_combine_pcd.points)

        np.save(os.path.join(self.parent_directory, 'multiview_data/pc_segments_combine.npy'), self.pc_segments_combine)
        np.save(os.path.join(self.parent_directory, 'multiview_data/pc_full_combine.npy'), self.pc_full_combine)
        self.pc_full_combine_colors = np.asarray(pc_full_combine_pcd.colors)
        np.save(os.path.join(self.parent_directory, 'multiview_data/pc_full_combine_colors.npy'), self.pc_full_combine_colors)
        
        self.pc_full_combine_noise = augment_depth_realsense(self.pc_full_combine, coefficient_scale=1)
        self.pc_segments_combine_noise = augment_depth_realsense(self.pc_segments_combine, coefficient_scale=1)
        np.save(os.path.join(self.parent_directory, 'multiview_data/pc_full_combine_noise.npy'), self.pc_full_combine_noise)
        np.save(os.path.join(self.parent_directory, 'multiview_data/pc_segments_combine_noise.npy'), self.pc_segments_combine_noise)
        # use open3d to visualize the augmented point cloud
        pc_full_combine_noise_pcd = o3d.geometry.PointCloud()
        pc_full_combine_noise_pcd.points = o3d.utility.Vector3dVector(self.pc_full_combine_noise)
        pc_full_combine_noise_pcd.colors = o3d.utility.Vector3dVector(pc_full_combine_pcd.colors)
        pc_full_combine_noise_pcd.transform(np.linalg.inv(self.origin_camera2world))
        if self.vis_pcd:
            o3d.visualization.draw_geometries([pc_full_combine_noise_pcd, robot_base, *camera])
        self.pc_segments_path = os.path.join(self.parent_directory, 'multiview_data/pc_segments_combine.npy')
        self.pc_segments_noise_path = os.path.join(self.parent_directory, 'multiview_data/pc_segments_combine_noise.npy')
    
    def gernerate_grasp(self):

        obs, _, _, _, _ = self.env._get_observation()
        color_image = obs[1][:3].T
        depth_image = obs[1][3].T
        mask_image = obs[1][4].T 
        bridge = CvBridge()
        color_msg = bridge.cv2_to_imgmsg(np.uint8(color_image), encoding='bgr8')
        depth_msg = bridge.cv2_to_imgmsg(depth_image)
        mask_msg = bridge.cv2_to_imgmsg(np.uint8(mask_image), encoding='mono8')    # Assuming mask is grayscale

        # Create a service request(depth image part)
        contact_request = GraspGroupRequest()
        contact_request.rgb = color_msg
        contact_request.depth = depth_msg
        contact_request.seg = mask_msg
        contact_request.K = self.intrinsic_matrix.flatten()  # Flatten the 3x3 matrix into a 1D array
        contact_request.segmap_id = 1
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base_link'  # Replace with your desired frame ID
        contact_request.pc_full = create_cloud_xyz32(header, self.pc_full_combine)
        contact_request.pc_target = create_cloud_xyz32(header, self.pc_segments_combine)
        contact_request.mode = 1
        grasp_poses = self.contact_client(contact_request).grasp_poses
        # 提取grasp_poses list中的pred_grasps_cam和scores
        grasp_list = []
        scores_list = []

        for grasp_pose in grasp_poses:
            grasp_list.append(np.array(grasp_pose.pred_grasps_cam).reshape(4, 4))
            scores_list.append(grasp_pose.score)
        # save the pred_grasps_cam and scores as npy
        np.save(os.path.join(self.parent_directory, 'results/pred_grasps_cam.npy'), np.array(grasp_list))
        np.save(os.path.join(self.parent_directory, 'results/scores.npy'), np.array(scores_list))
    
    def get_target_6d_pose(self, closest_target_index, gt=True):
        self.poseEstimate = PointCloudPoseEstimator(self.pc_segments_path, self.pc_segments_noise_path, self.init_ef_mat, self.cam_offset)
        self.env.target_idx = closest_target_index
        if gt == True:
            # ground truth
            self.target_pose_world = self.env._get_ef_pose(mat=True)@ self.env._get_target_relative_pose(option = 'ef')
            # rotate the target pose if the x of the target pose inner product with the x of the ef pose is negative
            print("===========target!!!!!!!!!!", self.target_placed_name)
            if self.target_placed_name == "025_mug_1.0" or self.target_placed_name == '021_bleach_cleanser_1.0':
            # if "025_mug_1.0" in self.placed_name or '021_bleach_cleanser_1.0' in self.placed_name:
                # rotate 90 degree around z axis of the target pose
                self.target_pose_world[:3, :3] = self.target_pose_world[:3, :3]@ np.array([[0, -1, 0],
                                                                                [1, 0, 0],
                                                                                [0, 0, 1]])
                print("rotate 90 degree around z axis of the target pose")
                self.result_z_rotation_angle += 90
            if np.dot(self.target_pose_world[:3, 0], self.env._get_ef_pose(mat=True)[:3, 2]) > 0:
                # rotate 180 degree around z axis of the target pose
                self.target_pose_world[:3, :3] = self.target_pose_world[:3, :3]@ np.array([[-1, 0, 0],
                                                                                [0, -1, 0],
                                                                                [0, 0, 1]])
                print("rotate 180 degree around z axis of the target pose")
                self.result_z_rotation_angle += 180
            print("result_z_rotation_angle", self.result_z_rotation_angle)
        else:
            self.target_pose_world = self.poseEstimate.get_6d_pose()

        print(self.target_pose_world)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # test 005_tomato_soup_can_1.0
        if self.target_placed_name == "005_tomato_soup_can_1.0":
            # 當前self.target_pose_world[:3, 0]和[0, 0, 1]利用旋轉z軸的方式讓他們一樣
            target_vector = self.target_pose_world[:3, 0]      
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(target_vector, z_axis)
            cos_theta = np.dot(target_vector, z_axis) / (np.linalg.norm(target_vector) * np.linalg.norm(z_axis))
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))      
            c = np.cos(angle)
            s = np.sin(angle)
            t = 1 - c
            
            x, y, z = rotation_axis / np.linalg.norm(rotation_axis)
            
            rotation_matrix = np.array([
                [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
                [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
                [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
            ])
            
            # 應用旋轉矩陣
            self.target_pose_world[:3, :3] = rotation_matrix @ self.target_pose_world[:3, :3]

            # 依據此ratation matrix依據z軸旋轉特定角度
            henry_angle = -0
            henry_angle = henry_angle * np.pi / 180
            rot_z = rotZ(henry_angle)
            self.target_pose_world = self.target_pose_world @ rot_z

        if self.vis_draw_coordinate:
            # can see the target pose, cabinet pose and ef pose in pybullet
            self.env.draw_ef_coordinate(self.env._get_ef_pose(mat=True), 5)
            self.env.draw_ef_coordinate(self.env._get_target_urdf_pose(option = 'cabinet_world', mat = True), 5)
            self.env.draw_ef_coordinate(self.target_pose_world, 5)

        

    def get_the_target_on_cabinet_pose(self):
        self.cabinet_pose_world = self.env._get_target_urdf_pose(option = 'cabinet_world', mat = True)
        self.env.draw_ef_coordinate(self.cabinet_pose_world, 5)
        cabinet_pose_world_stage1 = self.cabinet_pose_world.copy()
        cabinet_pose_world_stage2 = self.cabinet_pose_world.copy()
        # # stage1 
        # cabinet_pose_world_stage1[0, 3] += 0.
        # cabinet_pose_world_stage1[1, 3] += -0.3
        # cabinet_pose_world_stage1[2, 3] += 0.2

        # # stage2
        # cabinet_pose_world_stage2[0, 3] += 0.
        # cabinet_pose_world_stage2[1, 3] += -0.3
        # cabinet_pose_world_stage2[2, 3] += 0.4

        # stage1 
        cabinet_pose_world_stage1[0, 3] =0.75
        cabinet_pose_world_stage1[1, 3] += 0.
        cabinet_pose_world_stage1[2, 3] =0.25

        # stage2
        cabinet_pose_world_stage2[0, 3] =0.75
        cabinet_pose_world_stage2[1, 3] += 0.
        cabinet_pose_world_stage2[2, 3] =0.6

        # 利用self.placing_location來決定物體的放置位置y
        print('******************count_location = ', self.count_location)
        cabinet_pose_world_stage1[1, 3] += self.placing_location[self.count_location]
        cabinet_pose_world_stage2[1, 3] += self.placing_location[self.count_location]
        print('******************cabinet_pose_world_stage1 = ', cabinet_pose_world_stage1)
        print('******************cabinet_pose_world_stage2 = ', cabinet_pose_world_stage2)
        
            

        # chose the cabinet pose
        if self.placing_stage == 1:
            self.cabinet_pose_world = cabinet_pose_world_stage1
        elif self.placing_stage == 2:
            self.cabinet_pose_world = cabinet_pose_world_stage2
        
        z_translation = self.poseEstimate.get_normal_translation()
        y_translation = -0.
        x_translation = 0.
        print('z_translation = {}'.format(z_translation))
        self.place_pose_world = self.cabinet_pose_world.copy()
        self.place_pose_world[:3, 3] += np.array([x_translation, y_translation, z_translation])
        if self.vis_draw_coordinate:
            self.env.draw_ef_coordinate(self.cabinet_pose_world, 5)
            self.env.draw_ef_coordinate(self.place_pose_world, 5)

    def tranfrom_grasp_to_place(self):
        pred_grasps_cam = np.load(os.path.join(self.parent_directory, 'results/pred_grasps_cam.npy'), allow_pickle=True)
        scores = np.load(os.path.join(self.parent_directory, 'results/scores.npy'), allow_pickle=True)
        if len(pred_grasps_cam) == 0:
            print("No valid grasp found")
        else:
            new_pred_grasps_cam = self.init_ef_mat@ np.linalg.inv(self.cam_offset)@ pred_grasps_cam
        new_pred_grasps_cam_place = np.zeros_like(new_pred_grasps_cam)
        for i in range(new_pred_grasps_cam.shape[0]):
            relative_grasp_transform = np.linalg.inv(self.target_pose_world)@ new_pred_grasps_cam[i]
            new_pred_grasps_cam_place[i] = self.place_pose_world@ relative_grasp_transform
            #?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？?？
            new_pred_grasps_cam_place[i][2, 2] -= 0.03
        np.save(os.path.join(self.parent_directory, 'results/new_pred_grasps_cam_place.npy'), new_pred_grasps_cam_place)

    def tranfrom_place_to_grasp(self):
        relative_grasp_transform = np.linalg.inv(self.place_pose_world)@ self.final_place_grasp_pose
        self.final_grasp_pose = self.target_pose_world@ relative_grasp_transform
        print("final_grasp_pose = \n", self.final_grasp_pose)

    def execute_motion_and_check_pose(self, END_POINT, tolerance=0.04, gripper_state="open", repeat=150, visual_time=None):
        """
        執行動作計畫並檢查最後位姿是否正確。不正確則停5秒
        """
        execute = True
        if self.vis_draw_coordinate:
            self.env.draw_ef_coordinate(END_POINT, 5)
        plan = self.expert_plan(pack_pose(END_POINT), world=True, visual=False)
        _ = self.execute_motion_plan(plan, execute=execute, gripper_set=gripper_state, repeat=repeat)
        checker = check_pose_difference(self.env._get_ef_pose(mat=True), END_POINT, tolerance)
        if not checker:
            print("位姿不正確，請檢查位姿")
            time.sleep(5)
        print("=====================================================")

    def execute_robot_motion(self):
        # set the poses of the robot
        self.final_grasp_pose_z_bias = adjust_pose_with_bias(self.final_grasp_pose, -0.1, option="ef")
        if self.placing_stage == 1:
            self.mid_retract_pose = rotZ(-np.pi/2)@ transZ(0.50)@ transX(0.3)@ transY(0.3)@ np.eye(4)@ rotZ(np.pi/4*3)@ rotX(np.pi/4*3)
        elif self.placing_stage == 2:
            self.mid_retract_pose = rotZ(-np.pi/2)@ transZ(0.85)@ transX(0.3)@ transY(0.3)@ np.eye(4)@ rotZ(np.pi/4*3)@ rotX(np.pi/4*3)

        if self.placing_stage == 1:
            self.final_place_pose_z_bias_top = adjust_pose_with_bias(self.final_place_grasp_pose, 0.03, option="world")
            self.final_place_pose_z_bias_top = adjust_pose_with_bias(self.final_place_pose_z_bias_top, -0.0, option="world_x")
        elif self.placing_stage == 2:
            self.final_place_pose_z_bias_top = adjust_pose_with_bias(self.final_place_grasp_pose, 0.03, option="world")
        self.final_place_pose_z_bias_placing = adjust_pose_with_bias(self.final_place_grasp_pose, 0.015, option="world")
        self.final_place_pose_z_bias_release = adjust_pose_with_bias(self.final_place_grasp_pose, -0.05, option="ef")
        
        if(self.count_location == 2):
            self.placing_stage = 1 if self.placing_stage == 2 else 2
            self.count_location = 0
        else:
            self.count_location += 1

        # start execute the setting poses
        self.execute_motion_and_check_pose(self.final_grasp_pose_z_bias, gripper_state="open")
        img = ImageGrab.grab()
        random = np.random.randint(0, 1000)
        img.save(os.path.join(self.parent_directory, f'results/scene_grasp_{random}.png'))
        time.sleep(1)
        for i in range(10):
            _ = self.env.step([0, 0, 0.1/10, 0, 0, 0], repeat=100)
        self.reward_checker = self.retract()
        if not self.reward_checker:
            self.grasp_fail_rate += 1
            return
        self.execute_motion_and_check_pose(self.mid_retract_pose, tolerance = 0.1, gripper_state="close", repeat=100)
        img = ImageGrab.grab()
        random = np.random.randint(0, 1000)
        img.save(os.path.join(self.parent_directory, f'results/scene_mid_{random}.png'))
        self.execute_motion_and_check_pose(self.final_place_pose_z_bias_top, gripper_state="close")
        self.execute_motion_and_check_pose(self.final_place_pose_z_bias_placing, gripper_state="close", repeat=100)
        move_gripper_smoothly(self.env, p, 0.0, 0.085)  # open
        self.execute_motion_and_check_pose(self.final_place_pose_z_bias_release, gripper_state="open", repeat=50)
        img = ImageGrab.grab()
        random = np.random.randint(0, 1000)
        img.save(os.path.join(self.parent_directory, f'results/scene_place_{random}.png'))

    def retract(self):
        """
        Move the arm to lift the object.
        """
        reward = 0
        cur_joint = np.array(self.env._panda.getJointStates()[0])
        cur_joint[-1] = 0.8  # close finger
        observations = [self.env.step(cur_joint, repeat=300, config=True, vis=False)[0]]
        pos, orn = p.getLinkState(self.env._panda.pandaUid, self.env._panda.pandaEndEffectorIndex)[4:6]

        for i in range(10):
            pos = (pos[0], pos[1], pos[2] + 0.02)
            jointPoses = np.array(p.calculateInverseKinematics(self.env._panda.pandaUid,
                                                               self.env._panda.pandaEndEffectorIndex, pos,
                                                               maxNumIterations=500,
                                                               residualThreshold=1e-8))
            jointPoses[6] = 0.85
            jointPoses = jointPoses[:7].copy()
            obs = self.env.step(jointPoses, config=True)[0]
        self.reward_checker = self.grasp_reward(self.env.target_idx)
        if self.reward_checker:
            print("Grasp successfully")
            reward = 1
            return reward
        else:
            print("Grasp failed")
            return reward

    
    def grasp_reward(self, closest_target_index):
        '''
        check for success grasping or nor
        success: return True
        fail: return False
        '''
        self.env.target_idx = closest_target_index
        init_target_height = self.target_pose_world[2, 3]
        end_target_height = self.env._get_ef_pose(mat=True)@ self.env._get_target_relative_pose(option = 'ef')
        if end_target_height[2, 3] - init_target_height > 0.08:
            return True
        return False


    def final_reward(self, closest_target_index):
        '''
        check for success placing or nor
        success: return True
        fail: return False
        '''
        self.env.target_idx = closest_target_index
        target_final = self.env._get_ef_pose(mat=True)@ self.env._get_target_relative_pose(option = 'ef')
        # check the z axis of the target_final with the [0, 0, 1] vector
        z_axis_target = target_final[:3, 2]
        z_axis_target = z_axis_target/np.linalg.norm(z_axis_target)
        cos_theta = np.dot(np.array([0, 0, 1]), z_axis_target)
        if cos_theta < 0.9:
            print("The target is not vertical to the ground")
            return False
        else:   
            print("The target is vertical to the ground")
            return True
    
    def object_record_init(self):
        file = os.path.join(self.parent_directory, "object_index", 'contact_plane_object.json')
        with open(file) as f:
            file_dir = json.load(f)
        file_dir = file_dir['test_dataset']
        file_dir = [f[:-5] for f in file_dir]
        test_file_dir = list(set(file_dir))
        self.success_rates = {}
        for filename in test_file_dir:
            # 將標準化後的名稱用來初始化計數
            self.success_rates[filename] = {"total": 0, "successful": 0, "no_grasp_pose": 0}

    def object_record(self, target_name, no_grasp=False, success_bool=False):

        self.success_rates[target_name]["total"] += 1
        if success_bool:         
            self.success_rates[target_name]["successful"] += 1
        if no_grasp:
            self.success_rates[target_name]["no_grasp_pose"] += 1

        with open(os.path.join(self.parent_directory, 'results/obejct_record.txt'), 'a') as f:
            f.write(f"target_name: {target_name}\n")
            # save the stage and location
            f.write(f"stage: {self.placing_stage}; location: {self.count_location}\n")
            f.write(f"total: {self.success_rates[target_name]['total']}\n")
            f.write(f"successful: {self.success_rates[target_name]['successful']}\n")
            f.write(f"no_grasp_pose: {self.success_rates[target_name]['no_grasp_pose']}\n")
            f.write(f"success_rate: {self.success_rates[target_name]['successful']/self.success_rates[target_name]['total']}\n")
            f.write("=====================================================\n")
    
    def remove_target_object(self):
        # remove the self.env.target_idx
        self.placed_obj.pop(self.env.target_idx)
        print("================", self.placed_obj)
        self.env._reset_placed_objects()
        print("===================", self.placed_obj)

    def run(self):
        self.grasp_fail_rate = 0
        success_rate = 0
        no_grasp_pose_rate = 0
        while not rospy.is_shutdown():
            self.load_environment()
            test_index = 0
            for i in range(1, self.test_num + 1):
                self.initial()
                self.check_scene()
                
                for j in range (self.num_object):
                    self.initial()
                    test_index += 1
                    self.get_multiview_data()
                    closest_target_index, target_object = self.find_closet_target()
                    self.target_placed_name = self.placed_obj[closest_target_index]
                    time.sleep(3) 
                    tcp_utils.send_target_name('127.0.0.1', 19111, self.target_placed_name)
                    time.sleep(3)
                    self.get_multiview_pcd(target_object)
                    self.gernerate_grasp()
                    self.get_target_6d_pose(closest_target_index)
                    self.get_the_target_on_cabinet_pose()
                    self.tranfrom_grasp_to_place()
                    time.sleep(4)
                    tcp_utils.send_checker('127.0.0.1', 12346) 
                    time.sleep(3) 
                    final_place_target_matrix = self.place_pose_world@ get_rotation_matrix_z_4x4(-self.result_z_rotation_angle/90)
                    tcp_utils.send_matrix('127.0.0.1', 33333, final_place_target_matrix)
                    time.sleep(1)
                    tcp_utils.send_matrix('127.0.0.1', 55557, self.place_pose_world)
                    time.sleep(1)
                    tcp_utils.send_matrix('127.0.0.1', 56471, self.target_pose_world)
                    self.final_place_grasp_pose = tcp_utils.start_server('127.0.0.1', 44412)
                    if np.array_equal(self.final_place_grasp_pose, np.eye(4)):
                        print("The placing mode is not success")
                        no_grasp_pose_rate += 1
                        self.object_record(self.target_placed_name, no_grasp=True)
                        self.remove_target_object()
                        with open(os.path.join(self.parent_directory, 'results/ros_execution_final.txt'), 'a') as f:
                            f.write(f"Run: {test_index}; target_name: {self.target_placed_name}\n")
                            f.write(f"no_grasp_pose_in_placing_checker: {no_grasp_pose_rate}/{test_index}; rate: {no_grasp_pose_rate/test_index}\n")
                            f.write(f"grasp_fail_rate: {self.grasp_fail_rate}/{test_index}; rate: {self.grasp_fail_rate/test_index}\n")
                            f.write(f"place_success_count: {success_rate}/{test_index}; rate: {success_rate/test_index}\n")
                            # f.write(f"place_success_except_no_grasp: {success_rate}/{i-no_grasp_pose_rate}\n")    
                            f.write("=====================================================\n")
                        continue
                    self.tranfrom_place_to_grasp()
                    self.env._panda.reset()
                    self.execute_robot_motion()
                    success = self.final_reward(closest_target_index)
                    if success:
                        success_rate += 1
                    self.object_record(self.target_placed_name, success_bool=success)
                    self.remove_target_object()
                    

                    # save in the txt file
                    with open(os.path.join(self.parent_directory, 'results/ros_execution_final.txt'), 'a') as f:
                        f.write(f"Run: {test_index}; target_name: {self.target_placed_name}\n")
                        f.write(f"no_grasp_pose_in_placing_checker: {no_grasp_pose_rate}/{test_index}; rate: {no_grasp_pose_rate/test_index}\n")
                        f.write(f"grasp_fail_rate: {self.grasp_fail_rate}/{test_index}; rate: {self.grasp_fail_rate/test_index}\n")
                        f.write(f"place_success_count: {success_rate}/{test_index}; rate: {success_rate/test_index}\n")
                        # f.write(f"place_success_except_no_grasp: {success_rate}/{i-no_grasp_pose_rate}; rate: {success_rate/(i-no_grasp_pose_rate)}\n")    
                        f.write("=====================================================\n")

            with open(os.path.join(self.parent_directory, 'results/ros_execution_final.txt'), 'a') as f:
                f.write(f"=======total_success_rate: {success_rate}/{test_index}; rate: {success_rate/test_index}=======\n")
                f.write(f"place_success_except_no_grasp: {success_rate}/{test_index-no_grasp_pose_rate}; rate: {success_rate/(test_index-no_grasp_pose_rate)}\n")    
                f.write("=====================================================\n")

if __name__ == '__main__':
    rospy.init_node('robot_grasp_node', anonymous=True)
    robot = ExecutionNode()
    robot.run()
    rospy.spin()