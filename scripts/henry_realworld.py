#!/usr/bin/env python3
import numpy as np
import os
import sys
import time
import open3d as o3d
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utils import *
# from replay_buffer import ReplayMemoryWrapper
from ros_placing_final_realworld import PlacingNode
import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo, JointState
from std_msgs.msg import Int32
import std_msgs
import tf
import tf2_ros
import itertools
from tf.transformations import quaternion_matrix
from cv_bridge import CvBridge
from pybullet_ros.srv import GraspGroup, GraspGroupRequest
from pybullet_ros.msg import Robotiq2FGripper_robot_output
from sensor_msgs.point_cloud2 import create_cloud_xyz32
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Pose, TransformStamped
from utils.grasp_checker import ValidGraspChecker


class ros_node(object):
    def __init__(self, renders):
        self.actor = PlacingNode(renders=renders)
        self.start_sub = rospy.Subscriber("test_realworld_cmd", Int32, self.get_env_callback)
        self.joint_sub = rospy.Subscriber("joint_states", JointState, self.joint_callback)
        self.tm_pub = rospy.Publisher("/target_position", Pose, queue_size=1)
        self.tm_joint_pub = rospy.Publisher("/target_joint", JointState, queue_size=1)
        self.robotiq_pub = rospy.Publisher("/Robotiq2FGripperRobotOutput", Robotiq2FGripper_robot_output, queue_size=10)
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)  # Create a tf listener

        self.points_sub = rospy.Subscriber("/uoais/Pointclouds", PointCloud2, self.points_callback)
        self.obs_points_sub = rospy.Subscriber("/uoais/obs_pc", PointCloud2, self.obs_points_callback)
        self.seg_pub = rospy.Publisher("/uoais/data_init", Int32, queue_size=1)

        self.contact_client = rospy.ServiceProxy('contact_graspnet/get_grasp_result', GraspGroup)
        rospy.wait_for_service('contact_graspnet/get_grasp_result', timeout=None)
        self.target_points = None
        self.obs_points = None
        self.home_joint_point = [[0.2-np.pi/2, -1, 2, 0, 1.571, 0.0]]
        self.second_multi_view = [0.2-np.pi/2, -1, 2, 0, 1.571, 0.0]
        self.third_multi_view = [0.2-np.pi/2, -1, 2, 0, 1.571, 0.0]
        self.left_joint_list = [[-0.7347407781434788, 0.5118680836287178, 1.1827692285964984, 0.4877031591085841, 0.9902222157305788, 2.530426744504708]]
        self.right_joint_list = [[-2.1427776820230466, 0.4583241646482387, 0.7837185076979353, 0.7192889843407846, 2.2248681578861396, 0.8679910446001847]]

        self.vis_pcd = True

        self.placing_stage = None
        self.mid_retract_pose = None
        self.envir_joint = [1.6208316641972509-np.pi/2, -0.98, 1.46939, -0.119555, 1.63676, -0.05]
        rospy.loginfo("Init finished")


    def joint_callback(self, msg):
        cur_states = np.asarray(msg.position)
        cur_states= np.concatenate((cur_states, [0, 0, 0]))
        self.joint_states = cur_states

    def get_multiview_data(self, angle_list=None):
        # 初始化 multiview_pc 来存储所有角度的点云数据
        self.multiview_pc_obs_base = []
        self.multiview_pc_target_base = []

        # Get the pointcloud data in three degree and concatenate them
        # home joint放最後是為了移回camera的位置才可生成contact grasp
        angle_list = [self.left_joint_list, self.right_joint_list, self.home_joint_point]
        
        for multiview_joint in angle_list:
            # Reset the arm's position
            self.move_along_path(multiview_joint)

            # Set init_value to None
            self.target_points = None
            self.obs_points = None
            
            # Segmentation part
            seg_msg = Int32()
            seg_msg.data = 2
            self.seg_pub.publish(seg_msg)    
            time.sleep(2) # Sleep to wait for the segmentation pointcloud arrive

            print(f"self.obs_points: {self.obs_points}")

            self.obs_points_base = self.pc_cam2base(self.obs_points)
            self.target_points_base = self.pc_cam2base(self.target_points)

            # Append the transformed pointclouds to the multiview_pc lists
            if self.obs_points_base is not None:
                self.multiview_pc_obs_base.append(self.obs_points_base)
            if self.target_points_base is not None:
                self.multiview_pc_target_base.append(self.target_points_base)
                print(self.target_points_base.shape)

            if self.vis_pcd == True:
                self.visual_pc(self.target_points_base)
            self.move_along_path(self.home_joint_point)

            

        # Concatenate all pointclouds in multiview_pc_obs and multiview_pc_target
        if self.multiview_pc_obs_base:
            self.multiview_pc_obs_base = np.concatenate(self.multiview_pc_obs_base, axis=0)
        if self.multiview_pc_target_base:
            self.multiview_pc_target_base = np.concatenate(self.multiview_pc_target_base, axis=0)
        
        # transform the pointclouds to the camera frame
        self.multiview_pc_obs = self.pc_base2cam(self.multiview_pc_obs_base)
        self.multiview_pc_target = self.pc_base2cam(self.multiview_pc_target_base)

        # Visualize the concatenated pointclouds
        if self.vis_pcd == True:
            self.visual_pc(self.multiview_pc_obs_base)
            self.visual_pc(self.multiview_pc_target_base)
        print(self.multiview_pc_target_base.shape)
        # self.visual_pc(self.multiview_pc_obs)
        # self.visual_pc(self.multiview_pc_target)
        
        # save the pointclouds as npy file
        np.save("/home/user/henry_pybullet_ws/src/pybullet_ros/realworld_data/multiview_pc_obs.npy", self.multiview_pc_obs_base)
        np.save("/home/user/henry_pybullet_ws/src/pybullet_ros/realworld_data/multiview_pc_target.npy", self.multiview_pc_target_base)
        np.save("/home/user/henry_pybullet_ws/src/pybullet_ros/realworld_data/multiview_pc_obs_cam.npy", self.multiview_pc_obs)
        np.save("/home/user/henry_pybullet_ws/src/pybullet_ros/realworld_data/multiview_pc_target_cam.npy", self.multiview_pc_target)

    def get_the_target_on_cabinet_pose(self):
        self.placing_location = [0.15, 0, -0.15]
        if self.placing_stage == 1:
            place_pose_base = np.array([[-1, 0.,  0.,   0.7],
                                        [-0., -1,  0.,  0.0],
                                        [ 0.,  0.,  1.,  0.05],
                                        [ 0. ,         0. ,         0.,          1.        ]]
                                        )
        elif self.placing_stage == 2:
            place_pose_base = np.array([[-1, 0.,  0.,   0.7],
                                        [-0., -1,  0.,  0.0],
                                        [ 0.,  0.,  1.,  0.35],
                                        [ 0. ,         0. ,         0.,          1.        ]]
                                        )
        # 微調place pose
        place_pose_base[0, 3] += -0.1
        place_pose_base[1, 3] += self.placing_location[1]
        place_pose_base[2, 3] += self.get_normal_translation() 
        # place_pose_base[2, 3] += self.target_center[2]
        print('z_translation = {}'.format(self.get_normal_translation()))
        return place_pose_base

    def get_oriented_bounding_box(self):
        self.pc_segments_pcd = o3d.geometry.PointCloud()
        self.pc_segments_pcd.points = o3d.utility.Vector3dVector(self.multiview_pc_target)
        self.pc_segments_pcd.paint_uniform_color([0.0, 0.0, 0.0])
        self.pc_segments_pcd.estimate_normals()
        obb = self.pc_segments_pcd.get_oriented_bounding_box()
        obb.color = (1, 0, 0) 
        # if self.vis_pcd:
        o3d.visualization.draw_geometries([self.pc_segments_pcd, obb])
        return obb


    def get_normal_translation(self):
        obb = self.get_oriented_bounding_box()
        x_vec = obb.get_box_points()[0] - obb.get_box_points()[1]
        y_vec = obb.get_box_points()[0] - obb.get_box_points()[2]
        z_vec = obb.get_box_points()[0] - obb.get_box_points()[3]
        x_length = np.linalg.norm(obb.get_box_points()[0] - obb.get_box_points()[1])
        y_length = np.linalg.norm(obb.get_box_points()[0] - obb.get_box_points()[2])
        z_length = np.linalg.norm(obb.get_box_points()[0] - obb.get_box_points()[3])

        #define the x,y,z axis length in array and get the index number of the min length
        vector = np.array([x_vec, y_vec, z_vec])
        length = np.array([x_length, y_length, z_length])
        print('x = {}(m)\ny = {}(m)\nz = {}(m)'.format(length[0], length[1], length[2]))

        min_length_index = np.argmin(length)
        max_length_index = np.argmax(length)

        target_z_translation = length[max_length_index]/2
        return target_z_translation
    
    def get_env_callback(self, msg):
        if msg.data == 0:
            self.placing_stage = 2   # 放最上面
            self.mid_retract_pose = rotZ(-np.pi/2)@ transZ(0.65)@ transX(0.3)@ transY(0.3)@ np.eye(4)@ rotZ(np.pi/4*3)@ rotX(np.pi/4*3)
            # Reset the gripper
            self.control_gripper("reset")
            time.sleep(1)
            self.control_gripper("set_pose", 0.)
            time.sleep(1)
            self.control_gripper("set_pose", 0.085)
            print(f"finish grasping")

            # Pybullet setup
            self.actor.load_environment()
            self.actor.env._panda.reset(self.home_joint_point[0]+[0, 0, 0])
            self.actor.initial()
            self.actor.grasp_checker = ValidGraspChecker(self.actor.env)

            # multi-view data
            self.get_multiview_data()
            print(f"self.obs_points: {self.obs_points}")
            print("***********Finish get multiview data*************\n")

            # step 1: get the stable plane (6d pose) on target object
            '''
            # get the target pose (stable plane on target object)
            # self.target_pose_base = stable_plane_cvae_net
            '''
            # # pack object
            # target_pose_base = np.eye(4)
            # # get the center of the multiview_pc_target_base
            # self.target_center = np.mean(self.multiview_pc_target_base, axis=0)
            # target_pose_base[:3, 3] = self.target_center[:3]

            # 平躺object
            target_pose_base = np.eye(4)
            self.target_center = np.mean(self.multiview_pc_target_base, axis=0)
            target_pose_base[:3, 3] = self.target_center[:3]
            target_pose_base[:3, 0] = np.array([0, 0, 1])
            target_pose_base[:3, 1] = np.array([-1, 0, 0])
            target_pose_base[:3, 2] = np.array([0, -1, 0])
            # 手動旋轉x軸
            target_pose_base = target_pose_base@ rotX(np.pi/4)
            

            # get the place pose (can change to other place pose)
            # place_pose_base = self.actor.get_the_target_on_cabinet_pose()
            # get it from apriltag_inform.py
            place_pose_base = self.get_the_target_on_cabinet_pose()
            print("***********Finish get target pose*************\n")

            # step 2: get the contact grasp
            '''
            # 要放camrea frame的點雲來去生成contact grasp
            # grasp_poses_camera = self.setting_contact_req(obstacle_points=self.obs_points, target_points=self.target_points)
            '''
            grasp_poses_camera = self.setting_contact_req(obstacle_points=self.obs_points, target_points=self.target_points)

            # self.visual_pc(obs_points_base)
            
            self.grasp_list = []
            self.score_list = []
            for grasp_pose_cam in grasp_poses_camera:
                grasp_camera = np.array(grasp_pose_cam.pred_grasps_cam)
                grasp_world = self.pose_cam2base(grasp_camera.reshape(4,4))
                self.grasp_list.append(grasp_world)
                self.score_list.append(grasp_pose_cam.score)
            # if self.vis_pcd == True:
            self.visualize_points_grasppose(self.multiview_pc_obs_base, self.grasp_list)

            if len(self.grasp_list) == 0:
                print("No valid grasp found")
                return
            print("***********Finish generating grasp poses*************\n")
            
            # step 3: transform the grasp pose to the place pose
            self.grasp_place_list = []
            for grasp_pose in self.grasp_list:
                relative_grasp_transform = np.linalg.inv(target_pose_base)@ grasp_pose
                self.grasp_place_list.append(place_pose_base@ relative_grasp_transform)

            # adjust the raw grasp pose to pre-grasp pose
            grasp_place_list = self.grasp2pre_grasp(self.grasp_place_list, drawback_dis=0.05)
            
            # step 4: grasp pose filter
            self.actor.grasp_pose_checker_base(grasp_place_list)
            self.actor.refine_grasp_place_pose_base(self.score_list)
            succuss_result = self.actor.execute_placing_checker_base(place_pose_base, target_pose_base, self.mid_retract_pose)
            # 判断结果是否为单位矩阵
            if isinstance(succuss_result, np.ndarray) and np.array_equal(succuss_result, np.eye(4)):
                print("Result is the identity matrix.")
                raise ValueError("No valid grasp pose")
            else:
                print("Result is not the identity matrix.")
                success_grasp_pose, success_joint_grasp_list, success_joint_mid_list, success_joint_place_list = succuss_result
                print(f"success_grasp_pose: {success_grasp_pose}\n")
                print(f"success_joint_grasp_list: {success_joint_grasp_list}\n")
                print(f"success_joint_mid_list: {success_joint_mid_list}\n")
                print(f"success_joint_place_list: {success_joint_place_list}\n")

            print("***********Finish grasp poses filtered*************\n")

            # step 5: move to the grasp pose
            self.move_along_path(success_joint_grasp_list)
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] = 0.1
            ef_pose = ef_pose.dot(forward_mat)
            quat_pose = pack_pose(ef_pose)
            final_grasp_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
            self.set_pose(final_grasp_pose[0], final_grasp_pose[1])

            # Close gripper
            time.sleep(1)
            self.control_gripper("set_pose", 0.)
            time.sleep(1)
            print(f"***********Finish robot grasping***********\n")

            # step 6: move to the place pose
            

            # 可以微調place pose!!!!
            self.move_along_path(success_joint_mid_list)
            self.move_along_path(success_joint_place_list)
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] = 0.
            ef_pose = forward_mat.dot(ef_pose)
            quat_pose = pack_pose(ef_pose)
            final_place_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
            self.set_pose(final_place_pose[0], final_place_pose[1])
            time.sleep(2)
            # if self.placing_stage == 1:
            #     # step 6.5: move forward 5cm
            #     ef_pose = self.get_ef_pose()
            #     forward_mat = np.eye(4)
            #     forward_mat[2, 3] += 0.05
            #     ef_pose = forward_mat.dot(ef_pose)
            #     quat_pose = pack_pose(ef_pose)
            #     final_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
            #     self.set_pose(final_pose[0], final_pose[1])
            # Open gripper
            self.control_gripper("set_pose", 0.085)
            print(f"***********Finish robot placing***********\n")
            time.sleep(1)

            # step 7: move back 5cm
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] -= 0.05
            ef_pose = ef_pose.dot(forward_mat)
            quat_pose = pack_pose(ef_pose)
            final_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
            self.set_pose(final_pose[0], final_pose[1])

            # step 8: move back to the home joint
            self.move_along_path(self.home_joint_point)
            print("***********Finish the placing task***********\n")
        
        if msg.data == 11:
            self.placing_stage = 1
            self.mid_retract_pose = rotZ(-np.pi/2)@ transZ(0.45)@ transX(0.3)@ transY(0.3)@ np.eye(4)@ rotZ(np.pi/4*3)@ rotX(np.pi/4*3)
            # Reset the gripper
            self.control_gripper("reset")
            time.sleep(1)
            self.control_gripper("set_pose", 0.)
            time.sleep(1)
            self.control_gripper("set_pose", 0.085)
            print(f"finish grasping")

            # Pybullet setup
            self.actor.load_environment()
            self.actor.env._panda.reset(self.home_joint_point[0]+[0, 0, 0])
            self.actor.initial()
            self.actor.grasp_checker = ValidGraspChecker(self.actor.env)

            # multi-view data
            self.get_multiview_data()
            print(f"self.obs_points: {self.obs_points}")
            print("***********Finish get multiview data*************\n")

            # step 1: get the stable plane (6d pose) on target object
            '''
            # get the target pose (stable plane on target object)
            # self.target_pose_base = stable_plane_cvae_net
            '''
            # # pack object
            # target_pose_base = np.eye(4)
            # # get the center of the multiview_pc_target_base
            # self.target_center = np.mean(self.multiview_pc_target_base, axis=0)
            # target_pose_base[:3, 3] = self.target_center[:3]

            # 平躺object
            target_pose_base = np.eye(4)
            self.target_center = np.mean(self.multiview_pc_target_base, axis=0)
            target_pose_base[:3, 3] = self.target_center[:3]
            target_pose_base[:3, 0] = np.array([0, 0, 1])
            target_pose_base[:3, 1] = np.array([-1, 0, 0])
            target_pose_base[:3, 2] = np.array([0, -1, 0])
            # 手動旋轉x軸
            target_pose_base = target_pose_base@ rotX(np.pi/4)
            

            # get the place pose (can change to other place pose)
            # place_pose_base = self.actor.get_the_target_on_cabinet_pose()
            # get it from apriltag_inform.py
            place_pose_base = self.get_the_target_on_cabinet_pose()
            print("***********Finish get target pose*************\n")

            # step 2: get the contact grasp
            '''
            # 要放camrea frame的點雲來去生成contact grasp
            # grasp_poses_camera = self.setting_contact_req(obstacle_points=self.obs_points, target_points=self.target_points)
            '''
            grasp_poses_camera = self.setting_contact_req(obstacle_points=self.obs_points, target_points=self.target_points)

            # self.visual_pc(obs_points_base)
            
            self.grasp_list = []
            self.score_list = []
            for grasp_pose_cam in grasp_poses_camera:
                grasp_camera = np.array(grasp_pose_cam.pred_grasps_cam)
                grasp_world = self.pose_cam2base(grasp_camera.reshape(4,4))
                self.grasp_list.append(grasp_world)
                self.score_list.append(grasp_pose_cam.score)
            # if self.vis_pcd == True:
            self.visualize_points_grasppose(self.multiview_pc_obs_base, self.grasp_list)

            if len(self.grasp_list) == 0:
                print("No valid grasp found")
                return
            print("***********Finish generating grasp poses*************\n")
            
            # step 3: transform the grasp pose to the place pose
            self.grasp_place_list = []
            for grasp_pose in self.grasp_list:
                relative_grasp_transform = np.linalg.inv(target_pose_base)@ grasp_pose
                self.grasp_place_list.append(place_pose_base@ relative_grasp_transform)

            # adjust the raw grasp pose to pre-grasp pose
            grasp_place_list = self.grasp2pre_grasp(self.grasp_place_list, drawback_dis=0.05)
            
            # step 4: grasp pose filter
            self.actor.grasp_pose_checker_base(grasp_place_list)
            self.actor.refine_grasp_place_pose_base(self.score_list)
            succuss_result = self.actor.execute_placing_checker_base(place_pose_base, target_pose_base, self.mid_retract_pose)
            # 判断结果是否为单位矩阵
            if isinstance(succuss_result, np.ndarray) and np.array_equal(succuss_result, np.eye(4)):
                print("Result is the identity matrix.")
                raise ValueError("No valid grasp pose")
            else:
                print("Result is not the identity matrix.")
                success_grasp_pose, success_joint_grasp_list, success_joint_mid_list, success_joint_place_list = succuss_result
                print(f"success_grasp_pose: {success_grasp_pose}\n")
                print(f"success_joint_grasp_list: {success_joint_grasp_list}\n")
                print(f"success_joint_mid_list: {success_joint_mid_list}\n")
                print(f"success_joint_place_list: {success_joint_place_list}\n")

            print("***********Finish grasp poses filtered*************\n")

            # step 5: move to the grasp pose
            self.move_along_path(success_joint_grasp_list)
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] = 0.1
            ef_pose = ef_pose.dot(forward_mat)
            quat_pose = pack_pose(ef_pose)
            final_grasp_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
            self.set_pose(final_grasp_pose[0], final_grasp_pose[1])

            # Close gripper
            time.sleep(1)
            self.control_gripper("set_pose", 0.)
            time.sleep(1)
            print(f"***********Finish robot grasping***********\n")

            # step 6: move to the place pose
            

            # 可以微調place pose!!!!
            self.move_along_path(success_joint_mid_list)
            self.move_along_path(success_joint_place_list)
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] = 0.
            ef_pose = forward_mat.dot(ef_pose)
            quat_pose = pack_pose(ef_pose)
            final_place_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
            self.set_pose(final_place_pose[0], final_place_pose[1])
            time.sleep(2)
            # if self.placing_stage == 1:
            #     # step 6.5: move forward 5cm
            #     ef_pose = self.get_ef_pose()
            #     forward_mat = np.eye(4)
            #     forward_mat[2, 3] += 0.05
            #     ef_pose = forward_mat.dot(ef_pose)
            #     quat_pose = pack_pose(ef_pose)
            #     final_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
            #     self.set_pose(final_pose[0], final_pose[1])
            # Open gripper
            self.control_gripper("set_pose", 0.085)
            print(f"***********Finish robot placing***********\n")
            time.sleep(1)

            # step 7: move back 5cm
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] -= 0.05
            ef_pose = ef_pose.dot(forward_mat)
            quat_pose = pack_pose(ef_pose)
            final_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
            self.set_pose(final_pose[0], final_pose[1])

            # step 8: move back to the home joint
            self.move_along_path(self.home_joint_point)
            print("***********Finish the placing task***********\n")
        
        elif msg.data == 1:
            print(f"ruckig joint testing")
            self.actor.init_joint_pose = self.joint_states
            self.actor.env._panda.reset(self.actor.init_joint_pose)

            # Preset path
            predefined_path = [[-0.042302632795470585, -1.3297460265227181, 2.0443991797749743, -0.5891650026877226, 1.1305660780668196, -0.03845542196236371],
                               [-0.04581126430786592, -0.5309913778892226, 1.8918173598605195, -0.8026284161760975, 1.2111906169078785, -0.03845542196236371],
                               [-0.04229641181761173, -1.7921697281949702, 2.502655034341253, -0.5894170708848987, 1.5575706693473996, -0.0387850963381803]]
            
            self.move_along_path(predefined_path)

        elif msg.data == 2:
            print(f"ruckig cartesian testing")
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] = 0.02
            ef_pose = ef_pose.dot(forward_mat)
            quat_pose = pack_pose(ef_pose)
            RT_grasp = [quat_pose[:3], ros_quat(quat_pose[3:])]
        
            self.set_pose(RT_grasp[0], RT_grasp[1])
            print(f"move forward a little")
            self.control_gripper("reset")
            time.sleep(2)
            self.control_gripper("set_pose", 0.)
            time.sleep(2)
            self.control_gripper("set_pose", 0.085)
            print(f"finish grasping")

        elif msg.data == 3:
            # Reset the gripper
            self.control_gripper("reset")
            time.sleep(1)
            self.control_gripper("set_pose", 0.)
            time.sleep(1)
            self.control_gripper("set_pose", 0.085)
            print(f"finish grasping")

            # Reset the arm's position
            self.move_along_path(self.home_joint_point)

            # Set init_value to None
            self.target_points = None
            self.obs_points = None
        
            # Segmentation part
            seg_msg = Int32()
            seg_msg.data = 2
            self.seg_pub.publish(seg_msg)    

        elif msg.data == 4:
            self.move_along_path([self.envir_joint])

        elif msg.data == 5:
            # multi-view data
            self.get_multiview_data()
            self.get_oriented_bounding_box()
            print('z_translation = {}'.format(self.get_normal_translation()))
            print("***********Finish get multiview data*************\n")

        elif msg.data == 6:
            # step 7: move back 5cm
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] -= 0.03
            ef_pose = ef_pose.dot(forward_mat)
            quat_pose = pack_pose(ef_pose)
            final_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
            self.set_pose(final_pose[0], final_pose[1])

        elif msg.data == 7:
            # april tag callibration
            # Reset the gripper
            self.control_gripper("reset")
            time.sleep(1)
            # close the gripper
            self.control_gripper("set_pose", 0.)

    def points_callback(self, msg):
        self.target_points = self.pc2_tranfer(msg)

    def obs_points_callback(self, msg):
        self.obs_points = self.pc2_tranfer(msg)

    def pc2_tranfer(self, ros_msg):
        points = point_cloud2.read_points_list(
                ros_msg, field_names=("x", "y", "z"))
        return np.asarray(points)

    def pc_cam2base(self, pc, crop=True):
        transform_stamped = self.tf_buffer.lookup_transform('base', 'camera_color_optical_frame', rospy.Time(0))
        trans = np.array([transform_stamped.transform.translation.x,
                            transform_stamped.transform.translation.y,
                            transform_stamped.transform.translation.z])
        quat = np.array([transform_stamped.transform.rotation.x,
                        transform_stamped.transform.rotation.y,
                        transform_stamped.transform.rotation.z,
                        transform_stamped.transform.rotation.w])
        T = quaternion_matrix(quat)
        T[:3, 3] = trans
        T_inv = np.linalg.inv(T)
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pc)
        o3d_pc.transform(T)
        self.bounds = [[-0.05, 1.1], [-0.5, 0.5], [-0.12, 2]]  # set the bounds
        bounding_box_points = list(itertools.product(*self.bounds))  # create limit points
        self.bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object
        if crop:
            o3d_pc.crop(self.bounding_box)
        return np.asarray(o3d_pc.points)

    def pc_base2cam(self, pc, crop=True):
        # 获取从基座坐标系到相机坐标系的变换
        transform_stamped = self.tf_buffer.lookup_transform('camera_color_optical_frame', 'base', rospy.Time(0))
        trans = np.array([transform_stamped.transform.translation.x,
                        transform_stamped.transform.translation.y,
                        transform_stamped.transform.translation.z])
        quat = np.array([transform_stamped.transform.rotation.x,
                        transform_stamped.transform.rotation.y,
                        transform_stamped.transform.rotation.z,
                        transform_stamped.transform.rotation.w])
        T = quaternion_matrix(quat)
        T[:3, 3] = trans
        
        # 反转变换矩阵，从基座坐标系到相机坐标系
        T_inv = np.linalg.inv(T)
        
        # 创建点云对象并应用变换
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pc)
        o3d_pc.transform(T_inv)
            
        return np.asarray(o3d_pc.points)


    def pose_cam2base(self, poses):
        transform_stamped = self.tf_buffer.lookup_transform('base', 'camera_color_optical_frame', rospy.Time(0))
        trans = np.array([transform_stamped.transform.translation.x,
                            transform_stamped.transform.translation.y,
                            transform_stamped.transform.translation.z])
        quat = np.array([transform_stamped.transform.rotation.x,
                        transform_stamped.transform.rotation.y,
                        transform_stamped.transform.rotation.z,
                        transform_stamped.transform.rotation.w])
        T = quaternion_matrix(quat)
        T[:3, 3] = trans

        return np.dot(T, poses)


    def visual_pc(self, pc):
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pc)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([o3d_pc, axes])

    def create_grasp_geometry(self, grasp_pose, color=[0, 0, 0], length=0.08, width=0.08):
        """Create a geometry representing a grasp pose as a U shape."""
        # Define the grasp frame
        frame = grasp_pose.reshape(4, 4)
        # Define the U shape as a line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector([
            [0, 0, 0],
            [width/2, 0, 0],
            [-width/2, 0, 0],
            [width/2, 0, length/2],
            [-width/2, 0, length/2],
            [0, 0, -length/2]
        ])
        line_set.lines = o3d.utility.Vector2iVector([
            [0, 1], [0, 2], [1, 3], [2, 4], [0, 5]
        ])
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(line_set.lines))])
        line_set.transform(frame)

        return line_set
    
    def visualize_points_grasppose(self, scene_points, grasp_list=None, repre_idx=None):
        
        if grasp_list is None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(scene_points[:, :3]))
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([pcd, axes])
            # End of visualization
        else:
            # Visualize pointcloud and grasp pose part
            if repre_idx is None:
                grasp_geometries = [self.create_grasp_geometry(grasp_pose) for grasp_pose in grasp_list]
            else:
                grasp_geometries = [self.create_grasp_geometry(grasp_list[repre_idx], color=[0, 1, 0],
                                                            length=0.1, width=0.1)]
                remain_grasp_list = copy.deepcopy(grasp_list)
                del remain_grasp_list[repre_idx]
                if len(grasp_list) > 0:
                    grasp_geometries.extend([self.create_grasp_geometry(grasp_pose)
                                            for grasp_pose in remain_grasp_list])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(scene_points[:, :3]))
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([pcd, axes, *grasp_geometries])
            # End of visualization


    def setting_contact_req(self, obstacle_points, target_points):
        contact_request = GraspGroupRequest()
        contact_request.segmap_id = 1

        # Create a service request(pointcloud part)
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base_link'  # Replace with your desired frame ID
        full_pc = np.concatenate((obstacle_points[:, :3], target_points[:, :3]), axis=0)
        contact_request.pc_full = create_cloud_xyz32(header, full_pc)
        contact_request.pc_target = create_cloud_xyz32(header, target_points[:, :3])
        contact_request.mode = 1
        grasp_poses = self.contact_client(contact_request).grasp_poses

        return grasp_poses
    

    def grasp2pre_grasp(self, grasp_poses, drawback_dis=0.02):
        # This function will make the grasp poses retreat a little
        drawback_matrix = np.identity(4)
        drawback_matrix[2, 3] = -drawback_dis

        result_poses = []
        for i in range(len(grasp_poses)):
            grasp_candidate = np.dot(grasp_poses[i], drawback_matrix)
            result_poses.append(grasp_candidate)
        return np.array(result_poses)


    def adjust_joint_values(self, joint_values):
        # This function adjust the value outside the range into the range
        adjusted_values = []
        for value, min_limit, max_limit in zip(joint_values,
                                               self.actor.env._panda._joint_min_limit[:6],
                                               self.actor.env._panda._joint_max_limit[:6]):
            while value > max_limit:
                value -= 2 * np.pi
            while value < min_limit:
                value += 2 * np.pi
            adjusted_values.append(value)
        return adjusted_values

    def get_ef_pose(self):
        """
        (4, 4) end effector pose matrix from base
        """
        try:
            tf_pose = self.tf_buffer.lookup_transform("base",
                                                      # source frame:
                                                      "flange_link",
                                                      rospy.Time(0),
                                                      rospy.Duration(1.0))
            tf_pose = self.unpack_tf(tf_pose)
            pose = self.make_pose(tf_pose)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):

            pose = None
            print('cannot find end-effector pose')
            sys.exit(1)
        return pose
    
    def make_pose(self, tf_pose):
        """
        Helper function to get a full matrix out of this pose
        """
        trans, rot = tf_pose
        pose = tf.transformations.quaternion_matrix(rot)
        pose[:3, 3] = trans
        return pose
    
    def unpack_tf(self, transform):
        if isinstance(transform, TransformStamped):
            return np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]), \
                   np.array([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])
        elif isinstance(transform, Pose):
            return np.array([transform.position.x, transform.position.y, transform.position.z]), \
                   np.array([transform.orientation.x, transform.orientation.y, transform.orientation.z, transform.orientation.w])
    

    def set_joint(self, joint_position):
        """
        Send goal joint value to ruckig to move
        """
        target_joint = JointState()
        target_joint.position = joint_position
        self.joint_goal = np.concatenate((joint_position, [0, 0, 0]))
        print("Move tm joints to position: {}".format(target_joint.position))
        self.tm_joint_pub.publish(target_joint)
        return self.loop_confirm(mode="joint")


    def set_pose(self, pos, orn):
        """
        Send goal cartesian value to ruckig to move
        """
        target_pose = Pose()
        target_pose.position.x = pos[0]
        target_pose.position.y = pos[1]
        target_pose.position.z = pos[2]
        target_pose.orientation.x = orn[0]
        target_pose.orientation.y = orn[1]
        target_pose.orientation.z = orn[2]
        target_pose.orientation.w = orn[3]
        self.pose_goal = target_pose

        print("Move end effector to position: {}".format(target_pose))
        self.tm_pub.publish(target_pose)

        return self.loop_confirm(mode="cart")
        

    def loop_confirm(self, mode="joint"):
        if mode == "joint":
            threshold=0.01
            while True:
                dis = np.linalg.norm(self.joint_states-self.joint_goal)
                # print(f"dis: {dis}")
                if dis < threshold:
                    break
            return True
        else:
            threshold=0.01
            transform_ef = self.tf_buffer.lookup_transform("base",
                                                            # source frame:
                                                            "flange_link",
                                                            rospy.Time(0),
                                                            rospy.Duration(1.0))
            ef_pos, ef_orn = self.unpack_tf(transform_ef)
            target_pos, target_orn = self.unpack_tf(self.pose_goal)
            dis = np.abs(ef_pos - target_pos)
            while True:
                transform_ef = self.tf_buffer.lookup_transform("base",
                                                            # source frame:
                                                            "flange_link",
                                                            rospy.Time(0),
                                                            rospy.Duration(1.0))
                ef_pos, ef_orn = self.unpack_tf(transform_ef)
                target_pos, target_orn = self.unpack_tf(self.pose_goal)
                dis = np.sum(np.abs(ef_pos - target_pos))
                # print(f"dis: {dis}")
                if dis < threshold:
                    break
            return True


    def move_along_path(self, path):
        for waypoint in path:
            self.set_joint(waypoint)
            self.loop_confirm()

    def control_gripper(self, type, value=0):
        gripper_command = Robotiq2FGripper_robot_output()
        if type == "reset":
            gripper_command.rACT = 0
            gripper_command.rGTO = 0
            gripper_command.rATR = 0
            gripper_command.rSP = 0
            gripper_command.rFR = 0
            gripper_command.rPR = 0
        elif type == "set_pose":
            if value > 0.085 or value < 0:
                raise ValueError("Error invalid valur for gripper open length")

            uint_value = int(255 - value / 0.085 * 255)
            gripper_command.rACT = 1
            gripper_command.rGTO = 1
            gripper_command.rSP = 200
            gripper_command.rFR = 170
            gripper_command.rPR = uint_value

        self.robotiq_pub.publish(gripper_command)
    

if __name__ == "__main__":
    rospy.init_node("test_realworld")
    real_actor_node = ros_node(renders=False)
    rospy.spin()