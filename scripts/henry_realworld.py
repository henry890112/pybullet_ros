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
from utils.placement_utils import PointCloudProcessor

from pybullet_ros.srv import GetTargetMatrix, GetTargetMatrixRequest
from std_srvs.srv import Empty
import cv_bridge


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

        # self.points_sub = rospy.Subscriber("/uoais/Pointclouds", PointCloud2, self.points_callback)
        # self.obs_points_sub = rospy.Subscriber("/uoais/obs_pc", PointCloud2, self.obs_points_callback)
        # self.seg_pub = rospy.Publisher("/uoais/data_init", Int32, queue_size=1)

        self.contact_client = rospy.ServiceProxy('contact_graspnet/get_grasp_result', GraspGroup)
        rospy.wait_for_service('contact_graspnet/get_grasp_result', timeout=None)
        self.target_points = None
        self.obs_points = None
        self.home_joint_point = [[-1.3704622974806124, -1.27356437964057, 2.136768259023283, -0.0873246443471563, 1.4890664429434004, 0.09224064830473923]]
        self.home_joint_back = [[-0.7445512643880855, -0.16302066237350887, 1.341049414295396, -0.45060522525523944, 1.3288549328219774, 0.06511047700312719],
                                 [-0.875453022237505, -0.3015991318678931, 1.4738113235872086, -0.345972755240307, 1.3888166705553842, 0.04790928596987353],
                                [0.2-np.pi/2, -1, 2, 0, 1.571, 0.0]]
        self.second_multi_view = [0.2-np.pi/2, -1, 2, 0, 1.571, 0.0]
        self.third_multi_view = [0.2-np.pi/2, -1, 2, 0, 1.571, 0.0]
        self.left_joint_list = [[-0.7347407781434788, 0.5118680836287178, 1.1827692285964984, 0.4877031591085841, 0.9902222157305788, 2.530426744504708]]
        self.right_joint_list = [[-2.1427776820230466, 0.4583241646482387, 0.7837185076979353, 0.7192889843407846, 2.2248681578861396, 0.8679910446001847]]

        self.vis_pcd = True

        self.placing_stage = None
        self.mid_retract_pose_stage2 = [[-0.4500253552173883, -0.02267545597315525, 1.3331675227331974, -0.9159003139253015, 1.8160441684192035, -0.07735687347304526],
                                        [-0.7554130334730009, -0.4744738065250746, 1.7364733092634346, -0.9301731928924448, 1.7374461620096404, -0.04240180641775162]]
        self.mid_retract_pose_stage1 = [[-0.8220210226018364, -0.06652711225548122, 2.17589195964931, -1.8934204496786928, 1.6875297347499583, -0.03586651729361645]]

        self.stage2_home_to_mid  = [[-1.3704622974806124, -1.27356437964057, 2.136768259023283, -0.0873246443471563, 1.4890664429434004, 0.09224064830473923], [-1.3707963272857595, -1.0000000004669976, 1.9999999991003996, 7.948592259251724e-10, 1.5709999999913933, -2.7528697118704175e-10], [-0.5959123382374572, -0.9659030606149416, 1.848379229997479, 0.024420862371877163, 1.182712439435408, 0.5115630687104752], [-0.3420664951999188, -0.7755284357789677, 1.711846046952328, -0.17685853465629447, 1.0263177124245835, 0.5252561936826373], [-0.36310414109615835, -0.5065840203595358, 1.5319253997618798, -0.5175137556101873, 1.0823028274958202, 0.300974422962663]]
        self.stage1_home_to_mid = [[-1.3704622974806124, -1.27356437964057, 2.136768259023283, -0.0873246443471563, 1.4890664429434004, 0.09224064830473923], [-1.3707963272857595, -1.0000000004669976, 1.9999999991003996, 7.948592259251724e-10, 1.5709999999913933, -2.7528697118704175e-10], [-0.6013602631241829, -1.1127022267520306, 2.1322593012141002, -0.1144134804906389, 1.1860991538574264, 0.5069307269744772], [-0.3419576159396177, -1.0059191997076407, 2.279266126657858, -0.5138418598583432, 1.0262358581790991, 0.5253454950054585], [-0.3630999449382324, -0.6657111585294896, 2.3553511102421636, -1.181811222689202, 1.0822990860581434, 0.30097673718750256]]

        # self.envir_joint = [[1.6208316641972509-np.pi/2, -0.98, 1.46939, -0.119555, 1.63676, -0.05]]
        self.envir_joint = [[-0.013219572748584558, -1.0766580383862996, 0.9702233406826832, 0.501500963426771, 1.670231571363512, -0.025433325729500432]]
        rospy.loginfo("Init finished")
        self.use_cvae  =   False
        self.use_env_detect = False


    def joint_callback(self, msg):
        cur_states = np.asarray(msg.position)
        cur_states= np.concatenate((cur_states, [0, 0, 0]))
        self.joint_states = cur_states

    def get_multiview_data(self, angle_list=None, single_view = False, reset_pc = False):
        # 初始化 multiview_pc 来存储所有角度的点云数据
        self.multiview_pc_obs_base = []
        self.multiview_pc_target_base = []
         
        # Get the pointcloud data in three degree and concatenate them
        # home joint放最後是為了移回camera的位置才可生成contact grasp
        angle_list = [self.left_joint_list, self.right_joint_list, self.home_joint_point]
        
        if single_view == False:
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
                while(self.target_points is None):
                    print(f"wait for segmentation")
                    time.sleep(0.1) # Sleep to wait for the segmentation pointcloud arrive
                # time.sleep(2) # Sleep to wait for the segmentation pointcloud arrive

                print(f"self.obs_points: {self.obs_points}")

                self.obs_points_base = self.pc_cam2base(self.obs_points)
                self.target_points_base = self.pc_cam2base(self.target_points)

                # Append the transformed pointclouds to the multiview_pc lists
                if self.obs_points_base is not None:
                    self.multiview_pc_obs_base.append(self.obs_points_base)
                if self.target_points_base is not None:
                    self.multiview_pc_target_base.append(self.target_points_base)
                    print(self.target_points_base.shape)

                # if self.vis_pcd == True:
                # self.visual_pc(self.target_points_base)
                self.move_along_path(self.home_joint_point)
        if single_view == True:
            if reset_pc == False:
                for multiview_joint in angle_list:
                    # Reset the arm's position
                    self.move_along_path(multiview_joint)
                    time.sleep(0.5)
                    self.move_along_path(self.home_joint_point)
                
                # Set init_value to None
                self.target_points = None
                self.obs_points = None
                
                # Segmentation part
                seg_msg = Int32()
                seg_msg.data = 2
                self.seg_pub.publish(seg_msg)    
                while(self.target_points is None):
                    print(f"wait for segmentation")
                    time.sleep(0.1) # Sleep to wait for the segmentation pointcloud arrive
                # time.sleep(2) # Sleep to wait for the segmentation pointcloud arrive

                print(f"self.obs_points: {self.obs_points}")

                self.obs_points_base = self.pc_cam2base(self.obs_points)
                self.target_points_base = self.pc_cam2base(self.target_points)
                # Append the transformed pointclouds to the multiview_pc lists
                if self.obs_points_base is not None:
                    self.multiview_pc_obs_base.append(self.obs_points_base)
                if self.target_points_base is not None:
                    self.multiview_pc_target_base.append(self.target_points_base)
                    print(self.target_points_base.shape)
            
            if reset_pc == True:
                '''
                重新獲得一次multiview_pc_obs_base以便新獲得obb的z translation
                '''
                # for multiview_joint in angle_list:
                #     # Reset the arm's position
                #     self.move_along_path(multiview_joint)
                #     time.sleep(0.5)
                #     self.move_along_path(self.home_joint_point)
                
                # Set init_value to None
                self.target_points = None
                self.obs_points = None
                
                # Segmentation part
                seg_msg = Int32()
                seg_msg.data = 2
                self.seg_pub.publish(seg_msg)    
                while(self.target_points is None):
                    print(f"wait for segmentation")
                    time.sleep(0.1) # Sleep to wait for the segmentation pointcloud arrive
                # time.sleep(2) # Sleep to wait for the segmentation pointcloud arrive

                print(f"self.obs_points: {self.obs_points}")

                self.obs_points_base = self.pc_cam2base(self.obs_points)
                self.target_points_base = self.pc_cam2base(self.target_points)
                # Append the transformed pointclouds to the multiview_pc lists
                if self.obs_points_base is not None:
                    self.multiview_pc_obs_base.append(self.obs_points_base)
                if self.target_points_base is not None:
                    self.multiview_pc_target_base.append(self.target_points_base)
                    print(self.target_points_base.shape)

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
            # self.visual_pc(self.multiview_pc_obs_base)
            print('***********', self.multiview_pc_target_base.shape)
            # self.visual_pc(self.multiview_pc_target_base)
        # self.visual_pc(self.multiview_pc_obs)
        # self.visual_pc(self.multiview_pc_target)
        
        
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.multiview_pc_target_base)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=3)
        pcd_inlier = pcd.select_by_index(ind)
        self.multiview_pc_target_base = np.asarray(pcd_inlier.points)
        print('***********', self.multiview_pc_target_base.shape)
        # self.visual_pc(self.multiview_pc_target_base)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.multiview_pc_target)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=3)
        pcd_inlier = pcd.select_by_index(ind)
        self.multiview_pc_target = np.asarray(pcd_inlier.points)
        print('***********', self.multiview_pc_target.shape)
        # self.visual_pc(self.multiview_pc_target)


        # save the pointclouds as npy file
        np.save("/home/user/henry_pybullet_ws/src/pybullet_ros/realworld_data/multiview_pc_obs.npy", self.multiview_pc_obs_base)
        np.save("/home/user/henry_pybullet_ws/src/pybullet_ros/realworld_data/multiview_pc_target.npy", self.multiview_pc_target_base)
        np.save("/home/user/henry_pybullet_ws/src/pybullet_ros/realworld_data/multiview_pc_obs_cam.npy", self.multiview_pc_obs)
        np.save("/home/user/henry_pybullet_ws/src/pybullet_ros/realworld_data/multiview_pc_target_cam.npy", self.multiview_pc_target)

    def get_the_target_on_cabinet_pose(self):
        self.placing_location = [0.15, 0, -0.15]
        if self.placing_stage == 1:
            place_pose_base = np.array([[-1, 0.,  0.,   0.8],
                                        [-0., -1,  0.,  0.0],
                                        [ 0.,  0.,  1.,  0.2],
                                        [ 0. ,         0. ,         0.,          1.        ]]
                                        )
        elif self.placing_stage == 2:
            place_pose_base = np.array([[-1, 0.,  0.,   0.8],
                                        [-0., -1,  0.,  0.0],
                                        [ 0.,  0.,  1.,  0.6],
                                        [ 0. ,         0. ,         0.,          1.        ]]
                                        )
        # 微調place pose
        place_pose_base[0, 3] += 0
        place_pose_base[1, 3] += 0
        place_pose_base[2, 3] += self.get_normal_translation()
        # place_pose_base[2, 3] += self.target_center[2]
        print('z_translation = {}'.format(self.get_normal_translation()))
        return place_pose_base
    
    def get_the_target_on_cabinet_pose_new(self):
        self.placing_location = [0.15, 0, -0.15]
        print("!!!!!!!!!!!!!!!!!!!!!!", str(self.placing_stage)[0])
        if str(self.placing_stage)[0] == '1':
            place_pose_base = np.array([[-1, 0.,  0.,   0.8],
                                        [-0., -1,  0.,  0.0],
                                        [ 0.,  0.,  1.,  0.18],
                                        [ 0. ,         0. ,         0.,          1.        ]]
                                        )
        elif str(self.placing_stage)[0] == '2':
            print()
            place_pose_base = np.array([[-1, 0.,  0.,   0.8],
                                        [-0., -1,  0.,  0.0],
                                        [ 0.,  0.,  1.,  0.62],
                                        [ 0. ,         0. ,         0.,          1.        ]]
                                        )
        # 微調place pose
        place_pose_base[0, 3] += 0
        place_pose_base[1, 3] += 0
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
        # o3d.visualization.draw_geometries([self.pc_segments_pcd, obb])
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
    
    def deg2rad(self, deg):
        return deg * np.pi / 180

    def parse_input(self, input_value):
        # 将输入整数解析为 placing_stage 和 角度
        str_value = str(input_value)
        parallel_or_not = int(str_value[0]) # 第0个数字决定parallel or pack   (1代表parallel; 2代表pack)
        placing_stage = int(str_value[1:3])  # 第一个数字决定阶段
        angle_deg = int(str_value[3:])     # 其余数字决定角度
        angle_rad = self.deg2rad(angle_deg)  # 将角度转换为弧度
        return parallel_or_not, placing_stage, angle_rad
    
    
    def get_env_callback(self, msg):
        '''
        只有msg.data = 0, 11, 12, 13, 21, 22, 23
        或>100 才會跑這下面
        '''
        if msg.data > 100:
            input_value = msg.data  # 接收的整数值
            print("\n******************YEAH************************\n")
            self.parallel_or_not, self.placing_stage, self.angle_rad = self.parse_input(input_value)  # 解析阶段和角度
            print("parallel_or_not", self.parallel_or_not)
            print("placing_stage", self.placing_stage)
            print("angle_rad", self.angle_rad)

            print(str(self.placing_stage)[0])
            if str(self.placing_stage)[0] == '2':
                self.mid_retract_pose = rotZ(-np.pi/2)@ transZ(0.9)@ transX(0.3)@ transY(0.3)@ np.eye(4)@ rotZ(np.pi/4*3)@ rotX(np.pi/4*3)@ rotX(-np.pi/6)
            elif str(self.placing_stage)[0] == '1':
                self.mid_retract_pose = rotZ(-np.pi/2)@ transZ(0.45)@ transX(0.3)@ transY(0.3)@ np.eye(4)@ rotZ(np.pi/4*3)@ rotX(np.pi/4*3)@ rotX(-np.pi/6)
            
             # Pybullet setup
            self.actor.load_environment()
            self.actor.env._panda.reset(self.home_joint_point[0]+[0, 0, 0])
            self.actor.initial()
            self.actor.grasp_checker = ValidGraspChecker(self.actor.env)

           
            place_pose_base = self.get_the_target_on_cabinet_pose_new()

            if str(self.placing_stage)[1] == '1':
                place_pose_base[1, 3] += 0.25
            elif str(self.placing_stage)[1] == '2':
                place_pose_base[1, 3] += 0.
            elif str(self.placing_stage)[1] == '3':
                place_pose_base[1, 3] += -0.25

            # 用肉眼看決定角度20240719
            print("*********************************YEAH************************")
            if self.parallel_or_not == 1:
                target_pose_base = np.eye(4)
                self.target_center = np.mean(self.multiview_pc_target_base, axis=0)
                target_pose_base[:3, 3] = self.target_center[:3]
                target_pose_base[:3, 0] = np.array([0, 0, 1])
                target_pose_base[:3, 1] = np.array([-1, 0, 0])
                target_pose_base[:3, 2] = np.array([0, -1, 0])
            elif self.parallel_or_not == 2:
                target_pose_base = np.eye(4)
                self.target_center = np.mean(self.multiview_pc_target_base, axis=0)
                target_pose_base[:3, 3] = self.target_center[:3]
            elif self.parallel_or_not == 3 or self.parallel_or_not == 4:
                target_pose_base = np.eye(4)
                self.target_center = np.mean(self.multiview_pc_target_base, axis=0)
                target_pose_base[:3, 3] = self.target_center[:3]
                target_pose_base[:3, 0] = np.array([0, 0, 1])
                target_pose_base[:3, 1] = np.array([-1, 0, 0])
                target_pose_base[:3, 2] = np.array([0, -1, 0])

            if self.parallel_or_not == 1:
                # parral
                target_pose_base = target_pose_base @ rotX(self.angle_rad)
                target_pose_base = target_pose_base@ rotZ(0)
                place_pose_base[1, 3] += 0
                place_pose_base[2, 3] += 0.03
            elif self.parallel_or_not == 2:
                # pack
                target_pose_base = target_pose_base@ rotZ(self.angle_rad)
                place_pose_base[1, 3] += 0
                place_pose_base[2, 3] += 0.07
            elif self.parallel_or_not == 3:
                # pile1
                target_pose_base = target_pose_base@ rotX(self.deg2rad(90))@ rotY(self.angle_rad)
                place_pose_base[1, 3] += 0
                place_pose_base[2, 3] += 0.03
            elif self.parallel_or_not == 4:
                # pile2
                target_pose_base = target_pose_base@ rotX(self.deg2rad(45))@ rotY(self.angle_rad)
                place_pose_base[1, 3] += 0
                place_pose_base[2, 3] += 0.03
            
            
            self.actor.env.draw_ef_coordinate(place_pose_base, 0)
            self.actor.env.draw_ef_coordinate(target_pose_base, 0)
            print("Place pose:", place_pose_base)
            print("Target pose:", target_pose_base)
            print("***********Finish get place/target pose*************\n")

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
            # 只取斜下的姿態
            self.actor.refine_grasp_place_pose_base_in_target_pose(self.score_list)


            # 優先選擇中間的姿態
            if str(self.placing_stage)[0] == '2':
                self.actor.refine_grasp_place_pose_base_in_target_pose_stage2(self.score_list, place_pose_base[:3, 3])
            elif str(self.placing_stage)[0] == '1':
                self.actor.refine_grasp_place_pose_base_in_target_pose_stage1(self.score_list, place_pose_base[:3, 3])

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

            print(f"列表大小: {len(success_joint_place_list)}")
            success_joint_grasp_list = [success_joint_grasp_list[i] for i in [0, 3, 6, 9]]
            success_joint_mid_list = [success_joint_mid_list[i] for i in [0, 2, 3, 6, 9]]
            success_joint_place_list = [success_joint_place_list[i] for i in [0, 3, 6, 9]]

            # step 5: move to the grasp pose
            self.move_along_path_vel(np.array(success_joint_grasp_list))
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] = 0.05
            ef_pose = ef_pose.dot(forward_mat)
            quat_pose = pack_pose(ef_pose)
            final_grasp_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
            self.set_pose(final_grasp_pose[0], final_grasp_pose[1])

            # Close gripper
            time.sleep(1)
            self.control_gripper("set_pose", 0.045)
            time.sleep(1)
            print(f"***********Finish robot grasping***********\n")

            # # step 5.5: move to the grasp pose
            # self.move_along_path_vel(np.array(success_joint_grasp_list))
            # ef_pose = self.get_ef_pose()
            # forward_mat = np.eye(4)
            # forward_mat[2, 3] = -0.05
            # ef_pose = ef_pose.dot(forward_mat)
            # quat_pose = pack_pose(ef_pose)
            # final_grasp_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
            # self.set_pose(final_grasp_pose[0], final_grasp_pose[1])

            # step 6: move to the place pose
            # 可以微調place pose!!!!
            self.move_along_path_vel(np.array(success_joint_mid_list))
            self.move_along_path_vel(np.array(success_joint_place_list))

            # 往下輕放物體
            if(self.parallel_or_not == 1):
                ef_pose = self.get_ef_pose()
                forward_mat = np.eye(4)
                forward_mat[2, 3] = -0.01
                ef_pose = forward_mat.dot(ef_pose)
                quat_pose = pack_pose(ef_pose)
                final_place_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
                self.set_pose(final_place_pose[0], final_place_pose[1])
                time.sleep(2)
            else:
                ef_pose = self.get_ef_pose()
                forward_mat = np.eye(4)
                forward_mat[2, 3] = -0.01
                ef_pose = forward_mat.dot(ef_pose)
                quat_pose = pack_pose(ef_pose)
                final_place_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
                self.set_pose(final_place_pose[0], final_place_pose[1])
                time.sleep(2)

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
            reverse_path_list = []
            reverse_path_list = np.flip(np.array(success_joint_place_list), axis=0)
            print(reverse_path_list.shape)
            self.move_along_path_vel(reverse_path_list[-3:])
            if str(self.placing_stage)[0] == '1':
                reverse_path_list = []
                reverse_path_list = np.flip(np.array(self.stage1_home_to_mid), axis=0)
                self.move_along_path_vel(reverse_path_list)
            if str(self.placing_stage)[0] == '2':
                print("reverse path")
                reverse_path_list = []
                reverse_path_list = np.flip(np.array(self.stage2_home_to_mid), axis=0)
                self.move_along_path_vel(reverse_path_list)
            print("***********Finish the placing task***********\n")
            self.move_along_path(self.home_joint_point)

        if msg.data in {0, 11, 12, 13, 21, 22, 23} :
            
            input_value = msg.data  # 接收的整数值

            if msg.data in {0, 21, 22, 23}:
                self.placing_stage = 2
            else:
                self.placing_stage = 1

            if self.placing_stage == 2:
                self.mid_retract_pose = rotZ(-np.pi/2)@ transZ(0.85)@ transX(0.3)@ transY(0.3)@ np.eye(4)@ rotZ(np.pi/4*3)@ rotX(np.pi/4*3)@ rotX(-np.pi/6)
            elif self.placing_stage == 1:
                self.mid_retract_pose = rotZ(-np.pi/2)@ transZ(0.45)@ transX(0.3)@ transY(0.3)@ np.eye(4)@ rotZ(np.pi/4*3)@ rotX(np.pi/4*3)@ rotX(-np.pi/6)
            
            # # Reset the gripper
            # self.control_gripper("reset")
            # time.sleep(1)
            # self.control_gripper("set_pose", 0.)
            # time.sleep(1)
            # self.control_gripper("set_pose", 0.085)
            # print(f"finish grasping")

            # Pybullet setup
            self.actor.load_environment()
            self.actor.env._panda.reset(self.home_joint_point[0]+[0, 0, 0])
            self.actor.initial()
            self.actor.grasp_checker = ValidGraspChecker(self.actor.env)

            # get env data
            self.move_along_path(self.envir_joint)
            time.sleep(2)
            if self.use_env_detect:
                '''
                在下面
                '''

            self.move_along_path(self.home_joint_point)

            # multi-view data
            self.get_multiview_data(single_view = True)
            obb = self.get_oriented_bounding_box()
            print('z_translation = {}'.format(self.get_normal_translation()))
            o3d.visualization.draw_geometries([self.pc_segments_pcd, obb])
            print(f"self.multiview_pc_target_base: {self.multiview_pc_target_base}")
            print(f"self.obs_points: {self.obs_points}")
            print("***********Finish get multiview data*************\n")

            # get the place pose (can change to other place pose)
            # place_pose_base = self.actor.get_the_target_on_cabinet_pose()
            # get it from apriltag_inform.py
            place_pose_base = self.get_the_target_on_cabinet_pose()

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
            object_property = 'parallel'
            # object_property = 'pack'
            if object_property == 'parallel':
                target_pose_base = np.eye(4)
                self.target_center = np.mean(self.multiview_pc_target_base, axis=0)
                target_pose_base[:3, 3] = self.target_center[:3]
                target_pose_base[:3, 0] = np.array([0, 0, 1])
                target_pose_base[:3, 1] = np.array([-1, 0, 0])
                target_pose_base[:3, 2] = np.array([0, -1, 0])
            elif object_property == 'pack':
                target_pose_base = np.eye(4)
                self.target_center = np.mean(self.multiview_pc_target_base, axis=0)
                target_pose_base[:3, 3] = self.target_center[:3]
                
            if msg.data == 21:
                if object_property == 'parallel':
                    target_pose_base = target_pose_base@ rotX(-np.pi/4)
                    target_pose_base = target_pose_base@ rotZ(0)
                else:
                    target_pose_base = target_pose_base@ rotZ(0)
                place_pose_base[1, 3] += 0.25
                place_pose_base[2, 3] += 0.03

            if msg.data == 22:
                if object_property == 'parallel':
                    target_pose_base = target_pose_base@ rotX(np.pi/4)
                    target_pose_base = target_pose_base@ rotZ(0)
                else:
                    target_pose_base = target_pose_base@ rotZ(np.pi/2+np.pi/4)
                place_pose_base[1, 3] += 0
                place_pose_base[2, 3] += 0.04

            if msg.data == 23:
                if object_property == 'parallel':
                    target_pose_base = target_pose_base@ rotX(-np.pi/4)
                    target_pose_base = target_pose_base@ rotZ(0)
                else:
                    target_pose_base = target_pose_base@ rotZ(0)
                place_pose_base[1, 3] += -0.25
                place_pose_base[2, 3] += 0.02
            
            if msg.data == 11:
                if object_property == 'parallel':
                    target_pose_base = target_pose_base@ rotX(np.pi/2)
                    target_pose_base = target_pose_base@ rotZ(0)
                else:
                    target_pose_base = target_pose_base@ rotZ(0)
                place_pose_base[1, 3] += 0.25
                place_pose_base[2, 3] += 0.03

            if msg.data == 12:
                if object_property == 'parallel':
                    target_pose_base = target_pose_base@ rotX(-np.pi/3*2)
                    target_pose_base = target_pose_base@ rotZ(0)
                else:
                    target_pose_base = target_pose_base@ rotZ(0)
                place_pose_base[1, 3] += 0
                place_pose_base[2, 3] += 0.04

            if msg.data == 13:
                if object_property == 'parallel':
                    target_pose_base = target_pose_base@ rotX(np.pi/4)
                    target_pose_base = target_pose_base@ rotZ(0)
                else:
                    target_pose_base = target_pose_base@ rotZ(0)
                place_pose_base[1, 3] += -0.25
                place_pose_base[2, 3] += 0.04
            
                
            # TODO use cvae to get target 6d pose
            if self.use_cvae:
                target_pose_base = self.target_matrix

            self.actor.env.draw_ef_coordinate(place_pose_base, 0)
            self.actor.env.draw_ef_coordinate(target_pose_base, 0)
            print("Place pose:", place_pose_base)
            print("Target pose:", target_pose_base)
            print("***********Finish get place/target pose*************\n")

            # step 2: get the contact grasp
            '''
            # 要放camrea frame的點雲來去生成contact grasp
            # grasp_poses_camera = self.setting_contact_req(obstacle_points=self.obs_points, target_points=self.target_points)
            '''
            print(f"self.obs_points: {self.obs_points}")
            print(f"self.target_points: {self.target_points}")
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

            print(f"列表大小: {len(success_joint_place_list)}")
            success_joint_grasp_list = [success_joint_grasp_list[i] for i in [0, 3, 6, 9]]
            success_joint_mid_list = [success_joint_mid_list[i] for i in [0, 3, 6, 9]]
            success_joint_place_list = [success_joint_place_list[i] for i in [0, 3, 6, 9]]

            # step 5: move to the grasp pose
            self.move_along_path_vel(np.array(success_joint_grasp_list))
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] = 0.05
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
            self.move_along_path_vel(np.array(success_joint_mid_list))
            self.move_along_path_vel(np.array(success_joint_place_list))

            # 往下輕放物體
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] = -0.03
            ef_pose = forward_mat.dot(ef_pose)
            quat_pose = pack_pose(ef_pose)
            final_place_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
            self.set_pose(final_place_pose[0], final_place_pose[1])
            time.sleep(2)

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
            reverse_path_list = []
            reverse_path_list = np.flip(np.array(success_joint_place_list), axis=0)
            print(reverse_path_list.shape)
            self.move_along_path_vel(reverse_path_list[-3:])
            if self.placing_stage == 1:
                reverse_path_list = []
                reverse_path_list = np.flip(np.array(self.stage1_home_to_mid), axis=0)
                self.move_along_path_vel(reverse_path_list)
            if self.placing_stage == 2:
                print("reverse path")
                reverse_path_list = []
                reverse_path_list = np.flip(np.array(self.stage2_home_to_mid), axis=0)
                self.move_along_path_vel(reverse_path_list)
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

        elif msg.data == 3 or msg.data == 33:

            if msg.data == 3:
                # Reset the gripper
                self.control_gripper("reset")
                time.sleep(1)
                self.control_gripper("set_pose", 0.)
                time.sleep(1)
                self.control_gripper("set_pose", 0.085)
                print(f"finish grasping")

            # Reset the arm's position
            # self.move_along_path(self.mid_retract_pose_stage2)

                self.move_along_path(self.home_joint_point)
            if msg.data == 33:
                self.move_along_path(self.home_joint_back)

        elif msg.data == 4:
            self.move_along_path(self.envir_joint)
            env_pub = rospy.Publisher("/transform_start", Int32, queue_size=1)
            rospy.sleep(1)  # Ensure the publisher is connected
            env_msg = Int32()
            env_msg.data = 0  # Signal to start the callback execution
            env_pub.publish(env_msg)

            file_path = '/home/user/henry_pybullet_ws/src/pybullet_ros/env_data/captured_pointcloud.pcd'
            pcd = o3d.io.read_point_cloud(file_path)
            # o3d.visualization.draw_geometries([pcd])

            env_points_base = self.pc_cam2base(pcd.points)

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(env_points_base)
            point_cloud.colors = o3d.utility.Vector3dVector(pcd.colors)
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([point_cloud, origin_frame])
            print("*************get env data finished*******************")

            
            # Define the parameters
            voxel_size = 0.01  # Adjust based on your specific requirements
            # height_range = [0.59, 0.61]  # Minimum and maximum height to consider
            # plane_height_range = [0.59, 0.61]

            ######################################
            slice_width = 0.308

            height_range = [0.60, 0.65]  # Minimum and maximum height to consider
            plane_height_range = [0.60, 0.65]
            empth_threshold = 0.155
            # height_range = [0.20, 0.25]  # Minimum and maximum height to consider
            # plane_height_range = [0.20, 0.25]
            # empth_threshold = 0.2

            # Create an instance of the PointCloudProcessor
            processor = PointCloudProcessor(point_cloud, voxel_size, height_range, plane_height_range)

            criterion, filtered_point_cloud, unit_vec = processor.whole_pipe_realworld(
                min_ratio=0.05, threshold=0.02, iterations=2000, min_cluster=5000, max_cluster=70000, visualize=True
            )

            # empty_voxels_count, occupancy_grid, min_bound = processor.check_empty_voxels(filtered_point_cloud)
            # voxel_grid = processor.visualize_voxels(occupancy_grid, min_bound)
            # o3d.visualization.draw_geometries([voxel_grid])

            unit_vec = [0, -1, 0]
            sliced_point_clouds = processor.slice_point_cloud(filtered_point_cloud, unit_vec, slice_width=slice_width, empty_threshold=empth_threshold, display_voxels=True, check_slice_region=True)
            geometries = []
            for item in sliced_point_clouds:
                if isinstance(item, tuple):
                    sliced_pcd, voxel_grid = item
                    geometries.append(sliced_pcd)
                    # geometries.append(voxel_grid)
                else:
                    geometries.append(item)
            o3d.visualization.draw_geometries([*geometries, origin_frame])

            print('===========================================================================')
            sliced_point_clouds = processor.slice_point_cloud(filtered_point_cloud, unit_vec, slice_width=slice_width, empty_threshold=empth_threshold, display_voxels=True, check_slice_region=False)
            geometries = []
            for item in sliced_point_clouds:
                if isinstance(item, tuple):
                    sliced_pcd, voxel_grid = item
                    geometries.append(sliced_pcd)
                    # geometries.append(voxel_grid)
                else:
                    geometries.append(item)
            o3d.visualization.draw_geometries([*geometries, origin_frame])


        elif msg.data == 5:
            # # Reset the gripper
            # self.control_gripper("reset")
            # time.sleep(1)
            # self.control_gripper("set_pose", 0.0)
            # time.sleep(1)
            self.control_gripper("set_pose", 0.085)
            self.move_along_path(self.home_joint_point)
        
        elif msg.data == 55:
            # step 7: move back 5cm
            
            
            self.control_gripper("set_pose", 0.085)
            # step 7: move back 5cm
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] -= 0.05
            ef_pose = ef_pose.dot(forward_mat)
            quat_pose = pack_pose(ef_pose)
            final_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
            self.set_pose(final_pose[0], final_pose[1])
            self.move_along_path(self.mid_retract_pose_stage1)
            self.move_along_path(self.home_joint_point)

        elif msg.data == 6:
            self.points_sub = rospy.Subscriber("/uoais/Pointclouds", PointCloud2, self.points_callback)
            self.obs_points_sub = rospy.Subscriber("/uoais/obs_pc", PointCloud2, self.obs_points_callback)
            self.seg_pub = rospy.Publisher("/uoais/data_init", Int32, queue_size=1)

            # Set init_value to None
            self.target_points = None
            self.obs_points = None
            
            # Segmentation part
            seg_msg = Int32()
            seg_msg.data = 2
            self.seg_pub.publish(seg_msg)   
            while(self.target_points is None):
                print(f"wait for segmentation")
                time.sleep(0.1) # Sleep to wait for the segmentation pointcloud arrive 
            

            print(f"self.obs_points: {self.obs_points}")
            print("***********Init UOAIS*************\n")
            
            whole_point_cloud = None
            self.obs_points_base = self.pc_cam2base(self.obs_points)
            self.target_points_base = self.pc_cam2base(self.target_points)
            whole_point_cloud = np.concatenate([self.obs_points_base, self.target_points_base], axis=0)

            grasp_poses_camera = self.setting_contact_req(obstacle_points=self.obs_points, target_points=self.target_points)
            self.grasp_list = []
            self.score_list = []
            for grasp_pose_cam in grasp_poses_camera:
                grasp_camera = np.array(grasp_pose_cam.pred_grasps_cam)
                grasp_world = self.pose_cam2base(grasp_camera.reshape(4,4))
                self.grasp_list.append(grasp_world)
                self.score_list.append(grasp_pose_cam.score)
            # if self.vis_pcd == True:
            self.visualize_points_grasppose(whole_point_cloud, self.grasp_list)
            print("********Init Contact GraspNet***********\n")


        elif msg.data == 7 or msg.data == 70:
            '''
            only multiview to get multiview_pc_target_base points
            '''
            if msg.data == 7:
                
                self.points_sub = rospy.Subscriber("/uoais/Pointclouds", PointCloud2, self.points_callback)
                self.obs_points_sub = rospy.Subscriber("/uoais/obs_pc", PointCloud2, self.obs_points_callback)
                self.seg_pub = rospy.Publisher("/uoais/data_init", Int32, queue_size=1)

                # Set init_value to None
                self.target_points = None
                self.obs_points = None
                
                # Segmentation part
                seg_msg = Int32()
                seg_msg.data = 2
                self.seg_pub.publish(seg_msg)   
                while(self.target_points is None):
                    print(f"wait for segmentation")
                    time.sleep(0.1) # Sleep to wait for the segmentation pointcloud arrive 
                # time.sleep(2) # Sleep to wait for the segmentation pointcloud arrive

                print(f"self.obs_points: {self.obs_points}")
                # # multi-view data
                # self.get_multiview_data()
                # print(f"self.multiview_pc_target_base: {self.multiview_pc_target_base}")
                # self.get_oriented_bounding_box()
                # print('z_translation = {}'.format(self.get_normal_translation()))
                print("***********Init UOAIS*************\n")
            if msg.data == 70:
                self.shutdown_uoais_server()


        elif msg.data == 8 or msg.data == 80:
            '''
            only use CVAE model to get target 6d pose
            '''
            if msg.data == 8:
                # 等待服務可用
                rospy.wait_for_service('get_target_matrix')
                
                try:
                    get_target_matrix = rospy.ServiceProxy('get_target_matrix', GetTargetMatrix)
                    
                    # 構建請求
                    req = GetTargetMatrixRequest()
                    
                    # 假設multiview_pc_target_base是一個包含點雲數據的列表，這裡用隨機數據作為示例
                    # parent_directory = os.path.join(os.path.dirname(__file__))
                    # result_list = ['blue_bottle', 'mug', 'small_mug', 'long_box', 'tap', 'wash_hand']
                    # multiview_path = os.path.join(parent_directory, f'my_test_data/{result_list[0]}.npy')
                    # multiview_pointcloud = np.load(multiview_path)
                    multiview_pointcloud = self.multiview_pc_target_base
                    multiview_pc_target_base = multiview_pointcloud.flatten().tolist()
                    req.multiview_pc_target_base = multiview_pc_target_base
                    
                    # 呼叫服務並獲取響應
                    resp = get_target_matrix(req)
                    self.target_matrix = np.array(resp.target_matrix).reshape(4, 4)

                    # 打印接收到的目標矩陣
                    print("Received target matrix:")
                    print(self.target_matrix)
                
                except rospy.ServiceException as e:
                    print("Service call failed: %s" % e)
            
            if msg.data == 80:
                self.shutdown_bandu_server()

        elif msg.data == 9:

            # get env data
            self.move_along_path(self.envir_joint)
            time.sleep(2)
            self.move_along_path(self.home_joint_point)

            # multi-view data
            self.get_multiview_data(single_view = True)
            obb = self.get_oriented_bounding_box()
            print('z_translation = {}'.format(self.get_normal_translation()))
            o3d.visualization.draw_geometries([self.pc_segments_pcd, obb])
            print(f"self.multiview_pc_target_base: {self.multiview_pc_target_base}")
            print(f"self.obs_points: {self.obs_points}")
            print("***********Finish get multiview data*************\n")


        elif msg.data == 99:
            # get env data
            # self.move_along_path(self.envir_joint)
            # time.sleep(2)
            self.move_along_path(self.home_joint_point)

            # multi-view data
            self.get_multiview_data(single_view = True, reset_pc = True)
            obb = self.get_oriented_bounding_box()
            print('z_translation = {}'.format(self.get_normal_translation()))
            o3d.visualization.draw_geometries([self.pc_segments_pcd])
            print(f"self.multiview_pc_target_base: {self.multiview_pc_target_base}")
            print(f"self.obs_points: {self.obs_points}")
            print("***********Finish get multiview data*************\n")

        elif msg.data == 98:
            # get multiview pcd and save it 2024905
            self.move_along_path(self.home_joint_point)
            self.get_multiview_data(single_view = False, reset_pc = False)
            print('***********', self.multiview_pc_target_base.shape)
            file_name = "/home/user/henry_pybullet_ws/src/pybullet_ros/scripts/target_pcd/target_pcd.pcd"
            self.visual_pc(self.multiview_pc_target_base, save_pcd = True, file_name = file_name)
            print("***********Finish get & save multiview data*************\n")
        elif msg.data == 97:
            # 20241114加入
            self.points_sub = rospy.Subscriber("/uoais/Pointclouds", PointCloud2, self.points_callback)
            self.obs_points_sub = rospy.Subscriber("/uoais/obs_pc", PointCloud2, self.obs_points_callback)
            self.seg_pub = rospy.Publisher("/uoais/data_init", Int32, queue_size=1)

            # Set init_value to None
            self.target_points = None
            self.obs_points = None
            
            # Segmentation part
            seg_msg = Int32()
            seg_msg.data = 2
            self.seg_pub.publish(seg_msg)   
            while(self.target_points is None):
                print(f"wait for segmentation")
                time.sleep(0.1) # Sleep to wait for the segmentation pointcloud arrive 
            

            print(f"self.obs_points: {self.obs_points}")
            print("***********Init UOAIS*************\n")
            
            whole_point_cloud = None
            self.obs_points_base = self.pc_cam2base(self.obs_points)
            self.target_points_base = self.pc_cam2base(self.target_points)
            whole_point_cloud = np.concatenate([self.obs_points_base, self.target_points_base], axis=0)

            grasp_poses_camera = self.setting_contact_req(obstacle_points=self.obs_points, target_points=self.target_points)
            self.grasp_list = []
            self.score_list = []
            for grasp_pose_cam in grasp_poses_camera:
                grasp_camera = np.array(grasp_pose_cam.pred_grasps_cam)
                grasp_world = self.pose_cam2base(grasp_camera.reshape(4,4))
                self.grasp_list.append(grasp_world)
                self.score_list.append(grasp_pose_cam.score)
            # if self.vis_pcd == True:
            self.visualize_points_grasppose(whole_point_cloud, self.grasp_list)
            print("********Init Contact GraspNet***********\n")
            #### 
            
            # get env data
            self.move_along_path(self.envir_joint)
            time.sleep(2)
            self.move_along_path(self.home_joint_point)

            # multi-view data
            self.get_multiview_data(single_view = True)
            obb = self.get_oriented_bounding_box()
            print('z_translation = {}'.format(self.get_normal_translation()))
            o3d.visualization.draw_geometries([self.pc_segments_pcd, obb])
            print(f"self.multiview_pc_target_base: {self.multiview_pc_target_base}")
            print(f"self.obs_points: {self.obs_points}")
            print("***********Finish get multiview data*************\n")
        
        elif msg.data == 96:
            time.sleep(10)
            for i in range(3):
                # 20241114加入
                self.points_sub = rospy.Subscriber("/uoais/Pointclouds", PointCloud2, self.points_callback)
                self.obs_points_sub = rospy.Subscriber("/uoais/obs_pc", PointCloud2, self.obs_points_callback)
                self.seg_pub = rospy.Publisher("/uoais/data_init", Int32, queue_size=1)

                # Set init_value to None
                self.target_points = None
                self.obs_points = None
                
                # Segmentation part
                seg_msg = Int32()
                seg_msg.data = 2
                self.seg_pub.publish(seg_msg)   
                while(self.target_points is None):
                    print(f"wait for segmentation")
                    time.sleep(0.1) # Sleep to wait for the segmentation pointcloud arrive 
                

                print(f"self.obs_points: {self.obs_points}")
                print("***********Init UOAIS*************\n")
                
                whole_point_cloud = None
                self.obs_points_base = self.pc_cam2base(self.obs_points)
                self.target_points_base = self.pc_cam2base(self.target_points)
                whole_point_cloud = np.concatenate([self.obs_points_base, self.target_points_base], axis=0)

                grasp_poses_camera = self.setting_contact_req(obstacle_points=self.obs_points, target_points=self.target_points)
                self.grasp_list = []
                self.score_list = []
                for grasp_pose_cam in grasp_poses_camera:
                    grasp_camera = np.array(grasp_pose_cam.pred_grasps_cam)
                    grasp_world = self.pose_cam2base(grasp_camera.reshape(4,4))
                    self.grasp_list.append(grasp_world)
                    self.score_list.append(grasp_pose_cam.score)
                # if self.vis_pcd == True:
                # self.visualize_points_grasppose(whole_point_cloud, self.grasp_list)
                print("********Init Contact GraspNet***********\n")
                #### 
                
                # get env data
                self.move_along_path(self.envir_joint)
                time.sleep(2)
                self.move_along_path(self.home_joint_point)

                # multi-view data
                self.get_multiview_data(single_view = True)
                obb = self.get_oriented_bounding_box()
                print('z_translation = {}'.format(self.get_normal_translation()))
                # o3d.visualization.draw_geometries([self.pc_segments_pcd, obb])
                print(f"self.multiview_pc_target_base: {self.multiview_pc_target_base}")
                print(f"self.obs_points: {self.obs_points}")
                print("***********Finish get multiview data*************\n")
                
                ###################origin 97##############################

                if i == 0:
                    input_value = 42125  # 接收的整数值
                if i == 1:
                    input_value = 12390
                if i == 2:
                    input_value = 112315 
                self.parallel_or_not, self.placing_stage, self.angle_rad = self.parse_input(input_value)  # 解析阶段和角度
                print("parallel_or_not", self.parallel_or_not)
                print("placing_stage", self.placing_stage)
                print("angle_rad", self.angle_rad)

                print(str(self.placing_stage)[0])
                if str(self.placing_stage)[0] == '2':
                    self.mid_retract_pose = rotZ(-np.pi/2)@ transZ(0.9)@ transX(0.3)@ transY(0.3)@ np.eye(4)@ rotZ(np.pi/4*3)@ rotX(np.pi/4*3)@ rotX(-np.pi/6)
                elif str(self.placing_stage)[0] == '1':
                    self.mid_retract_pose = rotZ(-np.pi/2)@ transZ(0.45)@ transX(0.3)@ transY(0.3)@ np.eye(4)@ rotZ(np.pi/4*3)@ rotX(np.pi/4*3)@ rotX(-np.pi/6)
                
                # Pybullet setup
                self.actor.load_environment()
                self.actor.env._panda.reset(self.home_joint_point[0]+[0, 0, 0])
                self.actor.initial()
                self.actor.grasp_checker = ValidGraspChecker(self.actor.env)

            
                place_pose_base = self.get_the_target_on_cabinet_pose_new()
                if i == 0:
                    placement_rot_rad = self.deg2rad(0)
                    place_pose_base = place_pose_base@rotZ(placement_rot_rad)
                if i == 1:
                    placement_rot_rad = self.deg2rad(15)
                    place_pose_base = place_pose_base@rotZ(placement_rot_rad)
                if i == 2:
                    placement_rot_rad = self.deg2rad(15)
                    place_pose_base = place_pose_base@rotZ(placement_rot_rad) 

                if str(self.placing_stage)[1] == '1':
                    place_pose_base[1, 3] += 0.25
                elif str(self.placing_stage)[1] == '2':
                    place_pose_base[1, 3] += 0.
                elif str(self.placing_stage)[1] == '3':
                    place_pose_base[1, 3] += -0.25

                # 用肉眼看決定角度20240719
                print("*********************************YEAH************************")
                if self.parallel_or_not == 1:
                    target_pose_base = np.eye(4)
                    self.target_center = np.mean(self.multiview_pc_target_base, axis=0)
                    target_pose_base[:3, 3] = self.target_center[:3]
                    target_pose_base[:3, 0] = np.array([0, 0, 1])
                    target_pose_base[:3, 1] = np.array([-1, 0, 0])
                    target_pose_base[:3, 2] = np.array([0, -1, 0])
                elif self.parallel_or_not == 2:
                    target_pose_base = np.eye(4)
                    self.target_center = np.mean(self.multiview_pc_target_base, axis=0)
                    target_pose_base[:3, 3] = self.target_center[:3]
                elif self.parallel_or_not == 3 or self.parallel_or_not == 4:
                    target_pose_base = np.eye(4)
                    self.target_center = np.mean(self.multiview_pc_target_base, axis=0)
                    target_pose_base[:3, 3] = self.target_center[:3]
                    target_pose_base[:3, 0] = np.array([0, 0, 1])
                    target_pose_base[:3, 1] = np.array([-1, 0, 0])
                    target_pose_base[:3, 2] = np.array([0, -1, 0])

                if self.parallel_or_not == 1:
                    # parral
                    target_pose_base = target_pose_base @ rotX(self.angle_rad)
                    target_pose_base = target_pose_base@ rotZ(0)
                    place_pose_base[1, 3] += 0
                    place_pose_base[2, 3] += 0.03
                elif self.parallel_or_not == 2:
                    # pack
                    target_pose_base = target_pose_base@ rotZ(self.angle_rad)
                    place_pose_base[1, 3] += 0
                    place_pose_base[2, 3] += 0.07
                elif self.parallel_or_not == 3:
                    # pile1
                    target_pose_base = target_pose_base@ rotX(self.deg2rad(90))@ rotY(self.angle_rad)
                    place_pose_base[1, 3] += 0
                    place_pose_base[2, 3] += 0.03
                elif self.parallel_or_not == 4:
                    # pile2
                    target_pose_base = target_pose_base@ rotX(self.deg2rad(45))@ rotY(self.angle_rad)
                    place_pose_base[1, 3] += 0
                    place_pose_base[2, 3] += 0.03
                
                
                self.actor.env.draw_ef_coordinate(place_pose_base, 0)
                self.actor.env.draw_ef_coordinate(target_pose_base, 0)
                print("Place pose:", place_pose_base)
                print("Target pose:", target_pose_base)
                print("***********Finish get place/target pose*************\n")

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
               


                # 優先選擇中間的姿態
                if str(self.placing_stage)[0] == '2' and  str(self.placing_stage)[1] == '1':
                    self.actor.refine_grasp_place_pose_base_in_target_pose(self.score_list)
                    self.actor.refine_grasp_place_pose_base_in_target_pose_stage2(self.score_list, place_pose_base[:3, 3])
                if str(self.placing_stage)[0] == '2' and  str(self.placing_stage)[1] == '3':
                    self.actor.refine_grasp_place_pose_base_in_target_pose_stage2_newnew(self.score_list, place_pose_base[:3, 3])
                elif str(self.placing_stage)[0] == '1':
                    self.actor.refine_grasp_place_pose_base_in_target_pose_stage1(self.score_list, place_pose_base[:3, 3])

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

                print(f"列表大小: {len(success_joint_place_list)}")
                success_joint_grasp_list = [success_joint_grasp_list[i] for i in [0, 3, 6, 9]]
                success_joint_mid_list = [success_joint_mid_list[i] for i in [0, 2, 3, 6, 9]]
                success_joint_place_list = [success_joint_place_list[i] for i in [0, 3, 6, 9]]

                # step 5: move to the grasp pose
                self.move_along_path_vel(np.array(success_joint_grasp_list))
                ef_pose = self.get_ef_pose()
                forward_mat = np.eye(4)
                forward_mat[2, 3] = 0.05
                ef_pose = ef_pose.dot(forward_mat)
                quat_pose = pack_pose(ef_pose)
                final_grasp_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
                self.set_pose(final_grasp_pose[0], final_grasp_pose[1])

                # Close gripper
                time.sleep(1)
                self.control_gripper("set_pose", 0.045)
                time.sleep(1)
                print(f"***********Finish robot grasping***********\n")
                
                # # step 5.5: move to the grasp pose
                # self.move_along_path_vel(np.array(success_joint_grasp_list))
                # ef_pose = self.get_ef_pose()
                # forward_mat = np.eye(4)
                # forward_mat[2, 3] = -0.05
                # ef_pose = ef_pose.dot(forward_mat)
                # quat_pose = pack_pose(ef_pose)
                # final_grasp_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
                # self.set_pose(final_grasp_pose[0], final_grasp_pose[1])

                # step 6: move to the place pose
                # 可以微調place pose!!!!
                self.move_along_path_vel(np.array(success_joint_mid_list))
                self.move_along_path_vel(np.array(success_joint_place_list))

                # 往下輕放物體
                if(self.parallel_or_not == 1):
                    ef_pose = self.get_ef_pose()
                    forward_mat = np.eye(4)
                    forward_mat[2, 3] = -0.01
                    ef_pose = forward_mat.dot(ef_pose)
                    quat_pose = pack_pose(ef_pose)
                    final_place_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
                    self.set_pose(final_place_pose[0], final_place_pose[1])
                    time.sleep(2)
                else:
                    ef_pose = self.get_ef_pose()
                    forward_mat = np.eye(4)
                    forward_mat[2, 3] = -0.01
                    ef_pose = forward_mat.dot(ef_pose)
                    quat_pose = pack_pose(ef_pose)
                    final_place_pose = [quat_pose[:3], ros_quat(quat_pose[3:])]
                    self.set_pose(final_place_pose[0], final_place_pose[1])
                    time.sleep(2)

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
                reverse_path_list = []
                reverse_path_list = np.flip(np.array(success_joint_place_list), axis=0)
                print(reverse_path_list.shape)
                self.move_along_path_vel(reverse_path_list[-3:])
                if str(self.placing_stage)[0] == '1':
                    reverse_path_list = []
                    reverse_path_list = np.flip(np.array(self.stage1_home_to_mid), axis=0)
                    self.move_along_path_vel(reverse_path_list)
                if str(self.placing_stage)[0] == '2':
                    print("reverse path")
                    reverse_path_list = []
                    reverse_path_list = np.flip(np.array(self.stage2_home_to_mid), axis=0)
                    self.move_along_path_vel(reverse_path_list)
                print("***********Finish the placing task***********\n")
                self.move_along_path(self.home_joint_point)



    def points_callback(self, msg):
        self.target_points = self.pc2_tranfer(msg)

    def obs_points_callback(self, msg):
        self.obs_points = self.pc2_tranfer(msg)

    def pc2_tranfer(self, ros_msg):
        points = point_cloud2.read_points_list(
                ros_msg, field_names=("x", "y", "z"))
        return np.asarray(points)

    def pc_cam2base(self, pc, crop=True):
        print("*********", pc)
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


    def visual_pc(self, pc, save_pcd = False, file_name=None):
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pc)
        o3d_pc.colors = o3d.utility.Vector3dVector(np.zeros_like(pc))
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([o3d_pc, axes])
        if save_pcd == True:
            o3d.io.write_point_cloud(file_name, o3d_pc)
    
    def load_and_visualize_pcd(self, file_name, voxel_size=0.05):
        # Read the .pcd file
        pcd = o3d.io.read_point_cloud(file_name)
        
        # Apply voxel downsampling
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        # Create coordinate frame axes for reference
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Visualize the downsampled point cloud and axes
        o3d.visualization.draw_geometries([downsampled_pcd, axes])
        print(f"Loaded and visualized voxel-downsampled point cloud from {file_name} with voxel size {voxel_size}")




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
        target_joint.velocity = [0, 0, 0, 0, 0, 0]
        self.joint_goal = np.concatenate((joint_position, [0, 0, 0]))
        print("Move tm joints to position: {}".format(target_joint.position))
        self.tm_joint_pub.publish(target_joint)
        return self.loop_confirm(mode="joint_final")
    
    def set_joint_vel(self, joint_position, velocity):
        """
        Send goal joint and goal velocity to ruckig to move
        """
        target_joint = JointState()
        target_joint.position = joint_position
        target_joint.velocity = velocity
        self.joint_goal = np.concatenate((joint_position, [0, 0, 0]))
        print("Move tm joints to position: {}".format(target_joint.position))
        self.tm_joint_pub.publish(target_joint)
        # return self.loop_confirm(mode="joint")

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
        

    def loop_confirm(self, mode="joint_final"):
        '''
        joint_final代表只有一個joint要很準確
        joint_mid代表要跑很多waypoint, 由於我們自己求的velocity不准, 所以要在使用我們自己所求的vel之前換下一個點, 就可以保都是用ruckig算的vel
        '''
        last_state = None
        last_time = None
        if mode == "joint_mid":
            threshold=0.025
            while True:
                print("check for joint mid")
                dis = np.linalg.norm(self.joint_states-self.joint_goal)
                # print(f"dis: {dis}")
                if last_state is None or np.linalg.norm(self.joint_states - last_state) > 0.001:
                    last_time = time.time()

                if dis < threshold:
                    break
                if time.time() - last_time > 0.2:
                    break
                last_state = self.joint_states
            return True
        elif mode == "joint_final":
            threshold=0.005
            while True:
                print("check for joint final")
                dis = np.linalg.norm(self.joint_states-self.joint_goal)
                # print(f"dis: {dis}")
                if last_state is None or np.linalg.norm(self.joint_states - last_state) > 0.001:
                    last_time = time.time()

                if dis < threshold:
                    break
                if time.time() - last_time > 0.2:
                    break
                last_state = self.joint_states
            return True
        else:
            print("WTF")
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
            self.loop_confirm(mode="joint_final")
    
    def move_along_path_vel(self, path):
        '''
        此求出來的vel不准, 會停頓
        '''
        # First calculate the velocity
        joint_velocities = []
        delta_t = 3
        # Loop through waypoints to compute velocities
        for i in range(len(path) - 1):
            # Difference between consecutive waypoints
            delta_q = path[i+1] - path[i]
            
            # Compute the time interval needed for the max velocity constraint
            velocities = delta_q / delta_t
            joint_velocities.append(velocities)
            print("+++++", joint_velocities)
        joint_velocities.append([0, 0, 0, 0, 0, 0])
        for waypoint, velocity in zip(path, joint_velocities):
            self.set_joint_vel(waypoint, velocity)
            if all(v == 0 for v in velocity):
                # Use 'joint_final' mode if the velocity is all zeros
                self.loop_confirm(mode="joint_final")
            else:
                self.loop_confirm(mode="joint_mid")

    

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

    def shutdown_bandu_server(self):
        rospy.wait_for_service('shutdown_bandu')
        try:
            shutdown_service = rospy.ServiceProxy('shutdown_bandu', Empty)
            shutdown_service()
            rospy.loginfo("Shutdown bandu request sent.")
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e)
    
    def shutdown_uoais_server(self):
        rospy.wait_for_service('shutdown_uoais')
        try:
            shutdown_service = rospy.ServiceProxy('shutdown_uoais', Empty)
            shutdown_service()
            rospy.loginfo("Shutdown bandu request sent.")
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e) 
    
    def env_callback(self, data):
        # 轉換 PointCloud2 到 Open3D 的點雲格式
        points = np.array(list(pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)))
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        
        # 可選：為點雲添加顏色
        colors = self.decode_rgb_from_pcl2(data)
        cloud.colors = o3d.utility.Vector3dVector(colors)
    

        # 轉換點雲到 base_link 框架
        # cloud_transformed = self.transformer.transform_pointcloud(cloud, "camera_link", "base_link")
        
        # 設定過濾閾值（例如，移除距離大於 2 米的點）
        distance_threshold = 1.5
        # 使用點雲的位置過濾點雲
        points = np.asarray(cloud.points)
    
        colors = np.asarray(cloud.colors)
        # get the points 數量
        print("Number of points: ", points.shape[0])
        mask = np.linalg.norm(points, axis=1) < distance_threshold
        cloud_filtered = o3d.geometry.PointCloud()
        cloud_filtered.points = o3d.utility.Vector3dVector(points[mask])
        cloud_filtered.colors = o3d.utility.Vector3dVector(colors[mask])

        # 保存點雲
    
        o3d.io.write_point_cloud("/home/isci/henry_chen_ws/src/contact_graspnet/data/captured_pointcloud.pcd", cloud_filtered)

        # get origin frame
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

        # 顯示保存成功的消息
        o3d.visualization.draw_geometries([cloud_filtered, origin_frame])
        # self.env_points = 
        rospy.signal_shutdown('PointCloud captured and saved.')

    def decode_rgb_from_pcl2(self, point_cloud2_msg):
        colors = []
        for point in pc2.read_points(point_cloud2_msg, field_names=("rgb",), skip_nans=True):
            # 將 rgb 字段從浮點數解碼為整數
            rgb_packed = struct.unpack('I', struct.pack('f', point[0]))[0]
            # 提取單獨的 R、G、B 值
            r = (rgb_packed >> 16) & 0xFF
            g = (rgb_packed >> 8) & 0xFF
            b = (rgb_packed) & 0xFF
            colors.append((r / 255.0, g / 255.0, b / 255.0))  # 轉換為 0 到 1 範圍的浮點數
        return np.asarray(colors)
        

if __name__ == "__main__":
    rospy.init_node("test_realworld")
    real_actor_node = ros_node(renders=False)
    rospy.spin()