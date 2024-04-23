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
print(sys.path)
from env.ycb_scene import SimulatedYCBEnv
from utils.grasp_checker import ValidGraspChecker
from utils.utils import *
from utils.planner import GraspPlanner
import utils.tcp_utils as tcp_utils
import pyscreenshot as ImageGrab

parent_directory = os.path.join(os.path.dirname(__file__), '..')
print(parent_directory)

from utils.planner import GraspPlanner
planner = GraspPlanner()

def expert_plan(goal_pose, world=False, visual=False):
    if world:
        pos, orn = env._get_ef_pose()
        ef_pose_list = [*pos, *orn]
    else:
        ef_pose_list = [0, 0, 0, 0, 0, 0, 1]
    goal_pos = [*goal_pose[:3], *ros_quat(goal_pose[3:])]

    solver = planner.plan(ef_pose_list, goal_pos, path_length=30)
    if visual:
        path_visulization(solver)
    path = solver.getSolutionPath().getStates()
    planer_path = []
    for i in range(len(path)):
        waypoint = path[i]
        rot = waypoint.rotation()
        action = [waypoint.getX(), waypoint.getY(), waypoint.getZ(), rot.w, rot.x, rot.y, rot.z]
        planer_path.append(action)

    return planer_path

def path_visulization(ss):
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
    
def execute_motion_plan(env, plan, execute=False, gripper_set="close", repeat=100):
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
        jointPoses = env._panda.solveInverseKinematics(next_pos[:3], ros_quat(next_pos[3:]))
        if gripper_set == "close":
            jointPoses[6] = 0.85
        else:
            jointPoses[6] = 0.0
        jointPoses = jointPoses[:7].copy()  # Consider only the first 7 joint positions
                
        if execute:
            # Execute the action and obtain the observation
            obs = env.step(jointPoses, config=True, repeat=repeat)[0]
            # print("JointPoses = ", jointPoses)
        else:
            # Only reset the robot's joint positions
            env._panda.reset(joints=jointPoses)
            # 在path length中每一步都檢查是否會發生碰撞
            if(env._panda.check_for_collisions() == True):
                print(f"Collision detected in step {i} / {len(plan)}")
                return False
    return True

visual_simulation = False
### 將open3d下的點雲轉到world座標
world_frame_pose = np.array([[ 1.,    0.,    0.,   -0.05],
                            [ 0.,    1.,    0.,    0.  ],
                            [ 0.,    0.,    1.,   -0.65],
                            [ 0.,    0.,    0.,    1.  ]])

init_ef_mat = np.array([[-1.98785608e-01,  7.23231525e-01,  6.61377686e-01,  1.06898375e-01],
                        [9.80042993e-01,  1.46612626e-01,  1.34240345e-01, -9.29623842e-02],
                        [1.20530092e-04,  6.74863616e-01, -7.37942468e-01, -0.3],
                        [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

left_view_ef_mat = np.array([[ 0.98757027,  0.02243495,  0.15556875,  0.45691898],
                            [ 0.14573556, -0.501431,   -0.85283533,  0.36891946],
                            [ 0.05887368,  0.86490672, -0.49846791, -0.3],
                            [ 0.,          0.,          0.,          1.]])

right_view_ef_mat = np.array([[ 0.98691477, -0.16087768,  0.010845,    0.46446365],
                            [-0.10023915, -0.55945926,  0.82277424, -0.28816143],
                            [-0.12629867, -0.81309514, -0.56826485, -0.3],
                            [ 0.,          0.,          0.,          1.]])

cam_offset = np.eye(4)
# 先轉到pybullet座標後再往上移動0.13變相機座標
cam_offset[:3, 3] = (np.array([0., 0.1186, -0.0191344123493]))
# open3d 和 pybullet中的coordinate要對z旋轉180度才會一樣
cam_offset[:3, :3] = np.array([[-1, 0, 0],
                                [0, -1, 0],
                                [0, 0, 1]])

intrinsic_matrix = np.array([[320, 0, 320],
                             [0, 320, 320],
                             [0, 0, 1]])

### 轉換關係
origin_camera2world = cam_offset@ np.linalg.inv(init_ef_mat)@ world_frame_pose
left_camera2world = cam_offset@ np.linalg.inv(left_view_ef_mat)@ world_frame_pose
right_camera2world = cam_offset@ np.linalg.inv(right_view_ef_mat)@ world_frame_pose

'''
get data file name in json file and load mesh in pybullet
then reset robot and object position
'''
# tcp server
target_place_name = tcp_utils.start_server_target_name('127.0.0.1', 11111)
file = os.path.join(parent_directory, "object_index", 'contact_plane_object.json')   # 此json檔可以自己改
with open(file) as f: file_dir = json.load(f)
file_dir = file_dir[target_place_name]     #只取json檔中的"test"
file_dir = [f[:-5] for f in file_dir]
test_file_dir = list(set(file_dir))

env = SimulatedYCBEnv()
env._load_index_objs(test_file_dir)      #597

#ycb scene 中的 init 可以定義target_fixed, true代表target object不會自由落下 (第一次呼叫cabinet設定即可)
state = env.cabinet(save=False, enforce_face_target=True)  

'''
single release: (只在num_object = 1有用) true為以自己設定的角度放在桌上; (多object也可用)false就是pack放在桌上
if_stack: true代表旁邊有東西會擋住掉落下來的物體
'''
num_object = 0
single_release = False
if_stack = False
vis = True

state = env.cabinet(save=False, reset_free=True, num_object=num_object, if_stack=if_stack, single_release=single_release)

placed_obj = {}
placed_idx = np.where(np.array(env.placed_objects))[0]
placed_name = np.array(env.obj_indexes)[np.where(np.array(env.placed_objects))]
for i in range(num_object):
    placed_obj[placed_idx[i]] = placed_name[i]
print(placed_obj)

# ## Valid Grasp Pose Checker
# This API will take input as grasping pose group and validate collision between gripper and target object. It uses pybullet API to test if the gripper mesh has contact points with current environment when the gripper is set to fully opened.

'''
When user declare a checker, it will load a additional robot gripper at world origin.
Do NOT re-declare ValidGraspChecker.
'''

grasp_checker = ValidGraspChecker(env)

# # 將final中轉換後的所有的grasp pose load進來

# tcp 接收
checker_message = tcp_utils.start_server_checker('127.0.0.1', 22222)
if checker_message:
    print(f"成功接收到：{checker_message}")
new_pred_grasps_cam_place = np.load(os.path.join(parent_directory, 'results', 'new_pred_grasps_cam_place.npy'), allow_pickle=True)
print(new_pred_grasps_cam_place.shape)
scores = np.load(os.path.join(parent_directory, 'results', 'scores.npy'), allow_pickle=True)
print(scores.shape)

# ### 擺放target object到place pose

# relative pose of z axis rotation
def get_rotation_matrix_z_4x4(input_value):
    # 將輸入值轉換為旋轉角度（度）
    rotation_angle_degrees = 90 * input_value
    # 將旋轉角度轉換為弧度
    rotation_angle_radians = np.radians(rotation_angle_degrees)
    
    # 計算 4x4 旋轉矩陣
    rotation_matrix = np.array([
        [np.cos(rotation_angle_radians), -np.sin(rotation_angle_radians), 0, 0],
        [np.sin(rotation_angle_radians), np.cos(rotation_angle_radians), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    return rotation_matrix

# tcp client
final_place_target_matrix = tcp_utils.start_server('127.0.0.1', 33333)
place_pose_world = tcp_utils.start_server('127.0.0.1', 55557)
target_pose_world = tcp_utils.start_server('127.0.0.1', 56599)

print(env.obj_indexes)
state = env.cabinet(save=False, reset_free=True, num_object=1, 
if_stack=if_stack, single_release=single_release, place_target_matrix=final_place_target_matrix)
'''
此place_pose傳進去之後是改變原本urdf物品的6d pose, 但我們的6d pose有經過自己的演算法對z軸進行旋轉, 
所以要利用相對關係將此pose轉換回來
'''

'''
The extract_grasp() function takes grasp group[N, 4, 4] as input and outputs valid grasps.
The parameter "drawback_distance" is the distance to draw back the end effector pose along z-axis in validation process.
The parameter "filter_elbow" denote if checker use estimated elbow point and bounding box of table
    as one of the measurements to prevent collision of other joint.
Note: The estimated elbow point is NOT calculate by IK, so it's nearly a rough guess.
'''

grasp_array, grasp_index = grasp_checker.extract_grasp(new_pred_grasps_cam_place,
                                                       drawback_distance=0.03,
                                                       visual=visual_simulation,
                                                       filter_elbow=True,
                                                       time=0.05, distance=0.0001)
# 找到grasp_index中最大的score
max_score = 0
max_index = 0
print(grasp_index)
print("grasp_index = ", len(grasp_index))
print("scores = ", len(scores))
print(scores.shape)
for i in range(len(grasp_index)):
    if scores[grasp_index[i]] > max_score:
        max_score = scores[grasp_index[i]]
        max_index = grasp_index[i]
print("max_score = ", max_score)
print("max_index = ", max_index)

# print the number of valid grasp
print(f"Number of valid grasp: {len(grasp_index)}")


# Visualize validation process by setting "visual=True"
if len(grasp_index):
    _, _ = grasp_checker.extract_grasp(grasp_array,
                                       drawback_distance=0.03,
                                       visual=visual_simulation, time=0.1)
print(f"Number of valid grasp: {len(grasp_index)}")                                    


# # 測試夾取及擺放是否能成功
# 的到world ef的place pose和grasp pose

# ### sort grasp index & check the place pose 是否可以放置

env._panda.reset() 
 # 將best_grasp_pose的y軸和[1, 0, 0]做內積, 若<0則代表要旋轉180度
# 僅遍歷 grasp_index 中包含的索引, 且grasp index依照分數高低排好
grasp_index = np.array(grasp_index)
grasp_index = grasp_index[np.argsort(-scores[grasp_index])]
print("grasp_index = ", grasp_index)
for i in grasp_index:
    grasp_pose = new_pred_grasps_cam_place[i]
    # 檢查並根據條件進行旋轉
    if np.dot(grasp_pose[:3, 1], np.array([0, 0, 1])) < 0:
        # print(f"Grasp pose {i}：Rotate 180 degree")
        new_pred_grasps_cam_place[i] = np.dot(grasp_pose, rotZ(np.pi))

def execute_plan_with_check(env, pose, execute, draw_coordinate):
    if draw_coordinate:
        env.draw_ef_coordinate(pose, 1)
    plan = expert_plan(pack_pose(pose), world=True, visual=False)
    # checker true代表對的plan及pose
    plan_checker = execute_motion_plan(env, plan, execute=execute, gripper_set="open")
    checker = check_pose_difference(env._get_ef_pose(mat=True), pose, tolerance=0.04)
    return plan_checker, checker

def adjust_pose_with_bias(pose, bias):
    return pose.dot(transZ(bias))

execute = False
draw_coordinate = True
count = 0

# start time
start_time = time.time()
for i in grasp_index:
    env._panda.reset()
    checker_list = []
    
    grasp_pose = new_pred_grasps_cam_place[i]
    relative_grasp_transform = np.linalg.inv(place_pose_world) @ grasp_pose
    final_grasp_pose = target_pose_world @ relative_grasp_transform
    
    # 第一次執行計劃並檢查
    pose_z_bias = adjust_pose_with_bias(final_grasp_pose, -0.1)
    plan_checker, checker = execute_plan_with_check(env, pose_z_bias, execute, draw_coordinate)
    count += 1
    print(f"第 {count} / {len(grasp_index)} 個夾取姿態。")
    print("=====================================================")
    if visual_simulation:
        time.sleep(0.2)

    if not plan_checker or not checker:
        if not plan_checker and not checker:
            print("No.1存在碰撞且檢查失敗。")
        elif not plan_checker:
            print("No.1路徑存在碰撞。")
        elif not checker:
            print("No.1姿態錯誤。")
        continue

    # 第二次執行計劃並檢查
    mid_retract_pose = transZ(-0.2)@ transX(0.3)@ transY(0.3)@ np.eye(4)@ rotZ(np.pi/4*3)@ rotX(np.pi/4*3)
    plan_checker, checker = execute_plan_with_check(env, mid_retract_pose, execute, draw_coordinate)
    print("=====================================================")
    if visual_simulation:
        time.sleep(0.2)
    if not plan_checker or not checker:
        if not plan_checker and not checker:
            print("No.2存在碰撞且檢查失敗。")
        elif not plan_checker:
            print("No.2路徑存在碰撞。")
        elif not checker:
            print("No.2姿態錯誤。")
        continue
        
    # 第三次執行計劃並檢查
    pose_z_bias = adjust_pose_with_bias(grasp_pose, 0.015)
    plan_checker, checker = execute_plan_with_check(env, pose_z_bias, execute, draw_coordinate)
    print("=====================================================")
    if visual_simulation:
        time.sleep(0.2)
    if not plan_checker or not checker:
        if not plan_checker and not checker:
            print("No.3存在碰撞且檢查失敗。")
        elif not plan_checker:
            print("No.3路徑存在碰撞。")
        elif not checker:
            print("No.3姿態錯誤。")
        continue


    
    print(f"第 {count} / {len(grasp_index)} 個有效的夾取姿態。")
    tcp_utils.send_matrix('127.0.0.1', 44411, grasp_pose)
    print("找到合適的放置姿態。")
    img = ImageGrab.grab()
    random = np.random.randint(0, 1000)
    img.save(os.path.join(parent_directory, f'results/scene_placing_{random}.png'))
    _, _ = grasp_checker.extract_grasp(np.expand_dims(grasp_pose, axis=0),
                                    drawback_distance=0.03,
                                    visual=visual_simulation, time=3)
    break

if count == len(grasp_index):
    print("No valid grasp pose.")
    # 當執行到這代表沒有合適姿態
    grasp_pose = np.eye(4)
    tcp_utils.send_matrix('127.0.0.1', 44411, grasp_pose)

# end time
end_time = time.time()
print(f"!!!!!!!!!!!!!!!Time elapsed: {end_time - start_time} seconds!!!!!!!!!!!!!!!!!!!!!!!!")


