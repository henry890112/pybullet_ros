# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import os
import sys
from transforms3d.quaternions import *
from transforms3d.euler import *
from transforms3d.axangles import *
import random

from scipy import interpolate
import scipy.io as sio
import IPython
# from torch import nn
# from torch import optim

# import torch.nn.functional as F
import cv2

import matplotlib.pyplot as plt
import tabulate
import torch

from easydict import EasyDict as edict
import GPUtil
import open3d as o3d
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from PIL import Image
import copy



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.sum_2 = 0
        self.count_2 = 0
        self.means = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sum_2 += val * n
        self.count_2 += n

    def set_mean(self):
        self.means.append(self.sum_2 / self.count_2)
        self.sum_2 = 0
        self.count_2 = 0

    def std(self):
        return np.std(np.array(self.means) + 1e-4)

    def __repr__(self):
        return "{:.3f} ({:.3f})".format(self.val, self.avg)


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    return v / max(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def inv_lookat(eye, target=[0, 0, 0], up=[0, 1, 0]):
    """Generate LookAt matrix."""
    eye = np.float32(eye)
    forward = normalize(target - eye)
    side = normalize(np.cross(forward, up))
    up = np.cross(side, forward)
    R = np.stack([side, up, -forward], axis=-1)
    return R


def rand_sample_joint(env, init_joints=None, near=0.2, far=0.5):
    """
    randomize initial joint configuration
    """
    init_joints_ = env.randomize_arm_init(near, far)
    init_joints = init_joints_ if init_joints_ is not None else init_joints
    return init_joints


def check_scene(env, state, start_rot, object_performance=None, scene_name=None,
                init_dist_low=0.2, init_dist_high=0.5, run_iter=0):
    """
    check if a scene is valid by its distance, view, hand direction, target object state, and object counts
    """
    MAX_TEST_PER_OBJ = 10
    dist = np.linalg.norm(env._get_target_relative_pose('tcp')[:3, 3])
    dist_flag = dist > init_dist_low and dist < init_dist_high
    pt_flag = state[0][0].shape[1] > 100
    z = start_rot[:3, 0] / np.linalg.norm(start_rot[:3, 0])
    hand_dir_flag = z[-1] > -0.3
    target_obj_flag = env.target_name != 'noexists'
    if object_performance is None:
        full_flag = True
    else:
        full_flag = env.target_name not in object_performance or object_performance[env.target_name][0].count < (run_iter + 1) * MAX_TEST_PER_OBJ
    name_flag = 'pitcher' not in env.target_name
    return full_flag and target_obj_flag and pt_flag and name_flag


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def process_image_input(state):
    state[:, :3] *= 255
    if state.shape[1] >= 4:
        state[:, 3] *= 5000
    if state.shape[1] == 5:
        state[:, -1][state[:, -1] == -1] = 50
    return state.astype(np.uint16)


def check_ngc():
    GPUs = GPUtil.getGPUs()
    gpu_limit = max([GPU.memoryTotal for GPU in GPUs])
    return (gpu_limit > 14000)


def process_image_output(sample):
    sample = sample.astype(np.float32).copy()
    n = len(sample)
    if len(sample.shape) <= 2:
        return sample

    sample[:, :3] /= 255.0
    if sample.shape[0] >= 4:
        sample[:, 3] /= 5000
    sample[:, -1] = sample[:, -1] != 0
    return sample


def get_valid_index(arr, index):
    return arr[min(len(arr) - 1, index)]


def deg2rad(deg):
    if type(deg) is list:
        return [x/180.0*np.pi for x in deg]
    return deg/180.0*np.pi


def rad2deg(rad):
    if type(rad) is list:
        return [x/np.pi*180 for x in rad]
    return rad/np.pi*180


def make_video_writer(name, window_width, window_height):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # MJPG
    return cv2.VideoWriter(name, fourcc, 10.0, (window_width, window_height))


def projection_to_intrinsics(mat, width=224, height=224):
    intrinsic_matrix = np.eye(3)
    mat = np.array(mat).reshape([4, 4]).T
    fv = width / 2 * mat[0, 0]
    fu = height / 2 * mat[1, 1]
    u0 = width / 2
    v0 = height / 2

    intrinsic_matrix[0, 0] = fu
    intrinsic_matrix[1, 1] = fv
    intrinsic_matrix[0, 2] = u0
    intrinsic_matrix[1, 2] = v0
    return intrinsic_matrix


def view_to_extrinsics(mat):
    pose = np.linalg.inv(np.array(mat).reshape([4, 4]).T)
    return np.linalg.inv(pose.dot(rotX(np.pi)))


def safemat2quat(mat):
    quat = np.array([1, 0, 0, 0])
    try:
        quat = mat2quat(mat)
    except:
        print(f"{bcolors.FAIL}Mat to quat Error.{bcolors.RESET}")
    quat[np.isnan(quat)] = 0
    return quat


def se3_transform_pc(pose, point):
    if point.shape[1] == 3:
        return pose[:3, :3].dot(point) + pose[:3, [3]]
    else:
        point_ = point.copy()
        point_[:3] = pose[:3, :3].dot(point[:3]) + pose[:3, [3]]
        return point_


def _cross_matrix(x):
    """
    cross product matrix
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def a2e(q):
    p = np.array([0, 0, 1])
    r = _cross_matrix(np.cross(p, q))
    Rae = np.eye(3) + r + r.dot(r) / (1 + np.dot(p, q))
    return mat2euler(Rae)


def get_camera_constant(width):
    K = np.eye(3)
    K[0, 0] = K[0, 2] = K[1, 1] = K[1, 2] = width / 2.0

    offset_pose = np.zeros([4, 4])
    offset_pose[0, 1] = -1.
    offset_pose[1, 0] = offset_pose[2, 2] = offset_pose[3, 3] = 1.
    offset_pose[2, 3] = offset_pose[1, 3] = -0.036
    return offset_pose, K


def se3_inverse(RT):
    R = RT[:3, :3]
    T = RT[:3, 3].reshape((3, 1))
    RT_new = np.eye(4, dtype=np.float32)
    RT_new[:3, :3] = R.transpose()
    RT_new[:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new


def backproject_camera_target(im_depth, K, target_mask):
    Kinv = np.linalg.inv(K)

    width = im_depth.shape[1]
    height = im_depth.shape[0]
    depth = im_depth.astype(np.float32, copy=True).flatten()

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)  # each pixel

    # backprojection
    R = Kinv.dot(x2d.transpose())
    X = np.multiply(
        np.tile(depth.reshape(1, width * height), (3, 1)), R
    )
    X[1] *= -1  # flip y OPENGL. might be required for real-world
    if isinstance(target_mask, np.ndarray):
        mask = (depth != 0) * (target_mask.flatten() == 0)
        return X[:, mask]
    else:
        return X


def backproject_camera_target_realworld(im_depth, K, target_mask):
    Kinv = np.linalg.inv(K)

    width = im_depth.shape[1]
    height = im_depth.shape[0]
    depth = im_depth.astype(np.float32, copy=True).flatten()
    mask = (depth != 0) * (target_mask.flatten() == 0)

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)  # each pixel

    # backprojection
    R = Kinv.dot(x2d.transpose())
    X = np.multiply(
        np.tile(depth.reshape(1, width * height), (3, 1)), R
    )
    return X[:, mask]


def proj_point_img(img, K, offset_pose, points, color=(255, 0, 0), vis=False, neg_y=True, real_world=False):
    xyz_points = offset_pose[:3, :3].dot(points) + offset_pose[:3, [3]]
    if real_world:
        pass
    elif neg_y:
        xyz_points[:2] *= -1
    p_xyz = K.dot(xyz_points)
    p_xyz = p_xyz[:, p_xyz[2] > 0.03]
    x, y = (p_xyz[0] / p_xyz[2]).astype(np.int), (p_xyz[1] / p_xyz[2]).astype(np.int)
    valid_idx_mask = (x > 0) * (x < img.shape[1] - 1) * (y > 0) * (y < img.shape[0] - 1)
    img[y[valid_idx_mask], x[valid_idx_mask]] = (0, 255, 0)
    return img


def unpack_action(action):
    pose_delta = np.eye(4)
    pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
    pose_delta[:3, 3] = action[:3]
    return pose_delta


def unpack_pose(pose, rot_first=False):
    unpacked = np.eye(4)
    if rot_first:
        unpacked[:3, :3] = quat2mat(pose[:4])
        unpacked[:3, 3] = pose[4:]
    else:
        unpacked[:3, :3] = quat2mat(pose[3:])
        unpacked[:3, 3] = pose[:3]
    return unpacked


def quat2euler(quat):
    return mat2euler(quat2mat(quat))


def pack_pose(pose, rot_first=False):
    packed = np.zeros(7)
    if rot_first:
        packed[4:] = pose[:3, 3]
        packed[:4] = safemat2quat(pose[:3, :3])
    else:
        packed[:3] = pose[:3, 3]
        packed[3:] = safemat2quat(pose[:3, :3])
    return packed


def rotZ(rotz):
    RotZ = np.array(
        [
            [np.cos(rotz), -np.sin(rotz), 0, 0],
            [np.sin(rotz), np.cos(rotz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return RotZ


def rotY(roty):
    RotY = np.array(
        [
            [np.cos(roty), 0, np.sin(roty), 0],
            [0, 1, 0, 0],
            [-np.sin(roty), 0, np.cos(roty), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotY


def rotX(rotx):
    RotX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotx), -np.sin(rotx), 0],
            [0, np.sin(rotx), np.cos(rotx), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotX

def transZ(dist):
    Trans = np.eye(4)
    Trans[2, 3] = dist
    return Trans
def transX(dist):
    Trans = np.eye(4)
    Trans[0, 3] = dist
    return Trans
def transY(dist):
    Trans = np.eye(4)
    Trans[1, 3] = dist
    return Trans

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

def vector_angle(a, b):
    dot_product = np.dot(a, b)
    length_a = np.linalg.norm(a)
    length_b = np.linalg.norm(b)
    cos_angle = dot_product / (length_a * length_b)
    angle = np.arccos(cos_angle)
    return angle

def unpack_pose_rot_first(pose):
    unpacked = np.eye(4)
    unpacked[:3, :3] = quat2mat(pose[:4])
    unpacked[:3, 3] = pose[4:]
    return unpacked


def pack_pose_rot_first(pose):
    packed = np.zeros(7)
    packed[4:] = pose[:3, 3]
    packed[:4] = safemat2quat(pose[:3, :3])
    return packed


def inv_pose(pose):
    return pack_pose(np.linalg.inv(unpack_pose(pose)))


def relative_pose(pose1, pose2):
    return pack_pose(np.linalg.inv(unpack_pose(pose1)).dot(unpack_pose(pose2)))


def compose_pose(pose1, pose2):
    return pack_pose(unpack_pose(pose1).dot(unpack_pose(pose2)))


def skew_matrix(r):
    """
    Get skew matrix of vector.
    r: 3 x 1
    r_hat: 3 x 3
    """
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def inv_relative_pose(pose1, pose2, decompose=False):
    """
    pose1: b2a
    pose2: c2a
    relative_pose:  b2c
    shape: (7,)
    """

    from_pose = np.eye(4)
    from_pose[:3, :3] = quat2mat(pose1[3:])
    from_pose[:3, 3] = pose1[:3]
    to_pose = np.eye(4)
    to_pose[:3, :3] = quat2mat(pose2[3:])
    to_pose[:3, 3] = pose2[:3]
    relative_pose = se3_inverse(to_pose).dot(from_pose)
    return relative_pose


def ros_quat(tf_quat):  # wxyz -> xyzw
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat


def tf_quat(ros_quat):  # xyzw -> wxyz
    quat = np.zeros(4)
    quat[0] = ros_quat[-1]
    quat[1:] = ros_quat[:-1]
    return quat


def distance_by_translation_point(p1, p2):
    """
    Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def regularize_pc_point_count(pc, npoints, use_farthest_point=True):
    """
    If point cloud pc has less points than npoints, it oversamples.
    Otherwise, it downsample the input pc to have npoint points.
    use_farthest_point: indicates whether to use farthest point sampling
    to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        if use_farthest_point:
            pc = torch.from_numpy(pc).cuda()[None].float()
            new_xyz = (
                gather_operation(
                    pc.transpose(1, 2).contiguous(), furthest_point_sample(pc[..., :3].contiguous(), npoints)
                )
                .contiguous()
                )
            pc = new_xyz[0].T.detach().cpu().numpy()

        else:
            center_indexes = np.random.choice(
                range(pc.shape[0]), size=npoints, replace=False
            )
            pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


class bcolors:
    OK = '\033[92m'  # GREEN
    WARNING = '\033[93m'  # YELLOW
    FAIL = '\033[91m'  # RED
    RESET = '\033[0m'  # RESET COLOR


#Henry
# -*- coding: utf-8 -*-

import pybullet as p

class SlideBars():
    def __init__(self,Id):
        self.Id=Id
        self.motorNames=[]
        self.motorIndices=[]
        self.motorLowerLimits=[]
        self.motorUpperLimits=[]
        self.slideIds=[]

        self.numJoints=p.getNumJoints(self.Id)

        # self.init_joint = [0.2, -1, 2, 0, 1.571, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.init_joint = [0.2, -1.25, 2.48, -0.4, 1.571, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.left_view = [0.9424772262573242, 0.46297121047973633, 1.2897052764892578, 0.6944568157196045, 0.8928732872009277, 2.7282233238220215, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.right_view = [-0.44643640518188477, 0.2980003356933594, 1.2239999771118164, 0.5950002670288086, 2.4140639305114746, 1.0416851043701172, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        '''
        left_view_ef_mat = np.array([[ 0.98757027,  0.02243495,  0.15556875,  0.45691898],
                                    [ 0.14573556, -0.501431,   -0.85283533,  0.36891946],
                                    [ 0.05887368,  0.86490672, -0.49846791, -0.33324587],
                                    [ 0.,          0.,          0.,          1.]])

        right_view_ef_mat = np.array([[ 0.98691477, -0.16087768,  0.010845,    0.46446365],
                                    [-0.10023915, -0.55945926,  0.82277424, -0.28816143],
                                    [-0.12629867, -0.81309514, -0.56826485, -0.19399673],
                                    [ 0.,          0.,          0.,          1.]])

        '''

    def add_slidebars(self):
        for i in range(self.numJoints):
            jointInfo=p.getJointInfo(self.Id,i)
            jointName=jointInfo[1].decode('ascii')
            qIndex = jointInfo[3]
            lowerLimits=jointInfo[8]
            upperLimits=jointInfo[9]
            if qIndex > -1:
                self.motorNames.append(jointName)
                self.motorIndices.append(i)
                self.motorLowerLimits.append(lowerLimits)
                self.motorUpperLimits.append(upperLimits)

        for i in range(len(self.motorIndices)): 
            if self.motorLowerLimits[i]<=self.motorUpperLimits[i]:  
                slideId=p.addUserDebugParameter(self.motorNames[i],self.motorLowerLimits[i],self.motorUpperLimits[i],self.init_joint[i])
            else: 
                slideId=p.addUserDebugParameter(self.motorNames[i],self.motorUpperLimits[i],self.motorLowerLimits[i],self.init_joint[i])
            self.slideIds.append(slideId)

        return self.motorIndices
    



    def get_slidebars_values(self):
        slidesValues=[]
        for i in self.slideIds:
            value=p.readUserDebugParameter(i)
            slidesValues.append(value)
        return slidesValues
        
def depth2pc(depth, K, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    mask = np.where(depth > 0)
    x,y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]
        
    pc = np.vstack((world_x, world_y, world_z)).T
    return (pc, rgb)

def extract_point_clouds(depth, K, segmap=None, rgb=None, z_range=[0.5,1.8], segmap_id=0, skip_border_objects=False, margin_px=5):
        """
        Converts depth map + intrinsics to point cloud. 
        If segmap is given, also returns segmented point clouds. If rgb is given, also returns pc_colors.

        Arguments:
            depth {np.ndarray} -- HxW depth map in m
            K {np.ndarray} -- 3x3 camera Matrix

        Keyword Arguments:
            segmap {np.ndarray} -- HxW integer array that describes segeents (default: {None})
            rgb {np.ndarray} -- HxW rgb image (default: {None})
            z_range {list} -- Clip point cloud at minimum/maximum z distance (default: {[0.2,1.8]})
            segmap_id {int} -- Only return point cloud segment for the defined id (default: {0})
            skip_border_objects {bool} -- Skip segments that are at the border of the depth map to avoid artificial edges (default: {False})
            margin_px {int} -- Pixel margin of skip_border_objects (default: {5})

        Returns:
            [np.ndarray, dict[int:np.ndarray], np.ndarray] -- Full point cloud, point cloud segments, point cloud colors
        """

        '''
        table_rgb = Image.open('./paper_data/table_object/color_table1.png')
        table_rgb = np.array(table_rgb)
        table_rgb = table_rgb/255
        table_depth = np.load('./paper_data/table_object/depth_table1.npy')
        table_mask = Image.open('./paper_data/table_object/mask_table1.png')
        table_mask = np.array(table_mask)

        intrinsic_matrix = np.array([[240, 0, 240],
                                    [0, 240, 240],
                                    [0, 0, 1]])

        table_pc_full, table_pc_segments, table_pc_colors = extract_point_clouds(depth = table_depth,   K = intrinsic_matrix, segmap = table_mask, rgb = table_rgb, z_range = [0.2, 1.])
        '''

        if K is None:
            raise ValueError('K is required either as argument --K or from the input numpy file')
            
        # Convert to pc 
        pc_full, pc_colors = depth2pc(depth, K, rgb)

        # Threshold distance
        if pc_colors is not None:
            pc_colors = pc_colors[(pc_full[:,2] < z_range[1]) & (pc_full[:,2] > z_range[0])] 
        pc_full = pc_full[(pc_full[:,2] < z_range[1]) & (pc_full[:,2] > z_range[0])]
        
        # Extract instance point clouds from segmap and depth map
        pc_segments = {}
        if segmap is not None:
            pc_segments = {}
            obj_instances = [segmap_id] if segmap_id else np.unique(segmap[segmap>0])
            for i in obj_instances:
                if skip_border_objects and not i==segmap_id:
                    obj_i_y, obj_i_x = np.where(segmap==i)
                    if np.any(obj_i_x < margin_px) or np.any(obj_i_x > segmap.shape[1]-margin_px) or np.any(obj_i_y < margin_px) or np.any(obj_i_y > segmap.shape[0]-margin_px):
                        print('object {} not entirely in image bounds, skipping'.format(i))
                        continue
                inst_mask = segmap==i
                pc_segment,_ = depth2pc(depth*inst_mask, K)
                pc_segments[i] = pc_segment[(pc_segment[:,2] < z_range[1]) & (pc_segment[:,2] > z_range[0])] #regularize_pc_point_count(pc_segment, grasp_estimator._contact_grasp_cfg['DATA']['num_point'])

        return pc_full, pc_segments, pc_colors
        
# get the camera frame 圖像
def getCamera(
    transformation,
    fx = 320.,
    fy = 320.,
    cx = 320.,
    cy = 320.,
    scale=0.1,
    coordinate=False,
    shoot=False,
    length=4,
    color=np.array([0, 1, 0]),
    z_flip=True,
):
    '''
    camera = getCamera(identity_mat, length=0.1, color=np.array([1, 0, 0]))
    o3d.visualization.draw_geometries([environment_pcd, origin_frame, *camera])
    '''
    # Return the camera and its corresponding frustum framework
    if coordinate:
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        camera.transform(transformation)
    else:
        camera = o3d.geometry.TriangleMesh()
    # Add origin and four corner points in image plane
    points = []
    camera_origin = np.array([0, 0, 0, 1])
    points.append(np.dot(transformation, camera_origin)[0:3])
    # Calculate the four points for of the image plane
    magnitude = (cy**2 + cx**2 + fx**2) ** 0.5
    if z_flip:
        plane_points = [[-cx, -cy, fx], [-cx, cy, fx], [cx, -cy, fx], [cx, cy, fx]]
    else:
        plane_points = [[-cx, -cy, -fx], [-cx, cy, -fx], [cx, -cy, -fx], [cx, cy, -fx]]
    for point in plane_points:
        point = list(np.array(point) / magnitude * scale)
        temp_point = np.array(point + [1])
        points.append(np.dot(transformation, temp_point)[0:3])
    # Draw the camera framework
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [1, 3], [3, 4]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    meshes = [camera, line_set]

    if shoot:
        shoot_points = []
        shoot_points.append(np.dot(transformation, camera_origin)[0:3])
        shoot_points.append(np.dot(transformation, np.array([0, 0, -length, 1]))[0:3])
        shoot_lines = [[0, 1]]
        shoot_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(shoot_points),
            lines=o3d.utility.Vector2iVector(shoot_lines),
        )
        shoot_line_set.paint_uniform_color(color)
        meshes.append(shoot_line_set)

    return meshes



def create_arrow(vec, color, vis=None, vec_len=None, scale=.06, radius=.12, position=(0,0,0),
                 object_com=None, face_detection_x=False, face_detection_z = False):
    """
    #Henry: vec 為object stable plane 的法向量
    Creates an error, where the arrow size is based on the vector magnitude.
    :param vec:
    :param color:
    :param vis:
    :param scale:
    :param position:
    :return:
    :param face_detection: 選擇對我的論文來說較好的x和z軸, 皆和相機座標的z軸平行比較
        想要x軸朝內, face_detection_x = True
        想要z軸朝外, face_detection_z = True
    """
    if vec_len is None:
        vec_len = (np.linalg.norm(vec))

    if isinstance(vec, list):
        vec = np.array(vec)

    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * vec_len * scale,
        cone_radius=0.06 * vec_len * radius,
        cylinder_height=0.8 * vec_len * scale,
        cylinder_radius=0.04 * vec_len * radius
    )

    # the default arrow points straightup
    # therefore, we find the rotation that takes us from this arrow to our target vec, "vec"

    if object_com is not None:
        vec_endpoint = position + vec
        neg_vec_endpoint = position - vec
        #Henry：避免arrow的方向往object內部插出去
        if np.linalg.norm(vec_endpoint - object_com) < np.linalg.norm(neg_vec_endpoint - object_com):
            vec = -vec

    if face_detection_x:
        print("x axis face detection")
        if np.dot(vec, [0, 0, 1]) > 0:
            vec = -vec
    if face_detection_z:
        print("z axis face detection")
        if np.dot(vec, [0, 0, 1]) < 0:
            vec = -vec
    # print("ln554 create_arrow")
    # print(vec)
    rot_mat = get_rotation_matrix_between_vecs(vec, [0, 0, 1])
    #Henry: print the rotation matrix from table to the arrow on the object
    # print(rot_mat)
    mesh_arrow.rotate(rot_mat, center=np.array([0,0,0]))

    H = np.eye(4)
    H[:3, 3] = position
    mesh_arrow.transform(H)

    mesh_arrow.paint_uniform_color(color)
    return mesh_arrow, vec/np.linalg.norm(vec)

def get_rotation_matrix_between_vecs(target_vec, start_vec):
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    # rotation with theta = the cosine angle between the two vectors
    """

    :param target_vec:
    :param start_vec:
    :return: 3x3 rotation matrix representing relative rotation from start_vec to target_vec
    """
    # the formula fails when the vectors are collinear

    target_vec = np.array(target_vec)
    start_vec = np.array(start_vec)
    assert len(target_vec.shape) == 1 or len(target_vec.shape) == 2
    assert target_vec.shape == start_vec.shape, (target_vec.shape, start_vec.shape)
    target_vec = target_vec / np.linalg.norm(target_vec, axis=-1)
    start_vec = start_vec/np.linalg.norm(start_vec, axis=-1)

    # the formula doesn't work in this case...
    # TODO: make this an actual rotation instead of a flip...
    if np.all(np.isclose(start_vec, -1 * target_vec, atol=1.e-3)):
        # return -np.eye(3)
        return R.from_euler("x", np.pi).as_matrix()

    K = get_skew_symmetric_matrix(np.cross(start_vec, target_vec))
    rotation = np.eye(3,3) + K + \
               np.dot(K, K) * \
               1/(1+np.dot(start_vec, target_vec) + 1E-7)
    return rotation

def get_skew_symmetric_matrix(vec):
    return np.asarray([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0],
    ])

def check_pose_difference(current_pose, target_pose, tolerance=0.01):
    """
    檢查物體的當前姿態與目標姿態之間的差異是否在許可範圍內。
    
    參數:
        current_pose (numpy.ndarray): 物體的當前4x4轉換矩陣。
        target_pose (numpy.ndarray): 物體的目標4x4轉換矩陣。
        tolerance (float): 許可的最大差異範圍。
        
    返回:
        bool: 當前姿態是否達到目標姿態的範圍內。
    """
    # 計算從當前姿態到目標姿態的變換矩陣
    diff_matrix = np.dot(np.linalg.inv(current_pose), target_pose)
    
    # 從差異矩陣中提取平移差異
    translation_diff = np.linalg.norm(diff_matrix[:3, 3])
    print("translation_diff = ", translation_diff)
    
    # 從差異矩陣中提取旋轉差異，並確保輸入值位於[-1, 1]範圍內
    cos_angle = (np.trace(diff_matrix[:3, :3]) - 1) / 2
    cos_angle_clamped = np.clip(cos_angle, -1, 1)  # 限制cos_angle的值在[-1, 1]範圍內
    # get the rotation difference in radian
    rotation_diff = np.arccos(cos_angle_clamped)
    print("rotation_diff = ", rotation_diff)
    
    # 檢查平移和旋轉是否都在許可的範圍內
    return translation_diff <= tolerance and rotation_diff <= tolerance

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(5.0, 0.0)
    return False

def load_data(path, object_name, multiview_index):
    '''
    return type: numpy array(rgb, depth, mask)
    '''
    rgb_filename = os.path.join(path, f'color{multiview_index}_{object_name}.png')
    depth_filename = os.path.join(path, f'depth{multiview_index}_{object_name}.npy')
    mask_filename = os.path.join(path, f'mask{multiview_index}_{object_name}.png')

    rgb = Image.open(rgb_filename)
    depth = np.load(depth_filename)
    mask = Image.open(mask_filename)

    rgb = np.array(rgb)
    rgb = rgb/255
    depth = depth
    mask = np.array(mask)

    return rgb, depth, mask

def create_point_cloud_and_camera(rgb, depth, mask, K, camera_transform=np.eye(4), color=None):
    '''
    create the point cloud by rgbd data
    then, transfrom the numpy array to open3d point cloud
    return type: o3d.geometry.PointCloud
    '''
    pc_full, pc_segments, pc_colors = extract_point_clouds(depth=depth, K=K, segmap=mask, rgb=rgb, z_range=[0.2, 0.6])
    # print(pc_segments)
    # print(pc_segments[1])
    # print(pc_segments[1].shape) 
    pc_full_pcd = o3d.geometry.PointCloud()
    pc_full_pcd.points = o3d.utility.Vector3dVector(pc_full)
    pc_full_pcd.colors = o3d.utility.Vector3dVector(pc_colors)
    pc_full_pcd.transform(camera_transform)


    if pc_segments != {}:
        pc_segments_pcd = o3d.geometry.PointCloud()
        pc_segments_pcd.points = o3d.utility.Vector3dVector(pc_segments[1])
        pc_segments_pcd.colors = o3d.utility.Vector3dVector(pc_colors)
        pc_segments_pcd.transform(camera_transform)
    else:
        # build 一個空的pc_segments_pcd
        pc_segments_pcd = o3d.geometry.PointCloud()
        pc_segments_pcd.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
        pc_segments_pcd.colors = o3d.utility.Vector3dVector(np.zeros((1,3)))
    
    
    
    if color is not None:
        pc_full_pcd.paint_uniform_color(color)
        pc_segments_pcd.paint_uniform_color(color)

    return pc_full_pcd, pc_segments_pcd
    pc_full, pc_segments, pc_colors = extract_point_clouds(depth=depth, K=K, segmap=mask, rgb=rgb, z_range=[0.2, 0.6])

    pc_full_pcd = o3d.geometry.PointCloud()
    pc_full_pcd.points = o3d.utility.Vector3dVector(pc_full)
    pc_full_pcd.colors = o3d.utility.Vector3dVector(pc_colors)

    pc_segments_pcd = o3d.geometry.PointCloud()
    pc_segments_pcd.points = o3d.utility.Vector3dVector(pc_segments[1])
    pc_segments_pcd.colors = o3d.utility.Vector3dVector(pc_colors)
    
    
    pc_full_pcd.transform(camera_transform)
    pc_segments_pcd.transform(camera_transform)
    if color is not None:
        pc_full_pcd.paint_uniform_color(color)
        pc_segments_pcd.paint_uniform_color(color)

    return pc_full_pcd, pc_segments_pcd

def save_observation_images(path, env, placed_obj, multiview_index=0, visual=False):
    """
    從環境觀察中獲取RGB圖像、深度圖像和掩碼圖像，並將它們展示和保存。
    儲存同一個角度下的所有物體rgb, depth, mask圖片

    參數:
        env: 環境物件，用於獲取觀察資料。
        placed_name: 用於生成文件名的名稱列表。
    """
    num_objs = len(placed_obj)
    print(placed_obj)
    for i, index in enumerate(placed_obj.keys()):
        placed_name = placed_obj[index]
        env.target_idx = index

        obs, _, _, _, _ = env._get_observation()

        # 展示圖像
        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(num_objs, 3, i*3+1)
        plt.imshow(obs[1][:3].T)  # RGB

        fig.add_subplot(num_objs, 3, i*3+2)
        plt.imshow(obs[1][3].T)  # Depth

        fig.add_subplot(num_objs, 3, i*3+3)
        plt.imshow(obs[1][4].T)  # Mask
        plt.title(f"{index}: {placed_name}", fontsize=8) 

        # 保存RGB圖像
        rgb = cv2.cvtColor(obs[1][:3].T, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(path, f'color{multiview_index}_{placed_name}.png'), rgb*255)

        # 保存深度圖像
        depth = obs[1][3].T
        np.save(os.path.join(path, f'depth{multiview_index}_{placed_name}.npy'), depth)

        # 保存掩碼圖像
        mask = obs[1][4].T
        ones = np.ones(shape=(640, 640))
        mask = ones - mask
        cv2.imwrite(os.path.join(path, f'mask{multiview_index}_{placed_name}.png'), mask)

        print(index, placed_name)
    if visual:
        plt.show()

### combine the pcd
# combine the point cloud pc_init , pc_left, pc_right
def get_combine_pcd(pc_init, pc_left, pc_right):
    '''
    return o3d.geometry.PointCloud
    '''
    pc_combine = copy.deepcopy(pc_init)
    pc_combine.points = o3d.utility.Vector3dVector(np.vstack((pc_init.points, pc_left.points, pc_right.points)))
    pc_combine.colors = o3d.utility.Vector3dVector(np.vstack((np.asarray(pc_init.colors), np.asarray(pc_left.colors), np.asarray(pc_right.colors))))
    return pc_combine

def degree_between_vector(v1, v2):
    norm_a = np.linalg.norm(v1)
    norm_b = np.linalg.norm(v2)
    return np.arccos(np.dot(v1, v2) / (norm_a * norm_b)) * (180.0 / np.pi)