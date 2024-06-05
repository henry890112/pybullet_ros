import numpy as np
import open3d as o3d
import sys
print(sys.path)
from utils.utils import create_arrow
from utils.utils import *

class PointCloudPoseEstimator:
    '''
     PointCloudPoseEstimator，它將包含以下幾個主要功能：

    1. 載入和處理點雲數據。
    2. 計算點雲的有向邊界框（OBB）。
    3. 根據OBB計算點雲的6D姿態。
    4. 提供一個接口以獲取姿態信息。

    a. create_6d_pose_matrix 求出正交的y
    b. target_6d_pose 可以選擇用obb的還是gt的6d pose
    '''
    def __init__(self, pc_file_path, noise_pc_file_path, init_ef_matrix, cam_offset_matrix):
        self.pc_file_path = pc_file_path
        self.noise_pc_file_path = noise_pc_file_path
        self.init_ef_matrix = init_ef_matrix
        self.cam_offset_matrix = cam_offset_matrix
        self.pc_segments = None
        self.pc_segments_noise = None
        self.pc_segments_pcd = None
        self.pc_segments_noise_pcd = None
        self.load_point_clouds()
        self.gt = False  # 是否使用 ground truth
        self.vis = True  # 是否可視化

    def load_point_clouds(self):
        self.pc_segments = np.load(self.pc_file_path, allow_pickle=True)
        self.pc_segments_noise = np.load(self.noise_pc_file_path, allow_pickle=True)

    def get_oriented_bounding_box(self):
        if self.gt:
            self.pc_segments_pcd = o3d.geometry.PointCloud()
            self.pc_segments_pcd.points = o3d.utility.Vector3dVector(self.pc_segments)
            self.pc_segments_pcd.paint_uniform_color([0.0, 0.0, 0.0])
            self.pc_segments_pcd.estimate_normals()
            obb = self.pc_segments_pcd.get_oriented_bounding_box()
            obb.color = (1, 0, 0) 
            return obb
        else:
            self.pc_segments_noise_pcd = o3d.geometry.PointCloud()
            self.pc_segments_noise_pcd.points = o3d.utility.Vector3dVector(self.pc_segments_noise)
            self.pc_segments_noise_pcd.paint_uniform_color([0.0, 0.0, 0.0])
            self.pc_segments_noise_pcd.estimate_normals()
            obb = self.pc_segments_noise_pcd.get_oriented_bounding_box()
            obb.color = (1, 0, 0) 
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


    def create_6d_pose_matrix(self, x_axis, z_axis, translation):
        """
        創建一個表示 6D 姿態的 4x4 矩陣。

        :param x_axis: 歸一化的 X 軸向量 (numpy array)
        :param z_axis: 歸一化的 Z 軸向量 (numpy array)
        :param translation: 平移向量 (numpy array)
        :return: 4x4 的姿態矩陣 (numpy array)
        """
        # 確保 x 軸和 z 軸是正交且歸一化的
        assert np.isclose(np.linalg.norm(x_axis), 1), "X 軸向量應該是歸一化的"
        assert np.isclose(np.linalg.norm(z_axis), 1), "Z 軸向量應該是歸一化的"
        assert np.isclose(np.dot(x_axis, z_axis), 0), "X 軸和 Z 軸應該是正交的"

        # 計算 Y 軸向量
        y_axis = np.cross(x_axis, z_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)  # 歸一化 Y 軸向量

        # 檢查是否符合右手定則
        if np.dot(np.cross(x_axis, y_axis), z_axis) < 0:
            # 如果不符合，則反轉 Y 軸
            y_axis = -y_axis
            
        # 創建旋轉矩陣
        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T

        # 創建 4x4 的姿態矩陣
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = translation

        return pose_matrix

    def target_6d_pose(self, x_unit_vec, z_unit_vec, obb):
        """
        :param x_unit_vec: OBB x axis unit vector
        :param z_unit_vec: OBB z axis unit vector
        :param obb: OBB object
        :param gt: ground truth of target pose in pybullet or use OBB pose
        :param vis: visualize the target pose in pybullet or not
        """
        if not self.gt:
            # ground truth
            x_axis = x_unit_vec  # X 軸向量
            z_axis = z_unit_vec  # Z 軸向量
            translation = obb.get_center() # 平移向量
            pose_matrix = self.create_6d_pose_matrix(x_axis, z_axis, translation)
            # transform the pose_matrix from camera coordinate to world coordinate
            target_pose_world = self.init_ef_matrix@ np.linalg.inv(self.cam_offset_matrix)@ pose_matrix

        return target_pose_world

    def get_6d_pose(self, gt=True, vis=False):
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
        print('target_z_translation = {}(m)'.format(target_z_translation))

        normal_arrow, normal_unit_vec = create_arrow(vec = vector[max_length_index], color = [0., 0., 1.],
                                                    position=obb.get_center(), scale=1., radius=1.,
                                                    object_com=None, face_detection_z=True) # because the object has been centered
        # 最短邊的index作為朝向vector
        toward_arrow, toward_unit_vec = create_arrow(vec = vector[min_length_index], color = [1., 0., 0.],
                                                    position=obb.get_center(), scale=1., radius=1.,
                                                    object_com=None, face_detection_x=True) # because the object has been centered

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        if self.vis:
            o3d.visualization.draw_geometries([self.pc_segments_noise_pcd, obb, origin, normal_arrow, toward_arrow])
        return self.target_6d_pose(toward_unit_vec, normal_unit_vec, obb)

def augment_depth_realsense(depth, depth_adjustment=0, coefficient_scale=1):
    # augment depth according to the empirical realsense paper
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8768489&tag=1

    # print("ln250 augment depths")
    # print(np.max(depth))
    # print(np.min(depth))
    sigma = .001063*coefficient_scale + coefficient_scale*.0007278*(depth-depth_adjustment) \
            + coefficient_scale*.003949*(depth-depth_adjustment)**2
    new_depth = np.random.normal(loc=depth, scale=sigma)

    return new_depth

def adjust_pose_with_bias(pose, bias, option="ef"):
    if option == "ef":
        return pose@transZ(bias)
    elif option == "world":
        return transZ(bias)@pose
    elif option == "world_x":
        return transX(bias)@pose
    
def move_gripper_smoothly(env, p, start_opening, target_opening, steps=8000):
    """
    逐步移動夾爪從一個開合程度到另一個開合程度。

    參數:
        env: 包含夾爪控制函數的環境物件。
        p: 包含步進模擬函數的物理模擬物件。
        start_opening (float): 初始夾爪開合程度，範圍在0到0.085之間。
        target_opening (float): 目標夾爪開合程度，範圍在0到0.085之間。
        steps (int): 從初始開合程度到目標開合程度的步數。
        # 使用範例
        # 假設您已經有了env物件和p物件，以及想要設定的開始和目標開合程度
        move_gripper_smoothly(env, p, 0.085, 0.0)  # close
        move_gripper_smoothly(env, p, 0.0, 0.085)  # open
    """
    x = start_opening
    delta = (target_opening - start_opening) / steps  # 計算每一步的變化量

    for i in range(steps):
        env._panda.move_gripper(x)
        x += delta  # 更新夾爪的開合程度
        p.stepSimulation()  # 進行一次物理模擬步進