import numpy as np
import open3d as o3d

def load_and_display_point_cloud(npy_file):
    # 读取npy文件
    point_cloud_data = np.load(npy_file)
    
    # 确保数据的形状是正确的 (N, 3)
    if point_cloud_data.shape[1] != 3:
        raise ValueError("点云数据应该是 (N, 3) 的形状")

    # 将点云数据转换为Open3D格式
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)

    # 显示点云
    o3d.visualization.draw_geometries([point_cloud])

if __name__ == "__main__":
    # 指定你的npy文件路径
    npy_file_path = "/home/isci/henry_pybullet_ws/src/pybullet_ros/realworld_data/multiview_pc_target.npy"
    load_and_display_point_cloud(npy_file_path)
