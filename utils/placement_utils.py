import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import transform_util


def create_arrow(vec, color, vis=None, vec_len=None, scale=.06, radius=.12, position=(0,0,0),
                 object_com=None):
    """
    #Henry: vec 為object stable plane 的法向量
    Creates an error, where the arrow size is based on the vector magnitude.
    :param vec:
    :param color:
    :param vis:
    :param scale:
    :param position:
    :return:
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


    # print("ln554 create_arrow")
    # print(vec)
    rot_mat = transform_util.get_rotation_matrix_between_vecs(vec, [0, 0, 1])
    #Henry: print the rotation matrix from table to the arrow on the object
    # print(rot_mat)
    mesh_arrow.rotate(rot_mat, center=np.array([0,0,0]))

    H = np.eye(4)
    H[:3, 3] = position
    mesh_arrow.transform(H)

    mesh_arrow.paint_uniform_color(color)
    return mesh_arrow, vec/np.linalg.norm(vec)

def PCDToNumpy(pcd):
    """  convert open3D point cloud to numpy ndarray

    Args:
        pcd (open3d.geometry.PointCloud): 

    Returns:
        [ndarray]: 
    """

    return np.asarray(pcd.points)

def DownSample(pts, voxel_size=0.003):
    """ down sample the point clouds

    Args:
        pts (ndarray): N x 3 input point clouds
        voxel_size (float, optional): voxel size. Defaults to 0.003.

    Returns:
        [ndarray]: 
    """

    p = NumpyToPCD(pts).voxel_down_sample(voxel_size=voxel_size)

    return PCDToNumpy(p)

def NumpyToPCD(xyz):
    """ convert numpy ndarray to open3D point cloud 

    Args:
        xyz (ndarray): 

    Returns:
        [open3d.geometry.PointCloud]: 
    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    return pcd

def PlaneRegression(points, threshold=0.01, init_n=3, iter=1000):
    """ plane regression using ransac

    Args:
        points (ndarray): N x3 point clouds
        threshold (float, optional): distance threshold. Defaults to 0.003.
        init_n (int, optional): Number of initial points to be considered inliers in each iteration
        iter (int, optional): number of iteration. Defaults to 1000.

    Returns:
        [ndarray, List]: 4 x 1 plane equation weights, List of plane point index
    """

    pcd = NumpyToPCD(points)

    w, index = pcd.segment_plane(
        threshold, init_n, iter)

    return w, index

def DetectMultiPlanes(points, min_ratio=0.05, threshold=0.01, iterations=1000):
    """ Detect multiple planes from given point clouds

    Args:
        points (np.ndarray): 
        min_ratio (float, optional): The minimum left points ratio to end the Detection. Defaults to 0.05.
        threshold (float, optional): RANSAC threshold in (m). Defaults to 0.01.

    Returns:
        [List[tuple(np.ndarray, List)]]: Plane equation and plane point index
    """

    plane_list = []
    N = len(points)
    target = points.copy()
    count = 0

    while count < (1 - min_ratio) * N:
        w, index = PlaneRegression(
            target, threshold=threshold, init_n=3, iter=iterations)
    
        count += len(index)
        plane_list.append((w, target[index]))
        target = np.delete(target, index, axis=0)

    return plane_list

def plane_cluster(results, min_cluster = 5000, max_cluster = 50000):
    red_planes = []
    blue_planes = []
    red_colors = []
    blue_colors = []
    # print(len(results))
    # print(results[0])

    for _, plane in results:
        print(plane.shape, end = ' ')
        if plane.shape[0]>min_cluster and plane.shape[0]<max_cluster:
            # (5000~50000)之間設為紅色
            r = 1
            g = 0
            b = 0

            color = np.zeros((plane.shape[0], plane.shape[1]))
            color[:, 0] = r
            color[:, 1] = g
            color[:, 2] = b

            red_planes.append(plane)
            red_colors.append(color)
        # else:
        #     r = 0
        #     g = 0
        #     b = 1

        #     color = np.zeros((plane.shape[0], plane.shape[1]))
        #     color[:, 0] = r
        #     color[:, 1] = g
        #     color[:, 2] = b

        #     blue_planes.append(plane)
        #     blue_colors.append(color)
        
    red_plane_pcd = o3d.geometry.PointCloud()
    red_plane_pcd.points = o3d.utility.Vector3dVector(red_planes[0])
    red_plane_pcd.colors = o3d.utility.Vector3dVector(red_colors[0])

    # blue_plane_pcd = o3d.geometry.PointCloud()
    # blue_plane_pcd.points = o3d.utility.Vector3dVector(blue_planes[0])
    # blue_plane_pcd.colors = o3d.utility.Vector3dVector(blue_colors[0])

    # return red_plane_pcd, blue_plane_pcd
    return red_plane_pcd

def plane_cluster_in_height(results, min_height = 0.3, max_height = 0.5):
    green_planes = []
    green_colors = []

    for _, plane in results:
        # 將高度在0.5~1.5之間設為綠色
        if plane[0][2]>min_height and plane[0][2]<max_height:
            r = 0
            g = 1
            b = 0

            color = np.zeros((plane.shape[0], plane.shape[1]))
            color[:, 0] = r
            color[:, 1] = g
            color[:, 2] = b

            green_planes.append(plane)
            green_colors.append(color)
    
    green_plane_pcd = o3d.geometry.PointCloud()
    green_plane_pcd.points = o3d.utility.Vector3dVector(green_planes[0])
    green_plane_pcd.colors = o3d.utility.Vector3dVector(green_colors[0])

    return green_plane_pcd

import copy
base_y = [-1, 0, 0]
def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd



base_y = [-1, 0, 0]
def whole_pipe_virtual(pcd_path, min_ratio=0.05, threshold=0.0001, iterations=2000, min_cluster = 5000, max_cluster = 60000, visualize=True):

    '''RANSAC plane detection
    min_ratio：根據您期望的平面數量和點雲密度，選擇一個合適的值。較小的值可能檢測到更多的平面，但同時也可能增加偽議檢測的可能性。
    threshold：根據點雲的噪聲水平和平面特性，選擇一個合適的值。較小的值將導致更精確的檢測，但也可能導致平面被忽略。較大的值將允許更大的誤差，但同時也可能導致更多噪聲點被誤判為平面。
    iterations：根據您的計算資源和時間要求，選擇一個合適的值。較大的值將增加檢測的穩定性和準確性，但同時也增加了計算時間。
    '''
    #step1: get point cloud
    environment_pcd = o3d.io.read_point_cloud(pcd_path)
    print("Original point cloud", np.asarray(environment_pcd.points).shape)
    environment_pcd = environment_pcd.voxel_down_sample(voxel_size=0.005)
    print("Downsampled point cloud", np.asarray(environment_pcd.points).shape)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([environment_pcd, origin_frame])

    # step1.5: add the noise to the point cloud
    mu, sigma = 0, 0.001  # mean and standard deviation
    environment_pcd = apply_noise(environment_pcd, mu, sigma)
    if visualize:
        o3d.visualization.draw_geometries([environment_pcd, origin_frame])
    # convert the point cloud to numpy array
    environment_pc_full = np.asarray(environment_pcd.points)

    #step2: detect planes
    plane_list = DetectMultiPlanes(environment_pc_full, min_ratio=min_ratio, threshold=threshold, iterations=iterations)
    # 初始化一個用於存儲所有點雲的列表
    pcd_list = []  # 存儲所有點雲對象

    for plane_coefficients, points in plane_list:
        # 創建 Open3D 點雲對象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 生成隨機顏色
        color = np.random.rand(3)
        pcd.paint_uniform_color(color)
        
        # 將點雲對象添加到列表中
        pcd_list.append(pcd)
    pcd_list.append(origin_frame)
    # 使用 Open3D 視覺化所有點雲
    if visualize:
        o3d.visualization.draw_geometries(pcd_list)
    
    use_much_points = False
    if use_much_points:
        red_plane_pcd = plane_cluster(plane_list, min_cluster = min_cluster, max_cluster = max_cluster)
        # o3d.visualization.draw_geometries([red_plane_pcd, origin_frame])

        # step3: remove outliers in red plane
        points = np.asarray(red_plane_pcd.points)
        cl, ind = red_plane_pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=3.0) #(nb_neighbors=30, std_ratio=3.0)
        # cl, ind = red_plane_pcd.remove_radius_outlier(nb_points=1, radius=0.5)  # 半径5cm内至少要有16个点

        '''
        nb_neighbors，它指定在计算给定点的平均距离时要考虑的相邻要素数。
        std_ratio，允许根据点云中平均距离的标准偏差设置阈值水平。此数字越低，过滤器的激进程度就越高,删除的越多。
        '''
        filtered_point_cloud = red_plane_pcd.select_by_index(ind)
        # o3d.visualization.draw_geometries([filtered_point_cloud])
    else:
        green_plane_pcd = plane_cluster_in_height(plane_list, min_height = 0.3, max_height = 0.5)
        points = np.asarray(green_plane_pcd.points)
        cl, ind = green_plane_pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=3.0) #(nb_neighbors=30, std_ratio=3.0)
        filtered_point_cloud = green_plane_pcd.select_by_index(ind)

    # step4 : get obb in red plane filtered_point_cloud and print the arrow of the obb
    obb = filtered_point_cloud.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    vec = obb.get_box_points()[0] - obb.get_box_points()[2]  # 中長度的向量->y
    #TODO: why [1]-[0] & [0]-[1] 得到的arrow方向一樣，因為在vis_util中有利用if判斷來避免arrow往物體內部插入
    arrow, unit_vec = create_arrow(vec = -vec, color = [0., 0., 1.],
                                position=(obb.get_box_points()[0] + obb.get_box_points()[1])/2, scale=0.5, radius=1.2,
                                object_com=obb.get_center()) # because the object has been centered
    print("\nunit_vec: ", unit_vec)
    if visualize:
        o3d.visualization.draw_geometries([environment_pcd, origin_frame, arrow, filtered_point_cloud])

    # step5: get the criterion
    '''
    o3d.visualization.draw_geometries([obb, environment_pcd
    歐氏距離（Euclidean Distance）：衡量兩個向量在空間中的直線距離。對於兩個n維向量x和y，歐氏距離可以表示為：d(x, y) = √Σ(xi - yi)²。

    這個歐氏距離的計算可以幫助您了解兩個向量之間的距離或相似性。請注意，如果向量 base_y 和 unit_vec 都是單位向量（歐長為1），則它們之間的歐氏距離將是2，
    表示它們在方向上相互垂直。當兩個向量重疊時，歐氏距離將為0，表示它們相等。在其他情況下，歐氏距離將為正數，表示兩個向量之間的距離。
    '''
    criterion = np.linalg.norm(base_y-unit_vec)
    return criterion, filtered_point_cloud, unit_vec

def whole_pipe_realworld(pcd_path, min_ratio=0.05, threshold=0.0001, iterations=2000, min_cluster = 5000, max_cluster = 60000, visualize=True):

    '''RANSAC plane detection
    min_ratio：根據您期望的平面數量和點雲密度，選擇一個合適的值。較小的值可能檢測到更多的平面，但同時也可能增加偽議檢測的可能性。
    threshold：根據點雲的噪聲水平和平面特性，選擇一個合適的值。較小的值將導致更精確的檢測，但也可能導致平面被忽略。較大的值將允許更大的誤差，但同時也可能導致更多噪聲點被誤判為平面。
    iterations：根據您的計算資源和時間要求，選擇一個合適的值。較大的值將增加檢測的穩定性和準確性，但同時也增加了計算時間。
    '''
    # step1: get the point cloud from .pcd file
    environment_pcd = o3d.io.read_point_cloud(pcd_path)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    if visualize:
        o3d.visualization.draw_geometries([environment_pcd, origin_frame])

    # convert the point cloud to numpy array
    environment_pc_full = np.asarray(environment_pcd.points)

    #step2: detect planes
    plane_list = DetectMultiPlanes(environment_pc_full, min_ratio=min_ratio, threshold=threshold, iterations=iterations)
    # 初始化一個用於存儲所有點雲的列表
    pcd_list = []  # 存儲所有點雲對象

    for plane_coefficients, points in plane_list:
        # 創建 Open3D 點雲對象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 生成隨機顏色
        color = np.random.rand(3)
        pcd.paint_uniform_color(color)
        
        # 將點雲對象添加到列表中
        pcd_list.append(pcd)
    pcd_list.append(origin_frame)
    # 使用 Open3D 視覺化所有點雲
    if visualize:
        o3d.visualization.draw_geometries(pcd_list)
    
    red_plane_pcd = plane_cluster(plane_list, min_cluster = min_cluster, max_cluster = max_cluster)
    # o3d.visualization.draw_geometries([red_plane_pcd, origin_frame])

    # step3: remove outliers in red plane
    points = np.asarray(red_plane_pcd.points)
    cl, ind = red_plane_pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=3.0) #(nb_neighbors=30, std_ratio=3.0)
    # cl, ind = red_plane_pcd.remove_radius_outlier(nb_points=1, radius=0.5)  # 半径5cm内至少要有16个点

    '''
    nb_neighbors，它指定在计算给定点的平均距离时要考虑的相邻要素数。
    std_ratio，允许根据点云中平均距离的标准偏差设置阈值水平。此数字越低，过滤器的激进程度就越高,删除的越多。
    '''
    filtered_point_cloud = red_plane_pcd.select_by_index(ind)
    # o3d.visualization.draw_geometries([filtered_point_cloud])

    # step4 : get obb in red plane filtered_point_cloud and print the arrow of the obb
    obb = filtered_point_cloud.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    vec = obb.get_box_points()[0] - obb.get_box_points()[2]  # 中長度的向量->y
    #TODO: why [1]-[0] & [0]-[1] 得到的arrow方向一樣，因為在vis_util中有利用if判斷來避免arrow往物體內部插入
    arrow, unit_vec = create_arrow(vec = vec, color = [0., 0., 1.],
                                position=(obb.get_box_points()[0] + obb.get_box_points()[1])/2, scale=0.5, radius=1.2,
                                object_com=obb.get_center()) # because the object has been centered
    print("\nunit_vec: ", unit_vec)
    o3d.visualization.draw_geometries([environment_pcd, origin_frame, arrow, filtered_point_cloud])

    # step5: get the criterion
    '''
    o3d.visualization.draw_geometries([obb, environment_pcd
    歐氏距離（Euclidean Distance）：衡量兩個向量在空間中的直線距離。對於兩個n維向量x和y，歐氏距離可以表示為：d(x, y) = √Σ(xi - yi)²。

    這個歐氏距離的計算可以幫助您了解兩個向量之間的距離或相似性。請注意，如果向量 base_y 和 unit_vec 都是單位向量（歐長為1），則它們之間的歐氏距離將是2，
    表示它們在方向上相互垂直。當兩個向量重疊時，歐氏距離將為0，表示它們相等。在其他情況下，歐氏距離將為正數，表示兩個向量之間的距離。
    '''
    criterion = np.linalg.norm(base_y-unit_vec)
    return criterion, filtered_point_cloud, unit_vec

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

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

class PointCloudProcessor:
    def __init__(self, point_cloud, voxel_size, height_range, plane_height_range):
        self.point_cloud = point_cloud
        self.voxel_size = voxel_size
        self.height_range = height_range
        self.plane_height_range = plane_height_range

    def whole_pipe_realworld(self, min_ratio=0.05, threshold=0.0001, iterations=2000, min_cluster = 5000, max_cluster = 60000, visualize=True):

        '''RANSAC plane detection
        min_ratio：根據您期望的平面數量和點雲密度，選擇一個合適的值。較小的值可能檢測到更多的平面，但同時也可能增加偽議檢測的可能性。
        threshold：根據點雲的噪聲水平和平面特性，選擇一個合適的值。較小的值將導致更精確的檢測，但也可能導致平面被忽略。較大的值將允許更大的誤差，但同時也可能導致更多噪聲點被誤判為平面。
        iterations：根據您的計算資源和時間要求，選擇一個合適的值。較大的值將增加檢測的穩定性和準確性，但同時也增加了計算時間。
        '''
        # step1: get the point cloud from .pcd file
        environment_pcd = self.point_cloud
        print("Original point cloud", np.asarray(environment_pcd.points).shape)
        environment_pcd = environment_pcd.voxel_down_sample(voxel_size=0.005)
        print("Downsampled point cloud", np.asarray(environment_pcd.points).shape)
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

        # step1.5: add the noise to the point cloud
        mu, sigma = 0, 0.001  # mean and standard deviation
        environment_pcd = apply_noise(environment_pcd, mu, sigma)
        o3d.visualization.draw_geometries([environment_pcd, origin_frame])
        environment_pc_full = np.asarray(environment_pcd.points)

        #step2: detect planes
        plane_list = DetectMultiPlanes(environment_pc_full, min_ratio=min_ratio, threshold=threshold, iterations=iterations)
        # 初始化一個用於存儲所有點雲的列表
        pcd_list = []  # 存儲所有點雲對象

        for plane_coefficients, points in plane_list:
            # 創建 Open3D 點雲對象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 生成隨機顏色
            color = np.random.rand(3)
            pcd.paint_uniform_color(color)
            
            # 將點雲對象添加到列表中
            pcd_list.append(pcd)
        pcd_list.append(origin_frame)

        use_much_points = False
        if use_much_points:
            red_plane_pcd = plane_cluster(plane_list, min_cluster = min_cluster, max_cluster = max_cluster)
            points = np.asarray(red_plane_pcd.points)
            cl, ind = red_plane_pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=3.0) #(nb_neighbors=30, std_ratio=3.0)
            filtered_point_cloud = red_plane_pcd.select_by_index(ind)
        else:
            green_plane_pcd = plane_cluster_in_height(plane_list, min_height = self.plane_height_range[0], max_height = self.plane_height_range[1])
            points = np.asarray(green_plane_pcd.points)
            cl, ind = green_plane_pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=3.0) #(nb_neighbors=30, std_ratio=3.0)
            filtered_point_cloud = green_plane_pcd.select_by_index(ind)

        # step4 : get obb in red plane filtered_point_cloud and print the arrow of the obb
        obb = filtered_point_cloud.get_oriented_bounding_box()
        obb.color = (0, 1, 0)
        vec = obb.get_box_points()[0] - obb.get_box_points()[2]  # 中長度的向量->y
        #TODO: why [1]-[0] & [0]-[1] 得到的arrow方向一樣，因為在vis_util中有利用if判斷來避免arrow往物體內部插入
        arrow, unit_vec = create_arrow(vec = -vec, color = [0., 0., 1.],
                                    position=(obb.get_box_points()[2] + obb.get_box_points()[7])/2, scale=0.5, radius=1.2,
                                    object_com=obb.get_center()) # because the object has been centered
        print("\nunit_vec: ", unit_vec)
        o3d.visualization.draw_geometries([environment_pcd, filtered_point_cloud, arrow, origin_frame])

        # step5: get the criterion
        criterion = np.linalg.norm(base_y-unit_vec)
        return criterion, filtered_point_cloud, unit_vec

    @staticmethod
    def generate_colors(num_colors):
        colors = plt.cm.get_cmap("hsv", num_colors)
        return colors(np.linspace(0, 1, num_colors))[:, :3]

    def check_empty_voxels(self, sliced_pcd):
        points = np.asarray(sliced_pcd.points)
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        min_bound[2], max_bound[2] = self.height_range

        grid_dims = ((max_bound - min_bound) / self.voxel_size).astype(int) + 1
        occupancy_grid = np.zeros(grid_dims, dtype=bool)
        voxel_indices = ((points - min_bound) / self.voxel_size).astype(int)

        valid_indices = (points[:, 2] >= self.height_range[0]) & (points[:, 2] <= self.height_range[1])
        voxel_indices = voxel_indices[valid_indices]

        occupancy_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
        empty_voxels = np.argwhere(occupancy_grid == False)
        return len(empty_voxels), occupancy_grid, min_bound

    def visualize_voxels(self, occupancy_grid, min_bound):
        voxel_grid = o3d.geometry.VoxelGrid()
        voxel_grid.voxel_size = self.voxel_size
        voxel_grid.origin = min_bound

        for index, occupied in np.ndenumerate(occupancy_grid):
            if occupied:
                voxel = o3d.geometry.Voxel()
                voxel.grid_index = index
                voxel.color = [0.6, 0.6, 0.6]
                voxel_grid.add_voxel(voxel)
        return voxel_grid

    def slice_point_cloud(self, sliced_pcd, unit_vec, slice_width, empty_threshold, display_voxels=True, check_slice_region=True):
        points = np.asarray(sliced_pcd.points)
        projections = np.dot(points, unit_vec)
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        num_slices = int(np.ceil((max_proj - min_proj) / slice_width))
        sliced_point_clouds = []
        colors = self.generate_colors(num_slices)

        slice_details = []

        for i in range(num_slices):
            lower_bound = min_proj + i * slice_width
            upper_bound = lower_bound + slice_width
            mask = (projections >= lower_bound) & (projections < upper_bound)
            sliced_points = points[mask]

            if len(sliced_points) > 0:
                sliced_pcd = o3d.geometry.PointCloud()
                sliced_pcd.points = o3d.utility.Vector3dVector(sliced_points)
                empty_voxels, occupancy_grid, min_bound = self.check_empty_voxels(sliced_pcd)

                voxel_count = np.sum(occupancy_grid)
                total_voxels = voxel_count + empty_voxels
                slice_details.append({
                    "slice_index": i,
                    "voxel_count": voxel_count,
                    "empty_voxel_count": empty_voxels,
                    "total_voxel_count": total_voxels,
                    "ratio": voxel_count / total_voxels if total_voxels > 0 else 0
                })

                voxel_grid = self.visualize_voxels(occupancy_grid, min_bound) if display_voxels else None
                color = colors[i % num_slices] if check_slice_region else ([0, 1, 0] if voxel_count / total_voxels > empty_threshold else [1, 0, 0])
                sliced_pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(sliced_points), 1)))
                sliced_point_clouds.append((sliced_pcd, voxel_grid) if voxel_grid else sliced_pcd)

        for detail in slice_details:
            print(f"Slice {detail['slice_index'] + 1}: Voxel Count = {detail['voxel_count']}, Empty Voxel Count = {detail['empty_voxel_count']}, Total Voxel Count = {detail['total_voxel_count']}, Ratio = {detail['ratio']:.2f}")

        return sliced_point_clouds


