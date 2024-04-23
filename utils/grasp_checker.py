from time import sleep
import pybullet as p
from utils.utils import *
import numpy as np


class ValidGraspChecker():
    def __init__(self, env) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = current_dir.replace("utils", "env/models")
        print(current_dir)
        self.robot = p.loadURDF(os.path.join(current_dir, "tm5_900/robotiq_85.urdf"),
                                useFixedBase=True)
        self.env = env
        self.object_num = len(self.env._objectUids)

        # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        mimic_parent_id = []
        mimic_child_multiplier = {}
        for i in range(p.getNumJoints(self.robot)):
            inf = p.getJointInfo(self.robot, i)
            name = inf[1].decode('utf-8')
            if name == mimic_parent_name:
                mimic_parent_id.append(inf[0])
            if name in mimic_children_names:
                mimic_child_multiplier[inf[0]] = mimic_children_names[name]
            if inf[2] != p.JOINT_FIXED:
                p.setJointMotorControl2(self.robot, inf[0], p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        mimic_parent_id = mimic_parent_id[0]

        for joint_id, multiplier in mimic_child_multiplier.items():
            c = p.createConstraint(self.robot, mimic_parent_id,
                                   self.robot, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

        for i in range(-1, self.object_num + 1):
            p.setCollisionFilterGroupMask(self.robot, i, 0, 0)
            p.setCollisionFilterPair(0, self.robot, -1, i, 1)

        p.resetBasePositionAndOrientation(self.robot, [0, 0, -100],
                                          [0.000000, 0.000000, 0.000000, 1.000000])

    def draw_aabb(self, target_id, lifetime=5):
        """
        在 PyBullet 環境中繪製指定物體的 AABB（軸對齊包圍盒）。

        :param target_id: 要繪製 AABB 的物體ID。
        :param lifetime: 繪製的線條顯示時間（秒），默認為 5 秒。
        """
        # for i in range(p.getNumJoints(target_id)):
        #     print(i)
        #     target_box = p.getAABB(target_id, i)
        target_box = p.getAABB(target_id, 3)

        min_point = target_box[0]
        max_point = target_box[1]

        if target_id == self.env.cabinet_id:
            # 增加z軸高度
            # 用來微調AABB的高度 因為cabinet的AABB只會得到shelf_bottom_plate
            max_point[2]
        
        target_box = np.array([min_point, max_point])

        # 計算 AABB 的角點
        corners = [
            [min_point[0], min_point[1], min_point[2]],
            [max_point[0], min_point[1], min_point[2]],
            [max_point[0], max_point[1], min_point[2]],
            [min_point[0], max_point[1], min_point[2]],
            [min_point[0], min_point[1], max_point[2]],
            [max_point[0], min_point[1], max_point[2]],
            [max_point[0], max_point[1], max_point[2]],
            [min_point[0], max_point[1], max_point[2]]
        ]

        # 定義 AABB 的邊線
        lines = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        # 繪製 AABB 的邊線
        for start, end in lines:
            p.addUserDebugLine(corners[start], corners[end], lineColorRGB=[1, 0, 0], lineWidth=3, lifeTime=lifetime)
        
        return target_box
        # 使用示例
        # 假設 self.env.table_id 是您要繪製 AABB 的物體ID
        # draw_aabb(self.env.table_id, 5)


    def extract_grasp(self, grasp, drawback_distance=0., visual=False, filter_elbow=True, time=0.2, distance=0.0001):
        '''
        This API will take input as grasping pose group and validate collision between gripper and target object.
        It uses pybullet API to test if the gripper mesh has contact points with current environment when the gripper is set to fully opened.
        Input:
            grasp: Grasp pose candidate of multiple SE(3) matrix, shape [N, 4, 4]
        Output:
           valid_grasp: Valid grasp pose group of SE(3) matrix, shape [N, 4, 4]
           valid_grasp_index: valid grasp pose index of original grasp candidate, shape [N]
        Parameter:
        "drawback_distance" is the distance to draw back the end effector pose along z-axis in validation process.
        "filter_elbow" denote if checker use estimated elbow point and bounding box of table
            as one of the measurements to prevent collision of other joint.
        #######Note: The estimated elbow point is NOT calculated by IK, so it's nearly a rough guess.
        '''


        table_box = np.array(p.getAABB(self.env.table_id))
        # table_box = self.draw_aabb(self.env.table_id, 5)
        cabinet_box = self.draw_aabb(self.env.cabinet_id, 5)

        valid_grasp = []
        valid_grasp_index = []
        dist_bias = unpack_pose([0, 0, (-1) * drawback_distance, 1, 0, 0, 0])
        placed_uid = np.array(self.env._objectUids)[np.where(np.array(self.env.placed_objects))]

        for i in range(grasp.shape[0]):
            if_valid = True
            f_pose = grasp[i].dot(dist_bias)
            # if np.inner(f_pose[:3, 1], [0, 0, 1]) < 0:
            #     f_pose = f_pose.dot(rotZ(np.pi))

            p.resetBasePositionAndOrientation(self.robot, f_pose[:3, 3], ros_quat(mat2quat(f_pose[:3, :3])))
            if visual:
                sleep(time)

            elbow_point = f_pose.dot(unpack_pose([0., -0.16, -0.07, 1., 0., 0., 0.]))
            box_min_table, box_max_table = table_box - elbow_point[:3, 3]
            box_min_cabinet, box_max_cabinet = cabinet_box - elbow_point[:3, 3]

            if len(p.getClosestPoints(self.robot, self.env.table_id, distance)):
                continue
            elif len(p.getClosestPoints(self.robot, self.env.cabinet_id, distance)):
                continue
            elif (filter_elbow) and (len(box_min_table[box_min_table < 0]) == 3) and (len(box_max_table[box_max_table > 0]) == 3):
                continue
            elif (filter_elbow) and (len(box_min_cabinet[box_min_cabinet < 0]) == 3) and (len(box_max_cabinet[box_max_cabinet > 0]) == 3):
                continue

            for uid in placed_uid:
                if len(p.getClosestPoints(self.robot, uid, distance)):
                    if_valid = False
            if if_valid:
                valid_grasp.append(f_pose)
                valid_grasp_index.append(i)

        p.resetBasePositionAndOrientation(self.robot, [0, 0, -100], [0, 0, 0, 1])

        return np.array(valid_grasp), np.array(valid_grasp_index)
