# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import pybullet as p
import numpy as np
import IPython
import os
import math


class TM5:
    def __init__(self, stepsize=1e-3, realtime=0, init_joints=None, base_shift=[0, 0, 0], other_object=None):
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime
        self.control_mode = "position"

        self.position_control_gain_p = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.position_control_gain_d = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        f_max = 250
        self.max_torque = [f_max, f_max, f_max, f_max, f_max, f_max, 100, 100, 100]

        # connect pybullet
        p.setRealTimeSimulation(self.realtime)

        # load models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        p.setAdditionalSearchPath(current_dir + "/models")
        print(current_dir + "/models")
        self.robot = p.loadURDF("tm5_900/tm5_900_with_gripper.urdf",
                                useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION)
        self._base_position = [-0.05 - base_shift[0], 0.0 - base_shift[1], -0.65 - base_shift[2]]
        # self._base_position = [ -0.5, 0, 0]

        self.pandaUid = self.robot

        # robot parameters
        self.dof = p.getNumJoints(self.robot)

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

        self.mimic_parent_id = mimic_parent_id[0]

        for joint_id, multiplier in mimic_child_multiplier.items():
            c = p.createConstraint(self.robot, self.mimic_parent_id,
                                   self.robot, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

        p.setCollisionFilterPair(self.robot, self.robot, 11, 13, 0)
        p.setCollisionFilterPair(self.robot, self.robot, 16, 18, 0)

        self.joints = []
        self.q_min = []
        self.q_max = []
        self.target_pos = []
        self.target_torque = []
        self.pandaEndEffectorIndex = 7
        self._joint_min_limit = np.array([-4.712385, -3.14159, -3.14159, -3.14159, -3.14159, -4.712385, 0, 0, 0])
        self._joint_max_limit = np.array([4.712385, 3.14159, 3.14159,  3.14159,  3.14159,  4.712385, 0, 0, 0.8])
        self.gripper_range = [0, 0.085]

        for j in range(self.dof):
            p.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)
            joint_info = p.getJointInfo(self.robot, j)
            if j in range(1, 10):
                self.joints.append(j)
                self.q_min.append(joint_info[8])
                self.q_max.append(joint_info[9])
                self.target_pos.append((self.q_min[j-1] + self.q_max[j-1])/2.0)
                self.target_torque.append(0.)
        self.reset(init_joints)

        # collisionBoxPosition = [0, -0.0175, 0]  # xyz位置
        # collisionBoxOrientation = p.getQuaternionFromEuler([0, 0, 0])  # rpy轉四元數
        # collisionBoxSize = [0.2505/2, 0.9/2, 0.25/2]  # size參數是半邊長

        # # 創建一個對應的可視化物體（使用createVisualShape創建一個箱形，顏色和大小與碰撞框相符）
        # visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
        #                                     halfExtents=collisionBoxSize,
        #                                     rgbaColor=[0.5, 0.5, 0.5, 1])

        # # 使用createMultiBody將可視化物體添加到世界中，因為是純粹的可視化物體，不設置質量和碰撞形狀
        # visualObjId = p.createMultiBody(baseMass=0,
        #                                 baseVisualShapeIndex=visualShapeId,
        #                                 basePosition=collisionBoxPosition,
        #                                 baseOrientation=collisionBoxOrientation)
        self.cabinet = other_object

    def reset(self, joints=None):
        self.t = 0.0
        self.control_mode = "position"
        p.resetBasePositionAndOrientation(self.pandaUid, self._base_position,
                                          [0.000000, 0.000000, 0.000000, 1.000000])
        if joints is None:
            #Henry change the init joint of the robot
            self.target_pos = [0.2-np.pi/2, -1, 2, 0, 1.571, 0.0, 0.0, 0.0, 0.0]
            # self.target_pos = [0.2, -1.25, 2.48, -0.4, 1.571, 0.0, 0., 0., 0.]

            self.target_pos = self.standardize(self.target_pos)
            for j in range(1, 10):
                self.target_torque[j-1] = 0.
                p.resetJointState(self.robot, j, targetValue=self.target_pos[j-1])

        else:
            joints = self.standardize(joints)
            for j in range(1, 10):
                self.target_pos[j-1] = joints[j-1]
                self.target_torque[j-1] = 0.
                p.resetJointState(self.robot, j, targetValue=self.target_pos[j-1])
        self.resetController()
        self.setTargetPositions(self.target_pos)
        # p.stepSimulation()

    def step(self):
        self.t += self.stepsize
        p.stepSimulation()

    def resetController(self):
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[0. for i in range(1, 10)])

    def standardize(self, target_pos):
        if len(target_pos) == 7:
            if type(target_pos) == list:
                target_pos[6:6] = [0, 0]
            else:
                target_pos = np.insert(target_pos, 6, [0, 0])

        target_pos = np.array(target_pos)

        target_pos = np.minimum(np.maximum(target_pos, self._joint_min_limit), self._joint_max_limit)
        return target_pos

    def setTargetPositions(self, target_pos):
        self.target_pos = self.standardize(target_pos)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.target_pos,
                                    forces=self.max_torque,
                                    positionGains=self.position_control_gain_p,
                                    velocityGains=self.position_control_gain_d)

    def getJointStates(self):
        joint_states = p.getJointStates(self.robot, self.joints)

        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]

        del joint_pos[6:8]
        del joint_vel[6:8]

        return joint_pos, joint_vel

    def solveInverseKinematics(self, pos, ori):
        return list(p.calculateInverseKinematics(self.robot,
                    7, pos, ori,
                    maxNumIterations=500,
                    residualThreshold=1e-8))

    def move_gripper(self, open_length):
        open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(self.robot, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=100)

    # def check_for_collisions(self):
    #     # 假設 cameraLinkId 是 camera_link 的正確索引
    #     cameraLinkId = self.find_link_id("camera_link")  # 這需要您實現 find_link_id 方法
    #     # 利用for迴圈檢查self.robot中的camera link和其他elf.robot中個link是否有碰撞
    #     # print(p.getNumJoints(self.robot))
    #     threshold = 0.
    #     for i in range(p.getNumJoints(self.robot)):
    #         if i == cameraLinkId:
    #             continue  # 忽略 camera link 自己
    #         closest_points = p.getClosestPoints(bodyA=self.robot, bodyB=self.robot, distance = threshold, linkIndexA=cameraLinkId, linkIndexB=i)
    #         if closest_points:
    #             # print(closest_points[0][8])
    #             # find the target link name
    #             link_name = p.getJointInfo(self.robot, i)[12].decode('utf-8')
    #             # print(f"camera link 和連結 {link_name} 的最近距離小於 {threshold} 米")
    #             return True  # 如果有任何連結的最近點小於閾值，就認為是碰撞

    #     return False

    def check_for_collisions(self):
        # 定義需要檢查碰撞的連結名稱
        check_links = [
            "shoulder_1_link",
            "arm_1_link",
            "arm_2_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            "flange_link"
        ]

        # 建立一個空字典來存儲這些連結的ID
        link_ids = {name: self.find_link_id(name) for name in check_links}

        # 定義應該排除的連結對，即這些連結對之間的碰撞檢查將被忽略
        exclude_pairs = {
            ("shoulder_1_link", "arm_1_link"),
            ("arm_1_link", "arm_2_link"),
            ("arm_2_link", "wrist_1_link"),
            ("wrist_1_link", "wrist_2_link"),
            ("wrist_2_link", "wrist_3_link"),
            ("wrist_3_link", "flange_link"),
        }
        
        threshold = -0.0  # 定義檢查碰撞的距離閾值
        cabinet_threshold = 0.0  # 定義檢查與 cabinet 碰撞的距離閾值

        # 定義額外的 bounding boxes，以 (xmin, xmax, ymin, ymax, zmin, zmax) 的形式
        extra_bboxes = [
            (0.68, 1.0, -0.5, 0.5, 0.57, 0.598),
            (0.68, 1.0, -0.5, 0.5, 0.22, 0.248)
        ]

        # 检查 link 和 link 之间的碰撞
        for name_i in check_links:
            for name_j in check_links:
                if (name_i, name_j) in exclude_pairs or (name_j, name_i) in exclude_pairs:
                    continue  # 如果當前的連結對在排除列表中，則跳過不檢查

                link_id_i = link_ids[name_i]
                link_id_j = link_ids[name_j]
                if link_id_i == link_id_j:
                    continue  # 忽略自身的檢查

                closest_points = p.getClosestPoints(bodyA=self.robot, bodyB=self.robot, distance=threshold, linkIndexA=link_id_i, linkIndexB=link_id_j)
                if closest_points:
                    # 如果發現任何連結對的最近點小於閾值，認為發生了碰撞
                    print(f"連結 {name_i} 和連結 {name_j} 的最近距離小於 {threshold} 米")
                    for point in closest_points:
                        print(f"最近點距離：{point[8]} 米")
                    return True

        # 與 cabinet 進行碰撞檢查
        for name in check_links:
            link_id = link_ids[name]
            
            # 定义需要检查的 cabinet 的 link index 列表
            cabinet_link_indices = [2, 3]
            
            for cabinet_link_id in cabinet_link_indices:
                closest_points = p.getClosestPoints(bodyA=self.robot, bodyB=self.cabinet, distance=cabinet_threshold, linkIndexA=link_id, linkIndexB=cabinet_link_id)
                if closest_points:
                    # 如果發現與 cabinet 的最近點小於閾值，認為發生了碰撞
                    print(f"連結 {name} 和 cabinet 的 link {cabinet_link_id} 的最近距離小於 {cabinet_threshold} 米")
                    # 印出碰撞的最近點距離
                    for point in closest_points:
                        print(f"最近點距離：{point[8]} 米")
                    return True
        return False

    def is_point_in_bbox(self, point, bbox):
        """
        檢查點是否在包圍盒內
        :param point: 點的座標 (x, y, z)
        :param bbox: 包圍盒的範圍 (xmin, xmax, ymin, ymax, zmin, zmax)
        :return: 如果點在包圍盒內則返回 True，否則返回 False
        """
        x, y, z = point
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        return xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax






    def find_link_id(self, link_name):
        # print each link name
        for i in range(p.getNumJoints(self.robot)):
            # print(p.getJointInfo(self.robot, i)[12].decode('utf-8'))
            if p.getJointInfo(self.robot, i)[12].decode('utf-8') == link_name:
                # print("找到了", link_name, "的索引：", i)
                return i
        print("未找到", link_name, "的索引")
        return None
            

if __name__ == "__main__":
    robot = TM5(realtime=1)
    while True:
        pass
