import numpy as np
import torch
from src.utils import bmmt, bmv, bmtv, bbmv, bmtm
from src.lie_algebra import SO3
import matplotlib.pyplot as plt


class BaseLoss(torch.nn.Module):

    def __init__(self, min_N, max_N, dt):
        super().__init__()
        # windows sizes
        self.min_N = min_N
        self.max_N = max_N
        self.min_train_freq = 2 ** self.min_N
        self.max_train_freq = 2 ** self.max_N
        # sampling time
        self.dt = dt # (s)

class GyroLoss(BaseLoss):
    """Loss for low-frequency orientation increment"""

    def __init__(self, w, min_N, max_N, dt, target, huber):
        super().__init__(min_N, max_N, dt)
        # weights on loss
        self.w = w
        self.sl = torch.nn.SmoothL1Loss()
        if target == 'rotation matrix':
            self.forward = self.forward_with_rotation_matrices
        elif target == 'quaternion':
            self.forward = self.forward_with_quaternions
        elif target == 'rotation matrix mask':
            self.forward = self.forward_with_rotation_matrices_mask
        elif target == 'quaternion mask':
            self.forward = self.forward_with_quaternion_mask
        self.huber = huber
        self.weight = torch.ones(1, 1,
            self.min_train_freq).cuda()/self.min_train_freq
        self.N0 = 5 # remove first N0 increment in loss due not account padding

    def f_huber(self, rs):
        """Huber loss function"""
        loss = self.w*self.sl(rs/self.huber,
            torch.zeros_like(rs))*(self.huber**2)
        return loss

    def forward_with_rotation_matrices(self, xs, hat_xs):
        """Forward errors with rotation matrices"""
        N = xs.shape[0]
        Xs = SO3.exp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.exp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
        rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            Xs = Xs[::2].bmm(Xs[1::2])
            rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
            loss = loss + self.f_huber(rs)/(2**(k - self.min_N + 1))
        return loss

    def forward_with_quaternions(self, xs, hat_xs):
        """Forward errors with quaternion"""
        N = xs.shape[0]
        Xs = SO3.qexp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.qexp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
        rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs)).reshape(N,
                -1, 3)[:, self.N0:]
        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
            Xs = SO3.qmul(Xs[::2], Xs[1::2])
            rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs))
            rs = rs.view(N, -1, 3)[:, self.N0:]
            loss = loss + self.f_huber(rs)/(2**(k - self.min_N + 1))
        return loss

    def forward_with_rotation_matrices_mask(self, xs, hat_xs):
        """Forward errors with rotation matrices"""
        N = xs.shape[0]
        masks = xs[:, :, 3].unsqueeze(1)
        masks = torch.nn.functional.conv1d(masks, self.weight, bias=None,
            stride=self.min_train_freq).double().transpose(1, 2)
        masks[masks < 1] = 0
        Xs = SO3.exp(xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.exp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
        rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            Xs = Xs[::2].bmm(Xs[1::2])
            masks = masks[:, ::2] * masks[:, 1::2]
            rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
            rs = rs[masks[:, self.N0:].squeeze(2) == 1]
            loss = loss + self.f_huber(rs[:,2])/(2**(k - self.min_N + 1))
        return loss

    def forward_with_quaternion_mask(self, xs, hat_xs):
        """Forward errors with quaternion"""
        N = xs.shape[0]
        masks = xs[:, :, 3].unsqueeze(1)
        masks = torch.nn.functional.conv1d(masks, self.weight, bias=None,
            stride=self.min_train_freq).double().transpose(1, 2)
        masks[masks < 1] = 0
        Xs = SO3.qexp(xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.qexp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
        rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs)).reshape(N,
                -1, 3)[:, self.N0:]
        rs = rs[masks[:, self.N0:].squeeze(2) == 1]
        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
            Xs = SO3.qmul(Xs[::2], Xs[1::2])
            masks = masks[:, ::2] * masks[:, 1::2]
            rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs)).reshape(N,
                -1, 3)[:, self.N0:]
            rs = rs[masks[:, self.N0:].squeeze(2) == 1]
            loss = loss + self.f_huber(rs)/(2**(k - self.min_N + 1))
        return loss




# class INSIntegrationLoss(torch.nn.Module):
#     """
#     INS解算过程的损失计算，基于网络预测的加速度计和陀螺仪数据
#     和地面真值之间的误差。
#     """
#
#     # def __init__(self):
#     #     super(INSIntegrationLoss, self).__init__()
#     #     self.mse_loss = torch.nn.MSELoss()
#     def __init__(self, initial_conditions, dt):
#         super(INSIntegrationLoss, self).__init__()
#         self.initial_conditions = initial_conditions
#         self.dt = dt
#         self.mse_loss = torch.nn.MSELoss()
#
#     def forward(self,predicted, ground_truth):
#         """
#         参数:
#         predicted: 预测的加速度计和陀螺仪数据，格式为(batch_size, sequence_length, 6)。
#         ground_truth: 地面真值数据，包括速度、位置和姿态信息，格式为字典。
#         """
#         print(f"ground_truth keys: {list(ground_truth.keys())}")
#         for key in ground_truth:
#             print(f"Type of ground_truth['{key}']: {type(ground_truth[key])}")
#         print(f"Type of ground_truth['vs']: {type(ground_truth['vs'])}")
#         print(f"Type of ground_truth['ps']: {type(ground_truth['ps'])}")
#         print(f"Type of ground_truth['qs']: {type(ground_truth['qs'])}")
#         # 解包地面真值数据
#         assert isinstance(ground_truth['vs'], torch.Tensor), "ground_truth['vs'] must be a Tensor"
#         assert isinstance(ground_truth['ps'], torch.Tensor), "ground_truth['ps'] must be a Tensor"
#         assert isinstance(ground_truth['qs'], torch.Tensor), "ground_truth['qs'] must be a Tensor"
#         v_gt, p_gt, q_gt = ground_truth['vs'], ground_truth['ps'], ground_truth['qs']
#
#         # 执行INS解算，估计速度、位置和姿态
#         # 这里假设已经有了INS解算的函数，其使用预测的IMU数据进行解算
#         # 并返回估计的速度、位置和姿态
#         v_est, p_est, q_est = perform_ins_solve(predicted)
#
#         # 计算速度、位置和姿态的损失
#         loss_velocity = self.mse_loss(v_est, v_gt)
#         loss_position = self.mse_loss(p_est, p_gt)
#         loss_attitude = self.mse_loss(q_est, q_gt)
#
#         # 综合损失，这里简单地将三个损失相加
#         # 可以根据实际情况调整损失的权重
#         total_loss = loss_velocity + loss_position + loss_attitude
#
#         return total_loss

initial_conditions = {
    'vs': torch.zeros(3),  # Initial velocity
    'ps': torch.zeros(3),  # Initial position
    'qs': torch.tensor([1.0, 0.0, 0.0, 0.0])  # Initial orientation (quaternion)
}


class INSIntegrationLoss(torch.nn.Module):
    def __init__(self, initial_conditions, dt):
        super(INSIntegrationLoss, self).__init__()
        self.initial_conditions = initial_conditions
        self.dt = dt
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, predicted, ground_truth, us):
        v_gt = ground_truth[:, :, 0:3]  # Placeholder slicing
        p_gt = ground_truth[:, :, 3:6]  # Placeholder slicing
        q_gt = ground_truth[:, :, 6:10] # Placeholder slicing assuming quaternion representation for orientation

        # Perform INS solve to estimate velocity, position, and attitude
        v_est, p_est, q_est = perform_ins_solve(predicted, self.initial_conditions, self.dt, us)

        # Calculate loss for velocity, position, and attitude
        loss_velocity = self.mse_loss(v_est, v_gt)
        loss_position = self.mse_loss(p_est, p_gt)
        loss_attitude = self.mse_loss(q_est, q_gt)

        # Combine losses
        total_loss = loss_velocity + loss_position + loss_attitude

        return total_loss


def load_initial_conditions(dataset, sequence_index):
    """
    从给定序列索引的地面真值数据中加载初始条件。
    包括初始速度(vs)，位置(ps)和姿态(qs)。
    """
    # 确保序列索引在有效范围内
    if sequence_index < 0 or sequence_index >= len(dataset.sequences):
        raise ValueError("Sequence index out of range.")

    sequence_name = dataset.sequences[sequence_index]
    gt_data = dataset.load_gt(sequence_index)

    if not gt_data:
        raise ValueError(f"Ground truth data for sequence {sequence_name} could not be loaded.")

    # 从地面真值数据中提取初始条件
    initial_conditions = {
        'vs': gt_data['vs'][0],  # 取序列的第一个速度为初始速度
        'ps': gt_data['ps'][0],  # 取序列的第一个位置为初始位置
        'qs': gt_data['qs'][0],  # 取序列的第一个姿态为初始姿态
    }

    return initial_conditions


def perform_ins_solve(predicted, initial_conditions, dt, us):
    """
    执行INS解算。
    参数:
    predicted: 预测的加速度计和陀螺仪数据，格式为(batch_size, sequence_length, 6)。
    initial_conditions: 包含初始速度、位置和姿态（四元数）的字典。
    dt: 采样时间间隔。
    返回:
    估计的速度、位置和姿态四元数。
    """
    # Assuming initial_conditions contains 'vs', 'ps', and 'qs'

    print(f"Predicted tensor shape before accel extraction: {predicted.shape}")
    try:
        v0, p0, q0 = initial_conditions['vs'], initial_conditions['ps'], initial_conditions['qs']
    except KeyError as e:
        print(f"Missing key in initial_conditions: {e}")
        raise

    # 解包初始条件
    v0, p0, q0 = initial_conditions['vs'], initial_conditions['ps'], initial_conditions['qs']

    batch_size, sequence_length, _ = predicted.shape


    # v_est = torch.zeros_like(v0).repeat(1, sequence_length).reshape(batch_size, sequence_length, -1)
    # p_est = torch.zeros_like(p0).repeat(1, sequence_length).reshape(batch_size, sequence_length, -1)
    # q_est = q0.repeat(1, sequence_length).reshape(batch_size, sequence_length, -1)
    v_est = torch.zeros(batch_size, sequence_length, 3)
    p_est = torch.zeros(batch_size, sequence_length, 3)
    q_est = torch.zeros(batch_size, sequence_length, 4)
    print(q_est.shape)
    #q0 = q0.unsqueeze(0)
    q0_repented =q0.repeat(batch_size, sequence_length, 1)
    q_est[:, 0, :]= q0_repented[:, 0, :]

    print(f"predicted shape: {predicted.shape}")
    print(f"predicted data sample: {predicted[0, :, 3:6]}")

    for i in range(1, sequence_length):
        gyro = predicted[:, i-1, :3]
        accel = us[:, i-1, :3]
        print(f"Before processing, accel shape: {accel.shape}, type: {accel.dtype}")  # 打印处理前的accel形状和类型

        # 四元数更新
        for b in range(batch_size):
            q_est[b, i, :] = SO3.quaternion_update_batch(q_est[:, i - 1], gyro[b], dt)



        print("accel shape:", accel.shape)
        # 将加速度从体坐标系转换到导航坐标系，并扣除重力
        gravity = torch.tensor([0, 0, 9.81], device=accel.device)
        accel_ned = SO3.rotate_vector(accel, q_est[:, i]) - gravity
        print(f"After processing, accel_ned shape: {accel_ned.shape}, type: {accel_ned.dtype}")  # 打印处理后的accel形状和类型

        # 速度和位置积分
        v_est[:, i] = v_est[:, i-1] + accel_ned * dt
        p_est[:, i] = p_est[:, i-1] + v_est[:, i] * dt

    return v_est, p_est, q_est

# 加载初始条件
# def load_initial_conditions(self, sequence_index):
#     """
#     加载指定序列的初始条件，包括起始速度、位置和姿态（四元数）。
#     """
#     gt_data = self.load_gt(sequence_index)
#     initial_velocity = gt_data['vs'][0]  # 取序列的第一个速度为初始速度
#     initial_position = gt_data['ps'][0] - gt_data['ps'][0]  # 初始位置 0
#     initial_orientation = gt_data['qs'][0]  # 取序列的第一个姿态为初始姿态
#
#     return {
#         'velocity': initial_velocity,
#         'position': initial_position,
#         'orientation': initial_orientation,
#     }



def load_initial_conditions(dataset, sequence_index):
    gt_data = dataset.load_gt(sequence_index)
    if gt_data:
        initial_conditions = {
            'vs': gt_data['vs'],  # 注意这里的键匹配
            'ps': gt_data['ps'],
            'qs': gt_data['qs']
        }
        return initial_conditions
    else:
        print("Error: Missing required keys in ground truth data for initial conditions.")
        return None

