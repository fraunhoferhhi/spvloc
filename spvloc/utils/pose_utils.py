import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F

import transforms3d.quaternions as txq


# Converted from pytorch3d
def matrix_to_rotation_6d_np(matrix: np.ndarray) -> np.ndarray:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].copy().reshape(batch_dim + (6,))


# Converted from pytorch3d
def rotation_6d_to_matrix_np(d6: np.ndarray) -> np.ndarray:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


# From pytorch3d
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


# From pytorch3d
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def qlog(q):
    # Computes a log quaternion (3D) from quaternion (4D)
    if all(q[..., 1:] == 0):
        q = np.zeros([3])
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q


def qexp(q):
    # Computes a quaternion (4D) to a log quaternion (3D)
    if torch.is_tensor(q):
        n = torch.linalg.norm(q)
        q = torch.hstack((torch.cos(n), torch.sinc(n / torch.tensor(math.pi)) * q))
    else:
        n = np.linalg.norm(q)
        q = np.hstack((np.cos(n), np.sinc(n / np.pi) * q))
    return q


def angular_difference(yaw_gt, yaw_est):
    # Convert yaw values to radians
    yaw_gt_rad = np.radians(yaw_gt)
    yaw_est_rad = np.radians(yaw_est)

    # Calculate circular absolute angular difference
    diff_rad = np.arctan2(np.sin(yaw_gt_rad - yaw_est_rad), np.cos(yaw_gt_rad - yaw_est_rad))

    # Convert the result back to degrees and make sure it's in the range [-180, 180]
    diff_deg = np.degrees(diff_rad)
    # diff_deg = (diff_deg + 180) % 360 - 180

    return np.abs(diff_deg)


def get_pose_distances_batch(ref_batch, est_batch):
    t_est = est_batch[:, :, :3, 3]
    t_gt = ref_batch[:, :, :3, 3]
    r_est = est_batch[:, :, :3, :3]
    r_gt = ref_batch[:, :, :3, :3]

    # rotation distance in degree
    R = torch.matmul(r_est, torch.transpose(r_gt, dim0=2, dim1=3))
    trace_R = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    theta = torch.clamp((trace_R - 1) / 2, -1.0, 1.0)
    err_r = torch.acos(theta) * (180 / np.pi)

    # 2d rotation
    # query [1,2] instead [2,2] due to axis flip
    yaw_gt = np.degrees(np.arctan2(r_gt[:, :, 0, 2], r_gt[:, :, 1, 2]))
    yaw_est = np.degrees(np.arctan2(r_est[:, :, 0, 2], r_est[:, :, 1, 2]))
    err_yaw = torch.tensor(angular_difference(yaw_gt.numpy(), yaw_est.numpy()))
    err_t = torch.norm(t_gt - t_est, dim=-1)
    err_t_2d = torch.norm(t_gt[..., :2] - t_est[..., :2], dim=-1)

    return err_t, err_r, err_t_2d, err_yaw


def correct_pose(pose):
    """
    Corrects the pose by flipping the sign of the z-axis coordinate and transposing the rotation matrix.
    """
    if torch.is_tensor(pose):
        query = pose.clone()
    else:
        query = pose.copy()

    query[:3, 2] = -1 * pose[:3, 2]  # only rotation is changed
    query[:3, :3] = pose[:3, :3].T

    return query


def process_poses_spvloc(poses_in_query, poses_in_target, rot_type="log_quat"):
    """
    copy panorama poses and append the query pose at the end
    :param poses_in_query: 4 x 4
    :param poses_in_target: 1 x 3
    :return: processed poses (translation + quaternion) N x 7
    """
    N = len(poses_in_target)
    query = poses_in_query.copy()
    poses_out = np.zeros([N + 1, 6])

    if N > 0:
        poses_out[:N, 0:3] = poses_in_target

    poses_out[-1, 0:3] = query[0:3, 3]

    # IMPORTANT: ONLY IF AXIS IS FLIPPED AS HERE, THE OPERATION IS INVERTIBLE
    query[:, 2] = -1 * query[:, 2]  # remove mirroring
    R = query[:3, :3].transpose()
    if rot_type == "log_quat":
        q = txq.mat2quat(R)
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
    elif rot_type == "rotation_6d":
        q = matrix_to_rotation_6d_np(R)
    else:
        warnings.warn("Unsupported rotation format.")
    poses_out[-1, 3:] = q

    return poses_out


def decode_pose_spvloc(xyz, rot, scale_factor=1000.0):
    if rot.shape[-1] == 3:
        q = qexp(rot)

        if torch.is_tensor(q):
            q *= torch.sign(q[0])
            R = quaternion_to_matrix(q)
            pose = torch.eye(4, 4)
            pose[:3, :3] = R.t()
        else:
            q *= np.sign(q[0])
            R = txq.quat2mat(q)
            pose = np.eye(4, 4)
            pose[:3, :3] = R.transpose()
    elif rot.shape[-1] == 6:
        if torch.is_tensor(q):
            R = rotation_6d_to_matrix(q)
            pose[:3, :3] = R.t()
        else:
            R = rotation_6d_to_matrix_np(q)
            pose[:3, :3] = R.transpose()
    else:
        warnings.warn("Unsupported rotation format.")

    pose[:, 2] = pose[:, 2] * -1
    pose[0:3, 3] = xyz * scale_factor

    return pose
