import numpy as np


def combine_rotations(left, right):
    left = np.concatenate([np.array(left), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    right = np.concatenate([np.array(right), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

    return np.matmul(left, right)  # [0:3,:]


def normalize(vector):
    return vector / np.linalg.norm(vector)


def parse_camera_info(camera_info, height, width):
    """extract intrinsic and extrinsic matrix"""
    lookat = normalize(camera_info[3:6])
    up = normalize(camera_info[6:9])

    W = lookat
    U = normalize(np.cross(W, up))
    V = -np.cross(W, U)  # this flips image

    rot = np.vstack((U, V, W)).T
    trans = camera_info[:3]

    xfov = camera_info[9]
    yfov = camera_info[10]

    # intrinsics in device normalized coordinates
    K = prepare_camera_matrix(xfov, yfov, height, width)

    return rot, trans, K


def prepare_camera_matrix(xfov, yfov, height, width):
    K = np.diag([1.0, 1.0, 1.0])
    K[0, 2] = 0  # width / 2
    K[1, 2] = 0  # height / 2

    K[0, 0] = 1.0 / np.tan(xfov)  # K[0, 2] / np.tan(xfov)
    K[1, 1] = 1.0 / np.tan(yfov) * height / width  # K[1, 2] / np.tan(yfov)
    return K
