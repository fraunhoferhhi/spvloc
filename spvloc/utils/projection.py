import numpy as np

try:
    import cv2
    import torch
    from scipy import ndimage
except Exception:
    pass

from .panorama import coords_to_uv, coords_to_uv_persp, uvs_to_rays, uvs_to_rays_persp


def calculate_bb(xy, width, height, margin=3):
    """
    Calculate a 2D bounding box (BB) for a set of 3D points within an image frame.

    Parameters:
    xy (numpy.ndarray): A 2D array of shape (N, 2) representing the X and Y coordinates of 3D points.
    width (int): The width of the image frame.
    height (int): The height of the image frame.
    margin (int): The distance from the image frame border to the BB. Default is 3.

    Returns:
    numpy.ndarray: A 1D array of shape (5,) representing the 2D BB in the format [x_min, y_min, x_max, y_max, 1].
                  If the input array 'xy' is empty, it returns a zero-filled BB array.
    """
    if xy.size != 0:
        bb_min = np.min(xy, axis=0)
        bb_max = np.max(xy, axis=0)
        # correct bounding box if it exceeds image border in x direction
        if bb_max[0] > (width - margin) and bb_min[0] < margin:
            unique_x = np.unique(xy[:, 0])
            diff_left = unique_x - np.concatenate([[0], unique_x])[:-1]
            if np.max(diff_left) > (2 * margin):
                unique_x = np.where(unique_x < unique_x[np.argmax(diff_left)], unique_x + width, unique_x)
                bb_min[0] = np.min(unique_x)
                bb_max[0] = np.max(unique_x)

        # correct bounding box if it exceeds image border in y direction
        if bb_max[1] > (height - margin) and bb_min[1] < margin:
            unique_y = np.unique(xy[:, 0])
            diff_left = unique_y - np.concatenate([[0], unique_y])[:-1]
            if np.max(diff_left) > (2 * margin):
                unique_y = np.where(unique_y < unique_y[np.argmax(diff_left)], unique_y + height, unique_y)
                bb_min[1] = np.min(unique_y)
                bb_max[1] = np.max(unique_y)
        bb_2d = np.concatenate([bb_min, bb_max, [1]], axis=0)  # added 1 since this is a positive sample
    else:
        bb_2d = np.zeros([5])
    return bb_2d


def project_pano_to_persp_mask_pano(
    persp_depth,
    pano_depths,
    pano_poses,
    persp_pose,
    cam,
    closing_iteration=2,
    threshold=0.2,
    fix_rotation=True,
    scale_translation=True,
    projection_margin=0,
):
    """
    Project perspective depth to panoramic depth and generate a visibility mask.

    Parameters:
    - persp_depth (numpy.ndarray): Perspective depth map of shape (h, w).
    - pano_depths (numpy.ndarray): Panoramic depth maps of shape (bs, hp, wp).
    - pano_poses (numpy.ndarray): Panoramic camera poses of shape (bs, 3).
    - persp_pose (numpy.ndarray): Perspective camera pose of shape (3, 4) or (4, 4).
    - cam (numpy.ndarray): Camera matrix of shape (3, 3).
    - closing_iteration (int): Iterations of binary closing.
    - threshold (float): Allowed distance between reference and projected depth.
    - fix_rotation (bool): Scene rotation and axis swap.

    Returns:
    - visibility_mask (numpy.ndarray): Visibility mask of shape (bs, hp, wp).
    """
    bs, hp, wp = pano_depths.shape
    h, w = persp_depth.shape
    # aspect = w / h
    cam_in = np.expand_dims(cam, 0)

    if persp_pose.shape[0] == 3:
        pose_in = np.concatenate([np.array(persp_pose), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    else:
        pose_in = persp_pose.copy()

    if scale_translation:
        pose_in[:3, 3] /= 1000.0

    if fix_rotation:
        pose_in[:3, 2] = -1 * pose_in[:3, 2]

    pose_in = np.linalg.inv(pose_in)
    pose_in = np.expand_dims(pose_in, 0)

    # quite slow for many points
    ordered_pointclouds_world = depth_to_3d_pano(pano_depths, wp, hp, pano_poses, False)
    transformed_points, points_projected = project_back_with_pose(
        ordered_pointclouds_world, pose_in, -1 * cam_in, w, h
    )

    points_projected = points_projected.reshape(bs, hp, wp, 2)
    transformed_points = transformed_points.reshape(bs, hp, wp, 3)
    transformed_points_norm = np.linalg.norm(transformed_points, axis=-1)  # shape (bs, 128, 256)
    x_c = points_projected[..., 0]
    y_c = points_projected[..., 1]

    margin_x = 0  # int(((w - (h - (2 * projection_margin)) * aspect) / 2) + 0.5)

    # margin_x = int((projection_margin * aspect) + 0.5)
    valid_mask = (
        (y_c < h - projection_margin)
        * (y_c >= projection_margin)
        * (x_c < w - margin_x)
        * (x_c >= margin_x)
        * (transformed_points[:, :, :, -1] < 0)
    )

    reference_points = np.zeros_like(transformed_points_norm)

    xy = np.where(valid_mask)
    reference_points[xy] = persp_depth[y_c[xy], x_c[xy]]
    point_difference = np.abs(reference_points - transformed_points_norm)
    valid_mask = valid_mask * (point_difference <= threshold)

    if closing_iteration > 0:
        pad_width = ((0, 0), (closing_iteration, closing_iteration), (closing_iteration, closing_iteration))
        valid_mask = np.pad(valid_mask, pad_width, mode="wrap")

        for i in range(bs):
            valid_mask[i] = ndimage.binary_closing(
                valid_mask[i], structure=np.ones((3, 3)), iterations=closing_iteration
            )

        valid_mask = valid_mask[:, pad_width[1][0] : -pad_width[1][1], pad_width[2][0] : -pad_width[2][1]]

    # Calculate bounding boxes for each batch
    batched_bounding_boxes = []
    for batch_points in valid_mask:
        y, x = np.where(batch_points)
        bb = calculate_bb(np.stack([x, y]).transpose(), wp, hp)
        batched_bounding_boxes.append(bb)

    # Convert the list of bounding boxes to a NumPy array
    batched_bounding_boxes = np.array(batched_bounding_boxes)

    return batched_bounding_boxes, valid_mask


def projects_onto_floor(location, floors):
    return projects_onto_floor_efficient(location[np.newaxis, ...], floors)[0]


def projects_onto_floor_efficient(locations, floors):
    num_floors = len(floors.faces_list())
    verts_list = floors.verts_list()
    faces_list = floors.faces_list()
    locations = torch.Tensor(locations)
    locations_broadcasted = locations.unsqueeze(1).expand(-1, num_floors, -1)

    # Process vertices and faces
    verts_padded = torch.nn.utils.rnn.pad_sequence(verts_list, batch_first=True, padding_value=float("nan"))
    faces_padded = torch.nn.utils.rnn.pad_sequence(faces_list, batch_first=True, padding_value=-1)

    # Mask out invalid vertices
    verts_padded[torch.isnan(verts_padded)] = 0.0  # Replace NaN values with zeros for padding

    verts_expanded = verts_padded.unsqueeze(0).expand(locations.size(0), -1, -1, -1).numpy()
    faces_expanded = faces_padded.unsqueeze(0).expand(locations.size(0), -1, -1, -1).numpy()
    offset = np.array([np.arange(num_floors)])[..., np.newaxis, np.newaxis] * verts_expanded.shape[2]
    faces_expanded[faces_expanded >= 0] = (faces_expanded + offset)[faces_expanded >= 0]
    verts_expanded = verts_expanded.reshape(-1, 3)
    faces_expanded = faces_expanded.reshape(-1, 3)

    # Get the indices of the vertices for each triangle
    tri_indices = faces_expanded

    # Gather the vertices using the indices
    triangles = np.take(verts_expanded, tri_indices, axis=0)
    triangles = torch.tensor(
        triangles.reshape(locations_broadcasted.shape[0], faces_padded.shape[0], faces_padded.shape[1], 3, 3)
    )
    # Compute barycentric coordinates
    bary = barycentric_coordinates_batched(
        locations_broadcasted.reshape(-1, 3).detach().numpy(),
        triangles.reshape(-1, faces_padded.shape[1], 3, 3).detach().numpy(),
    )
    bary = torch.tensor(bary.reshape(locations_broadcasted.shape[0], faces_padded.shape[0], faces_padded.shape[1], 3))

    # Determine if each location is inside a triangle
    is_inside = (bary >= 0).all(dim=3) & (bary <= 1).all(dim=3) & (torch.abs(bary.sum(dim=3) - 1) < 1e-5)

    reduced_tensor = is_inside.any(dim=-1)
    indices = np.argmax(reduced_tensor, axis=-1, keepdims=True)
    floor_indices = np.where(reduced_tensor, indices, -1)

    return np.max(floor_indices, axis=1)


def barycentric_coordinates_batched(p, v):
    """
    Compute the barycentric coordinates of a point relative to multiple triangles.
    A batched version of the Pytorch3d function barycentric_coordinates

    Args:
        p: Coordinates of a point with shape (batch_size, 3).
        v: Coordinates of the triangle vertices with shape (batch_size, n, 3, 3).

    Returns:
        bary: Barycentric coordinates with shape (batch_size, n, 3) in the range [0, 1].
    """

    def edge_function(p, v0, v1):
        return (p[..., 0] - v0[..., 0]) * (v1[..., 1] - v0[..., 1]) - (p[..., 1] - v0[..., 1]) * (
            v1[..., 0] - v0[..., 0]
        )

    def compute_area(v0, v1, v2):
        return edge_function(v2, v0, v1) + np.finfo(float).eps  # Add epsilon for numerical stability.

    areas = compute_area(v[..., 2, :], v[..., 0, :], v[..., 1, :])  # Shape: (batch_size, n)
    w0 = edge_function(p[:, None], v[..., 1, :], v[..., 2, :]) / areas[..., None].squeeze(-1)
    w1 = edge_function(p[:, None], v[..., 2, :], v[..., 0, :]) / areas[..., None].squeeze(-1)
    w2 = edge_function(p[:, None], v[..., 0, :], v[..., 1, :]) / areas[..., None].squeeze(-1)
    bary = np.stack([w0, w1, w2], axis=-1)  # Shape: (batch_size, n, 3)

    return bary


def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out


def pano2persp_get_camera_parameter(fov, yaw, pitch, roll):
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)  # New: Define z-axis orientation
    R1, _ = cv2.Rodrigues(y_axis * np.radians(yaw))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(pitch))
    R3, _ = cv2.Rodrigues(np.dot(R2, z_axis) * np.radians(roll))  # New: Calculate rotation matrix for z-axis
    R = R3 @ R2 @ R1  # Updated: Combine all rotation matrices

    R_f = np.copy(R)  # fixed rotation matrix to be compatible with the renderers
    R_f[1] = -R_f[1]  # negate second row
    R_f[:, 1] = -R_f[:, 1]  # negate second col
    R_f[[1, 2]] = R_f[[2, 1]]  # flip second and third row

    cam = np.eye(3)
    cam[0, 0] = cam[1, 1] = 1.0 / np.tan(np.deg2rad(fov / 2))

    return R, R_f, cam


# modified from https://github.com/fuenwang/Equirec2Perspec
def pano2persp_from_rotation(img, cam, R, height, width):
    #
    # THETA is left/right angle, PHI is up/down angle, both in degree
    # cam is the same as used in redner
    #

    fov = np.rad2deg(np.arctan(1 / cam[0, 0])) * 2

    f = 0.5 * width * 1 / np.tan(0.5 * fov / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array(
        [
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ],
        np.float32,
    )
    K_inv = np.linalg.inv(K)

    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    xyz = xyz @ K_inv.T

    xyz = xyz @ R.T
    lonlat = xyz2lonlat(xyz)
    XY = lonlat2XY(lonlat, shape=img.shape).astype(np.float32)
    persp = cv2.remap(img, XY[..., 0], XY[..., 1], cv2.INTER_AREA, borderMode=cv2.BORDER_WRAP)

    return persp


def depth_to_3d_pano(depth_img, width, height, poses, save=False, filename="point_cloud.obj"):
    xys_camera = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=2).reshape((-1, 2)) + 0.5

    uvs = coords_to_uv(xys_camera, width, height)
    rays = uvs_to_rays(uvs)[np.newaxis, ...]

    depth = depth_img.reshape(depth_img.shape[0], -1)
    depth = depth[..., np.newaxis]

    points_3d = depth * rays

    # transform to world coords
    points_3d = points_3d + np.expand_dims(poses, 1)

    if save:
        save_point_cloud_as_obj(filename, points_3d.reshape(-1, 3))

    return points_3d  # .reshape(points_3d.shape[0], height, width, 3)


# Unused
def depth_to_3d(depth_img, K, width, height, save=False, filename="point_cloud.obj"):
    xys_camera = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=2).reshape((-1, 2)) + 0.5

    uvs = coords_to_uv_persp(xys_camera, width, height, K[0, 0, 0], K[0, 1, 1])
    rays = uvs_to_rays_persp(uvs, normalize=True)[np.newaxis, ...]

    depth = depth_img.reshape(depth_img.shape[0], -1)
    depth = depth[..., np.newaxis]

    points_3d = depth * rays

    if save:
        save_point_cloud_as_obj(filename, points_3d[0])

    return points_3d


def project_back_with_pose(points_3d, pose, K, width, height, from_left=False, save=False, filename="point_cloud.obj"):
    aspect = width / height
    K_ = K.copy()
    K_[:, 1, 1] = -K_[:, 1, 1] * aspect

    # make homogeneous
    ones_column = np.ones((points_3d.shape[0], points_3d.shape[1], 1))
    points_3d_homogeneous = np.concatenate((points_3d, ones_column), axis=-1)

    if from_left:
        transformed_points = np.matmul(pose, points_3d_homogeneous.transpose(0, 2, 1)).transpose(0, 2, 1)[..., :3]
    else:
        transformed_points = np.matmul(points_3d_homogeneous, pose.transpose(0, 2, 1))[..., :3]

    normalized_points_3d = transformed_points / transformed_points[..., -1:]

    # Project points
    projected_points = np.matmul(normalized_points_3d, K_)
    points_projected = ((projected_points[:, :, :2] + 1) * 0.5 * np.array([width, height]) + 0.5).astype(np.int32)

    if save:
        save_point_cloud_as_obj(filename, transformed_points[0])

    return transformed_points, points_projected


def save_point_cloud_as_obj(filename, pts, color=None, add_normal=False):
    normals = []
    if add_normal:
        center = np.mean(np.array(pts), axis=0)
        for pt in pts:
            normal = np.array(pt) - center
            normal /= np.linalg.norm(normal)
            normals.append(normal)

    with open(filename, "w") as f:
        for i, pt in enumerate(pts):
            if color is not None:
                f.write(f"v {pt[0]} {pt[1]} {pt[2]} {color[0]} {color[1]} {color[2]} 255\n")
            else:
                f.write(f"v {pt[0]} {pt[1]} {pt[2]}\n")
            if add_normal:
                f.write(f"vn {normals[i][0]} {normals[i][1]} {normals[i][2]}\n")
