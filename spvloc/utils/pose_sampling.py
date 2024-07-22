import warnings

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
from skimage.draw import polygon as skpolygon

from ..utils.plot_floorplan import convert_lines_to_vertices
from ..utils.projection import projects_onto_floor, projects_onto_floor_efficient


def _sample_near_poses_optimized(pose_panorama, floor_geometry, max_offset, num_samples, sample_in_room=True):
    max_attempts = 100
    gt_room_idx = projects_onto_floor(pose_panorama, floor_geometry)
    if gt_room_idx < 0:
        # warnings.warn("pose outside room during near pose sampling")
        gt_room_idx = 0  # TODO: Use actual room index int(room_id)

    poses = np.tile(pose_panorama, (num_samples, 1))
    rooms = np.full(num_samples, gt_room_idx)

    valid_poses = np.full((num_samples), False)
    for i in range(max_attempts):
        pose_offsets = (2 * (np.random.rand(num_samples, 3) - 0.5)) * max_offset * 1000.0
        poses_pano_new = pose_panorama + pose_offsets
        near_room_indices = projects_onto_floor_efficient(poses_pano_new, floor_geometry)

        if sample_in_room:
            valid = near_room_indices == gt_room_idx
        else:
            valid = near_room_indices >= 0  # == gt_room_idx

        valid_poses = np.where(valid, True, valid_poses)
        rooms = np.where(valid, near_room_indices, rooms)
        poses = np.where(valid[:, np.newaxis], poses_pano_new, poses)

        if np.all(valid_poses):
            break
        # if i == (max_attempts - 1): # TODO activate this again and get sure it is not triggered
        #     warnings.warn("No pose sampled in function _sample_near_poses_optimized")
    poses_out = list(zip(list(poses), list(rooms)))
    return poses_out


def sample_far_poses(pose_panorama, floor_geometry, limits, offset=500, num_samples=1, far_min_dist=1000):
    poses = []
    for _ in range(num_samples):
        far_room_idx = -1
        num_attempts = 0
        while far_room_idx < 0:
            x_location = np.random.uniform(limits[0] + (offset / 2), limits[1] - (offset / 2))
            y_location = np.random.uniform(limits[2] + (offset / 2), limits[3] - (offset / 2))
            pose_far = np.array([x_location, y_location, pose_panorama[-1]])

            room_idx = projects_onto_floor(pose_far, floor_geometry)
            if np.linalg.norm(pose_panorama - pose_far) < far_min_dist:
                if room_idx >= 0:
                    pose_near = pose_far
                continue

            far_room_idx = room_idx
            if num_attempts >= 100:
                pose_far = pose_near
                continue
            #     pose_far = pose_near
            # far_room_idx = projects_onto_floor(pose_far, floor_geometry)
            # num_attempts += 1
        poses.append((pose_far, far_room_idx))
    return poses


def _sample_test_poses(limits, z, floor_geometry, step=1000):
    x_locations = np.arange(limits[0] + (step / 2), limits[1] + (step / 2), step)
    y_locations = np.arange(limits[2] + (step / 2), limits[3] + (step / 2), step)

    poses = np.meshgrid(x_locations, y_locations)
    poses = np.stack(poses, axis=2).reshape(-1, 2)
    poses = np.concatenate([poses, np.full((poses.shape[0], 1), z)], axis=1)

    room_idxs = [projects_onto_floor(pose, floor_geometry) for pose in poses]
    pose_grid = [(p, i) for p, i in zip(poses, room_idxs) if i >= 0]
    return pose_grid


def get_grid(poly, step_size, walls_padding, rooms, planeID):
    x1, y1 = np.array(poly.exterior.coords.xy)
    radius_x = (np.max(x1) - np.min(x1)) / 2 - 1
    radius_y = (np.max(y1) - np.min(y1)) / 2 - 1
    walls_padding_poly = min(walls_padding, radius_x, radius_y)

    # padded = poly.buffer(-walls_padding_poly - 4)  # Inward padding
    room_out = poly.buffer(-walls_padding_poly)  # Inward padding

    try:
        coords = [room_out.exterior.coords.xy]
    except Exception:
        coords = [poly.exterior.coords.xy for poly in room_out.geoms]
    grid_points = []

    def get_meshgrid(x1, y1, grid_points):
        x_min, x_max = np.min(x1), np.max(x1)
        y_min, y_max = np.min(y1), np.max(y1)
        # Grid spacing, adjust as needed

        step_sizex = min((x_max - x_min) / 2, step_size)
        step_sizey = min((y_max - y_min) / 2, step_size)
        # step_sizex = step_size
        # step_sizey = step_size
        grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max, step_sizex), np.arange(y_min, y_max, step_sizey))
        # Flatten and pair x and y grid coordinates
        grid_points.extend(np.c_[grid_x.ravel(), grid_y.ravel()])
        return grid_points

    for x1, y1 in coords:
        if len(x1) == 0:
            rooms[planeID] = poly
            grid_points.extend(np.array(poly.centroid.xy).reshape([1, 2]))
            return grid_points
        grid_points = get_meshgrid(x1, y1, grid_points)

    grid_points_filtered = [p for p in grid_points if room_out.contains(Point(p))]

    if len(grid_points_filtered) == 0:
        import largestinteriorrectangle as lir
        from numba import config

        config.DISABLE_JIT = True
        x, y, w, h = lir.lir(np.expand_dims(np.array(room_out.exterior.coords), 0).astype(np.int32))
        config.DISABLE_JIT = False

        room_out = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        coords = room_out.exterior.coords.xy
        grid_points = get_meshgrid(coords[0], coords[1], [])
        grid_points_filtered = [p for p in grid_points if room_out.contains(Point(p))]

    rooms[planeID] = room_out
    return grid_points_filtered


def _sample_test_poses_v2(
    limits,
    z,
    annos,
    floor_geometry,
    step_size,
    walls_padding_size,
    visualize=False,
):
    """Generate floorplan instance mask with distance around walls and columns"""
    x0, x1, y0, y1 = limits
    visualize = False
    if visualize:
        visualization_mask = np.zeros((round(x1 - x0) + 1, round(y1 - y0) + 1), dtype=np.uint8)

    # step_size = density * ((x1 - x0) + (y1 - y0)) / 2
    # walls_padding_size = walls_padding_density * ((x1 - x0) + (y1 - y0)) / 2
    junctions = np.array([junc["coordinate"][:2] for junc in annos["junctions"]])
    sample = []
    rooms = {}
    columns = []

    columns_x, columns_y = [], []
    for semantic in annos["semantics"]:
        if semantic["type"] not in ["outwall", "door", "window", "wall", "bb"]:
            for planeID in semantic["planeID"]:
                if annos["planes"][planeID]["type"] == "floor":
                    lineIDs = np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist()
                    junction_pairs = [
                        np.where(np.array(annos["lineJunctionMatrix"][lineID]))[0].tolist() for lineID in lineIDs
                    ]
                    polygon = convert_lines_to_vertices(junction_pairs)
                    polygon_coords = np.array([junctions[vertex] for vertex in polygon[0]])
                    x = np.around(polygon_coords[:, 0] - x0) - 1
                    y = np.around(polygon_coords[:, 1] - y0) - 1

                    coords = np.column_stack([x, y])
                    poly = Polygon(coords)

                    if visualize:
                        x_polygon, y_polygon = skpolygon(x, y)

                    if semantic["type"] == "column":
                        padded = poly.buffer(walls_padding_size)  # Outward padding
                        columns.append(padded)

                        columns_x.extend[x_polygon]
                        columns_y.extend[y_polygon]
                    else:
                        if visualize:
                            visualization_mask[x_polygon, y_polygon] = 1
                        grid_points = get_grid(poly, step_size, walls_padding_size, rooms, planeID)
                        sample.extend([(planeID, grid_point) for grid_point in grid_points])
                        # Check if each grid point is inside the polygon

    if visualize:
        visualization_mask[columns_x, columns_y] = 0
    sample_filtered = []
    planes = [planeID for planeID, _ in sample]
    valid_planes = []

    for planeID, grid_point in sample:
        x, y = grid_point
        point = Point(x, y)
        in_poly = rooms[planeID].contains(point)
        in_col = any(col.contains(point) for col in columns)

        if in_poly and not in_col:
            planes.remove(planeID)
            valid_planes.append(planeID)
            sample_filtered.append(np.array([x + x0, y + y0, z]))
            if visualize:
                visualization_mask[int(x) - 15 : int(x) + 15, int(y) - 15 : int(y) + 15] = 3

    poses = np.stack(sample_filtered, axis=0)
    # poses = np.concatenate([poses, np.full((poses.shape[0], 1), z)], axis=1)
    room_idxs = [projects_onto_floor(pose, floor_geometry) for pose in poses]
    pose_grid = [(p, i) for p, i in zip(poses, room_idxs) if i >= 0]

    if visualize:
        plt.imshow(visualization_mask)
        plt.colorbar()
        plt.title("Instance Mask")
        plt.show()

    return pose_grid
