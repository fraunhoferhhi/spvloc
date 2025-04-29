"""
Parts of this code are modified from: https://github.com/bertjiazheng/Structured3D
Copyright (c) 2019 Structured3D Group
"""

import subprocess

import numpy as np
import trimesh
from tqdm import tqdm


def project(x, meta):
    """project 3D to 2D for polygon clipping"""
    proj_axis = max(range(3), key=lambda i: abs(meta["normal"][i]))

    return tuple(c for i, c in enumerate(x) if i != proj_axis)


def project_inv(x, meta):
    """recover 3D points from 2D"""
    # Returns the vector w in the walls' plane such that project(w) equals x.
    proj_axis = max(range(3), key=lambda i: abs(meta["normal"][i]))

    w = list(x)
    w[proj_axis:proj_axis] = [0.0]
    c = -meta["offset"]
    for i in range(3):
        c -= w[i] * meta["normal"][i]
    c /= meta["normal"][proj_axis]
    w[proj_axis] = c
    return tuple(w)


def triangulate(points):
    """triangulate the plane for operation and visualization"""

    num_points = len(points)
    if not num_points:
        return [], []
    indices = np.arange(num_points, dtype=np.int32)
    segments = np.vstack((indices, np.roll(indices, -1))).T
    points = np.array(points)
    if points.shape[1] == 3:
        plane_origin, plane_normal = trimesh.points.plane_fit(points)
        plane_transform = trimesh.geometry.plane_transform(plane_origin, plane_normal)
        point_cloud = trimesh.points.PointCloud(points)
        point_cloud.apply_transform(plane_transform)
        points2d = point_cloud.vertices
    else:
        points2d = points
    if len(np.unique(np.float32(points2d), axis=0)) > 2:
        try:
            polygon = trimesh.path.polygons.edges_to_polygons(segments, points2d)
            vertices, faces = trimesh.creation.triangulate_polygon(
                polygon[0], engine="triangle"
            )  # , triangle_args='a40000q')
        except Exception:
            points2d = order_points(points2d)
            polygon = trimesh.path.polygons.edges_to_polygons(segments, points2d)

            try:
                vertices, faces = trimesh.creation.triangulate_polygon(polygon[0], engine="triangle")
            except Exception as e:
                print(f"Triangulation failed: {e}")
                return [], []

        if points.shape[1] == 3:
            points3d = np.c_[vertices, np.zeros(vertices.shape[0])]
            point_cloud = trimesh.points.PointCloud(points3d)
            plane_transform_inv = np.linalg.inv(plane_transform)
            point_cloud.apply_transform(plane_transform_inv)
            vertices = point_cloud.vertices
        return vertices, faces
    else:
        return [], []


def order_points(points):
    # Calculate the centroid of the polygon
    centroid = np.mean(points, axis=0)

    # Calculate the angle between each point and the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort the points by the angle
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]


def clip_polygon(polygons, vertices_hole, junctions, meta, clip_holes=True):
    """clip polygon the hole"""
    mesh_holes = []
    if len(polygons) == 1:
        junctions = [junctions[vertex] for vertex in polygons[0]]

        mesh_wall_vertices, mesh_wall_faces = triangulate(junctions)
        return [mesh_wall_vertices, mesh_wall_faces], [mesh_wall_vertices, mesh_wall_faces], mesh_holes
    else:
        wall = []
        holes = []
        for polygon in polygons:
            if np.any(np.intersect1d(polygon, vertices_hole)):
                holes.append(polygon)
            else:
                wall.append(polygon)

        # extract junctions on this plane
        indices = []
        junctions_wall = []
        for plane in wall:
            for vertex in plane:
                indices.append(vertex)
                junctions_wall.append(junctions[vertex])
        junctions_wall = [project(x, meta) for x in junctions_wall]
        mesh_wall_vertices, mesh_wall_faces = triangulate(junctions_wall)

        if len(mesh_wall_vertices) > 0:
            mesh_wall_vertices_clipped = mesh_wall_vertices
            mesh_wall_faces_clipped = mesh_wall_faces

            if clip_holes:
                junctions_holes = []
                hole_vertices = []
                for plane in holes:
                    hole_vertices.append(plane)
                    junctions_hole = []
                    for vertex in plane:
                        indices.append(vertex)
                        junctions_hole.append(junctions[vertex])
                    junctions_holes.append(junctions_hole)

                junctions_holes = [[project(x, meta) for x in junctions_hole] for junctions_hole in junctions_holes]

                for hole, hole_ids in zip(junctions_holes, hole_vertices):
                    mesh_hole_vertices, mesh_hole_faces = triangulate(hole)
                    if len(mesh_hole_vertices) > 0:
                        try:
                            mesh_wall = trimesh.creation.extrude_triangulation(
                                mesh_wall_vertices_clipped, mesh_wall_faces_clipped, 1.0
                            )  # a dransformation could be added here
                            mesh_hole = trimesh.creation.extrude_triangulation(
                                mesh_hole_vertices, mesh_hole_faces, 2.0
                            )

                            " N.G. it is not really clear why this is needed"
                            mesh_hole.vertices = mesh_hole.vertices - [0.0, 0.0, 0.5]
                            # translate to center of axis aligned bounds
                            transform = np.eye(4)
                            transform[:3, 3] = -1.0 * mesh_hole.bounds.mean(axis=0)
                            mesh_hole.apply_transform(transform)
                            # scale minimally
                            mesh_hole.apply_scale(0.999)
                            # transform back
                            transform[:3, 3] = -1 * transform[:3, 3]
                            mesh_hole.apply_transform(transform)

                            # mesh_cut = trimesh.boolean.difference([mesh_wall, mesh_hole], engine="scad")  # really slow trimesh 3.10.0
                            mesh_cut = trimesh.boolean.difference([mesh_wall, mesh_hole], engine="manifold")

                            # filter mesh again (does not work because of cause vertex index changes as well)
                            mask = np.argwhere(mesh_cut.vertices[:, 2] < 0.1)  # keep those
                            mesh_cut.faces = mesh_cut.faces[np.isin(mesh_cut.faces[:, 0], mask)]
                            mesh_cut.faces = mesh_cut.faces[np.isin(mesh_cut.faces[:, 1], mask)]
                            mesh_cut.faces = mesh_cut.faces[np.isin(mesh_cut.faces[:, 2], mask)]
                            mesh_cut.remove_unreferenced_vertices()
                            mesh_wall_vertices_clipped = mesh_cut.vertices[:, :2]
                            mesh_wall_faces_clipped = (
                                mesh_cut.faces
                            )  # this includes too many faces, since we have all faces of the extruded mesh
                            mesh_hole_vertices = [project_inv(vertex, meta) for vertex in mesh_hole_vertices]
                            mesh_holes.append([mesh_hole_vertices, np.array(mesh_hole_faces), hole_ids])

                        except subprocess.CalledProcessError as err:
                            print("CalledProcessError: {0}".format(err))
                        except IndexError as err:
                            print("Index Error: {0}".format(err))

            vertices = [project_inv(vertex, meta) for vertex in mesh_wall_vertices]
            vertices_clipped = [project_inv(vertex, meta) for vertex in mesh_wall_vertices_clipped]
            return (
                [vertices, np.array(mesh_wall_faces)],
                [vertices_clipped, np.array(mesh_wall_faces_clipped)],
                mesh_holes,
            )
        else:
            return (
                [mesh_wall_vertices, np.array([mesh_wall_faces])],
                [mesh_wall_vertices, np.array([mesh_wall_faces])],
                mesh_holes,
            )  # return all empty


def convert_lines_to_vertices(lines):
    """convert line representation to polygon vertices"""
    polygons = []
    lines = np.array(lines)
    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()  # init line
            lines = np.delete(lines, 0, 0)
        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons


# visualize_plane
def generate_3d_model(annotations, clip_holes=False, display_bbox=True, color_by_normal=False):  # , args, eps=0.9):
    junctions = [item["coordinate"] for item in annotations["junctions"]]

    # extract hole vertices
    lines_holes = []
    for semantic in annotations["semantics"]:
        if semantic["type"] in ["window", "door", "opening"]:
            for planeID in semantic["planeID"]:
                lines_holes.extend(np.where(np.array(annotations["planeLineMatrix"][planeID]))[0].tolist())

    lines_holes = np.unique(lines_holes)
    if len(lines_holes) > 0:
        _, vertices_holes = np.where(np.array(annotations["lineJunctionMatrix"])[lines_holes])
        vertices_holes = np.unique(vertices_holes)
    else:
        vertices_holes = []

    has_rooms = False
    for semantic in annotations["semantics"]:
        if semantic["type"] in ["staircase", "corridor", "office", "outwall"]:
            has_rooms = True
            break

    add_elements = ["column", "window", "door", "opening", "bb"]
    if has_rooms:
        add_elements.extend(["staircase", "corridor", "office", "outwall"])
    else:
        add_elements.extend(["wall"])
    if not has_rooms:
        raise ValueError
    # load polygons

    all_geometries = []
    print("Creating Planes")
    for planeID in tqdm(range(len(annotations["planes"]))):

        plane_anno = annotations["planes"][planeID]
        lineIDs = np.where(np.array(annotations["planeLineMatrix"][planeID]))[0].tolist()
        junction_pairs = [
            np.where(np.array(annotations["lineJunctionMatrix"][lineID]))[0].tolist() for lineID in lineIDs
        ]
        polygon = convert_lines_to_vertices(junction_pairs)
        geometry, geometry_clipped, _ = clip_polygon(
            polygon, vertices_holes, junctions, plane_anno, clip_holes=clip_holes
        )
        all_geometries.append(geometry_clipped if clip_holes else geometry)

    polygons = []
    print("Creating Polygons")
    for semantic in tqdm(annotations["semantics"]):
        if semantic["type"] in add_elements:
            for planeID in semantic["planeID"]:
                plane_anno = annotations["planes"][planeID]
                if "bb" in plane_anno["type"] and not display_bbox:
                    continue
                geometry = all_geometries[planeID]
                polygons.append(
                    [geometry[0], geometry[1], planeID, plane_anno["normal"], plane_anno["type"], semantic["type"]]
                )

    plane_mesh_set = []
    print("Building mesh")
    for i, (vertices, faces, planeID, normal, plane_type, semantic_type) in tqdm(enumerate(polygons)):
        # ignore the room ceiling
        if plane_type == "ceiling" and semantic_type not in ["door", "window"]:
            continue
        if not np.any(vertices):
            continue
        if semantic_type == "opening":
            continue
        if semantic_type == "bb" and not display_bbox:
            continue

        face_normals = np.broadcast_to(np.array(normal), np.array(faces).shape)
        eps = 0.95
        if color_by_normal:
            # Colors depend on normals.
            if np.dot(normal, [1, 0, 0]) > eps:
                color = np.array([255, 0, 0])
            elif np.dot(normal, [-1, 0, 0]) > eps:
                color = np.array([0, 255, 0])
            elif np.dot(normal, [0, 1, 0]) > eps:
                color = np.array([0, 0, 255])
            elif np.dot(normal, [0, -1, 0]) > eps:
                color = np.array([255, 255, 0])
            elif np.dot(normal, [0, 0, 1]) > eps:
                color = np.array([0, 255, 255])
            elif np.dot(normal, [0, 0, -1]) > eps:
                color = np.array([125, 125, 125])
            else:
                color = np.array([255, 255, 255])
        else:
            # Colors depend on semantics.
            if semantic_type == "office":
                if np.dot(normal, [0, 0, 1]) > eps or np.dot(normal, [0, 0, -1]) > eps:
                    if np.all(vertices[:, 2] > 2000):
                        color = np.array([41, 120, 142])  # floor
                    else:
                        color = np.array([64, 67, 135])  # floor
                else:
                    color = np.array([34, 167, 132])  # wall
            elif semantic_type == "door":
                color = np.array([121, 209, 81])
            elif semantic_type == "window":
                color = np.array([253, 231, 36])
            else:
                color = np.array([192, 192, 192])  # gray

        face_color = np.broadcast_to(np.array(color), np.array(faces).shape)

        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, face_normals=face_normals, face_colors=face_color, process=False
        )
        plane_mesh_set.append(mesh)

    return trimesh.util.concatenate(plane_mesh_set)
