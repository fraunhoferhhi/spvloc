"""
Parts of this code are modified from: https://github.com/bertjiazheng/Structured3D
Copyright (c) 2019 Structured3D Group
"""

import json
import os

import numpy as np
import torch

from ..utils.polygons import clip_polygon, convert_lines_to_vertices


def load_scene_annos(root, scene_id):
    with open(os.path.join(root, f"scene_{scene_id:05d}", "annotation_3d.json")) as file:
        annos = json.load(file)
    return annos


def prepare_geometry_from_annos(annos, use_torch=True):
    # create array with all coordinates from annos["junctions"]]
    junctions = [item["coordinate"] for item in annos["junctions"]]
    # extract hole vertices
    lines_holes = []
    hole_polygons_labeled = []
    for semantic in annos["semantics"]:
        # select windows and doors
        if semantic["type"] in ["window", "door"]:
            # select corresponding planes
            for planeID in semantic["planeID"]:
                # with plane_id get coordinates of lines from planeLineMatrix and append to array
                lines_holes.extend(np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist())

                # lineIDs contains the same data as previously added to lines_holes
                lineIDs = np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist()

                # with lineIds select junction IDs on that line and build array with all of them
                junction_pairs = [
                    np.where(np.array(annos["lineJunctionMatrix"][lineID]))[0].tolist() for lineID in lineIDs
                ]

                # create polygons inside plane a polygon has a type
                hole_polygons_labeled.append([convert_lines_to_vertices(junction_pairs), semantic["type"]])

    hole_polygons_labeled = [(np.array(polygon[0]), label) for polygon, label in hole_polygons_labeled]
    hole_polygons, hole_labels = zip(*hole_polygons_labeled)
    hole_polygons = np.array(hole_polygons)

    lines_holes = np.unique(lines_holes)
    if len(lines_holes) > 0:
        _, vertices_holes = np.where(np.array(annos["lineJunctionMatrix"])[lines_holes])
        vertices_holes = np.unique(vertices_holes)
    else:
        vertices_holes = []

    # load polygons
    rooms = []
    rooms_clipped = []

    all_materials = []
    all_materials_clipped = []

    floor_verts = []
    floor_faces = []
    floor_ids = []

    min_x = 1e15
    max_x = -1e15
    min_y = 1e15
    max_y = -1e15

    all_walls = []
    all_walls_clipped = []
    all_holes = []
    all_hole_annos = []

    # start = time.time()

    for planeID in range(len(annos["planes"])):
        plane_anno = annos["planes"][planeID]
        lineIDs = np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist()
        junction_pairs = [np.where(np.array(annos["lineJunctionMatrix"][lineID]))[0].tolist() for lineID in lineIDs]
        polygon = convert_lines_to_vertices(junction_pairs)

        wall, walls_clipped, holes = clip_polygon(polygon, vertices_holes, junctions, plane_anno, clip_holes=True)
        all_walls.append(wall)
        all_walls_clipped.append(walls_clipped)
        all_holes.append(holes)
        hole_annos = []

        for hole in holes:
            intersection_mask = np.any(np.isin(hole_polygons, np.array(hole[2])), axis=1)
            hole_anno = hole_labels[np.argmax(intersection_mask, axis=0)] if np.any(intersection_mask) else ""
            hole_annos.append(hole_anno)

        all_hole_annos.append(hole_annos)

    for semantic in annos["semantics"]:
        # room and wall are not needed in converted file, column should be handled seperatly
        if semantic["type"] in ["outwall", "door", "window", "wall", "bb", "room", "column"]:
            continue
        polygons = []
        polygons_semantic = []
        # the rooms are defined via planes
        for planeID in semantic["planeID"]:
            plane_anno = annos["planes"][planeID]

            wall = all_walls[planeID]
            walls_clipped = all_walls_clipped[planeID]
            holes = all_holes[planeID]
            anno_normal = plane_anno["corrected_normal"] if "corrected_normal" in plane_anno else plane_anno["normal"]

            if len(wall[0]) > 0:
                polygons_semantic.append(
                    [
                        walls_clipped[0],
                        walls_clipped[1],
                        planeID,
                        anno_normal,
                        plane_anno["type"],
                        semantic["type"],
                        semantic["ID"],
                    ]
                )

                polygons.append(
                    [
                        wall[0],
                        wall[1],
                        planeID,
                        anno_normal,
                        plane_anno["type"],
                        semantic["type"],
                        semantic["ID"],
                    ]
                )

                hole_annos = all_hole_annos[planeID]
                for hole_idx, hole in enumerate(holes):
                    polygons_semantic.append(
                        [
                            hole[0],
                            hole[1],
                            planeID,
                            anno_normal,
                            hole_annos[hole_idx],
                            semantic["type"],
                            semantic["ID"],
                        ]
                    )
            else:
                print("SKIPPED INVALID POLYGON")

        room_verts = []
        room_faces = []
        room_normals = []

        materials = []

        current_floor_verts = []
        current_floor_faces = []
        current_vertex_count = 0

        for vertices, faces, planeID, normal, plane_type, semantic_type, semantic_id in polygons:
            num_vertices = len(vertices)
            vis_verts = np.array(vertices)
            vis_faces = np.array(faces)
            vis_normal = np.repeat(np.expand_dims(np.array(normal), 0), num_vertices, axis=0)

            if len(vis_faces) == 0:
                continue

            if use_torch:
                room_verts.append(torch.Tensor(vertices))
                room_faces.append(torch.Tensor(faces))
                room_normals.append(torch.Tensor(vis_normal))
            else:
                room_verts.append(np.array(vertices))
                room_faces.append(np.array(faces))
                room_normals.append(np.array(vis_normal))

            materials.append(plane_type)
            min_x = min(min_x, np.min(vis_verts[:, 0]))
            max_x = max(max_x, np.max(vis_verts[:, 0]))
            min_y = min(min_y, np.min(vis_verts[:, 1]))
            max_y = max(max_y, np.max(vis_verts[:, 1]))

            if plane_type == "floor":
                if use_torch:
                    current_floor_verts.append(torch.Tensor(vertices))
                    current_floor_faces.append(torch.Tensor(faces) + current_vertex_count)
                else:
                    current_floor_verts.append(np.array(vertices))
                    current_floor_faces.append(np.array(faces) + current_vertex_count)
                current_vertex_count += len(vertices)

                floor_ids.append(semantic_id)

        # Merge all floor vertices of one room to a single 3d object
        if len(current_floor_verts) > 0:
            if use_torch:
                current_floor_verts = torch.cat(current_floor_verts, axis=0)
                current_floor_faces = torch.cat(current_floor_faces, axis=0)
            else:
                current_floor_verts = np.concatenate(current_floor_verts, axis=0)
                current_floor_faces = np.concatenate(current_floor_faces, axis=0)
            floor_verts.append(current_floor_verts)
            floor_faces.append(current_floor_faces)

        room = {"verts": room_verts, "faces": room_faces, "normals": room_normals}

        rooms.append(room)
        all_materials.append(materials)

        room_verts_clipped = []
        room_faces_clipped = []
        room_normals_clipped = []

        materials_clipped = []

        for vertices, faces, planeID, normal, plane_type, semantic_type, semantic_id in polygons_semantic:
            num_vertices = len(vertices)
            vis_verts = np.array(vertices)
            vis_faces = np.array(faces)
            vis_normal = np.repeat(np.expand_dims(np.array(normal), 0), num_vertices, axis=0)

            if len(vis_faces) == 0:
                continue

            if use_torch:
                room_verts_clipped.append(torch.Tensor(vertices))
                room_faces_clipped.append(torch.Tensor(faces))
                room_normals_clipped.append(torch.Tensor(vis_normal))
            else:
                room_verts_clipped.append(np.array(vertices))
                room_faces_clipped.append(np.array(faces))
                room_normals_clipped.append(np.array(vis_normal))
            materials_clipped.append(plane_type)
        room = {"verts": room_verts_clipped, "faces": room_faces_clipped, "normals": room_normals_clipped}

        rooms_clipped.append(room)
        all_materials_clipped.append(materials_clipped)

    floor_ids = [x for i, x in enumerate(floor_ids) if x not in floor_ids[:i]]
    # print("It took: ", time.time() - start)

    floors = {
        "verts": floor_verts,
        "faces": floor_faces,
        "ids": floor_ids,
    }
    limits = (min_x, max_x, min_y, max_y)
    return rooms, rooms_clipped, floors, limits, all_materials, all_materials_clipped
