# #!/usr/bin/env python3
# """
# Copyright [2021] <Zillow Inc.>
#
# Script to transfer data for the Zillow Indoor Dataset (ZInD) to Structure3D data format

# Example usage:
# python export_structure3d_cli.py -i <input_folder> -o <output_folder>

# Remark: This script has been adapted to transform ZInD to the appropriate S3D representation to train/test SPVLoc.

import copy
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shapely
import trimesh
from trimesh.constants import tol
import transforms3d.axangles as txa

from shapely.geometry import LineString, MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils
from pano_image import PanoImage
from transformations_zind import Transformation2D

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOG = logging.getLogger(__name__)

FORMATING_SCALE = 1000
MAXIMUM_OPENING_WIDTH = 250

class MyExtrusion(trimesh.primitives.Extrusion):
    def _create_mesh(self):
        # extrude the polygon along Z
        mesh = trimesh.creation.extrude_polygon(
            polygon=self.primitive.polygon,
            height=self.primitive.height,
            transform=self.primitive.transform,
            engine="triangle",
        )

        # check volume here in unit tests
        if tol.strict and mesh.volume < 0.0:
            raise ValueError("matrix inverted mesh!")

        # cache mesh geometry in the primitive
        self._cache["vertices"] = mesh.vertices
        self._cache["faces"] = mesh.faces


class ConvertS3D:
    """Class that loads ZInD data formats and transfers to Structure3D data formats"""

    def __init__(
        self,
        json_file_name,
        fuse_rooms=True,
        add_bb2rooms=False,
        add_individual_floors=False,
        plot_extended_rooms=False,
        include_nonflat_ceiling_panos=True,
        stretch_size=1,
    ):
        """Create a floor map polygon object from ZinD JSON file."""
        with open(json_file_name) as json_file:
            self._floor_map_json = json.load(json_file)

        parent_dir_name = os.path.basename(os.path.dirname(json_file_name))
        self._public_guid = int(parent_dir_name)

        self._input_folder = Path(json_file_name).resolve().parent
        self._panos_list = []

        # Global counters to put all local primitives into global IDs
        self.junctions_idx_counter = 0
        self.lines_idx_counter = 0
        self.planes_idx_counter = 0

        self.open_semantics = {}
        self.open_walls = {}

        self.add_bb2rooms = add_bb2rooms
        self.fuse_rooms = fuse_rooms
        self.add_individual_floors = add_individual_floors
        self.plot_extended_rooms = plot_extended_rooms
        self.stretch_size = stretch_size
        self.include_nonflat_ceiling_panos = include_nonflat_ceiling_panos

    def _to_trimesh(self, pano_id_int, vertices_list, ceiling_height, openings, transformation):
        """Prepare a redner scene by setting the geometry and camera
        (with the specified resolution)"""
        num_vertices = len(vertices_list)
        LOG.debug("Number of vertices: {}".format(num_vertices))

        metric_scale = self.global_scale * transformation.scale

        if num_vertices == 0:
            return None, None
        # Create the polygon for the room
        room_2d = Polygon([[p[0], p[1]] for p in vertices_list])
        if openings and self.fuse_rooms:
            # If the current room has some opening, dont process it yet,
            # we'll save the room in the next possible index in
            # self.rooms_w_opening (if there's any yet, we save it with 0)
            if self.rooms_w_opening:
                i = max(self.rooms_w_opening.keys()) + 1
            else:
                i = 0

            self.rooms_w_opening[i] = (room_2d, ceiling_height, pano_id_int, [])

            # Transform the local opening vertices to the global frame of reference
            num_open = len(openings) // 3
            opening_vertices_local = []
            opening_vertices_top_down_bound = []

            for open_idx in range(num_open):
                opening_vertices_local.extend(openings[open_idx * 3 : open_idx * 3 + 2])
                opening_vertices_top_down_bound.extend(openings[open_idx * 3 + 2 : open_idx * 3 + 3])

            opening_vertices_global = (
                metric_scale * self.to_global_s3d(transformation, np.array(opening_vertices_local))
            ).tolist()
            for p1, p2, top_down in zip(
                opening_vertices_global[::2],
                opening_vertices_global[1::2],
                opening_vertices_top_down_bound,
            ):
                opening_line = LineString([p1, p2])
                y_bottom = (top_down[0] + 1) * metric_scale  # camera height
                y_top = (top_down[1] + 1) * metric_scale
                self.opening_lines.append((i, opening_line, (y_bottom, y_top)))
                x, y = opening_line.coords.xy

            # We return empty meshes, so we know we process it later
            return []

        else:
            # If there's no opening we process it normally

            # Revising transform_mat could change the coordinates if required.
            # Current config does not change coordinates.
            transform_mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            room_3d = MyExtrusion(polygon=room_2d, height=ceiling_height, transform=transform_mat)
            # room_3d_obj = trimesh.exchange.obj.export_obj(room_3d)
            return room_3d

    # CUSTOM FUNCTIONS FOR FINDING THE OPENINGS

    def find_opening_pairs(self):
        """Find the paired openings"""
        connected_room_edges = []
        room_bridges = {}

        room_bridge_heights = {}
        paired_openings = []
        merged_bridges = False
        # fig, ax = plt.subplots()

        # for current_idx, current_opening, crrent_height in self.opening_lines:
        #     x, y = current_opening.xy
        #     ax.plot(x, y, label="LineString", marker="o")

        # plt.show()
        # non_paired_openings=self.opening_lines
        for current_idx, current_opening, crrent_height in self.opening_lines:
            # If the current opening has been paired already, skip it
            if current_opening in paired_openings:
                if current_idx not in room_bridges.keys():
                    room_bridges[current_idx] = []
                    room_bridge_heights[current_idx] = []
                continue

            # The possible pairs for teh current opening must be in a different
            # room and facing the oposite direction (around 180 degrees)
            possible_pairs = [
                (idx, opening, height)
                for (idx, opening, height) in self.opening_lines
                if idx != current_idx and self.is_collinear(opening, current_opening)
            ]
            if not possible_pairs:
                connected_room_edges.append((current_idx, current_idx))
                paired_openings.append(current_opening)
            # else, append it to the list of paired openings and add the height aswell
            else:

                def line_dist(l1, l2):
                    s1 = Point(l1.coords[0])
                    e1 = Point(l1.coords[-1])
                    s2 = Point(l2.coords[0])
                    e2 = Point(l2.coords[-1])
                    d1 = (s1.distance(s2) + e1.distance(e2)) / 2
                    d2 = (s1.distance(e2) + e1.distance(s2)) / 2
                    return min(d1, d2)

                # Sort the possible pairs by distance and get the closest
                posible_pair_sorted = sorted(possible_pairs, key=lambda line: current_opening.distance(line[1]))
                pair_idx, pair_opening, pair_height = posible_pair_sorted[0]

                paired_openings.append(current_opening)
                # If the closest opening is to far, this opening is considered isolated
                # if current_opening.distance(pair_opening) > 250:
                # print(line_dist(current_opening, pair_opening))
                if line_dist(current_opening, pair_opening) > MAXIMUM_OPENING_WIDTH:
                    connected_room_edges.append((current_idx, current_idx))
                    continue
                paired_openings.append(pair_opening)
                connected_room_edges.append((current_idx, pair_idx))

                # Check if tehre's any other close enough opening to pair_opening
                # This is implicitly only chckeing if there is any paralel opening that
                # is close enough, both in the parallel and perpendicular
                # directions. In this case, we consider it was one bigger opening
                # that was split in two becasue of teh labelling
                extra_openings = [pair_opening]
                pair_heights = [pair_height]
                opening_distance = 0
                for pair_idx2, pair_opening2, pair_height2 in posible_pair_sorted[1:]:
                    opening_distance = pair_opening.distance(pair_opening2)
                    if opening_distance < 20:
                        # If tehre are two openings too clsoe, we consider them
                        # the same. We also consider it paired and
                        extra_openings.append(pair_opening2)
                        connected_room_edges.append((current_idx, pair_idx2))
                        pair_heights.append(pair_height2)
                    else:
                        break

                # Now calculate the bridges coords and heights, managing the case
                # when the opening was split, merging the subopenings
                x1, y1 = current_opening.coords.xy
                x2, y2 = pair_opening.coords.xy

                if len(extra_openings) == 1:
                    merged_bridges = False
                    bridge_coords = np.array([list(x1) + list(x2), list(y1) + list(y2)]).T
                    bridge_heights = np.array([crrent_height, pair_height])
                else:
                    merged_bridges = True
                    all_x = list(x1) + list(x2)
                    all_y = list(y1) + list(y2)
                    for extra_opening in extra_openings:
                        x3, y3 = extra_opening.coords.xy
                        all_x.extend(x3)
                        all_y.extend(y3)
                    vertices = list(np.array([all_x, all_y]).T)
                    # Create MultiPoint object from all coordinates
                    multi_point = MultiPoint(vertices)

                    # Get the rotated bounding box
                    rotated_bbox = multi_point.minimum_rotated_rectangle
                    # Update the
                    bridge_coords = np.array(rotated_bbox.exterior.coords.xy).T[:-1]
                    bridge_heights = np.array([crrent_height, min(pair_heights)])

                bridge_coords = self.stretch_bridges(bridge_coords, self.stretch_size)
                bridge_coords = self.order_coordinates_clockwise(bridge_coords)

                room_bridge = Polygon(bridge_coords)
                if current_idx in room_bridges.keys():
                    room_bridges[current_idx].append(room_bridge)
                    room_bridge_heights[current_idx].append(bridge_heights)
                else:
                    room_bridges[current_idx] = [room_bridge]
                    room_bridge_heights[current_idx] = [bridge_heights]

        return room_bridges, connected_room_edges, room_bridge_heights, merged_bridges

    def order_coordinates_clockwise(self, coordinates):
        # Calculate the center of the coordinates
        center = np.mean(coordinates, axis=0)

        # Convert coordinates to polar coordinates with respect to the center
        polar_coords = np.arctan2(coordinates[:, 1] - center[1], coordinates[:, 0] - center[0])

        # Sort polar coordinates in ascending order
        sorted_indices = np.argsort(polar_coords)

        # Reorder the coordinates based on the sorted indices
        clockwise_ordered_coords = coordinates[sorted_indices]

        return clockwise_ordered_coords

    def is_collinear_bridge(self, points):
        if len(points) != 4:
            return False  # Requires exactly four points

        # Calculate vectors between the points
        vector1 = points[1] - points[0]
        vector2 = points[2] - points[1]
        vector3 = points[3] - points[2]

        # Check if all cross products of consecutive vectors are nearly zero
        cross_product1 = np.cross(vector1, vector2)
        cross_product2 = np.cross(vector2, vector3)
        eps = 1e-5
        return np.allclose(cross_product1, 0.0, atol=eps) and np.allclose(cross_product2, 0.0, atol=eps)

    def stretch_bridges(self, bridge_coords, strech_size=1):
        bridge_coords = self.order_coordinates_clockwise(bridge_coords)
        center = np.mean(bridge_coords, axis=0)

        # We get the longest side and we calculate the stretch direction as the
        # normal. We could use the short side directly, but sometimes this side
        # is degenerate (the openings are overlapping)
        lengths = [np.sum((bridge_coords[i % 4] - bridge_coords[(i + 1) % 4]) ** 2) for i in range(4)]

        max_length_index = np.argmax(lengths)

        p0 = bridge_coords[(max_length_index) % 4]
        p1 = bridge_coords[(max_length_index + 1) % 4]

        long_side = p1 - p0

        # Perform the stretching sequentially, using the center of teh rectangle
        #  to find out the signed stretch direction
        stretch_dir = np.array([-long_side[1], long_side[0]])
        stretch_dir /= np.linalg.norm(stretch_dir)

        # check if the bridge is colinear
        collinear_bridge = False
        if self.is_collinear_bridge(bridge_coords):
            collinear_bridge = True
            # print(collinear_bridge)

        stretch_dir_signs = [-1, -1, -1, -1]
        stretch_dir_signs[(max_length_index) % 4] = 1
        stretch_dir_signs[(max_length_index + 1) % 4] = 1

        stretched_bridge = bridge_coords.copy()
        for i, point in enumerate(bridge_coords):
            # Vector from center to the current point
            center_to_point = point - center

            if collinear_bridge:
                stretch_dir_sgn = stretch_dir_signs[i]
            else:
                # This check if we have to invert teh stretch dir or not
                stretch_dir_sgn = 1 - 2 * int(np.dot(center_to_point, stretch_dir) < 0)

            # Calculate the new point position
            stretched_bridge[i] += stretch_dir_sgn * (stretch_dir * strech_size)

        return stretched_bridge

    def is_collinear(self, opening, current_opening):
        # Convert LineStrings to vectors
        v1 = np.array(opening.coords[1]) - np.array(opening.coords[0])
        v2 = np.array(current_opening.coords[1]) - np.array(current_opening.coords[0])

        # Get angle between the vectors
        angle = self.angle_between(v1, v2)

        # Check if angle is close to 0 or 180
        return 175 < angle % 180 or angle % 180 < 5

    def angle_between(self, v1, v2):
        # Calculate dot product
        dot_product = np.dot(v1, v2)

        # Calculate magnitudes of the vectors
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        # Calculate cosine of angle
        cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)

        # Calculate the angle in degrees
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

        return angle

    def connected_components(self, edges):
        visited = set()
        components = []

        # Get all unique vertices in the graph
        vertices = set([i for i, j in edges] + [j for i, j in edges])

        for v in vertices:
            if v not in visited:
                component = self.dfs(v, visited, edges)
                components.append(component)

        return components

    def dfs(self, v, visited, edges):
        component = []
        stack = [v]

        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                component.append(vertex)
                # Add unvisited neighbors
                for i, j in edges:
                    if i == vertex and j not in visited:
                        stack.append(j)
                    elif j == vertex and i not in visited:
                        stack.append(i)
        return component

    # def delete_ghost_rooms(self):

    #     room_polygons=[poly for poly,_,_,_ in self.rooms_w_opening.values()]
    #     n=len(room_polygons)
    #     for i in range(n):

    #         union_list = room_polygons[:]
    #         room=union_list[i]
    #         union_list.pop(i)
    #         union=unary_union(union_list)
    #         inter=shapely.intersection(room,union)
    #         if inter.area>0.6*room.area:
    #             print("GHOST")
    #             self.rooms_w_opening.pop(i)

    def delete_room_intersections(self, room_polygons0):
        print("Deleting room intersections")
        # Do a copy in case there are ghost rooms
        room_polygons = room_polygons0[:]
        intersect = False
        n = len(room_polygons)
        for i in range(n):
            # For each room, do the inetrsection with the otehr rooms in teh list
            # (from that index on, to not do duplicates)
            one_room = room_polygons[i]

            # x, y = one_room.exterior.xy

            for j in range(i + 1, n):
                other_room = room_polygons[j]
                # Enlarge the intersection
                inters = shapely.intersection(one_room, other_room.buffer(1e-1)).buffer(1e-1)

                # If the inetrsection is bigger than zero but smaller than a threshold
                # MAke the boolean difference between the rooms and teh inetrsections
                # Also simplify geometry to delete degenerate geom
                if one_room.intersects(other_room.buffer(1e-1)) and inters.area < 100000:
                    intersect = True
                    room_polygons[i] = shapely.difference(one_room, inters).simplify(0.05, preserve_topology=True)
                    room_polygons[j] = shapely.difference(other_room, inters).simplify(0.05, preserve_topology=True)
                    one_room = room_polygons[i]
        return room_polygons, intersect

    def process_opening_rooms(
        self,
        trimesh_scene,
        junctions_list,
        lines_list,
        planes_list,
        junctions_lines_list,
        lines_planes_list,
        semantics_list,
    ):
        print("Calculating connected rooms:")
        # self.delete_ghost_rooms()
        (
            room_bridges,
            connected_room_edges,
            room_bridge_heights,
            merged_bridges,
        ) = self.find_opening_pairs()
        # Note: N.G. connected_room_edges stores which romms belong together
        connected_rooms = self.connected_components(connected_room_edges)
        print(connected_rooms)

        for connected_rooms_ids in connected_rooms:
            if self.plot_extended_rooms:
                fig, ax = plt.subplots()
            # 0) prepraration
            room_polygons = [
                self.rooms_w_opening[room_id][0] for room_id in connected_rooms_ids  # .buffer(-1,join_style=2)
            ]

            ceiling_height = np.max([self.rooms_w_opening[room_id][1] for room_id in connected_rooms_ids])

            pano_ids = [self.rooms_w_opening[room_id][2] for room_id in connected_rooms_ids]
            pano_id_int = np.min(pano_ids)
            # Once we know the pano_id of teh emrged room, if tehre was some
            # merged opening, add it to the info json
            if merged_bridges:
                self.info_json["merged_openings"].append(str(pano_id_int))

            if len(connected_rooms_ids) == 1:
                final_room_2d = room_polygons[0]
                x, y = final_room_2d.exterior.xy
                if self.plot_extended_rooms:
                    ax.plot(x, y, color="blue")
                bridges_polygons = None
            else:
                # get the final room polygon as a union of all the rooms plus
                # teh bridges
                room_polygons, intersects = self.delete_room_intersections(room_polygons)

                new_room_polygons = []

                # Iterate through the room polygons
                for polygon in room_polygons:
                    if isinstance(polygon, MultiPolygon):
                        # If it's a MultiPolygon, add its individual geometries to the new list
                        largest_area_polygon = max(polygon.geoms, key=lambda p: p.area)
                        new_room_polygons.append(largest_area_polygon)
                    else:
                        # If it's a regular Polygon, add it to the new list as is
                        new_room_polygons.append(polygon)
                room_polygons = new_room_polygons

                if intersects:
                    self.info_json["intersecting_walls"].append(str(pano_id_int))
                bridges_polygons = [polygon for room_id in connected_rooms_ids for polygon in room_bridges[room_id]]

                bridges_heights = [
                    heights for room_id in connected_rooms_ids for heights in room_bridge_heights[room_id]
                ]

                if self.plot_extended_rooms:
                    for poly in room_polygons:
                        x, y = poly.exterior.xy
                        ax.plot(x, y, color="blue")
                    for poly in bridges_polygons:
                        x, y = poly.exterior.xy
                        ax.plot(x, y, color="red")
                    plt.show()

                    # for poly in room_polygons:
                    #     x, y = poly.exterior.xy
                    #     ax.plot(x, y, color="blue")
                    # for poly in bridges_polygons:
                    #     x, y = poly.exterior.xy
                    #     ax.plot(x, y, color="red")
                    # plt.show()

                final_room_2d = unary_union(room_polygons + bridges_polygons)
                # Clean final_room_2d
                final_room_2d = Polygon(
                    final_room_2d.exterior,
                    [hole for hole in final_room_2d.interiors if Polygon(hole).area > 100],
                )
                final_room_2d = final_room_2d.simplify(0.05, preserve_topology=True)
                if self.plot_extended_rooms:
                    try:
                        x, y = final_room_2d.exterior.xy

                        ax.plot(x, y, color="green", linewidth=2)

                        for poly in final_room_2d.interiors:
                            x, y = poly.xy
                            ax.plot(x, y, color="green", linewidth=2)
                        plt.show()
                    except Exception:
                        for room in final_room_2d.geoms:
                            x, y = room.exterior.xy

                            ax.plot(x, y, color="green", linewidth=2)

                            plt.show()
                            raise ValueError

            # plt.close('all')
            # 1) _to_trimesh processing
            rooms_3d = []

            transform_mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

            room_3d = MyExtrusion(polygon=final_room_2d, height=ceiling_height, transform=transform_mat)
            rooms_3d.append(room_3d)

            # room_3d.apply_scale(1.0 / 1000)
            # room_3d.show()
            # room_3d_obj = trimesh.exchange.obj.export_obj(rooms_3d[0])

            ceiling_height = np.max([self.rooms_w_opening[room_id][1] for room_id in connected_rooms_ids])

            openings_3d = []
            if bridges_polygons:
                print("Getting top and bottom boundaries")
                for idx, bridge_polygon in enumerate(bridges_polygons):
                    bridge_height = bridges_heights[idx]
                    bottom_boundary = np.min(bridge_height[:, 0])
                    top_boundary = np.min(bridge_height[:, 1])
                    # print(bottom_boundary, top_boundary, ceiling_height)
                    if bottom_boundary > 50.0:  # only if the gap is above 5 cm
                        # print("Bottom", 0)
                        # print("Height ", bottom_boundary)

                        extruded_bridge_bottom = MyExtrusion(
                            polygon=bridge_polygon,
                            height=bottom_boundary,
                            transform=transform_mat,
                        )
                        openings_3d.append(extruded_bridge_bottom)
                    if top_boundary < (ceiling_height - 10.0):
                        transform_mat_ = copy.deepcopy(transform_mat)
                        transform_mat_[2][3] = top_boundary
                        # print("Bottom", top_boundary)
                        # print("Height ", ceiling_height - top_boundary)
                        extruded_top = MyExtrusion(
                            polygon=bridge_polygon,
                            height=ceiling_height - top_boundary,
                            transform=transform_mat_,
                        )
                        openings_3d.append(extruded_top)

            if self.add_individual_floors:
                print("Adding individual floors...")
                for idx, room_2d in enumerate(room_polygons):
                    # Note (N.G.) different ceiling height for different room parts
                    ceiling_height_local = self.rooms_w_opening[connected_rooms_ids[idx]][1]
                    room_3d = MyExtrusion(
                        polygon=room_2d,
                        height=ceiling_height_local,
                        transform=transform_mat,
                    )
                    # room_3d_obj = trimesh.exchange.obj.export_obj(room_3d)
                    rooms_3d.append(room_3d)

            planes_list_idx = []
            for t, room_3d in enumerate(rooms_3d + openings_3d):
                # 2) _arrange_trimesh processing
                trimesh_scene.add_geometry(room_3d)
                facet_edges_dict = defaultdict(list)

                for facet_idx, facet_boundary in enumerate(room_3d.facets_boundary):
                    for facet_edge in facet_boundary:
                        facet_edges_dict[tuple(facet_edge)].append(facet_idx)

                facet_vertices_dict = defaultdict(list)
                facet_edges_list = sorted(facet_edges_dict.items(), key=lambda x: x[0])
                for facet_edge_idx, facet_edge in enumerate(facet_edges_list):
                    vert0, vert1 = facet_edge[0]
                    facet_vertices_dict[vert0].append(facet_edge_idx)
                    facet_vertices_dict[vert1].append(facet_edge_idx)
                facet_vertices_list = sorted(facet_vertices_dict.items(), key=lambda x: x[0])

                # 3) self._export process:

                # Populate the "junctions" primitives
                self._populate_junctions(room_3d, facet_vertices_list, junctions_list)

                # Populate the "lines" primitives
                self._populate_lines(
                    room_3d,
                    facet_edges_list,
                    lines_list,
                    junctions_lines_list,
                    lines_planes_list,
                )

                # Populate the "planes" primitives
                planes_list_idx.extend(
                    self._populate_planes(room_3d, planes_list, floor=t, opening=(t >= len(rooms_3d)))
                )

                self.junctions_idx_counter = len(junctions_list)
                self.lines_idx_counter = len(lines_list)
                self.planes_idx_counter = len(planes_list)

            for pano_id in pano_ids:
                self._transfer_wdo(
                    self.pano_datas[pano_id],
                    junctions_list,
                    planes_list,
                    semantics_list,
                    pano_id_int,
                    lines_list,
                    junctions_lines_list,
                    lines_planes_list,
                )

            # Add walls of the final room to semantics list

            semantic_dict = {}
            semantic_dict["ID"] = pano_id_int
            semantic_dict["type"] = "office"
            semantic_dict["planeID"] = planes_list_idx

            # All planes of the given pano form a room type.

            semantics_list.append(semantic_dict)
            self.junctions_idx_counter = len(junctions_list)
            self.lines_idx_counter = len(lines_list)
            self.planes_idx_counter = len(planes_list)
        return (
            trimesh_scene,
            junctions_list,
            lines_list,
            planes_list,
            junctions_lines_list,
            lines_planes_list,
            semantics_list,
        )

    def to_global_s3d(self, transformation, coordinates):
        """
        Apply transformation on a list of 2D points to transform them from local to global frame of reference.

        :param coordinates: List of 2D coordinates in local frame of reference.

        :return: The transformed list of 2D coordinates.
        """
        coordinates = coordinates.dot(transformation.rotation_matrix)
        coordinates += transformation.translation / transformation.scale
        return (-1, 1) * coordinates

    # Compute a canonical pose
    def _canonical_pose(self, partial_room_data, room_vertices_global_all):
        for pano_id, pano_data in partial_room_data.items():
            if not pano_data["is_primary"]:
                continue
            transformation = Transformation2D.from_zind_data(pano_data["floor_plan_transformation"])
            room_vertices_local = np.asarray(pano_data["layout_raw"]["vertices"])
            # in case if there is collinear in the data
            room_vertices_local = utils.remove_collinear(room_vertices_local)
            num_vertices = len(room_vertices_local)
            if num_vertices == 0:
                continue
            room_vertices_global = self.to_global_s3d(transformation, room_vertices_local)
            room_vertices_global_all.extend(room_vertices_global)
            return room_vertices_global_all

    def _get_camera_center_global(self, pano_data):
        rotation_angle = pano_data["floor_plan_transformation"]["rotation"]
        transformation = Transformation2D.from_zind_data(pano_data["floor_plan_transformation"])
        metric_scale = self.global_scale * transformation.scale

        camera_center_global = self.to_global_s3d(transformation, np.asarray([[0, 0]]))
        camera_center_global = camera_center_global[0].tolist()
        camera_center_global.append(pano_data["camera_height"])
        camera_center_global = np.asarray(camera_center_global) * metric_scale
        # Align the pano texture to the expected Structure3D coordinate system, Z is up and looking at Y
        rot_axis = np.array([0, 1, 0])
        rotation_angle_rad = np.deg2rad(rotation_angle)

        # Replaced to remove dependency from pyquaternion
        # rot = Quaternion(axis=rot_axis, angle=rotation_angle_rad).rotation_matrix
        rot = txa.axangle2mat(rot_axis, rotation_angle_rad)

        pano_image_name = os.path.join(self._input_folder, pano_data["image_path"])
        pano_image = PanoImage.from_file(pano_image_name)

        # transfer the image from original to global position
        pano_image = pano_image.rotate_pano(rot)
        return pano_image_name, pano_image, camera_center_global

    def _arrange_trimesh(self, pano_id_int, pano_data, trimesh_scene):
        transformation = Transformation2D.from_zind_data(pano_data["floor_plan_transformation"])
        room_vertices_local = np.asarray(pano_data["layout_raw"]["vertices"])
        # in case if there is collinear in the data
        room_vertices_local = utils.remove_collinear(room_vertices_local)

        num_vertices = len(room_vertices_local)
        if num_vertices == 0:
            return False

        # # Calculate the centroid of the polygon
        # centroid = np.mean(room_vertices_local, axis=0)

        # # Calculate the angle between each point and the centroid
        # angles = np.arctan2(room_vertices_local[:, 1] - centroid[1],
        #                     room_vertices_local[:, 0] - centroid[0])

        # # Sort the points by the angle
        # sorted_indices = np.argsort(angles)
        # room_vertices_local=room_vertices_local[sorted_indices]

        metric_scale = self.global_scale * transformation.scale

        room_vertices_global = self.to_global_s3d(transformation, room_vertices_local) * metric_scale
        room_vertices_global = room_vertices_global.tolist()

        trimesh_geometry = self._to_trimesh(
            pano_id_int,
            room_vertices_global,
            pano_data["ceiling_height"] * metric_scale,
            openings=pano_data["layout_raw"]["openings"],
            transformation=transformation,
        )

        if trimesh_geometry:
            trimesh_scene.add_geometry(trimesh_geometry)
            facet_edges_dict = defaultdict(list)

            for facet_idx, facet_boundary in enumerate(trimesh_geometry.facets_boundary):
                for facet_edge in facet_boundary:
                    facet_edges_dict[tuple(facet_edge)].append(facet_idx)

            facet_vertices_dict = defaultdict(list)
            facet_edges_list = sorted(facet_edges_dict.items(), key=lambda x: x[0])
            for facet_edge_idx, facet_edge in enumerate(facet_edges_list):
                vert0, vert1 = facet_edge[0]
                facet_vertices_dict[vert0].append(facet_edge_idx)
                facet_vertices_dict[vert1].append(facet_edge_idx)
            facet_vertices_list = sorted(facet_vertices_dict.items(), key=lambda x: x[0])
            return (
                trimesh_scene,
                facet_vertices_list,
                trimesh_geometry,
                facet_edges_list,
            )
        else:
            return trimesh_scene, [], [], []

    # Populate the junctions primitives
    def _populate_junctions(
        self,
        trimesh_geometry,
        facet_vertices_list,
        junctions_list,
    ):
        trimesh_vertices = trimesh_geometry.vertices
        for facet_vertex in facet_vertices_list:
            vertex_idx = facet_vertex[0]
            junction_dict = {}
            junction_dict["ID"] = vertex_idx + self.junctions_idx_counter

            junction_dict["coordinate"] = list(trimesh_vertices[vertex_idx, :])
            junctions_list.append(junction_dict)

    # Populate the lines primitives
    def _populate_lines(
        self,
        trimesh_geometry,
        facet_edges_list,
        lines_list,
        junctions_lines_list,
        lines_planes_list,
    ):
        trimesh_vertices = trimesh_geometry.vertices
        for edge_idx, facet_edge in enumerate(facet_edges_list):
            v0_idx = facet_edge[0][0]
            v1_idx = facet_edge[0][1]
            edge_point = trimesh_vertices[v0_idx, :]
            edge_direction = trimesh_vertices[v1_idx, :] - trimesh_vertices[v0_idx, :]
            line_dict = {}
            edge_idx_global = edge_idx + self.lines_idx_counter
            line_dict["ID"] = edge_idx_global
            line_dict["point"] = list(edge_point)
            line_dict["direction"] = list(edge_direction)
            lines_list.append(line_dict)
            v0_idx_global = v0_idx + self.junctions_idx_counter
            v1_idx_global = v1_idx + self.junctions_idx_counter
            junctions_lines_list.append((v0_idx_global, edge_idx_global))
            junctions_lines_list.append((v1_idx_global, edge_idx_global))
            for plane_idx in facet_edge[1]:
                plane_idx_global = plane_idx + self.planes_idx_counter
                lines_planes_list.append((edge_idx_global, plane_idx_global))

    # Populate the "planes" primitives
    def _populate_planes(self, trimesh_geometry, planes_list, floor=-1, opening=False):
        facets_normal = -trimesh_geometry.facets_normal
        facets_origin = trimesh_geometry.facets_origin
        planes_list_idx = []
        for facet_idx, facet_normal in enumerate(facets_normal):
            facet_normal /= np.linalg.norm(facet_normal) + 1e-16
            facet_origin = facets_origin[facet_idx]
            # Solve for D from the plane equation Ax + By + Cz + D = 0
            facet_offset = -np.dot(facet_origin, facet_normal)
            # Compute the plane type by comparing the normal vector with the gravity vector
            angular_dist_to_gravity = np.dot(facet_normal, [0.0, 0.0, 1.0])
            if abs(angular_dist_to_gravity + 1.0) < 1e-1:
                plane_type = "ceiling"
            elif abs(angular_dist_to_gravity - 1.0) < 1e-1:
                plane_type = "floor"
            elif opening:
                facet_normal = -1 * facet_normal
                plane_type = "wall"
            elif floor > 0:
                plane_type = "false_wall"
            else:
                plane_type = "wall"

            plane_dict = {}
            plane_dict["ID"] = facet_idx + self.planes_idx_counter
            plane_dict["type"] = plane_type
            plane_dict["normal"] = list(facet_normal)
            plane_dict["offset"] = facet_offset
            planes_list.append(plane_dict)
            if plane_type == "false_wall" or (floor == 0 and self.add_individual_floors and plane_type != "wall"):
                continue
            planes_list_idx.append(plane_dict["ID"])

        return planes_list_idx

    def _add_global_bbox(
        self,
        junctions_list,
        lines_list,
        planes_list,
        semantics_list,
        junctions_lines_list,
        lines_planes_list,
    ):
        xy_points = [point["coordinate"][:2] for point in junctions_list]
        min_x, min_y = np.min(xy_points + [2 * [sys.maxsize]], axis=0)
        max_x, max_y = np.max(xy_points + [2 * [0]], axis=0)

        bb_2D = np.array([[max_x, min_y], [max_x, max_y], [min_x, max_y], [min_x, min_y]])

        # We will generate the 3D cuboid that bounds the object, and then we'll
        # extract the relationships of junctions, lines, planes from it.
        # See init_generating_ids to understadn the flow
        z_coords = [point["coordinate"][2] for point in junctions_list]
        h1 = np.array([4 * [min(z_coords)]]).T
        bottom_points = np.concatenate([bb_2D, h1], axis=1)
        h2 = np.array([4 * [max(z_coords)]]).T
        top_points = np.concatenate([bb_2D, h2], axis=1)
        cuboid = np.concatenate([bottom_points, top_points])
        center = np.sum(cuboid, axis=0) / 8
        cuboid = 1.01 * (cuboid - center) + center

        junctions_in_plane_ids = [
            (2, 1, 0),  # D: floor
            (6, 2, 1),  # R: right wall
            (5, 1, 0),  # F: front wall
            (0, 3, 7),  # L: left wall
            (7, 3, 2),  # B: back wall
            (4, 5, 6),
        ]  # U: ceiling
        # We'll need to label them for the semantic segmentation
        planes_type = ["floor", "wall", "wall", "wall", "wall", "ceiling"]
        # Simialarly, each line is generated as the intersection of
        # two planes, or alternatively, as a segment between two junctions.

        # To capture the first relation (line as a segment between 2 junctions),
        # we identify each line with the two junctions thats share its 2 tags
        junctions_in_line_ids = [
            (0, 1),  # FD
            (1, 2),  # RD
            (2, 3),  # BD
            (3, 0),  # LD
            (1, 5),  # RF
            (2, 6),  # RB
            (3, 7),  # LB
            (0, 4),  # LF
            (4, 5),  # FU
            (5, 6),  # RU
            (6, 7),  # BU
            (7, 4),
        ]  # LU
        # About the second relation (lines as intersection of two
        # planes), we're more interested in the inverse relation:
        # planes (faces) as bounded by four lines (edges).
        # Thus, we capture this relation
        # we identify each line with the two junctions thats share its 2 tags
        lines_in_plane_ids = [
            (0, 1, 2, 3),  # D
            (1, 4, 5, 9),  # R
            (0, 4, 7, 8),  # F
            (3, 6, 7, 11),  # L
            (2, 5, 6, 10),  # B
            (8, 9, 10, 11),
        ]  # U

        # JUNCTIONS #

        junctions_idx_0 = self.junctions_idx_counter
        for i in range(8):
            junctions_list.append(
                dict(
                    {
                        "coordinate": cuboid[i].tolist(),
                        "IDs": self.junctions_idx_counter,
                    }
                )
            )
            self.junctions_idx_counter += 1

        # LINES #
        lines_idx_0 = self.lines_idx_counter
        # Calculate the initial index of this batch of 12 lines:
        for i in range(12):
            id0, id1 = junctions_in_line_ids[i]
            # Get the coordinate of teh extremes
            p0, p1 = cuboid[[id0, id1], :]
            # Get the director vector of the line
            dir = p0 - p1
            dir /= np.linalg.norm(dir)
            # Add a the new line to teh list self.lines
            lines_list.append(
                dict(
                    {
                        "point": p0.tolist(),
                        "direction": dir.tolist(),
                        "ID": self.lines_idx_counter,
                    }
                )
            )
            # We fill the lineJunctionMatrix with a  1 in the corresponding id
            for junction_id in (id0, id1):
                junctions_lines_list.append((junctions_idx_0 + junction_id, self.lines_idx_counter))
            self.lines_idx_counter += 1

        # PLANES #

        plane_ids = []

        for i in range(6):
            id0, id1, id2 = junctions_in_plane_ids[i]
            # The normal vectors for walls point outwards, and for bbs
            # point inwards
            n, d = self.calculate_normal(cuboid[[id0, id1, id2], :], invert=(planes_type[i] != "wall"))
            # Note: I had to flip it so that it was looking ok.
            n = n * -1
            planes_list.append(
                dict(
                    {
                        "normal": n.tolist(),
                        "offset": d,
                        "ID": self.planes_idx_counter,
                        # NOTE: For walls and doors we dont
                        # put _ object type??
                        "type": (planes_type[i] + "_bb"),
                    }
                )
            )
            plane_ids.append(self.planes_idx_counter)

            for line_id in lines_in_plane_ids[i]:
                lines_planes_list.append((lines_idx_0 + line_id, self.planes_idx_counter))
            # Update the ID
            self.planes_idx_counter += 1

        # SEMANTICS #

        semantics_list.append(dict({"ID": len(semantics_list), "planeID": plane_ids, "type": "bb"}))
        if self.add_bb2rooms:
            for semantic_dict in semantics_list:
                if semantic_dict["type"] == "office":
                    semantic_dict["planeID"].extend(plane_ids)

        return (
            junctions_list,
            lines_list,
            planes_list,
            semantics_list,
            junctions_lines_list,
            lines_planes_list,
        )

    def calculate_normal(self, points, invert):
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]

        # Calculate the normal vector of the plane using the cross product
        normal = np.cross(v1, v2)

        # normalize vector
        normal = normal / (np.linalg.norm(normal) + 1e-16)
        normal *= 2 * invert - 1
        d = np.dot((-points[0]), normal)

        return normal, d

    def _add_wall_element(
        self,
        wdo_ordered_list,
        wdo_type,
        junctions_list,
        planes_list,
        semantics_list,
        pano_id_int,
        lines_list,
        junctions_lines_list,
        lines_planes_list,
    ):
        """
        Add wall into the lists of junctions, junction lines, plans, lines planes, and semantics.
        These lists are used for final Structure3D json generation
        """

        junctions_idx_base_counter = len(junctions_list)
        for wdo_idx, wdo_junction in enumerate(wdo_ordered_list):
            junction_dict = {}
            junction_dict["ID"] = len(junctions_list)
            junction_dict["coordinate"] = (wdo_junction).tolist()
            junctions_list.append(junction_dict)

        wdo_normal = -np.cross(
            wdo_ordered_list[3] - wdo_ordered_list[0],
            wdo_ordered_list[1] - wdo_ordered_list[0],
        )

        wdo_offset = -np.dot(wdo_normal, wdo_ordered_list[0])

        # Initialize variables to keep track of the best distance
        best_dist = -1

        # Compute the center point of the window
        wdo_center = np.mean(np.array(wdo_ordered_list), axis=0)

        # Iterate through planes
        for plane_wall in planes_list:
            # Skip if not a wall
            if plane_wall["type"] != "wall":
                continue

            # Compute the distance from the center of the window to the plane, along the plane's normal
            plane_norm_factor = np.linalg.norm(plane_wall["normal"])
            curr_dist = abs(np.dot(plane_wall["normal"], wdo_center) + plane_wall["offset"]) / plane_norm_factor

            # Update best distance and corresponding plane
            if best_dist == -1 or curr_dist < best_dist:
                best_dist = curr_dist

                wdo_plane_id = plane_wall["ID"]

        # To avoid floating windows/doors
        if best_dist > 0.1 and wdo_type != "opening":
            self.info_json["floating_window"].append(str(pano_id_int))
            print("FLOATING WINDOW")
            return planes_list

        plane_dict = {}
        plane_dict["ID"] = len(planes_list)
        plane_dict["type"] = wdo_type
        plane_dict["normal"] = (wdo_normal / (np.linalg.norm(wdo_normal) + 1e-16)).tolist()
        plane_dict["offset"] = wdo_offset.tolist()

        # If its an openening, we save it to see if tehre is a floating wdoor/window inside.
        # If its a door or windo, we check that there's no opening too close.

        planes_list.append(plane_dict)

        semantic_dict = {}
        semantic_dict["ID"] = pano_id_int
        semantic_dict["type"] = wdo_type
        semantic_dict["planeID"] = [plane_dict["ID"]]
        semantics_list.append(semantic_dict)

        for idx in range(4):
            idx_next = idx + 1
            if idx_next >= 4:
                idx_next = 0

            line_dict = {}
            line_dict["ID"] = len(lines_list)
            line_dict["point"] = wdo_ordered_list[idx].tolist()
            line_dict["direction"] = (wdo_ordered_list[idx_next] - wdo_ordered_list[idx]).tolist()
            lines_list.append(line_dict)
            v0_idx_global = junctions_idx_base_counter + idx
            v1_idx_global = junctions_idx_base_counter + idx_next
            junctions_lines_list.append((v0_idx_global, line_dict["ID"]))
            junctions_lines_list.append((v1_idx_global, line_dict["ID"]))
            lines_planes_list.append((line_dict["ID"], plane_dict["ID"]))
            # If we were able to assign the window to an existing plane, we assing it
            if wdo_plane_id != -1:
                lines_planes_list.append((line_dict["ID"], wdo_plane_id))

        return planes_list

    # Transform windows / doors / openings
    def _transfer_wdo(
        self,
        pano_data,
        junctions_list,
        planes_list,
        semantics_list,
        pano_id_int,
        lines_list,
        junctions_lines_list,
        lines_planes_list,
    ):
        transformation = Transformation2D.from_zind_data(pano_data["floor_plan_transformation"])
        metric_scale = self.global_scale * transformation.scale
        for wdo_type in ["openings", "windows", "doors"]:
            wdo_vertices_local = np.asarray(pano_data["layout_raw"][wdo_type])

            # Skip if there are no elements of this type
            if len(wdo_vertices_local) == 0:
                continue
            # If we're not fusing rooms, we add the opening as a door:
            if wdo_type == "openings" and (not self.fuse_rooms):
                wdo_type = "doors"
            elif wdo_type == "openings" and self.fuse_rooms:
                continue
            # Transform the local W/D/O vertices to the global frame of reference
            num_wdo = len(wdo_vertices_local) // 3
            wdo_left_right_bound = []
            # save top/down list, note: door down is close to -1 due to camera height
            wdo_top_down_bound = []
            for wdo_idx in range(num_wdo):
                wdo_left_right_bound.extend(wdo_vertices_local[wdo_idx * 3 : wdo_idx * 3 + 2])
                wdo_top_down_bound.extend(wdo_vertices_local[wdo_idx * 3 + 2 : wdo_idx * 3 + 3])
            wdo_vertices_global = self.to_global_s3d(transformation, np.array(wdo_left_right_bound))
            wdo_vertices_global = wdo_vertices_global.tolist()
            for wdo_points in zip(
                wdo_vertices_global[::2],
                wdo_vertices_global[1::2],
                wdo_top_down_bound,
            ):
                top_down = wdo_points[2::3]
                top_down = top_down[0]
                wdo_points_0 = wdo_points[0]
                wdo_points_1 = wdo_points[1]
                wdo_points = [wdo_points_0, wdo_points_1]
                y_bottom = top_down[0] + 1  # camera height
                y_top = top_down[1] + 1

                bottom_left = np.asarray([wdo_points[0][0], wdo_points[0][1], y_bottom])
                top_left = np.asarray([wdo_points[0][0], wdo_points[0][1], y_top])
                bottom_right = np.asarray([wdo_points[1][0], wdo_points[1][1], y_bottom])
                top_right = np.asarray([wdo_points[1][0], wdo_points[1][1], y_top])
                wdo_type_structure3d = wdo_type[:-1]

                wdo_ordered_list = [
                    bottom_left * metric_scale,
                    bottom_right * metric_scale,
                    top_right * metric_scale,
                    top_left * metric_scale,
                ]

                self._add_wall_element(
                    wdo_ordered_list,
                    wdo_type_structure3d,
                    junctions_list,
                    planes_list,
                    semantics_list,
                    pano_id_int,
                    lines_list,
                    junctions_lines_list,
                    lines_planes_list,
                )

    # export to json file
    def _export_json(
        self,
        trimesh_scene,
        junctions_list,
        lines_list,
        planes_list,
        semantics_list,
        lines_planes_list,
        junctions_lines_list,
        output_folder_scene,
    ):
        self.junctions_idx_counter = len(junctions_list)
        self.lines_idx_counter = len(lines_list)
        self.planes_idx_counter = len(planes_list)

        if not trimesh_scene.is_empty:
            structure3d_dict = {}
            structure3d_dict["junctions"] = junctions_list
            structure3d_dict["lines"] = lines_list
            structure3d_dict["planes"] = planes_list
            structure3d_dict["semantics"] = semantics_list
            plane_line_matrix = np.zeros((self.lines_idx_counter, self.planes_idx_counter), dtype=int)
            for line_plane in lines_planes_list:
                line_idx = line_plane[0]
                plane_idx = line_plane[1]
                plane_line_matrix[line_idx][plane_idx] = 1
            structure3d_dict["planeLineMatrix"] = plane_line_matrix.T.tolist()
            line_junction_matrix = np.zeros(
                (self.junctions_idx_counter, self.lines_idx_counter),
                dtype=int,
            )
            for junction_line in junctions_lines_list:
                junction_idx = junction_line[0]
                line_idx = junction_line[1]
                line_junction_matrix[junction_idx][line_idx] = 1
            structure3d_dict["lineJunctionMatrix"] = line_junction_matrix.T.tolist()
            structure3d_dict["cuboids"] = []
            structure3d_dict["manhattan"] = []

            def convert(o):
                if isinstance(o, np.int64) or isinstance(o, np.int32):
                    return int(o)
                raise TypeError

            with open(
                os.path.join(output_folder_scene, "annotation_3d.json"),
                "w",
            ) as outfile:
                json.dump(structure3d_dict, outfile, default=convert)

    def export(self, output_folder: str):
        merger_data = self._floor_map_json["merger"]
        for floor_idx, (floor_id, floor_data) in enumerate(merger_data.items()):
            self.rooms_w_opening = {}
            self.connected_rooms = {}
            self.opening_lines = []
            self.pano_datas = {}
            self.global_scale = self._floor_map_json["scale_meters_per_coordinate"][floor_id]

            if self.global_scale is None:
                continue
            else:
                self.global_scale = self.global_scale * FORMATING_SCALE  # scale from foot to m
                # print(self.global_scale)

            output_folder_scene = os.path.join(
                output_folder,
                "scene_{:04d}{:d}".format(int(self._public_guid), floor_idx),
            )
            os.makedirs(output_folder_scene, exist_ok=True)
            output_folder_2d_rendering = os.path.join(output_folder_scene, "2D_rendering")
            os.makedirs(output_folder_2d_rendering, exist_ok=True)
            # Create a list of all the ZInD polygons for this floor: rooms, windows, doors, openings
            trimesh_scene = trimesh.scene.scene.Scene()

            # Global counters to put all local primitives into global IDs
            self.junctions_idx_counter = 0
            self.lines_idx_counter = 0
            self.planes_idx_counter = 0

            # Keep track of the binary relationship
            junctions_lines_list = []
            lines_planes_list = []
            # Prepare the Structure3D data
            junctions_list = []
            lines_list = []
            planes_list = []
            semantics_list = []
            # room_vertices_global_all = []

            for complete_room_id, complete_room_data in tqdm(floor_data.items()):
                # Initialize zind json
                self.info_json = {
                    "original_image": {},
                    "floating_window": [],
                    "intersecting_walls": [],
                    "merged_openings": [],
                }
                for (
                    partial_room_id,
                    partial_room_data,
                ) in complete_room_data.items():

                    # Inner level data is per-pano (for each floor)
                    for pano_id, pano_data in partial_room_data.items():
                        pano_id_int = int(pano_id.split("_")[-1])

                        if pano_data["image_path"] != "":
                            # Get the integer portion of the pano name, which is "pano_{:03d}"
                            output_folder_room_id = os.path.join(
                                output_folder_2d_rendering,
                                "{:d}".format(pano_id_int),
                            )
                            os.makedirs(output_folder_room_id, exist_ok=True)
                            output_folder_pano = os.path.join(output_folder_room_id, "panorama")
                            os.makedirs(output_folder_pano, exist_ok=True)
                            output_folder_pano_full = os.path.join(output_folder_pano, "full")
                            os.makedirs(output_folder_pano_full, exist_ok=True)

                            # get_camera_center_global
                            (
                                pano_image_name,
                                pano_image,
                                camera_center_global,
                            ) = self._get_camera_center_global(pano_data)
                            # Dont save the images from the panos that have a non
                            # flat ceiling, if include_nonflat_ceiling_panos=True.
                            #

                            # The key is not available, we assume it is not flat
                            is_ceiling_flat = pano_data.get("is_ceiling_flat", False)

                            if is_ceiling_flat or self.include_nonflat_ceiling_panos:
                                output_file_name = os.path.join(output_folder_pano_full, "rgb_rawlight.jpg")
                                pano_image.write_to_file(output_file_name)
                                camera_xyz_file_path = os.path.join(output_folder_pano, "camera_xyz.txt")
                                np.savetxt(camera_xyz_file_path, camera_center_global)

                                self.info_json["original_image"][pano_id] = pano_image_name
                            else:
                                self.info_json["original_image"] = "flat_ceiling"
                        # The primary and secodnary panos
                        if not pano_data["is_primary"]:
                            continue

                        (
                            trimesh_scene,
                            facet_vertices_list,
                            trimesh_geometry,
                            facet_edges_list,
                        ) = self._arrange_trimesh(pano_id_int, pano_data, trimesh_scene)

                        if trimesh_geometry:
                            # Populate the "junctions" primitives
                            self._populate_junctions(
                                trimesh_geometry,
                                facet_vertices_list,
                                junctions_list,
                            )

                            # Populate the "lines" primitives
                            self._populate_lines(
                                trimesh_geometry,
                                facet_edges_list,
                                lines_list,
                                junctions_lines_list,
                                lines_planes_list,
                            )

                            # Populate the "planes" primitives
                            planes_list_idx = self._populate_planes(trimesh_geometry, planes_list)
                            semantic_dict = {}
                            semantic_dict["ID"] = pano_id_int
                            semantic_dict["type"] = "office"
                            semantic_dict["planeID"] = planes_list_idx

                            # All planes of the given pano form a room type.
                            semantics_list.append(semantic_dict)

                            # Transform windows / doors / openings
                            self._transfer_wdo(
                                pano_data,
                                junctions_list,
                                planes_list,
                                semantics_list,
                                pano_id_int,
                                lines_list,
                                junctions_lines_list,
                                lines_planes_list,
                            )
                        else:
                            self.pano_datas[pano_id_int] = pano_data
                        self.junctions_idx_counter = len(junctions_list)
                        self.lines_idx_counter = len(lines_list)
                        self.planes_idx_counter = len(planes_list)

            (
                trimesh_scene,
                junctions_list,
                lines_list,
                planes_list,
                junctions_lines_list,
                lines_planes_list,
                semantics_list,
            ) = self.process_opening_rooms(
                trimesh_scene,
                junctions_list,
                lines_list,
                planes_list,
                junctions_lines_list,
                lines_planes_list,
                semantics_list,
            )

            (
                junctions_list,
                lines_list,
                planes_list,
                semantics_list,
                junctions_lines_list,
                lines_planes_list,
            ) = self._add_global_bbox(
                junctions_list,
                lines_list,
                planes_list,
                semantics_list,
                junctions_lines_list,
                lines_planes_list,
            )

            self.junctions_idx_counter = len(junctions_list)
            self.lines_idx_counter = len(lines_list)
            self.planes_idx_counter = len(planes_list)
            print("\n CONVERSION SUCESSFUL, EXPORTING S3D...")
            self._export_json(
                trimesh_scene,
                junctions_list,
                lines_list,
                planes_list,
                semantics_list,
                lines_planes_list,
                junctions_lines_list,
                output_folder_scene,
            )
            # Save original zind json
            with open(
                os.path.join(output_folder_scene, "zind_data.json"),
                "w",
            ) as outfile:
                json.dump(self._floor_map_json, outfile)
            # Save info json
            with open(
                os.path.join(output_folder_scene, "info.json"),
                "w",
            ) as outfile:
                json.dump(self.info_json, outfile)


def main(
    index,
    zind_folder,
    out_folder,
    fuse_rooms=True,
    plot_extended_rooms=False,
    add_bb2rooms=False,
    add_individual_floors=False,
    stretch_size=1,
):
    args = {
        "input": os.path.join(zind_folder, "{:04d}".format(index), "zind_data.json"),
        "output": out_folder,
    }

    # Collect all the feasible input JSON files
    input = args["input"]
    input_files_list = [input]
    if Path(input).is_dir():
        input_files_list = sorted(Path(input).glob("**/zind_data.json"))
    for input_file in input_files_list:
        zindass3d = ConvertS3D(
            input_file, fuse_rooms, add_bb2rooms, add_individual_floors, plot_extended_rooms, stretch_size=stretch_size
        )
        zindass3d.export(args["output"])


if __name__ == "__main__":
    for index in range(1):
        print("converting {}".format(index))
        main(index)
