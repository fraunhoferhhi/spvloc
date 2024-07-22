import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from descartes.patch import PolygonPatch
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Polygon

semantics_cmap = {
    "living room": "#e6194b",
    "kitchen": "#3cb44b",
    "bedroom": "#ffe119",
    "bathroom": "#0082c8",
    "balcony": "#f58230",
    "corridor": "#911eb4",
    "dining room": "#46f0f0",
    "study": "#f032e6",
    "studio": "#d2f53c",
    "store room": "#fabebe",
    "garden": "#008080",
    "laundry room": "#e6beff",
    "office": "#aa6e28",
    "basement": "#fffac8",
    "garage": "#800000",
    "undefined": "#aaffc3",
    "door": "#de3a00",
    "window": "#00a4de",
    "outwall": "#000000",
    "floor": "#FF0000",
    "wall": "#070000",
    "column": "#030023",
    "staircase": "#F40400",
}

BLUE = "#004873"
GRAY = "#999999"
DARKGRAY = "#646464"
YELLOW = "#ffcc33"
GREEN = "#008e5e"
RED = "#a52019"
BLACK = "#000000"
VIOLET = "#FF00FF"


def plot_coords(ax, ob, color=BLACK, zorder=1, alpha=1, linewidth=1.0):
    x, y = ob.xy
    ax.plot(x, y, color=color, zorder=zorder, alpha=alpha, linewidth=linewidth)


def convert_lines_to_vertices(lines):
    """convert line representation to polygon vertices"""
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
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


def plot_floorplan(
    annos,
    polygons,
    circles=None,
    dpi=200,
    linewidth=0.2,
    crop_floorplan=False,
    zoom=False,
    show_only_samples=False,
):
    """plot floorplan"""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    draw_elems = ["office", "staircase", "corridor", "collumn"]
    additional_elems = ["door", "window"]  # only draw lines
    junctions = np.array([junc["coordinate"][:2] for junc in annos["junctions"]])
    for polygon, poly_type in polygons:
        polygon = Polygon(junctions[np.array(polygon)])
        if poly_type in additional_elems:
            plot_coords(
                ax,
                polygon.exterior,
                alpha=1.0,
                linewidth=3 if zoom else 2,
                color=semantics_cmap[poly_type],
            )
        else:
            plot_coords(ax, polygon.exterior, alpha=0.5, linewidth=linewidth)
        if poly_type == "outwall":
            patch = PolygonPatch(polygon, facecolor=semantics_cmap[poly_type], alpha=0, linewidth=linewidth)
            ax.add_patch(patch)
        else:
            if poly_type in draw_elems:
                patch = PolygonPatch(polygon, facecolor=semantics_cmap[poly_type], alpha=0.5, linewidth=0)
                ax.add_patch(patch)
    roi_points = []
    add_colorbar = False
    if circles is not None:
        for (circle, bb, score), circle_type in circles:
            if score is not None:
                # gradient from red to green
                colors = [(1, 0, 0), (1, 0.5, 0), (1, 1, 0), (0.5, 1, 0), (0, 1, 0)]  # Red to Green
                color_stops = [0, 0.5, 0.70, 0.85, 1]
                # Create the colormap
                cmap = LinearSegmentedColormap.from_list("CustomMap", list(zip(color_stops, colors)))
                color = cmap(score)
                add_colorbar = True
            else:
                color = (0.0, 0.0, 0.0, 0.0)
                score = 0.0
            if bb is not None:
                bb_1 = ((bb[0] / 256.0) - 0.5) * -360.0
                bb_2 = ((bb[2] / 256.0) - 0.5) * -360.0

            pt_2d = (circle[0], circle[1])

            if show_only_samples:
                if circle_type == "sample_pose":
                    circle_patch = plt.Circle(pt_2d, 60.0, color=BLACK)
                    roi_points.append(pt_2d)
            else:
                if circle_type == "init_pose":
                    roi_points.append(pt_2d)
                    circle_patch = plt.Circle(pt_2d, 150.0, color=GREEN)
                    if bb is not None:
                        arc_patch = patches.Arc(pt_2d, 600, 600, 90, bb_2, bb_1, color=color)
                        ax.add_patch(arc_patch)
                elif circle_type == "init_pose_ring":
                    roi_points.append(pt_2d)
                    circle_patch = plt.Circle(pt_2d, 80.0, fc=(0, 0.6, 0.4, 0.6), ec=(0, 0, 0, 0.5), aa=True)
                elif circle_type == "sample_pose_score":
                    circle_patch = plt.Circle(pt_2d, 30.0, color=DARKGRAY)
                    ax.add_patch(plt.Circle(pt_2d, 120.0, color=color))
                    if bb is not None and score > 0.5:
                        arc_patch = patches.Arc(pt_2d, 300, 300, 90, bb_2, bb_1, color=DARKGRAY)
                        ax.add_patch(arc_patch)

                elif circle_type == "sample_pose":
                    circle_patch = plt.Circle(pt_2d, 60.0, color=DARKGRAY)

                    if bb is not None and score > 0.6:
                        arc_patch = patches.Arc(pt_2d, 300, 300, 90, bb_2, bb_1, color=color)
                        ax.add_patch(arc_patch)
                elif circle_type == "refined_pose":
                    roi_points.append(pt_2d)
                    circle_patch = plt.Circle(pt_2d, 150.0, color=YELLOW)
                elif circle_type == "selected_reference":
                    roi_points.append(pt_2d)
                    circle_patch = plt.Circle(pt_2d, 120.0, fc=(0, 0, 0, 0), ec=VIOLET, lw=2, aa=True)
                    # ax.add_patch(plt.Circle(pt_2d, 100.0, fc=(0, 0, 0, 0.0), ec=DARKGRAY, aa=True))
                elif circle_type == "gt_pose":
                    roi_points.append(pt_2d)
                    if bb is not None:
                        circle_patch = patches.Arc(
                            pt_2d, 1900.0, 1900.0, 90, bb_2, bb_1, fc=(0, 0, 0, 0.1), ec=(0, 0, 0, 0.5), aa=True
                        )
                        for i in range(3, 20):
                            ax.add_patch(
                                patches.Arc(
                                    pt_2d,
                                    i * 100.0,
                                    i * 100.0,
                                    90,
                                    bb_2,
                                    bb_1,
                                    fc=(0, 0, 0, 0.1),
                                    ec=(0, 0, 0, 0.5),
                                    aa=True,
                                )
                            )
                    else:
                        circle_patch = plt.Circle(pt_2d, 1000.0, fc=(0, 0, 0, 0.1), ec=(0, 0, 0, 0.5), aa=True)
                        ax.add_patch(plt.Circle(pt_2d, 500.0, fc=(0, 0, 0, 0.1), ec=(0, 0, 0, 0.5), aa=True))

                    ax.add_patch(plt.Circle(pt_2d, 100.0, fc=(0, 0, 0, 0.1), ec=(0, 0, 0, 0.5), aa=True))

                elif circle_type == "alternative_pose":
                    roi_points.append(pt_2d)
                    circle_patch = plt.Circle(pt_2d, 150.0, color=BLUE)
                elif circle_type == "alternative_pose_ring":
                    roi_points.append(pt_2d)
                    circle_patch = plt.Circle(pt_2d, 80.0, fc=(0, 0.28, 0.8, 0.6), ec=(0, 0, 0, 0.5), aa=True)
                elif circle_type == "fail_pose":
                    roi_points.append(pt_2d)
                    circle_patch = plt.Circle(pt_2d, 150.0, color=RED)

            ax.add_patch(circle_patch)

        if zoom:
            roi_points = np.array(roi_points)
            min_x, min_y = np.min(roi_points, axis=0)
            max_x, max_y = np.max(roi_points, axis=0)
            padding = 2000
            padding_x = min(padding, max(0.1 * padding, (2 * padding - (max_x - min_x)) / 2))
            padding_y = min(padding, max(0.1 * padding, (2 * padding - (max_y - min_y)) / 2))
            ax.set_xlim(min_x - padding_x, max_x + padding_x)
            ax.set_ylim(min_y - padding_y, max_y + padding_y)
            ax.set_aspect("equal", adjustable="box")
    if not zoom:
        plt.axis("equal")
    plt.axis("off")
    if add_colorbar:
        plt.colorbar(cm.ScalarMappable(cmap=cmap))
    fig.set_dpi(dpi)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # plt.close(fig)
    plt.close("all")
    if crop_floorplan:
        data = crop_non_white(data)
    return data


def crop_non_white(image):
    # Find the indices of non-white pixels
    non_white_indices = np.any(image != [255, 255, 255], axis=-1)

    # Find the bounding box of non-white pixels
    non_white_rows, non_white_cols = np.where(non_white_indices)
    min_row, max_row = np.min(non_white_rows), np.max(non_white_rows)
    min_col, max_col = np.min(non_white_cols), np.max(non_white_cols)

    # Crop the image
    cropped_image = image[min_row : max_row + 1, min_col : max_col + 1, :]

    return cropped_image


def get_floorplan_polygons(annos):
    """visualize floorplan"""
    # extract the floor in each semantic for floorplan visualization
    planes = []
    outerwall_planes = []
    for semantic in annos["semantics"]:
        # print(semantic["type"])
        if semantic["type"] != "wall":  # or semantic["type"] == "column":
            for planeID in semantic["planeID"]:
                if annos["planes"][planeID]["type"] == "floor":
                    planes.append({"planeID": planeID, "type": semantic["type"]})

        if semantic["type"] == "outwall":
            if isinstance(semantic["planeID"], list):
                outerwall_planes.extend(semantic["planeID"])
            else:
                outerwall_planes = semantic["planeID"]
    outerwall_planes = np.unique(outerwall_planes)
    # extract hole vertices
    lines_holes = []
    for semantic in annos["semantics"]:
        if semantic["type"] in ["window", "door"]:
            for planeID in semantic["planeID"]:
                planes.append({"planeID": planeID, "type": semantic["type"]})
                lines_holes.extend(np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist())
    lines_holes = np.unique(lines_holes)

    # junctions on the floor
    junctions = np.array([junc["coordinate"] for junc in annos["junctions"]])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

    # construct each polygon
    polygons = []
    for plane in planes:
        lineIDs = np.where(np.array(annos["planeLineMatrix"][plane["planeID"]]))[0].tolist()
        junction_pairs = [np.where(np.array(annos["lineJunctionMatrix"][lineID]))[0].tolist() for lineID in lineIDs]
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane["type"]])

    outerwall_floor = []
    for planeID in outerwall_planes:
        lineIDs = np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist()
        lineIDs = np.setdiff1d(lineIDs, lines_holes)
        junction_pairs = [np.where(np.array(annos["lineJunctionMatrix"][lineID]))[0].tolist() for lineID in lineIDs]
        for start, end in junction_pairs:
            if start in junction_floor and end in junction_floor:
                outerwall_floor.append([start, end])

    if len(outerwall_floor) > 0:
        outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
        polygons.append([outerwall_polygon[0], "outwall"])
    return polygons
