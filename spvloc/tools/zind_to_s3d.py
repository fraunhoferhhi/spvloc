import argparse
import json
import os
import sys
import time

from zind.s3d_visualization.export_structure3d_cli import main as zind_to_s3d

from ..utils.polygons import generate_3d_model

# (Do not change) In some cases model generation requires an extended stretch distance during room fusion.
EXTENDED_STRETCH = [32, 347, 969]


def create_mesh(
    structured3d_json,
    s3d_filepath=None,
    out_path=None,
    clip_holes=True,
    display_bbox=False,
):
    print("Loading sd3...")
    if s3d_filepath:
        with open(s3d_filepath) as f:
            # Load JSON data from file
            structured3d_json = json.load(f)

    print("Creating mesh...")
    plane_mesh = generate_3d_model(structured3d_json, clip_holes=clip_holes, display_bbox=display_bbox)
    if out_path:
        mesh_path = os.path.join(out_path, "scene_mesh.obj")
        obj_mesh = plane_mesh.export(mesh_path, file_type="obj", include_texture=False)
    else:
        obj_mesh = plane_mesh.export(None, file_type="obj", include_texture=False)
        return obj_mesh


def convert_zind(
    zind_path,
    out_folder,
    index_list,
    mesh_out=False,
    fuse_rooms=True,
    plot_extended_rooms=False,
    display_bbox=True,
    add_bb2rooms=True,
    add_individual_floors=False,
    clip_holes=True,
):
    print(os.path.join(out_folder, "conversion.log"))
    log_file_path = os.path.join(out_folder, "conversion.log")
    log = open(log_file_path, "w")
    print("\n ---------Starting conversion from zind to s3d----------")
    # Write log messages
    log.write("Failed indices: ")

    index_error_list = []
    for index in index_list:
        print("\n\n Converting {}...".format(index))

        try:
            zind_to_s3d(
                index,
                zind_folder=zind_path,
                out_folder=out_folder,
                fuse_rooms=fuse_rooms,
                plot_extended_rooms=plot_extended_rooms,
                add_bb2rooms=add_bb2rooms,
                add_individual_floors=add_individual_floors,
                stretch_size=1 if index not in EXTENDED_STRETCH else 20,
            )
        except Exception as e:
            print(e, "\n FAILED \n")
            index_error_list.append(index)
            log.write("{}, ".format(index))

    log.close()
    if mesh_out:
        print("----------------Starting mesh generation----------------")
        # Filter out only the folder names using list comprehension
        for scene in os.listdir(out_folder):
            try:
                scene_id = int(scene[-5:-1])
            except Exception:
                continue
            if int(scene_id) in index_list and scene_id not in index_error_list:
                print("\n\n Creating mesh of scene {}...".format(scene))
                scene_path = os.path.join(out_folder, scene)
                if os.path.isdir(scene_path):
                    create_mesh(
                        structured3d_json=None,
                        s3d_filepath=os.path.join(scene_path, "annotation_3d.json"),
                        out_path=scene_path,
                        clip_holes=clip_holes,
                        display_bbox=display_bbox,
                    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert S3D files")
    parser.add_argument("--path_in", "-i", type=str, help="Input directory path", required=True)
    parser.add_argument("--path_out", "-o", type=str, help="Output directory path", required=True)
    parser.add_argument("--mesh_out", "-m", action="store_true", help="Boolean flag for mesh output")
    parser.add_argument("--index", "-idx", type=int, default=-1, help="Optional integer argument, default is -1")
    return parser.parse_args()


def main():
    print("Start Conversion")
    args = parse_arguments()

    path_in = args.path_in
    path_out = args.path_out
    mesh_out = args.mesh_out
    index_list = [args.index] if args.index >= 0 else [*range(0, 1575)]

    print("Input Path:", path_in)
    print("Output Path:", path_out)
    print("Mesh Output:", mesh_out)
    print("Index:", [args.index] if args.index >= 0 else "full dataset")
    os.makedirs(os.path.normpath(path_out), exist_ok=True)

    timer = time.time()
    convert_zind(
        path_in,
        path_out,
        index_list=index_list,
        mesh_out=mesh_out,
        fuse_rooms=True,
        plot_extended_rooms=False,
        display_bbox=False,
        add_bb2rooms=True,
        add_individual_floors=True,
        clip_holes=True,
    )
    print(f"it took {time.time() - timer} seconds")


if __name__ == "__main__":
    main()
