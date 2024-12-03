import argparse
import csv
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

from spvloc.utils.projection import pano2persp_from_rotation, pano2persp_get_camera_parameter


def save_color(path, pano_path, cam, pano_R, img_size, filename="rgb_rawlight.png"):
    img = Image.open(pano_path).convert("RGB")
    img_persp = pano2persp_from_rotation(
        np.array(img),
        cam,
        pano_R,
        height=img_size[0],
        width=img_size[1],
    )
    img_persp = Image.fromarray(img_persp)
    img_persp = img_persp.save(os.path.join(path, filename))


def normalize(vector):
    return vector / np.linalg.norm(vector)


def save_camera_pose(path, cam, pano_R, pano_pose, img_size, filename="camera_pose.txt"):
    aspect = img_size[0] / img_size[1]
    cam_fx = np.arctan(1.0 / cam[0, 0])
    cam_fy = np.arctan(1.0 / (cam[1, 1] / aspect))
    intrinsics = np.array([cam_fx, cam_fy, 1])

    rot_in = pano_R

    W = normalize(rot_in[:, 2])
    up = normalize(rot_in[:, 1])

    output = np.concatenate((pano_pose, W, up, intrinsics))
    output = [round(value, 12) for value in output]
    pose_string = " ".join(map(str, output))
    # print(pose_string)
    with open(os.path.join(path, filename), "w") as file:
        file.write(pose_string)


def count_rows(path_to_csv):
    # Count the total number of rows in the CSV file
    with open(path_to_csv, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        total_rows = sum(1 for _ in csv_reader)
    return total_rows


def process_csv(path_to_csv, path_in, path_out):
    # Load CSV file
    anno_filename = "annotation_3d.json"
    zillow_filename = "zind_data.json"
    scene_info_filename = "scene_info.csv"
    column_headers = [
        "path",
        "furniture",
        "fov",
        "yaw",
        "pitch",
        "roll",
        "pitch_range",
        "roll_range",
        "x",
        "y",
        "z",
    ]
    img_size = (320, 320)

    scenes = {}

    total_rows = count_rows(path_to_csv)

    with open(path_to_csv, "r") as csv_file:
        csv_reader = csv.reader(csv_file)

        first_row = True
        # Iterate over each line
        for row in tqdm(csv_reader, total=total_rows):
            if first_row:
                first_row = False
                continue
            # Process each line here

            sample_path, furniture, fov, yaw, pitch, roll, pitch_range, roll_range, x, y, z = row
            sample_path = os.path.normpath(sample_path)
            if os.path.sep == "/":
                sample_path = sample_path.replace("\\", os.path.sep)

            scene = sample_path.split(os.path.sep, 2)[-2]
            sample_path_scene = sample_path.replace(os.path.sep + scene, "")
            if not scene in scenes:
                scenes[scene] = []

            scenes[scene].append(
                {
                    "path": sample_path_scene,
                    "furniture": furniture,
                    "fov": int(fov),
                    "yaw": int(yaw),
                    "pitch": int(pitch),
                    "roll": int(roll),
                    "pitch_range": int(pitch_range),
                    "roll_range": int(roll_range),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                }
            )

            fov_f = float(fov)
            yaw_f = float(yaw)
            pitch_f = float(pitch)
            roll_f = float(roll)
            pano_pose = np.array([float(x), float(y), float(z)])

            # Transform the folder path
            transformed_path = sample_path.replace("perspective", "panorama")

            # Cut after "panorama\" if it exists
            index = transformed_path.find("panorama" + os.path.sep)
            if index != -1:
                transformed_path = transformed_path[: index + len("panorama" + os.path.sep)]

            # Add furniture component and file name
            transformed_path = os.path.join(transformed_path, furniture, "rgb_rawlight.png")
            if not os.path.exists(transformed_path):
                transformed_path = transformed_path.replace(".png", ".jpg")

            img_path = os.path.normpath(path_in) + os.path.normpath(transformed_path)

            pano_R, render_R, cam = pano2persp_get_camera_parameter(fov_f, yaw_f, pitch_f, roll_f)
            persp_pose = np.concatenate([render_R, pano_pose[..., np.newaxis]], axis=1)
            persp_pose = np.concatenate([persp_pose, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

            data_path = os.path.normpath(path_out) + os.path.normpath(sample_path)

            os.makedirs(data_path, exist_ok=True)
            save_camera_pose(
                data_path,
                cam,
                render_R,
                pano_pose,
                img_size,
                filename="camera_pose.txt",
            )

            save_color(
                data_path,
                img_path,
                cam,
                pano_R,
                img_size,
                filename="rgb_rawlight.png",
            )

    zind_partition_name = "zind_partition.json"
    zind_partition_path = os.path.join(os.path.dirname(path_to_csv), zind_partition_name)

    if not os.path.exists(os.path.join(path_out, zind_partition_name)) and os.path.exists(zind_partition_path):
        shutil.copy(
            os.path.join(zind_partition_path),
            os.path.join(path_out, zind_partition_name),
        )

    for scene_name in tqdm(scenes):
        output_data = scenes[scene_name]
        scene_path_out = os.path.join(path_out, scene_name)
        scene_path_in = os.path.join(path_in, scene_name)

        if not os.path.exists(os.path.join(scene_path_out, anno_filename)):
            shutil.copy(
                os.path.join(scene_path_in, anno_filename),
                os.path.join(scene_path_out, anno_filename),
            )

        if not os.path.exists(os.path.join(scene_path_out, zillow_filename)):
            if os.path.exists(os.path.join(scene_path_in, zillow_filename)):
                shutil.copy(
                    os.path.join(scene_path_in, zillow_filename),
                    os.path.join(scene_path_out, zillow_filename),
                )

        csv_file_path = os.path.join(scene_path_out, scene_info_filename)

        if not os.path.exists(csv_file_path):
            with open(csv_file_path, "w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=column_headers)
                writer.writeheader()

                for new_data in output_data:
                    writer.writerow(new_data)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert dataset from CSV format.")
    parser.add_argument("-d", "--path_to_csv", type=str, help="Path to the CSV file")
    parser.add_argument("-i", "--path_in", type=str, help="Input path")
    parser.add_argument("-o", "--path_out", type=str, help="Output path")
    return parser.parse_args()


def main():
    args = parse_arguments()
    path_to_csv = args.path_to_csv
    path_in = args.path_in
    path_out = args.path_out
    print("Input Path:", path_in)
    print("Output Path:", path_out)
    print("CSV Path:", path_to_csv)
    process_csv(path_to_csv, path_in, path_out)


if __name__ == "__main__":
    main()
