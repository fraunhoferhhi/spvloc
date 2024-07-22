import csv
import os
import shutil

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from spvloc.config.defaults import get_cfg_defaults
from spvloc.config.parser import parse_args
from spvloc.data.dataset import load_scene_data
from spvloc.data.split import scenes_split
from spvloc.utils.projection import pano2persp_from_rotation, pano2persp_get_camera_parameter, projects_onto_floor
from spvloc.utils.render import render_scene
from spvloc.utils.render_pyrender import render_scene_pyrender


def save_normal(path, normalmap, filename="normal.png"):
    # Map the normal values from range [-1, 1] to [0, 255]
    normal_mapped = ((normalmap + 1) * 127.5).astype(np.uint8)
    # Create an RGB image from the mapped normal values
    normal_image = Image.fromarray(normal_mapped, "RGB")
    # Save the image as a PNG file
    normal_image.save(os.path.join(path, filename))


def save_semantics(path, semantics, filename="semantic.png"):
    # Define a color mapping for each semantic label
    color_mapping = {
        1: (255, 0, 0),  # Label 1 to Red
        2: (0, 255, 0),  # Label 2 to Green
        3: (0, 0, 255),  # Label 3 to Blue
        4: (255, 255, 0),  # Label 4 to Yellow
        5: (255, 0, 255),  # Label 5 to Magenta
    }

    # Map semantic labels to colors
    height, width = semantics.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)
    for label, color in color_mapping.items():
        colored_image[semantics == label] = color
    # Create an RGB image from the colored data
    colored_image = Image.fromarray(colored_image, "RGB")
    # Save the image as a PNG file
    colored_image.save(os.path.join(path, filename))


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


def import_dict_from_cvs(
    filename_csv_in,
):
    data_list = []  # Initialize an empty list to store the data

    with open(filename_csv_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                # Use the first row as keys for the dictionary
                headers = row
            else:
                # Create a dictionary for each row using the headers
                row_data = {}
                for header, value in zip(headers, row):
                    # Try to convert the value to a number, if not, keep it as a string
                    try:
                        row_data[header] = int(value)
                    except ValueError:
                        try:
                            row_data[header] = float(value)
                        except ValueError:
                            row_data[header] = value

                # Append the dictionary to the list or do whatever you need with it
                # If you want to store each row's dictionary in a list, you can do:
                data_list.append(row_data)
            line_count += 1

    return data_list


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


# Dataloader
class Dataset_Exporter(Dataset):
    def __init__(
        self,
        config,
        output_path,
        sample_fov,
        angle_offset,
        prepared_output_path=None,
        max_render_trys=10,
        num_samples=1,
    ):
        scene_ids = scenes_split("test", config.DATASET.NAME, dataset_folder=config.DATASET.PATH)

        self.dataset_path = config.DATASET.PATH

        self.precompute_path = config.DATASET.PREPARED_DATA_PATH
        self.use_prepared_data = config.DATASET.USE_PREPARED_DATA
        self.dataset_name = config.DATASET.NAME
        self.output_path = output_path
        self.max_render_trys = max_render_trys
        self.num_samples = num_samples
        self.sample_fov = sample_fov
        self.angle_offset = angle_offset

        # add data to existing dataset, create only the elements which do not exist in this path
        self.prepared_output_path = prepared_output_path
        self.extend_base_dataset = self.prepared_output_path is not None

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.visualise = True
        self.scenes = []
        delete_all = False
        # scene_ids = scene_ids[42:]

        for scene_id in tqdm(scene_ids, desc="Preparing"):
            output_path_scene = os.path.join(output_path, f"scene_{scene_id:05d}")
            print(output_path_scene)
            if os.path.exists(output_path_scene):
                if not delete_all:
                    user_input = input(f"Folder '{output_path_scene}' exists. Do you want to delete it? (y/n/(a)ll): ")

                if user_input.lower() == "a":
                    delete_all = True
                if user_input.lower() == "y" or delete_all:
                    # Delete the folder and its contents
                    shutil.rmtree(output_path_scene)
                    print(f"Folder '{output_path_scene}' deleted.")
                else:
                    print("Deletion cancelled.")

            data = load_scene_data(
                [scene_id],
                self.dataset_path,
                self.precompute_path,
                self.use_prepared_data,
                for_visualisation=self.visualise,
                load_perspective=False,
                config=config,
            )[0]
            if data is not None:
                rooms = data["rooms"]
                for room_idx, room in enumerate(rooms):
                    self.scenes.append([scene_id, room_idx, -1])
        print("No samples: ", len(self.scenes))
        self.config = config

    def __len__(self):
        return len(self.scenes)

    def get_valid_yaw(self, geometry, materials, pano_pose, room_idx):
        layout_pano = render_scene(
            self.config,
            geometry,
            pano_pose,
            materials,
            room_idx=room_idx,
            mode="depth",
            img_size_in=(16, 32),
        )

        distances = torch.tensor(layout_pano[7:8, :]).unsqueeze(0)
        distances = torch.nn.functional.interpolate(distances, (360))[0].numpy()
        distances = np.clip(distances, 0.1, 10.0)
        if distances.mean() == 10.0 or distances.mean() == 0.1:
            return np.array([])
        else:
            return np.where(distances > distances.mean())[1]

    def __getitem__(self, idx):
        data_example, room_example, _ = self.scenes[idx]

        data = load_scene_data(
            [data_example],
            self.dataset_path,
            self.precompute_path,
            self.use_prepared_data,
            self.visualise,
            load_perspective=False,
            config=self.config,
        )[0]

        geometry = data["geometry"]
        geometry_clipped = data["geometry_clipped"]
        materials = data["materials"]
        materials_clipped = data["materials_clipped"]
        rooms = data["rooms"]
        floor = data["floor_planes"]

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
        # furniture_configs = ["empty", "full"]
        room = rooms[room_example]

        # replace_path = "/mnt/datasets/ZillowIndoorS3D/"
        # scene_path = room["path"].replace(replace_path, self.output_path)
        room_path = os.path.normpath(room["path"])
        room_path_out = room_path.replace(os.path.normpath(self.dataset_path), os.path.normpath(self.output_path))

        scene_path_in = os.path.normpath(os.path.join(room_path, "..", ".."))
        sample_path = room_path.replace(scene_path_in, "")
        scene_path_out = os.path.normpath(os.path.join(room_path_out, "..", ".."))

        if self.extend_base_dataset:
            scene_path_prep = scene_path_out.replace(
                os.path.normpath(self.output_path), os.path.normpath(self.prepared_output_path)
            )

        if not os.path.exists(scene_path_out):
            os.makedirs(scene_path_out)

        # Copy files only if prepared_output_path is not None, indicating the generation of a fresh dataset.
        if not self.extend_base_dataset:
            if not os.path.exists(os.path.join(scene_path_out, anno_filename)):
                shutil.copy(
                    os.path.join(scene_path_in, anno_filename),
                    os.path.join(scene_path_out, anno_filename),
                )

            if not os.path.exists(os.path.join(scene_path_out, zillow_filename)) and self.dataset_name == "Zillow":
                shutil.copy(
                    os.path.join(scene_path_in, zillow_filename),
                    os.path.join(scene_path_out, zillow_filename),
                )
        else:
            csv_file_path_prep = os.path.join(scene_path_prep, scene_info_filename)
            output_data_prep = import_dict_from_cvs(csv_file_path_prep)

        csv_file_path = os.path.join(scene_path_out, scene_info_filename)

        if not os.path.exists(csv_file_path):
            with open(csv_file_path, "w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=column_headers)
                writer.writeheader()

        if self.dataset_name == "Zillow":
            furniture_configs = ["full"]  # Zillow
        else:
            furniture_configs = ["empty", "full"]

        pano_pose = room["pose"]

        # If a base dataset is available, it is assumed that the yaw has already been chosen,
        # and no further rendering will be performed later.
        if not self.extend_base_dataset:
            room_idx = projects_onto_floor(pano_pose, floor)
            if room_idx < 0:
                print(f"Test sample outside room: {room_path_out}")
                return {}

            valid_yaw = self.get_valid_yaw(geometry, materials, pano_pose, room_idx)

            if len(valid_yaw) == 0:
                print(f"No valid yaw: {room_path_out}")
                return {}
        else:
            valid_yaw = np.array([0])

        output_data = []
        for sample in range(self.num_samples):
            valid_image = False
            yaw = np.random.choice(valid_yaw - 180)  # each sample has a different yaw

            for offset_idx, (pitch_range, roll_range) in enumerate(self.angle_offset):
                pitch = 0 if pitch_range == 0 else np.random.randint(-pitch_range, pitch_range)
                roll = 0 if roll_range == 0 else np.random.randint(-roll_range, roll_range)

                for fov_idx, fov in enumerate(self.sample_fov):
                    if fov == 80:
                        if pitch_range == 5:
                            continue
                        if pitch_range == 15:
                            continue
                        if pitch_range == 20:
                            continue
                    if pitch_range == 5 and fov != 90:
                        continue
                    if pitch_range == 15 and fov != 90:
                        continue

                    for furniture_idx, furniture in enumerate(furniture_configs):
                        # path_out = os.path.join(scene_path, "perspective", "empty", str(sample))
                        # path_out = scene_path
                        # identifier = f"{pano_id}_{sample}_"
                        setup_identifier = f"{fov}_{pitch_range}_{roll_range}"
                        relative_image_path = os.path.join(
                            "perspective",
                            f"{furniture}_{setup_identifier}",
                            str(sample),
                        )

                        if self.extend_base_dataset:
                            precomputed_sample = False
                            relative_image_path_test = os.path.join(sample_path, relative_image_path)
                            for row_data in output_data_prep:
                                if os.path.normpath(row_data["path"]) == os.path.normpath(relative_image_path_test):
                                    precomputed_sample = True
                                    yaw = row_data["yaw"]
                                    pitch = row_data["pitch"]
                                    roll = row_data["roll"]
                                    break
                            if not precomputed_sample:
                                # This sample is invalid and was skipped initially.
                                if offset_idx == 0 and fov_idx == 0 and furniture_idx == 0:
                                    print("Skip: ", room["panorama"].format(furniture, "raw"))
                                    return {}
                                else:
                                    valid_image = True
                            else:
                                output_data.append(row_data)
                                continue

                        if valid_image:
                            path_out = os.path.join(room_path_out, relative_image_path)
                            relative_image_path = os.path.join(sample_path, relative_image_path)
                            if not os.path.exists(path_out):
                                os.makedirs(path_out)

                        pano_R, render_R, cam = pano2persp_get_camera_parameter(fov, yaw, pitch, roll)

                        persp_pose = np.concatenate([render_R, pano_pose[..., np.newaxis]], axis=1)
                        persp_pose = np.concatenate([persp_pose, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

                        # only save meta data for smallest fov and furniture_idx
                        if offset_idx == 0 and fov_idx == 0 and furniture_idx == 0:
                            render_trys = 0
                            while not valid_image and render_trys < self.max_render_trys:
                                normal_persp, depth_persp, semantics_persp = render_scene_pyrender(
                                    geometry,
                                    geometry_clipped,
                                    persp_pose,
                                    materials,
                                    materials_clipped,
                                    self.config.RENDER.IMG_SIZE,
                                    cam,
                                    room_idx=room_idx,
                                )

                                size = semantics_persp.shape[-1] * semantics_persp.shape[-2]
                                histogram = torch.zeros(6, dtype=torch.float32)
                                # Flatten the semantic mask for the current sample
                                flat_semantics = semantics_persp.flatten()
                                for value in range(self.config.MODEL.DECODER_SEMANTIC_CLASSES):
                                    histogram[value] = (flat_semantics == value).sum().item() / size
                                # print(histogram)
                                valid_filter = histogram[0] < 0.05  # filter invalid
                                three_classes_filter = (histogram[1:] > 0.002).sum(axis=-1) > 2
                                if valid_filter and three_classes_filter:
                                    valid_image = True

                                    path_out = os.path.join(room_path_out, relative_image_path)
                                    relative_image_path = os.path.join(sample_path, relative_image_path)
                                    if not os.path.exists(path_out):
                                        os.makedirs(path_out)
                                else:
                                    render_trys = render_trys + 1
                                    print("Try to render again")
                                    # update angles
                                    yaw = np.random.choice(valid_yaw - 180)
                                    pitch = 0 if pitch_range == 0 else np.random.randint(-pitch_range, pitch_range)
                                    roll = 0 if roll_range == 0 else np.random.randint(-roll_range, roll_range)

                                    pano_R, render_R, cam = pano2persp_get_camera_parameter(fov, yaw, pitch, roll)
                                    persp_pose = np.concatenate([render_R, pano_pose[..., np.newaxis]], axis=1)
                                    persp_pose = np.concatenate([persp_pose, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

                            # calculate histogram
                            # check three classes in semantic try 5 variations skip if no valid image is found
                            if valid_image:
                                save_normal(path_out, normal_persp, "normal.png")
                                save_semantics(
                                    path_out,
                                    semantics_persp,
                                    "semantic.png",
                                )
                            else:
                                print("Skip: ", room["panorama"].format(furniture, "raw"))

                        if valid_image:
                            save_color(
                                path_out,
                                room["panorama"].format(furniture, "raw"),
                                cam,
                                pano_R,
                                self.config.RENDER.IMG_SIZE,
                                filename="rgb_rawlight.png",
                            )
                            save_camera_pose(
                                path_out,
                                cam,
                                render_R,
                                pano_pose,
                                self.config.RENDER.IMG_SIZE,
                                filename="camera_pose.txt",
                            )

                            output_data.append(
                                {
                                    "path": relative_image_path,
                                    "furniture": furniture,
                                    "fov": fov,
                                    "yaw": yaw,
                                    "pitch": pitch,
                                    "roll": roll,
                                    "pitch_range": pitch_range,
                                    "roll_range": roll_range,
                                    "x": pano_pose[0],
                                    "y": pano_pose[1],
                                    "z": pano_pose[2],
                                }
                            )

                # exprint(path_out)
                # poses_and_rooms = []
                # poses_and_rooms.append((persp_pose, room_idx))

                # sampled_depth_normals = render_scene_batched(
                #     self.config, geometry, poses_and_rooms, materials, mode="depth_normal"
                # )
                # sampled_normals = sampled_depth_normals[..., :3]
                # sampled_layouts = sampled_depth_normals[..., 3:]

                # sampled_semantics = render_scene_batched(
                #     self.config, geometry_clipped, poses_and_rooms, materials_clipped, mode="semantic"
                # )

                # room_sample["id"] = 0
                # room_sample["cam"] = cam
                # room_sample["pose"] = persp_pose
                # room_sample["euler"] = np.deg2rad(np.array([yaw, pitch, roll]))
                # room_sample["path"] = room["panorama"]
                # room_sample["rotation_pano"] = pano_R

        # test_fov = sample_fov[0]
        # # samples_per_pano = 1
        # max_tests = 10

        # pano_pose = room["pose"]

        # room_idx = projects_onto_floor(pano_pose, floor)

        # yaw = np.random.randint(-180, 180)

        # layout_pano = render_scene(
        #     self.config,
        #     geometry,
        #     pano_pose,
        #     materials,
        #     room_idx=room_idx,
        #     mode="depth",
        #     img_size_in=(16, 32),
        # )

        # distances = torch.tensor(layout_pano[7:8, :]).unsqueeze(0)
        # distances = torch.nn.functional.interpolate(distances, (360))[0].numpy()
        # distances = np.clip(distances, 0.1, 10.0)
        # if distances.mean() == 10.0:
        #     return {}
        # else:
        #     yaw = np.random.choice(np.where(distances > distances.mean())[1] - 180)
        # # yaw = np.random.choice(np.where((distances > 1.2) | (distances > mean_distance))[1] - 180)
        # yaw = np.random.choice(np.where(distances > distances.mean())[1] - 180)
        # pano_R, render_R, cam = pano2persp_get_camera_parameter(test_fov, yaw, 0, 0)

        # persp_pose = np.concatenate([render_R, pano_pose[..., np.newaxis]], axis=1)

        ######
        # room_sample["id"] = 0
        # room_sample["cam"] = cam
        # room_sample["pose"] = persp_pose
        # room_sample["euler"] = np.deg2rad(np.array([yaw, pitch, roll]))
        # room_sample["path"] = room["panorama"]
        # room_sample["rotation_pano"] = pano_R

        with open(csv_file_path, "a", newline="") as csv_file:
            # print(output_data)
            writer = csv.DictWriter(csv_file, fieldnames=column_headers)
            for new_data in output_data:
                writer.writerow(new_data)
        return {}


if __name__ == "__main__":
    args = parse_args()

    config = get_cfg_defaults()
    config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)
    config.freeze()
    pl.seed_everything(config.SEED)

    output_path = config.CONVERSION.INPUT_PATH

    # To expand the dataset, utilize the base dataset from this location and
    # generate additional samples in a separate folder.
    # Note: Ensure that the folder contains the smallest field of view (FOV) with an angle offset of 0.
    # Any new data can be added to this folder at a later time.
    output_path_prep = config.CONVERSION.OUTPUT_PATH

    if output_path == "" or output_path_prep == "":
        raise ValueError("Please provide a valid input and output path.")

    sample_fov = config.CONVERSION.SAMPLE_FOV
    angle_offset = config.CONVERSION.ANGLE_OFFSET

    loader = Dataset_Exporter(
        config,
        output_path,
        sample_fov,
        angle_offset,
        prepared_output_path=output_path_prep,
    )

    for i in tqdm(loader, desc="Preparing"):
        print(i)
        # exit()

    # config = {}
    # config["data_path"] = "G:/Datasets/ZillowIndoor" # already in config

    # config["pitch_range"] = 10  # already in config
    # config["roll_range"] = 10 # already in config
    # config["min_distance"] = 1.5 # already in config

    # config["angles"] = [60, 90, 120]
    # config["check_invalid"] = True
    # config["check_large_elements"] = True
    # config["check_three_classes"] = True
    # config["min_class_percentage"] = 2
    # config["image_only"] = False
    # config["samples_per_pano"] = False
