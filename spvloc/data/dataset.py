import os
import pickle
import random
import warnings

import numpy as np
from tqdm import tqdm

import PIL
import torch
import torchvision
from PIL import Image

# from pytorch3d.structures import Meshes, join_meshes_as_scene
from torch.utils.data import Dataset

from ..utils.pose_sampling import _sample_near_poses_optimized, sample_far_poses
from ..utils.pose_utils import process_poses_spvloc
from ..utils.projection import (
    pano2persp_from_rotation,
    pano2persp_get_camera_parameter,
    project_pano_to_persp_mask_pano,
    projects_onto_floor,
)
from ..utils.pytorch3d.meshes import Meshes, join_meshes_as_scene
from ..utils.render import render_scene, render_scene_batched
from ..utils.render_pyrender import render_scene_pyrender
from ..utils.transformations import parse_camera_info
from .load import load_scene_annos, prepare_geometry_from_annos
from .split import scenes_split
from .transform import build_transform


def load_scene_data(
    scene_ids,
    root_path,
    precompute_path="",
    use_prepared_data=False,
    for_visualisation=False,
    load_perspective=True,
    config=None,
    filter_invalid=False,
):
    scenes = []
    scene_precompute_path = os.path.join(root_path, precompute_path)
    # scene_precompute_path = precompute_path
    if not os.path.exists(scene_precompute_path):
        os.makedirs(scene_precompute_path)
    for scene_id in scene_ids:
        precomputed_scene = os.path.join(scene_precompute_path, f"temp_{scene_id:05d}")
        reload_scene = True
        if use_prepared_data:
            if os.path.exists(precomputed_scene):
                scene = pickle.load(open(precomputed_scene, "rb"))
                reload_scene = False
        if reload_scene:
            try:
                annos = load_scene_annos(root_path, scene_id)
            except FileNotFoundError:
                print(f"Scene {scene_id} not available")
                return [None]
            (
                scene_geometry,
                scene_geometry_clipped,
                floor_planes,
                limits,
                scene_materials,
                scene_materials_clipped,
            ) = prepare_geometry_from_annos(annos)
            scene_path = os.path.join(root_path, f"scene_{scene_id:05d}", "2D_rendering")

            scene_rooms = []
            for room_id in np.sort(os.listdir(scene_path)):
                room_path = os.path.join(scene_path, room_id)

                if config.DATASET.NAME == "Zillow":
                    panorama_path = os.path.join(room_path, "panorama", "{}", "rgb_{}light.jpg")
                else:
                    panorama_path = os.path.join(room_path, "panorama", "{}", "rgb_{}light.png")

                pano_pose_path = os.path.join(room_path, "panorama", "camera_xyz.txt")
                if os.path.isfile(pano_pose_path):
                    panorama_pose = np.loadtxt(pano_pose_path)
                else:
                    panorama_pose = -1

                floor_mesh = Meshes(floor_planes["verts"], floor_planes["faces"])
                scene_mesh_clipped = []
                for geometry in scene_geometry_clipped:
                    scene_mesh_clipped.append(
                        Meshes(geometry["verts"], geometry["faces"], verts_normals=geometry["normals"])
                    )

                if config.DATASET.PERSP_FROM_PANO and config.DATASET.MIN_AVERAGE_DISTANCE > 0:
                    # Filter small rooms where the average distance to the panoramic camera is smaller then threshold
                    min_average_distance = config.DATASET.MIN_AVERAGE_DISTANCE
                    room_idx = projects_onto_floor(panorama_pose, floor_mesh)
                    layout_pano = render_scene(
                        config,
                        scene_mesh_clipped,
                        panorama_pose,
                        scene_materials,
                        room_idx=room_idx,
                        mode="depth",
                        img_size_in=(16, 32),
                    )

                    distances = torch.tensor(layout_pano[7:8, :]).unsqueeze(0)
                    if distances.mean() < min_average_distance:
                        print(f"Filter tight room: {room_id} {scene_id}")
                        continue

                perspective_samples = []
                if load_perspective:
                    # TODO: Currently we assume that the number of images for furniture types is the same
                    samples_path = os.path.join(
                        room_path, "perspective"
                    )  # empty and full should contain the same number of samples

                    if os.path.exists(samples_path):  # some rooms might have no samples
                        samples_path_subfolders = os.listdir(samples_path)
                        if len(samples_path_subfolders) > 0:
                            samples_path = os.path.join(samples_path, samples_path_subfolders[0])
                            for sample_id in np.sort(os.listdir(samples_path)):
                                sample_path = os.path.join(samples_path, sample_id)

                                perspective_path = os.path.join(
                                    room_path, "perspective", "{}", sample_id, "rgb_{}light.png"
                                )  # empty and full could still be exchanged
                                pose = np.loadtxt(os.path.join(sample_path, "camera_pose.txt"))

                                img_size = config.RENDER.IMG_SIZE
                                rot, trans, K = parse_camera_info(pose, img_size[0], img_size[1])

                                trans = trans[np.newaxis].T
                                pose = np.concatenate([rot, trans], axis=1)

                                if config.DATASET.FILTER_EMPTY_IMAGES:
                                    # prepare filtering
                                    pose_pano = pose[0:3, 3]

                                    room_idx = projects_onto_floor(pose_pano, floor_mesh)
                                    if room_idx < 0:
                                        warnings.warn("pose outside room: {}".format(room_path))
                                        continue
                                    _pose = np.concatenate([np.array(pose), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
                                    semantics_persp = render_scene(
                                        config,
                                        scene_mesh_clipped,
                                        _pose,
                                        scene_materials_clipped,
                                        K,
                                        mode="semantic",
                                        room_idx=room_idx,
                                    )

                                    filter_empty_image = np.min(semantics_persp) == np.max(semantics_persp)
                                else:
                                    filter_empty_image = False

                                if filter_empty_image:
                                    print("filter empty image: {}".format(sample_path))
                                else:
                                    perspective_samples.append(
                                        {"id": sample_id, "path": perspective_path, "pose": pose, "cam": K}
                                    )

                if not load_perspective or len(perspective_samples) > 0:  # ignore rooms without perspective samples
                    scene_rooms.append(
                        {
                            "id": room_id,
                            "path": room_path,
                            "panorama": panorama_path,
                            "pose": panorama_pose,
                            "perspective": perspective_samples,
                        }
                    )

            scene = {
                "id": scene_id,
                "geometry": scene_geometry,
                "geometry_clipped": scene_geometry_clipped,
                "rooms": scene_rooms,
                "floor_planes": floor_planes,
                "limits": limits,
                "materials": scene_materials,
                "materials_clipped": scene_materials_clipped,
                "annos": annos,
            }

            if use_prepared_data:
                pickle.dump(scene, open(precomputed_scene, "wb"))

        scenes.append(scene)

    for scene in scenes:
        geometry_meshes = []
        geometry_meshes_clipped = []
        for geometry in scene["geometry"]:
            room = Meshes(geometry["verts"], geometry["faces"], verts_normals=geometry["normals"])
            if not for_visualisation:
                room = join_meshes_as_scene(room)
            geometry_meshes.append(room)
        for geometry in scene["geometry_clipped"]:
            room = Meshes(geometry["verts"], geometry["faces"], verts_normals=geometry["normals"])
            if not for_visualisation:
                room = join_meshes_as_scene(room)
            geometry_meshes_clipped.append(room)
        scene["geometry"] = geometry_meshes
        scene["geometry_clipped"] = geometry_meshes_clipped
        # TODO: Inconsistent, floor planes do not have to be Meshes here.
        scene["floor_plane_ids"] = scene["floor_planes"]["ids"]
        scene["floor_planes"] = Meshes(scene["floor_planes"]["verts"], scene["floor_planes"]["faces"])
        # TODO: Add flag
        # Post ECCV fix: Filter rooms without valid gt (2024.06.13: this can only be used in certain cases)
        if filter_invalid:
            scene["rooms"] = [
                room for room in scene["rooms"] if projects_onto_floor(room["pose"], scene["floor_planes"]) != -1
            ]

    return scenes


class Structured3DPlans_Perspective(Dataset):
    def __init__(
        self,
        config,
        split="train",
        visualise=False,
    ):
        scene_ids = scenes_split(split, config.DATASET.NAME, dataset_folder=config.DATASET.PATH)
        self.dataset_path = config.DATASET.PATH
        self.precompute_path = config.DATASET.PREPARED_DATA_PATH
        self.use_prepared_data = config.DATASET.USE_PREPARED_DATA
        self.pano_mode = config.DATASET.PERSP_FROM_PANO
        self.dataset_name = config.DATASET.NAME
        self.load_perspective = self.dataset_name != "Zillow"  # Zillow does not have perspective images

        self.visualise = visualise
        self.is_train = split == "train"
        self.is_test = split == "test" or (split == "val" and config.TEST.VAL_AS_TEST)

        self.scenes = []
        # Hack to allow reproducing training behavior if perspective images are not downloaded.
        ignore_persp_images = (
            self.dataset_name == "S3D" and self.pano_mode and config.DATASET.S3D_NO_PERSP_IMAGES and self.is_train
        )

        if ignore_persp_images:
            self.load_perspective = False  # perspective images are not available
            file_path = f"data/datasets/s3d/scenes_{split}.pkl"  # Load prepared scenes from pickle file
            with open(file_path, "rb") as f:
                prepared_scenes = pickle.load(f)

        for scene_id in tqdm(scene_ids, desc=f"Prepare {split} data", unit="scene", mininterval=3.0):
            data = load_scene_data(
                [scene_id],
                self.dataset_path,
                self.precompute_path,
                self.use_prepared_data,
                self.visualise,
                load_perspective=self.load_perspective,
                config=config,
            )[0]
            if data is not None:
                rooms = data["rooms"]
                for room_idx, room in enumerate(rooms):
                    if self.load_perspective:
                        for persp_idx in range(len(room["perspective"])):
                            if (not self.pano_mode) or np.any(room["pose"] != -1):
                                # print(scene_id, room_idx, persp_idx)
                                self.scenes.append([scene_id, room_idx, persp_idx])
                    elif ignore_persp_images:
                        selected_samples = [row for row in prepared_scenes if (row[0], row[1]) == (scene_id, room_idx)]
                        for sample in selected_samples:
                            self.scenes.append(sample)
                    else:
                        self.scenes.append([scene_id, room_idx, -1])

        print("No samples: ", len(self.scenes))

        self.pad_height = int((config.RENDER.IMG_SIZE[0] - config.INPUT.IMG_SIZE[0]) / 2)

        self.transform = build_transform(
            config,
            augment=config.DATASET.AUGMENT_QUERYS,
            pad_height=self.pad_height,
        )

        self.layout_transform = build_transform(config, is_layout=True)
        self.semantic_transform = build_transform(config, is_layout=True)
        self.normal_layout_transform = build_transform(config, is_layout=True)

        self.precomputed = None

        if self.is_train:
            furniture_levels = config.DATASET.TRAIN_FURNITURE
            lighting_levels = config.DATASET.TRAIN_LIGHTING
        else:
            furniture_levels = config.DATASET.TEST_SET_FURNITURE
            lighting_levels = config.DATASET.TEST_LIGHTING

        self.furniture_levels = furniture_levels
        self.lighting_levels = lighting_levels
        self.config = config
        self.device = torch.device("cpu")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx, room_id=None, room_sample_id=None):
        # start_time = time.time()

        # if the room is specified index will represent the scene id
        if room_sample_id or room_id:
            data_example = idx
        else:
            data_example, room_example, persp_example = self.scenes[idx]

        data = load_scene_data(
            [data_example],
            self.dataset_path,
            self.precompute_path,
            self.use_prepared_data,
            self.visualise,
            load_perspective=self.load_perspective,
            config=self.config,
        )[0]

        geometry = data["geometry"]
        geometry_clipped = data["geometry_clipped"]
        materials = data["materials"]
        materials_clipped = data["materials_clipped"]
        rooms = data["rooms"]
        floor = data["floor_planes"]
        limits = data["limits"]

        room_count = len(rooms)  # Stats
        aspect_crop = self.pad_height

        if room_id is None:
            room = rooms[room_example]
            # if PANO mode create room_sample
            if self.pano_mode:
                room_sample = {}

                yaw_range, pitch_range, roll_range = self.config.DATASET.PERSP_FROM_PANO_RANGE
                yaw = np.random.randint(-yaw_range, yaw_range)
                pitch = np.random.randint(-pitch_range, pitch_range)
                roll = np.random.randint(-roll_range, roll_range)
                if self.config.MODEL.LEARN_FOV:
                    fov = np.random.uniform(45, 135)
                else:
                    fov = self.config.DATASET.PERSP_FROM_PANO_FOV

                img_size = self.config.RENDER.IMG_SIZE

                if self.config.MODEL.LEARN_ASPECT and img_size[0] == img_size[1]:
                    x = img_size[0]
                    perspective_crops = np.array(
                        [0, (x - (x / 4 * 3)) / 2, (x - (x / 16 * 10)) / 2, (x - (x / 16 * 9)) / 2]
                    )
                    aspect_crop = int(np.random.choice(perspective_crops))

                pano_pose = room["pose"]

                if self.config.DATASET.PERSP_FROM_PANO_FILTER_VALID_YAW:
                    room_idx = projects_onto_floor(pano_pose, floor)
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

                    valid_yaw = np.where(distances > distances.mean())[1]
                    yaw = 0 if len(valid_yaw) == 0 else np.random.choice(valid_yaw - 180)

                pano_R, render_R, cam = pano2persp_get_camera_parameter(
                    fov, yaw, pitch, roll, self.config.DATASET.PERSP_FROM_PANO_CORRECT_ROLL
                )

                persp_pose = np.concatenate([render_R, pano_pose[..., np.newaxis]], axis=1)

                room_sample["id"] = 0
                room_sample["cam"] = cam
                room_sample["pose"] = persp_pose
                room_sample["euler"] = np.deg2rad(np.array([yaw, pitch, roll]))
                room_sample["path"] = room["panorama"]
                room_sample["rotation_pano"] = pano_R

            else:
                room_sample = room["perspective"][persp_example]

        else:
            room = rooms[room_id % room_count]
            if room_sample_id is None:
                room_sample = random.choice(room["perspective"])
            else:
                room_sample = room["perspective"][room_sample_id % len(room["perspective"])]

        room_sample["aspect_crop"] = aspect_crop

        sample_count = len(room["perspective"])  # Stats

        combined_poses = torch.tensor([])

        cam = room_sample["cam"]
        cam_fov = np.arctan(1.0 / cam[0, 0]) * 2.0  # * (180.0 / np.pi)
        pose = room_sample["pose"]

        cam_location = pose[0:3, 3]
        near_samples = self.config.TRAIN.NUM_NEAR_SAMPLES
        far_samples = self.config.TRAIN.NUM_FAR_SAMPLES

        offset = np.asarray(self.config.TRAIN.PANO_POSE_OFFSETS)
        sampled_poses_and_rooms = _sample_near_poses_optimized(cam_location, floor, offset, near_samples)
        if far_samples > 0:
            sampled_poses_and_rooms += sample_far_poses(
                cam_location,
                floor,
                limits,
                num_samples=far_samples,
                far_min_dist=offset[0] * 1000,
            )

        sampled_layouts = sampled_semantics = sampled_normals = []
        sampled_semantics = None
        sampled_normals = None

        if sampled_poses_and_rooms is not None:
            sampled_depth_normals = render_scene_batched(
                self.config, geometry, sampled_poses_and_rooms, materials, mode="depth_normal"
            )
            sampled_normals = sampled_depth_normals[..., :3]
            sampled_layouts = sampled_depth_normals[..., 3:]
            sampled_semantics = render_scene_batched(
                self.config, geometry_clipped, sampled_poses_and_rooms, materials_clipped, mode="semantic"
            )

        else:
            warnings.warn("The grid size is set too large for the scene leading to there being no valid test poses.")

        sampled_poses_tensor = torch.cat([torch.tensor(i[0]).unsqueeze(0) for i in sampled_poses_and_rooms], dim=0)
        sampled_rooms_tensor = torch.cat([torch.tensor(i[1]).unsqueeze(0) for i in sampled_poses_and_rooms], dim=0)

        pano_layout = pano_normal = pano_semantics = torch.tensor([])

        room_data = self._process_room_sampled_layouts(
            room["id"],
            room_sample,
            sampled_layouts,
            geometry,
            geometry_clipped,
            materials,
            materials_clipped,
            cam,
            floor,
            sampled_poses_tensor,
            sampled_rooms_tensor,
        )

        if far_samples > 0:
            sampled_poses_and_rooms = sampled_poses_and_rooms[:-far_samples]

        sampled_poses_and_rooms = [(i[0] / 1000.0, i[1]) for i in sampled_poses_and_rooms]
        image = room_data["image"]
        image_orig = room_data["image_orig"]
        persp_pose = room_data["pose"]
        pano_pose = room_data["pose_pano"]
        room_idx = [room_data["room_idx"]]
        boundingbox_2d = room_data["boundingbox_2d"]
        bounding_mask = room_data["bounding_mask"]
        persp_layouts = self.layout_transform(room_data["layout"])
        persp_semantics = self.semantic_transform(room_data["semantics"].astype(np.float32))
        persp_normal = self._apply_normal_layout_transform(room_data["normal"])

        if len(sampled_semantics) > 0:
            sampled_semantics = torch.stack([torch.tensor(q) for q in sampled_semantics], dim=0)
            sampled_semantics = sampled_semantics.unsqueeze(1)

        if len(sampled_layouts) > 0:
            sampled_layouts = torch.stack([torch.tensor(q) for q in sampled_layouts], dim=0)
            sampled_layouts = sampled_layouts.permute(0, 3, 1, 2)

        if len(sampled_normals) > 0:
            sampled_normals = torch.stack([torch.tensor(q) for q in sampled_normals], dim=0)
            sampled_normals = sampled_normals.permute(0, 3, 1, 2)
            # else:

        combined_poses = torch.Tensor(
            process_poses_spvloc(persp_pose, np.array([p for p, _ in sampled_poses_and_rooms]))
        )

        # TODO: Experimental, estimate euler angles directly
        if self.config.TRAIN.PERSP_FROM_PANO_RANGE_OPTIMIZE_EULER:
            combined_poses[-1, 3:] = torch.tensor(room_sample["euler"])

        persp_semantics = np.stack(persp_semantics)

        persp_pose = torch.Tensor(persp_pose)
        boundingbox_2d = torch.Tensor(boundingbox_2d)
        bounding_mask = torch.Tensor(bounding_mask)

        sampled_poses = torch.Tensor(np.array([p for p, _ in sampled_poses_and_rooms]))
        sampled_room_idxs = torch.Tensor(np.array([r for _, r in sampled_poses_and_rooms]))

        pano_pose = torch.Tensor(pano_pose)  # n.a. in test

        if not self.is_test:
            geometry = torch.Tensor()
            floor = torch.Tensor()

        room_idx = torch.tensor(np.array(room_idx))
        persp_semantics = torch.tensor(persp_semantics)

        scene_idx = torch.tensor(idx)

        # Create Reference Histogram
        size = persp_semantics.shape[-1] * persp_semantics.shape[-2]
        histogram = torch.zeros(6, dtype=torch.float32)
        # Flatten the semantic mask for the current sample
        flat_semantics = persp_semantics.flatten()
        for value in range(self.config.MODEL.DECODER_SEMANTIC_CLASSES):
            histogram[value] = (flat_semantics == value).sum().item() / size

        aspect_visibility_mask = torch.ones_like(image[0:1, :, :])
        if aspect_crop > 0:
            # Set the top and bottom n lines to 0 (not visible)
            aspect_visibility_mask[0, :aspect_crop, :] = 0
            aspect_visibility_mask[0, -aspect_crop:, :] = 0

        # encoded_data = pickle.dumps(data)
        # encoding_length = len(encoded_data)
        # encoded_data += bytearray(1000000 - encoding_length)
        # encoded_data = np.frombuffer(encoded_data, dtype=np.uint8)
        # encoded_data_tensor = torch.tensor(encoded_data)
        # elapsed_time = time.time() - start_time
        # print(f"Data loading took {elapsed_time:.6f} seconds to run.")
        return {
            "image": image,
            "image_orig": image_orig,
            "scene_idx": scene_idx,
            "room_idx": room_idx,
            "pano_layout": pano_layout,
            "pano_normal": pano_normal,
            "pano_semantics": pano_semantics,
            "pano_pose": pano_pose,
            "persp_layout": persp_layouts,
            "persp_normal": persp_normal,
            "persp_semantics": persp_semantics,
            "persp_pose": persp_pose,
            "persp_fov": cam_fov,
            "sampled_normals": sampled_normals,
            "sampled_depth": sampled_layouts,
            "sampled_semantics": sampled_semantics,
            "sampled_poses": sampled_poses,
            "sampled_room_idxs": sampled_room_idxs,
            "boundingbox_2d": boundingbox_2d,
            "bounding_mask": bounding_mask,
            "geometry": geometry,
            "floor": floor,
            "combined_poses": combined_poses,
            "reference_histogram": histogram,
            "stats": (room_count, sample_count),
            "aspect_visibility_mask": aspect_visibility_mask,
            # "encoded_data": encoded_data_tensor,
            # "encoding_length": encoding_length,
        }

    def _get_scene(self, idx):
        return load_scene_data(
            [idx],
            self.dataset_path,
            self.precompute_path,
            self.use_prepared_data,
            self.visualise,
            config=self.config,
        )[0]

    def check_load_image(self, furniture_levels, lightning_levels, path):
        for furniture in furniture_levels:
            for lightning in lightning_levels:
                img_path = path.format(furniture, lightning)
                try:
                    img = Image.open(img_path).convert("RGB")
                    return img
                except (PIL.UnidentifiedImageError, Exception):
                    continue
        return False

    def _load_image(self, room_sample):
        furniture = random.choice(self.furniture_levels)
        lighting = random.choice(self.lighting_levels)
        img_path = room_sample["path"].format(furniture, lighting)

        # If a combination does not exist, default to "empty" with "raw"
        if not os.path.isfile(img_path):
            furniture = "empty"
            lighting = "raw"

            img_path = room_sample["path"].format(furniture, lighting)

        try:
            img = Image.open(img_path).convert("RGB")
        except (PIL.UnidentifiedImageError, Exception):
            if self.pano_mode:
                img = self.check_load_image(self.furniture_levels, self.lighting_levels, room_sample["path"])
                if not img:
                    img = self.check_load_image(
                        ["empty", "simple", "full"], ["raw", "warm", "cold"], room_sample["path"]
                    )
                    if not img:
                        raise Exception("No valid image path found.")
            else:
                raise Exception(f"{img_path} does not exist.")

        if self.pano_mode:
            img_size = self.config.RENDER.IMG_SIZE
            img = pano2persp_from_rotation(
                np.array(img),
                room_sample["cam"],
                room_sample["rotation_pano"],
                height=img_size[0],
                width=img_size[1],
            )
            crop = room_sample["aspect_crop"]
            if crop > 0:
                img[:crop, :, :] = 0  # Set the top n lines to black
                img[-crop:, :, :] = 0  # Set the bottom n lines to black
            img = Image.fromarray(img)

        img_transf = self.transform(img)

        # Noise could be added to the image if this is commented in
        # img_transf = img_transf + (torch.randn_like(img_transf) * (torch.rand(1) * 0.05))

        if self.is_test:
            img = torchvision.transforms.ToTensor()(img)
            if not self.pano_mode:
                img = torchvision.transforms.Resize(
                    tuple([4 * x for x in self.config.INPUT.IMG_SIZE]), antialias=None
                )(img)
        else:
            img = torch.tensor([])

        return img_transf, img

    def _process_room_sampled_layouts(
        self,
        room_id,
        room_sample,
        sampled_layouts,
        geometry,
        geometry_clipped,
        materials,
        materials_clipped,
        cam,
        floor,
        sampled_poses,
        samples_rooms,
    ):
        # start_time = time.time()
        (
            pose_pano,
            pose,
            layout_persp,
            normal_persp,
            semantics_persp,
            room_idx,
        ) = self._get_room_data(room_sample, geometry, geometry_clipped, materials, materials_clipped, cam, floor)

        pano_poses = (sampled_poses / 1000.0).numpy()

        bb_2d, mask = project_pano_to_persp_mask_pano(
            layout_persp,
            sampled_layouts.squeeze(-1),
            pano_poses,
            pose,
            cam,
            scale_translation=False,
            projection_margin=room_sample["aspect_crop"],
        )

        # Filter all bounding boxes, where the camera is in a different room
        bb_filter = room_idx != samples_rooms
        bb_2d[bb_filter] = 0
        mask[bb_filter] = False

        img_transf, img = self._load_image(room_sample)

        return {
            "image": img_transf,
            "image_orig": img,
            "pose": pose,
            "pose_pano": pose_pano,
            "layout": layout_persp,
            "normal": normal_persp,
            "semantics": semantics_persp,
            "boundingbox_2d": bb_2d,
            "bounding_mask": mask,
            "room_idx": room_idx,
        }

    def _get_room_data(self, room_sample, geometry, geometry_clipped, materials, materials_clipped, cam, floor):
        pose = np.concatenate([np.array(room_sample["pose"]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
        pose_pano = room_sample["pose"][0:3, 3]

        room_idx = projects_onto_floor(pose_pano, floor)
        if room_idx < 0:
            # warnings.warn("pose outside room: {}".format(room_sample))
            # TODO: Do not set this index to 0 but keep -1 and filter later
            # With ECCV fix this should not be triggered anyway.
            room_idx = 0

        # Render with pyrender (outputs are nearly the same as with redner, but with less noise)
        normal_persp, layout_persp, semantics_persp = render_scene_pyrender(
            geometry,
            geometry_clipped,
            pose,
            materials,
            materials_clipped,
            self.config.RENDER.IMG_SIZE,
            cam,
            room_idx=room_idx,
        )

        pose[0:3, 3] = pose[0:3, 3] / 1000.0
        pose_pano = pose_pano / 1000.0
        return (
            pose_pano,
            pose,
            layout_persp,
            normal_persp,
            semantics_persp,
            room_idx,
        )

    def _apply_normal_layout_transform(self, x):
        y = []

        # This has to be changed. at the moment it only works if the input is not a tensor,
        # otherwise the normals will be wrong.
        for i in range(x.shape[-1]):
            y.append(self.normal_layout_transform(x[:, :, i]))

        return torch.cat(y)
