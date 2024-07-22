import hashlib
import json
import os
import pickle
import time
import warnings

import numpy as np
import torch
import torchvision
from PIL import Image

# from pytorch3d.structures import Meshes
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils.eval import get_pickle_name
from ..utils.pose_sampling import _sample_test_poses, _sample_test_poses_v2
from ..utils.pose_utils import process_poses_spvloc
from ..utils.projection import projects_onto_floor
from ..utils.pytorch3d.meshes import Meshes
from ..utils.render import render_scene_batched
from ..utils.transformations import parse_camera_info
from .dataset import load_scene_data
from .load import load_scene_annos, prepare_geometry_from_annos
from .split import scenes_split
from .transform import build_transform


def json_hash(json_scene):
    json_str = json.dumps(json_scene)
    m = hashlib.md5()
    m.update(json_str.encode("utf-8"))
    name = str(int(m.hexdigest(), 16))[0:12]
    return name


def check_path(folder, file):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return os.path.exists(file)


def prepare_image(config, image, keep_original=False, pad_height=0):
    transform = build_transform(config, augment=False, pad_height=pad_height)
    img_transf = transform(image)

    if keep_original:
        image = torchvision.transforms.ToTensor()(image)
        image = torchvision.transforms.Resize(config.RENDER.IMG_SIZE, antialias=True)(image)
        return img_transf, image
    else:
        return img_transf


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def scene_from_json(json_scene, tmp_path="./temp", name=""):
    if name == "":
        name = json_hash(json_scene)

    precomputed_scene = os.path.join(tmp_path, name) + ".json"

    if check_path(tmp_path, precomputed_scene):
        start = time.time()
        print("load {}".format(precomputed_scene))
        with open(precomputed_scene, "rb") as file:
            scene = json.load(file)
        print("loading of prepared annotations took {} seconds.".format(time.time() - start))

    else:
        start = time.time()
        (
            geometry,
            geometry_clipped,
            floor,
            limits,
            materials,
            materials_clipped,
        ) = prepare_geometry_from_annos(json_scene, use_torch=False)

        print("preparation of annotations took {} seconds.".format(time.time() - start))

        scene = {
            "geometry": geometry,
            "geometry_clipped": geometry_clipped,
            "floor_planes": floor,
            "limits": limits,
            "materials": materials,
            "materials_clipped": materials_clipped,
        }
        with open(precomputed_scene, "w") as file:
            json.dump(scene, file, cls=NumpyEncoder)

    def to_tensor(list, dtype=torch.float32):
        return [torch.tensor(item, dtype=dtype) for item in list]

    # transform meshes
    geometries = []
    geometries_clipped = []

    for geo in scene["geometry"]:
        geometries.append(
            Meshes(
                to_tensor(geo["verts"]),
                to_tensor(geo["faces"], dtype=torch.int32),
                verts_normals=to_tensor(geo["normals"]),
            )
        )
    for geo in scene["geometry_clipped"]:
        geometries_clipped.append(
            Meshes(
                to_tensor(geo["verts"]),
                to_tensor(geo["faces"], dtype=torch.int32),
                verts_normals=to_tensor(geo["normals"]),
            )
        )
    scene["geometry"] = geometries
    scene["geometry_clipped"] = geometries_clipped
    scene["floor_planes"]["verts"] = to_tensor(scene["floor_planes"]["verts"])
    scene["floor_planes"]["faces"] = to_tensor(scene["floor_planes"]["faces"], dtype=torch.int32)

    return scene, name


class Structured3DPlans_Inference(Dataset):
    def __init__(
        self,
        config,
        split="test",
        visualise=False,
    ):
        scene_ids = scenes_split(split, config.DATASET.NAME, dataset_folder=config.DATASET.PATH)
        self.dataset_path = config.DATASET.PATH
        self.precompute_path = config.DATASET.PREPARED_DATA_PATH
        self.precompute_path_test = os.path.join(config.DATASET.PATH, config.DATASET.PREPARED_DATA_PATH_TEST)
        self.use_prepared_data = config.DATASET.USE_PREPARED_DATA
        self.visualise = visualise
        self.pad_height = int((config.RENDER.IMG_SIZE[0] - config.INPUT.IMG_SIZE[0]) / 2)

        self.scenes = []
        sample_count = 0
        for scene_id in tqdm(scene_ids, desc="Prepare test data"):
            data = load_scene_data(
                [scene_id],
                self.dataset_path,
                self.precompute_path,
                self.use_prepared_data,
                self.visualise,
                config=config,
            )[0]

            rooms = data["rooms"]
            samples = []

            for room_idx, room in enumerate(rooms):
                for persp_idx in range(len(room["perspective"])):
                    # print(scene_id, room_idx, persp_idx)
                    samples.append([scene_id, room_idx, persp_idx])

            sample_count += len(samples)
            if len(samples) > 0:
                self.scenes.append(samples)

        print("No scenes: ", len(self.scenes))
        print("Images: ", sample_count)

        self.transform = build_transform(config, augment=False)
        self.semantic_transform = build_transform(config, is_layout=True)
        self.normal_layout_transform = build_transform(config, is_layout=True)  # , skip_pil=True)

        self.furniture = config.DATASET.TEST_SET_FURNITURE[0]
        self.lighting = config.DATASET.TEST_LIGHTING[0]
        self.config = config
        self.device = torch.device("cpu")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx, room_id=None):
        start_time = time.time()

        pose_sample_step = self.config.TEST.POSE_SAMPLE_STEP
        cache_data = self.config.DATASET.CACHE_TEST_BATCHES

        sample_list = self.scenes[idx]
        scene_id = sample_list[0][0]
        scene = load_scene_data(
            [scene_id],
            self.dataset_path,
            self.precompute_path,
            self.use_prepared_data,
            self.visualise,
            config=self.config,
        )[0]

        rooms = scene["rooms"]
        floor = scene["floor_planes"]

        floor_ids_int = [int(x) if isinstance(x, str) and x.isnumeric() else x for x in scene["floor_plane_ids"]]
        room_id = room_id if (room_id in floor_ids_int) else -1

        if cache_data:
            scene_name = f"temp_{scene_id:05d}"
            pickle_name = get_pickle_name(self.config, prefix=scene_name)

            if room_id > 0:
                pickle_name += "_" + str(int(room_id))

            tmp_path = self.precompute_path_test
            precomputed_data_path = os.path.join(tmp_path, pickle_name)
            if check_path(tmp_path, precomputed_data_path):
                output = pickle.load(open(precomputed_data_path, "rb"))
                images, cams, poses = self.prepare_all_images(sample_list, rooms)
                output["image"] = images
                output["gt_poses"] = poses
                output["cams"] = cams

                return output

        # Sample poses on map.
        sampling_time_start = time.time()
        if self.config.TEST.ADVANCED_POSE_SAMPLING:
            annos = load_scene_annos(self.dataset_path, scene_id)
            sampled_poses_and_rooms = _sample_test_poses_v2(
                scene["limits"],
                self.config.TEST.CAMERA_HEIGHT,
                annos,
                floor,
                step_size=pose_sample_step if pose_sample_step > 0 else self.config.TEST.POSE_SAMPLE_STEP,
                walls_padding_size=200,
            )
        else:
            sampled_poses_and_rooms = _sample_test_poses(
                scene["limits"],
                self.config.TEST.CAMERA_HEIGHT,
                floor,
                step=pose_sample_step if pose_sample_step > 0 else self.config.TEST.POSE_SAMPLE_STEP,
            )

        # Filter by room
        if room_id > 0:
            room_id = floor_ids_int.index(room_id)
            sampled_poses_and_rooms = [(i) for i in sampled_poses_and_rooms if i[1] == room_id]

        sampling_time = time.time() - sampling_time_start

        # Render all pano images.
        render_time_start = time.time()
        (
            sampled_normals,
            sampled_semantics,
            sampled_depth,
            sampled_poses_and_rooms,
            sampled_poses,
        ) = prepare_pano_renderings(self.config, sampled_poses_and_rooms, scene)
        render_time = time.time() - render_time_start

        # Prepare local samples
        combined_poses = []

        for sample_pose in sampled_poses_and_rooms:
            locally_sampled_poses_and_rooms = [sample_pose]
            combined_poses.append(
                torch.Tensor(
                    process_poses_spvloc(np.eye(4), np.array([p for p, _ in locally_sampled_poses_and_rooms]))
                )
            )

        combined_poses = torch.stack(combined_poses)

        images, cams, poses = self.prepare_all_images(sample_list, rooms)

        if self.config.TEST.EVAL_GT_MAP_POINT:
            gt_map_points = self.prepare_gt_map_points(sample_list, rooms, floor, sampled_poses_and_rooms, scene_id)
        else:
            gt_map_points = torch.tensor([0])

        refine_iterations = self.config.POSE_REFINE.MAX_ITERS
        prepare_plot = self.config.TEST.PLOT_OUTPUT or self.config.TEST.SAVE_PLOT_OUTPUT

        if refine_iterations > 0 or prepare_plot:
            scene["annos"] = {}
            encoded_data = pickle.dumps(scene)
            encoding_length = len(encoded_data)
            encoded_data += bytearray(1000000 - encoding_length)
            encoded_data = np.frombuffer(encoded_data, dtype=np.uint8)
            encoded_data_tensor = torch.tensor(encoded_data)
        else:
            encoded_data_tensor = torch.tensor([0])
            encoding_length = torch.tensor([0])

        output = {
            "scene_id": torch.tensor(scene_id),
            "image": images,
            "gt_poses": poses,
            "gt_map_points": gt_map_points,
            "cams": cams,
            "sampled_normals": sampled_normals,
            "sampled_semantics": sampled_semantics,
            "sampled_depth": sampled_depth,
            "sampled_poses": sampled_poses,
            "combined_poses": combined_poses,
            "encoded_data": encoded_data_tensor,
            "encoding_length": encoding_length,
            "sampling_time": torch.tensor(sampling_time),
            "render_time": torch.tensor(render_time),
        }

        print(
            "preparation of scene {} with {} samples took {} seconds. Rendering took {} seconds.".format(
                sample_list[0][0], len(sampled_poses_and_rooms), time.time() - start_time, render_time
            )
        )

        if cache_data:
            # print(f"Dump output to {precomputed_data_path}")
            pickle.dump(output, open(precomputed_data_path, "wb"))

        return output

    def prepare_gt_map_points(self, sample_list, rooms, floor, sampled_poses_and_rooms, scene_id):
        gt_map_points = []
        for sample in sample_list:
            _, room_example, persp_example = sample

            room = rooms[room_example]
            room_sample = room["perspective"][persp_example]

            gt_room_idx = projects_onto_floor(room_sample["pose"][:, -1], floor)
            pose_2d = room_sample["pose"][:2, -1]

            closest_index = -1

            # Use a list comprehension to find the closest index and distance
            matching_distances = [
                (idx, np.linalg.norm(pose_2d_sample[:2] - (pose_2d / 1000.0)))
                for idx, (pose_2d_sample, room_idx) in enumerate(sampled_poses_and_rooms)
                if room_idx == gt_room_idx
            ]

            if matching_distances:
                closest_index, _ = min(matching_distances, key=lambda x: x[1])
            else:
                print(f"No matching room was found for {gt_room_idx} in scene {scene_id}")
                print(room_sample["path"])
                # Just take the closest TODO generate at least one pose in each room
                matching_distances = [
                    (idx, np.linalg.norm(pose_2d_sample[:2] - (pose_2d / 1000.0)))
                    for idx, (pose_2d_sample, room_idx) in enumerate(sampled_poses_and_rooms)
                ]

            closest_index, _ = min(matching_distances, key=lambda x: x[1])
            gt_map_points.append(closest_index)

        gt_map_points = torch.Tensor(gt_map_points)

        return gt_map_points

    def prepare_all_images(self, sample_list, rooms):
        images = []
        poses = []
        cams = []
        for sample in sample_list:
            _, room_example, persp_example = sample

            room = rooms[room_example]
            room_sample = room["perspective"][persp_example]

            # To test with cluster pickle, remove later
            # room_sample["path"] = room_sample["path"].replace("/mnt/", "G:/")
            img_path = room_sample["path"].format(self.furniture, self.lighting)

            sample_path, _ = os.path.split(img_path)
            camera_data = np.loadtxt(os.path.join(sample_path, "camera_pose.txt"))

            img_size = self.config.RENDER.IMG_SIZE
            rot, trans, K = parse_camera_info(camera_data, img_size[0], img_size[1])
            trans = trans[np.newaxis].T
            pose = np.concatenate([rot, trans], axis=1)
            cams.append(torch.tensor(K))
            poses.append(torch.tensor(pose))

            # If a combination does not exist, default to "empty" with "raw"
            if not os.path.isfile(img_path):
                img_path = room_sample["path"].format("empty", "raw")
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(img_path)
                print(e)

            img_transf = prepare_image(self.config, image, keep_original=False, pad_height=self.pad_height)
            images.append(img_transf)

        images = torch.stack(images)
        cams = torch.stack(cams)
        poses = torch.stack(poses)

        return images, cams, poses


def prepare_pano_renderings(config, sampled_poses_and_rooms, scene):
    sampled_semantics = []
    sampled_normals = []
    sampled_depth = []

    prepare_plot = config.TEST.PLOT_OUTPUT or config.TEST.SAVE_PLOT_OUTPUT

    if sampled_poses_and_rooms is not None:
        if prepare_plot or config.MODEL.PANO_ENCODE_DEPTH or config.MODEL.PANO_ENCODE_NORMALS:
            sampled_depth_normals = render_scene_batched(
                config, scene["geometry"], sampled_poses_and_rooms, scene["materials"], mode="depth_normal"
            )
            sampled_normals = sampled_depth_normals[..., :3]
            sampled_depth = sampled_depth_normals[..., 3:]

        sampled_semantics = render_scene_batched(
            config, scene["geometry_clipped"], sampled_poses_and_rooms, scene["materials_clipped"], mode="semantic"
        )
    else:
        warnings.warn("The grid size is set too large for the scene leading to there being no valid test poses.")

    if len(sampled_semantics) > 0:
        sampled_semantics = torch.stack([torch.tensor(q) for q in sampled_semantics], dim=0)

    if len(sampled_normals) > 0:
        sampled_normals = torch.stack([torch.tensor(q) for q in sampled_normals], dim=0)

    if len(sampled_depth) > 0:
        sampled_depth = torch.stack([torch.tensor(q) for q in sampled_depth], dim=0)

    sampled_semantics = sampled_semantics.unsqueeze(1)

    if prepare_plot or config.MODEL.PANO_ENCODE_DEPTH or config.MODEL.PANO_ENCODE_NORMALS:
        sampled_normals = sampled_normals.permute(0, 3, 1, 2)
        sampled_depth = sampled_depth.permute(0, 3, 1, 2)
    else:
        sampled_normals = torch.zeros([1, 1, 1, 1])
        sampled_depth = torch.zeros([1, 1, 1, 1])

    sampled_poses_and_rooms = [(i[0] / 1000.0, i[1]) for i in sampled_poses_and_rooms]  # only needed in stage 1
    sampled_poses = torch.Tensor(np.array([p for p, _ in sampled_poses_and_rooms]))  # only needed in stage 1

    return sampled_normals, sampled_semantics, sampled_depth, sampled_poses_and_rooms, sampled_poses
