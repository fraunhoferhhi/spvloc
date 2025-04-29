import numpy as np
import cv2
import torch
import glob
import os
import time
import matplotlib

matplotlib.use("Agg")  # Use a non-gui backend for matplotlib, as images are rasterized before display

from torch.utils.data import Dataset
from spvloc.data.split import scenes_split
from spvloc.config.defaults import get_cfg_defaults
from spvloc.config.parser import parse_args
from spvloc.utils.projection import pano2persp_from_rotation, projects_onto_floor, pano2persp_get_camera_parameter
from spvloc.data.dataset import load_scene_data
from spvloc.data.data_inference import prepare_pano_renderings
from spvloc.utils.render import render_scene, render_scene_batched
from spvloc.data.transform import build_transform
from spvloc.utils.pose_utils import get_pose_distances_batch, process_poses_spvloc
from spvloc.utils.plot_floorplan import get_floorplan_polygons, plot_floorplan
from spvloc.utils.pose_sampling import _sample_test_poses_v2

from spvloc.model.spvloc import PerspectiveImageFromLayout

import imgui
import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer

from PIL import Image
from tqdm import tqdm


def load_test_checkpoint(checkpoint_path, model):
    if not checkpoint_path.endswith(".ckpt"):  # If it's a directory
        # Use the code to find and load the newest checkpoint within the directory
        checkpoint_files = glob.glob(os.path.join(checkpoint_path, "*.ckpt"))
        sorted_checkpoints = sorted(checkpoint_files)
        checkpoint_path = sorted_checkpoints[-1]
    print("Load test checkpoint ", checkpoint_path)
    load = torch.load(checkpoint_path)
    # Change strict to false if something has changed about the model.
    model.load_state_dict(load["state_dict"], strict=True)
    model.eval()


# Dataloader
class Dataset_Inference(Dataset):
    def __init__(
        self,
        config,
    ):
        scene_ids = scenes_split(
            "test", config.DATASET.NAME, dataset_folder=config.DATASET.PATH, split_filename="zind_partition.json"
        )

        self.dataset_path = config.DATASET.PATH
        self.precompute_path = config.DATASET.PREPARED_DATA_PATH
        self.use_prepared_data = config.DATASET.USE_PREPARED_DATA
        self.dataset_name = config.DATASET.NAME

        self.visualise = True
        self.scenes = []
        for scene_id in tqdm(scene_ids, desc="Preparing"):

            data = load_scene_data(
                [scene_id],
                self.dataset_path,
                self.precompute_path,
                self.use_prepared_data,
                for_visualisation=self.visualise,
                load_perspective=False,
                config=config,
                filter_invalid=True,
            )[0]
            if data is not None:
                rooms = data["rooms"]
                for room_idx, room in enumerate(rooms):
                    self.scenes.append([scene_id, room_idx, -1])
        print("No samples: ", len(self.scenes))
        self.config = config

    def __len__(self):
        return len(self.scenes)

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
            filter_invalid=True,
        )[0]

        return data, data_example, room_example

    def prepare_scene(self, scene):
        floor = scene["floor_planes"]

        # sample floorplan
        sampled_poses_and_rooms = _sample_test_poses_v2(
            scene["limits"],
            self.config.TEST.CAMERA_HEIGHT,
            scene["annos"],
            floor,
            step_size=self.config.TEST.POSE_SAMPLE_STEP,
            walls_padding_size=200,
        )

        # Render all pano images.
        (
            _,
            sampled_semantics,
            _,
            sampled_poses_and_rooms,
            _,
        ) = prepare_pano_renderings(self.config, sampled_poses_and_rooms, scene)

        # Prepare local samples
        combined_poses = []
        for sample_pose in sampled_poses_and_rooms:
            locally_sampled_poses_and_rooms = [sample_pose]
            combined_poses.append(
                torch.Tensor(
                    process_poses_spvloc(np.eye(4), np.array([p for p, _ in locally_sampled_poses_and_rooms]))
                )
            )

        combined_poses = torch.stack(combined_poses)  #

        return sampled_semantics, combined_poses


def impl_glfw_init(window_name="SPVLoc Demo", width=1500, height=720):
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


def draw_image_with_pyimgui(opencv_img):
    height, width, depth = opencv_img.shape
    if depth == 3:
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2RGBA)

    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, 0)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, opencv_img)

    imgui.image(texture, width, height)


def get_semantics(semantic_rendering):
    # Define a color mapping for each semantic label
    color_mapping = {
        1: (64, 67, 135),
        2: (41, 120, 142),
        3: (34, 167, 132),
        4: (121, 209, 81),
        5: (253, 231, 36),
    }

    # Map semantic labels to colors
    height, width = semantic_rendering.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)
    for label, color in color_mapping.items():
        colored_image[semantic_rendering == label] = color

    return colored_image


class GUI(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backgroundColor = (0, 0, 0, 1)
        self.window = impl_glfw_init()
        gl.glClearColor(*self.backgroundColor)
        imgui.create_context()

        # Load layout
        with open("data/misc/demo_gui.ini", "r") as f:
            ini_data = f.read()
        imgui.load_ini_settings_from_memory(ini_data)

        # Comment out if manual changes to the interface should be saved
        imgui.get_io().ini_file_name = None

        self.impl = GlfwRenderer(self.window)
        self.dataloader = Dataset_Inference(self.config)
        self.transform = build_transform(self.config, augment=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PerspectiveImageFromLayout(config).to(device)

        if args.test_ckpt:
            load_test_checkpoint(args.test_ckpt, self.model)
        else:
            exit()

        self.values = {
            "roll": 0,
            "pitch": 0,
            "yaw": 0,
            "FoV": 90,
            "panorama offset x": 0,
            "panorama offset y": 0,
            "panorama offset z": 0,
        }
        self.ranges = {
            "roll": (-20, 20),
            "pitch": (-20, 20),
            "yaw": (-179, 179),
            "FoV": (45, 135),
            "panorama offset x": (-1000, 1000),
            "panorama offset y": (-1000, 1000),
            "panorama offset z": (-300, 300),
        }

        self.update_pano_keys = [
            "panorama offset x",
            "panorama offset y",
            "panorama offset z",
        ]
        # self.has_changed = False  # Flag to track changes

        self.batch = {}

        result = {}
        for index, sublist in enumerate(self.dataloader.scenes):
            key = sublist[0]
            value = (sublist[1], index)
            if key not in result:
                result[key] = []
            result[key].append(value)

        self.scene_names = list(result.keys())
        self.all_scenes = result

        self.selected_idx = 0
        self.selected_scene = 0
        self.selected_image = 0

        self.load_sample(self.selected_idx, True)

        self.img_persp = None
        self.semantics_persp = None
        self.semantics_persp_gt = None
        self.semantics_pano_draw = None
        self.semantics_pano = None
        self.floorplan_img = None
        self.last_persp_pose = None
        self.last_fov = None
        self.sample_local = True
        self.draw_map = True
        self.draw_viewport_mask = True
        self.estimate_pose = True
        self.render_gt_pose = True
        self.render_estimated_pose = True
        self.floorplan_bg = None
        self.floorplan_limits = None

        self.sem = False

        self.loop()

    def load_sample(self, idx, changed_scene):
        data, data_example, room_example = self.dataloader[idx]
        self.geometry = data["geometry"]
        self.geometry_clipped = data["geometry_clipped"]
        self.materials = data["materials"]
        self.materials_clipped = data["materials_clipped"]
        self.rooms = data["rooms"]
        self.floor = data["floor_planes"]
        furniture = "full"
        self.room = self.rooms[room_example]
        pano_path = self.room["panorama"].format(furniture, "raw")
        self.scene = data

        self.img = np.array(Image.open(pano_path).convert("RGB"))

        if changed_scene:
            self.floorplan_polygons = get_floorplan_polygons(self.scene["annos"])
            self.renderd_panoramas, self.combined_poses = self.dataloader.prepare_scene(self.scene)

        self.batch = {}

    def loop(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()
            imgui.new_frame()

            imgui.begin("Settings", True)

            slider_changed = False
            reference_pano_changed = False

            for key in self.values:
                min_val, max_val = self.ranges[key]

                pano_key = key in self.update_pano_keys

                if self.sample_local or not pano_key:
                    _, new_value = imgui.slider_int(key, self.values[key], min_val, max_val)
                    if new_value != self.values[key]:
                        self.values[key] = new_value
                        slider_changed = True
                        if pano_key:
                            reference_pano_changed = True
                else:
                    imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 0.5, 0.5)
                    imgui.input_int(f"{key}", 0, step=0, flags=imgui.INPUT_TEXT_READ_ONLY)
                    imgui.pop_style_color()

            changed_scene, selected_scene = imgui.slider_int(
                f"scene: ({self.scene_names[self.selected_scene]})", self.selected_scene, 0, len(self.scene_names) - 1
            )
            changed_image, self.selected_image = imgui.slider_int(
                "image", self.selected_image, 0, len(self.all_scenes[self.scene_names[self.selected_scene]]) - 1
            )

            changed_sampling_type, self.sample_local = imgui.checkbox("sample local", self.sample_local)
            changed_estimate_pose, self.estimate_pose = imgui.checkbox("estimate pose", self.estimate_pose)
            changed_draw_map, self.draw_map = imgui.checkbox("draw map", self.draw_map)
            changed_draw_viewport_mask, self.draw_viewport_mask = imgui.checkbox(
                "draw viewport mask", self.draw_viewport_mask
            )
            changed_render_gt_pose, self.render_gt_pose = imgui.checkbox("render gt pose", self.render_gt_pose)
            changed_render_est_pose, self.render_estimated_pose = imgui.checkbox(
                "render estimated pose", self.render_estimated_pose
            )

            checkbox_changed = any(
                [
                    changed_sampling_type,
                    changed_draw_map,
                    changed_estimate_pose,
                    changed_draw_viewport_mask,
                    changed_render_gt_pose,
                    changed_render_est_pose,
                ]
            )

            if changed_scene:
                self.selected_image = 0
                self.selected_scene = selected_scene

            changed_sample = False
            if any([changed_scene, changed_image, checkbox_changed]):
                if changed_scene or changed_image or changed_sampling_type:
                    self.selected_idx = self.all_scenes[self.scene_names[self.selected_scene]][self.selected_image][1]
                    self.load_sample(self.selected_idx, changed_scene)
                    reference_pano_changed = True
                self.floorplan_bg = None
                changed_sample = True

            imgui.end()

            if slider_changed or changed_sample or self.img_persp is None:
                # start_time = time.time()

                pano_R, render_R, cam = pano2persp_get_camera_parameter(
                    self.values["FoV"],
                    self.values["yaw"],
                    self.values["pitch"],
                    self.values["roll"],
                    self.config.DATASET.PERSP_FROM_PANO_CORRECT_ROLL,
                )

                persp_pose = np.concatenate([render_R, self.room["pose"][..., np.newaxis]], axis=1)
                persp_pose = np.concatenate([persp_pose, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

                render_gt = False
                if (
                    not np.array_equal(persp_pose, self.last_persp_pose)
                    or self.last_fov != self.values["FoV"]
                    or changed_render_gt_pose
                ):
                    # Prepare perspective image
                    self.img_persp = pano2persp_from_rotation(
                        self.img,
                        cam,
                        pano_R,
                        height=self.config.RENDER.IMG_SIZE[0],
                        width=self.config.RENDER.IMG_SIZE[1],
                    )
                    self.last_persp_pose = persp_pose
                    self.last_fov = self.values["FoV"]
                    render_gt = True
                    self.batch.pop("cached_image_feat", None)

                img = Image.fromarray(self.img_persp)
                img_transf = self.transform(img)

                if reference_pano_changed or self.semantics_pano is None:
                    if self.sample_local:

                        self.pano_pose = self.room["pose"]
                        pano_pose_test = self.pano_pose + np.array(
                            [
                                self.values["panorama offset x"],
                                self.values["panorama offset y"],
                                self.values["panorama offset z"],
                            ]
                        )

                        room_idx = projects_onto_floor(self.pano_pose, self.floor)

                        room_idx_test = projects_onto_floor(pano_pose_test, self.floor)
                        if room_idx == room_idx_test:
                            room_idx = room_idx_test
                            self.pano_pose = pano_pose_test

                        self.semantics_pano = render_scene(
                            config,
                            self.geometry_clipped,
                            self.pano_pose,
                            self.materials_clipped,
                            None,
                            mode="semantic",
                            room_idx=room_idx,
                            output_numpy=True,
                        )
                        self.batch["combined_poses"] = torch.zeros([1, 2, 6])  # relative to center
                        self.batch["sampled_semantics"] = torch.tensor(self.semantics_pano).unsqueeze(0).unsqueeze(0)
                    else:
                        self.batch["sampled_semantics"] = self.renderd_panoramas
                        self.batch["combined_poses"] = self.combined_poses

                    self.batch.pop("cached_pano_feat", None)

                if self.estimate_pose:
                    self.batch["image"] = img_transf.unsqueeze(0)
                    self.batch["cams"] = torch.tensor(cam).unsqueeze(0)

                    self.batch["sampled_normals"] = torch.tensor([])  # unused
                    self.batch["sampled_depth"] = torch.tensor([])  # unused

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():  # No idea if this makes execution really quicker
                            output, output_plot = self.model.test_step_full_epoch(
                                self.batch,
                                refine_iterations=0,
                                top_n=1,
                                scene=self.scene,
                                prepare_plot=self.draw_viewport_mask,
                                cache_features=True,
                            )

                    # Copy for reuse
                    self.batch["cached_pano_feat"] = output["cached_pano_feat"]
                    self.batch["cached_image_feat"] = output["cached_image_feat"]

                    render_pose = output["initial_estimates"][0][0].numpy()

                    render_pose[:3, 3] = (
                        render_pose[:3, 3] + self.pano_pose if self.sample_local else render_pose[:3, 3]
                    )

                    err_t, err_r, err_t_2d, err_yaw = get_pose_distances_batch(
                        torch.tensor(persp_pose[np.newaxis, np.newaxis, :, :]).float(),
                        torch.tensor(render_pose[np.newaxis, np.newaxis, :, :]).float(),
                    )
                    max_score_idx = output["max_score_idx"][0, 0]
                    max_score = output["max_score"][max_score_idx]
                    poses = [render_pose] if self.render_estimated_pose else []
                else:
                    poses = []
                    err_t = torch.tensor([[0.0]])
                    err_t_2d = torch.tensor([[0.0]])
                    err_r = torch.tensor([[0.0]])
                    err_yaw = torch.tensor([[0.0]])
                    max_score = torch.tensor([0.0])

                if render_gt and self.render_gt_pose:
                    poses.append(persp_pose)

                if len(poses) > 0:
                    gt_poses_and_rooms = [(pose, int(projects_onto_floor(pose[:3, 3], self.floor))) for pose in poses]
                    rendered_semantics = render_scene_batched(
                        config,
                        self.geometry_clipped,
                        gt_poses_and_rooms,
                        self.materials_clipped,
                        cam,
                        mode="semantic",
                    )
                    if self.estimate_pose:
                        self.semantics_persp = get_semantics(rendered_semantics[0])

                if render_gt and self.render_gt_pose:
                    self.semantics_persp_gt = get_semantics(rendered_semantics[-1])

                if self.sample_local:
                    self.semantics_pano_draw = get_semantics(self.semantics_pano)
                else:
                    if self.estimate_pose:
                        self.semantics_pano_draw = get_semantics(self.renderd_panoramas[max_score_idx][0])
                    else:
                        self.semantics_pano_draw = get_semantics(self.renderd_panoramas[0][0])

                if self.estimate_pose and self.draw_viewport_mask:
                    mask = torch.clamp(
                        (output_plot["vp_masks_est"][0, 0, 0] > 0.5).unsqueeze(-1).float() + 0.5, min=0.0, max=1.0
                    )
                    self.semantics_pano_draw = (self.semantics_pano_draw * np.array(mask)).astype(np.uint8)

                if self.draw_map:

                    plot_poses = []

                    plot_poses.append(((persp_pose[:3, 3], None, None), "gt_pose"))
                    if self.estimate_pose:
                        plot_poses.append(((render_pose[:3, 3], None, None), "init_pose"))

                    if self.sample_local:
                        if self.floorplan_bg is None:
                            plot_poses.append(((self.pano_pose, None, None), "sample_pose"))
                        if self.estimate_pose:
                            plot_poses.append(((self.pano_pose, None, None), "selected_reference"))
                    else:
                        if self.floorplan_bg is None:
                            for _, pose in enumerate(self.combined_poses):
                                plot_poses.append(((pose[0, :3] * 1000.0, None, None), "sample_pose"))
                        if self.estimate_pose:
                            plot_poses += [
                                (
                                    (self.combined_poses[max_score_idx][0, :3] * 1000.0, None, None),
                                    "selected_reference",
                                )
                            ]

                    if self.floorplan_bg is None:
                        self.floorplan_bg, self.floorplan_limits = plot_floorplan(
                            self.scene["annos"],
                            self.floorplan_polygons,
                            plot_poses,
                            70,
                            crop_floorplan=False,
                            show_only_samples=True,
                            fill_frame=True,
                            export=True,
                        )

                    self.floorplan_img = plot_floorplan(
                        self.scene["annos"],
                        self.floorplan_polygons,
                        plot_poses,
                        70,
                        crop_floorplan=True,
                        show_only_samples=False,
                        limits=self.floorplan_limits,
                        fill_frame=True,
                        background_image=self.floorplan_bg,
                    )

                # print(f"Everything took {time.time() - start_time} seconds.")

            imgui.begin("Camera Image", True)
            draw_image_with_pyimgui(self.img_persp)
            imgui.end()

            if self.estimate_pose and self.render_estimated_pose:
                imgui.begin("Estimated Pose", True)
                draw_image_with_pyimgui(self.semantics_persp)
                imgui.end()

            if self.render_gt_pose:
                imgui.begin("GT Pose", True)
                draw_image_with_pyimgui(self.semantics_persp_gt)
                imgui.end()

            imgui.begin("Reference Panorama", True)
            draw_image_with_pyimgui(self.semantics_pano_draw)
            imgui.end()

            if self.draw_map:
                imgui.begin("Floorplan", True)
                draw_image_with_pyimgui(self.floorplan_img)
                imgui.end()

            imgui.begin("Result Info")
            imgui.text(f"distance: {(err_t[0,0].numpy() / 10):.2f} cm")
            imgui.text(f"distance 2d: {(err_t_2d[0,0].numpy() / 10):.2f} cm")
            imgui.text(f"angle offset: {err_r[0,0].numpy():.2f} deg.")
            imgui.text(f"yaw difference: {err_yaw[0,0].numpy():.2f} deg.")
            # if imgui.is_item_hovered():
            #    imgui.set_tooltip("Difference between estimated and true yaw.")
            imgui.text(f"top-1 match score: {max_score[0].numpy():.2f}")

            imgui.end()

            imgui.render()

            gl.glClearColor(*self.backgroundColor)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

        self.impl.shutdown()
        glfw.terminate()


if __name__ == "__main__":
    args = parse_args()

    config = get_cfg_defaults()
    config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)
    config.freeze()

    gui = GUI(config)
