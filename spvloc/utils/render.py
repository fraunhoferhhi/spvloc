import os
import numpy as np
import pyredner
import torch
import warnings
import re
from ..utils.pose_utils import decode_pose_spvloc

# Pyrender has some problems with device index.
# Filter warning (which does not make sense) and set cpu index for linux
warnings.filterwarnings("ignore", message=".*from cpu to cpu:0, this can be inefficient.*")

device = torch.device("cpu" if os.name == "nt" else "cpu:0")

pyredner.set_print_timing(False)
pyredner.set_device(device)


def create_camera(pose, img_size):
    # look at x axis equivalent to an offset x'
    look_at = pose.clone() + torch.Tensor([1, 0, 0])
    up = torch.Tensor([0.0, 0.0, 1.0])

    return pyredner.Camera(
        position=pose,
        look_at=look_at,
        up=up,
        camera_type=pyredner.camera_type.panorama,
        resolution=img_size,
    )


def create_perspective_camera(pose, cam, img_size):
    return pyredner.Camera(
        cam_to_world=pose,
        camera_type=pyredner.camera_type.perspective,
        intrinsic_mat=cam,
        resolution=img_size,
    )


def create_objects(mesh, material_names=None, clip_holes=False, r=None, t=None, recompute_normals=False):
    objects = []
    material_map = {}
    material_map["floor"] = 1
    material_map["floor_bb"] = 1
    material_map["floor_wall"] = 1
    material_map["floor_column"] = 1

    material_map["ceiling"] = 2
    material_map["ceiling_bb"] = 2
    material_map["ceiling_wall"] = 2
    material_map["ceiling_column"] = 2

    material_map["wall"] = 3
    material_map["wall_bb"] = 3
    material_map["wall_wall"] = 3
    material_map["wall_column"] = 3

    material_map["door"] = 4
    material_map["window"] = 5

    if material_names is not None:
        for verts, faces, normals, material_name in zip(
            mesh.verts_list(), mesh.faces_list(), mesh.verts_normals_list(), material_names
        ):
            if r is not None:
                verts = (torch.matmul(verts, torch.transpose(r, 0, 1))) + t
                normals = torch.matmul(normals, torch.transpose(r, 0, 1))

            if recompute_normals:
                normals = pyredner.compute_vertex_normal(verts, faces)

            colors = torch.full(verts.shape, float(material_map[material_name]))

            if not clip_holes or material_name not in ["door", "window"]:
                objects.append(
                    pyredner.Shape(
                        verts.to(device),
                        faces.to(device).int(),
                        material_map[material_name],
                        normals=normals.to(device),
                        colors=colors,
                    )
                )
    else:
        material = pyredner.Material()
        objects = []
        for verts, faces, normals in zip(mesh.verts_list(), mesh.faces_list(), mesh.verts_normals_list()):
            if r is not None:
                verts = (torch.matmul(verts, torch.transpose(r, 0, 1))) + t
                normals = torch.matmul(normals, torch.transpose(r, 0, 1))

            if recompute_normals:
                normals = pyredner.compute_vertex_normal(verts, faces)
            objects.append(
                pyredner.Object(verts.to(device), faces.to(device).int(), material, normals=normals.to(device))
            )
    return objects


def render_multiple_images(config, data, room_idx, poses, mode="color", cam=None, output_numpy=False):
    images = []
    for pose in poses:
        images.append(
            render_single_image(config, data, room_idx, pose, mode, cam, output_numpy).cpu().detach().numpy()
        )
    return images


def render_single_image(config, data, room_idx, pose, mode="color", cam=None, output_numpy=True):
    if pose.shape[0] == 6:
        xyz = pose[0:3].unsqueeze(0).cpu()
        rot = pose[3:].cpu()
        pose = decode_pose_spvloc(xyz, rot)

    if cam is None:
        room = data["rooms"][room_idx]
        cam = room["perspective"][0]["cam"]

    if mode == "color" or mode == "semantic":
        semantics_persp = render_scene(
            config,
            data["geometry_clipped"],
            pose,
            data["materials_clipped"],
            cam,
            mode=mode,
            room_idx=room_idx,
            output_numpy=output_numpy,
        )[:, :, 0]
    elif mode == "normal" or mode == "depth":
        semantics_persp = render_scene(
            config,
            data["geometry"],
            pose,
            data["materials"],
            cam,
            mode=mode,
            room_idx=room_idx,
            output_numpy=output_numpy,
        )

    return semantics_persp


def render_scene(
    config,
    geometry,
    pose,
    materials=None,
    cam=None,
    clip_holes=False,
    mode="depth",
    room_idx=None,
    output_numpy=True,
    img_size_in=None,
):
    render_pano = cam is None

    pose = torch.Tensor(pose)
    r_wtc = None
    t_wtc = None

    if render_pano:
        if img_size_in is None:
            img_size = config.RENDER.PANO_SIZE
        else:
            img_size = img_size_in
        camera = create_camera(pose, img_size)
    else:
        img_size = config.RENDER.IMG_SIZE
        if mode == "normal":
            world_to_cam = torch.linalg.inv(pose)  # is this transofmration needed
            r_wtc = world_to_cam[0:3, 0:3]
            t_wtc = world_to_cam[0:3, 3]
            pose = torch.eye(4)
        camera = create_perspective_camera(pose, torch.Tensor(cam), img_size)

    if isinstance(geometry, list):
        objects = []
        if materials is not None:
            for room, (m, mat) in enumerate(zip(geometry, materials)):
                if room_idx is None or room in [room_idx]:
                    objects += create_objects(m, mat, clip_holes=clip_holes, r=r_wtc, t=t_wtc)
        else:
            for room, m in enumerate(geometry):
                if room_idx is None or room in [room_idx]:
                    objects += create_objects(m, r=r_wtc, t=t_wtc)
    else:
        objects = create_objects(geometry)

    if materials is not None:
        scene = pyredner.Scene(camera=camera, shapes=objects)
    else:
        scene = pyredner.Scene(camera=camera, objects=objects)

    if mode == "depth":
        render_target = pyredner.channels.depth
    elif mode == "semantic":
        if materials is not None:
            render_target = pyredner.channels.material_id
        else:
            render_target = pyredner.channels.shape_id
    elif mode == "normal":
        render_target = pyredner.channels.geometry_normal
    elif mode == "color":
        render_target = pyredner.channels.vertex_color
    else:
        raise NotImplementedError("Only depth and semantic and normal rendering are implemented.")

    seed = 0 if config.RENDER.FIX_SEED else None
    img = pyredner.render_g_buffer(scene, [render_target], device=device, num_samples=1, seed=seed)
    # flip x axis for equivalent to a flipped x'
    if render_pano:
        img = img.flip(dims=[1])

    if output_numpy:
        img = img.cpu().detach().squeeze(2).numpy()
    if mode == "semantic":
        if output_numpy:
            img = img.astype(int)
        else:
            img.int()

    elif mode == "depth":
        img = img / 1000.0

    if output_numpy:
        img = np.ascontiguousarray(img)

    return img


def render_scene_batched(config, geometry, poses, materials=None, cam=None, clip_holes=False, mode="depth"):
    scenes = []
    objects = {}
    render_pano = cam is None

    if render_pano:
        img_size = config.RENDER.PANO_SIZE
    else:
        img_size = config.RENDER.IMG_SIZE

    for pose, room in poses:
        pose = torch.Tensor(pose)
        r_wtc = None
        t_wtc = None

        if render_pano:
            camera = create_camera(pose, img_size)
        else:
            cam = torch.Tensor(cam)
            if mode == "normal" or mode == "depth_normal":
                world_to_cam = torch.linalg.inv(pose)
                r_wtc = world_to_cam[0:3, 0:3]
                t_wtc = world_to_cam[0:3, 3]
                pose = torch.eye(4)
            camera = create_perspective_camera(pose, cam, img_size)

        if isinstance(geometry, list):
            objects = []
            if materials is not None:
                for room_, (m, mat) in enumerate(zip(geometry, materials)):
                    if room_ == room:
                        objects += create_objects(m, mat, clip_holes=clip_holes, r=r_wtc, t=t_wtc)
            else:
                for room_, m in enumerate(geometry):
                    if room_ == room:
                        objects += create_objects(m, r=r_wtc, t=t_wtc)
        else:
            raise NotImplementedError("Batch rendering can only be called with a list of meshes")

        if materials is not None:
            scenes.append(pyredner.Scene(camera=camera, shapes=objects))  # objects))
        else:
            scenes.append(pyredner.Scene(camera=camera, objects=objects))  # objects))

    if mode == "depth":
        render_target = [pyredner.channels.depth]
    elif mode == "semantic":
        if materials is not None:
            render_target = [pyredner.channels.material_id]
        else:
            render_target = [pyredner.channels.shape_id]
    elif mode == "normal":
        render_target = [pyredner.channels.geometry_normal]
    elif mode == "color":
        render_target = [pyredner.channels.vertex_color]
    elif mode == "depth_normal":
        render_target = [pyredner.channels.geometry_normal, pyredner.channels.depth]
    else:
        raise NotImplementedError("Only depth and semantic rendering are implemented.")

    seeds = [0] * len(scenes) if config.RENDER.FIX_SEED else None
    imgs = pyredner.render_g_buffer(scenes, render_target, device=device, num_samples=1, seed=seeds)
    # flip x axis for equivalent to a flipped x'
    if render_pano:
        imgs = imgs.flip(dims=[2])
    imgs = imgs.cpu().squeeze(3).numpy()
    if mode == "semantic":
        imgs = imgs.astype(int)
    if mode == "depth":
        imgs = imgs / 1000.0
    elif mode == "depth_normal":
        imgs[..., -1] = imgs[..., -1] / 1000.0
    imgs = np.ascontiguousarray(imgs)

    return imgs
