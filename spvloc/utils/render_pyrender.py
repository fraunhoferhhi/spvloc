import sys

import numpy as np
import torch

# Manuipulate OpenGL to disable antialiasing:
# https://github.com/marian42/mesh_to_sdf/blob/66036a747e82e7129f6afc74c5325d676a322114/mesh_to_sdf/pyrender_wrapper.py#L20
if "pyrender" in sys.modules or "OpenGL" in sys.modules:
    raise ImportError("This file must be imported before pyrender is imported.")

# Disable antialiasing:
import OpenGL.GL

suppress_multisampling = False
old_gl_enable = OpenGL.GL.glEnable


def new_gl_enable(value):
    if suppress_multisampling and value == OpenGL.GL.GL_MULTISAMPLE:
        OpenGL.GL.glDisable(value)
    else:
        old_gl_enable(value)


OpenGL.GL.glEnable = new_gl_enable

old_glRenderbufferStorageMultisample = OpenGL.GL.glRenderbufferStorageMultisample


def new_glRenderbufferStorageMultisample(target, samples, internalformat, width, height):
    if suppress_multisampling:
        OpenGL.GL.glRenderbufferStorage(target, internalformat, width, height)
    else:
        old_glRenderbufferStorageMultisample(target, samples, internalformat, width, height)


OpenGL.GL.glRenderbufferStorageMultisample = new_glRenderbufferStorageMultisample

import pyrender

device = torch.device("cpu")


class CustomShaderCache:
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram(
                "spvloc/utils/shader/mesh.vert",
                "spvloc/utils/shader/mesh.frag",
                defines=defines,
            )
        return self.program

    def clear(self):
        self.program = None


def z_to_xyz_buffer(y_fov, image_resolution, z_map):
    height, width = image_resolution

    # Calculate the focal length
    focal_length = (height / 2) / np.tan(y_fov / 2)

    # Create arrays of x and y coordinates
    x_coords = np.arange(width) - (width / 2)
    y_coords = np.arange(height) - (height / 2)

    # Create a grid of x and y coordinates
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Compute the x and y coordinates in the camera coordinate system
    x_camera = (x_grid * (z_map / focal_length)).astype(np.float32)
    y_camera = (y_grid * (z_map / focal_length)).astype(np.float32)

    # Create the XYZ buffer
    xyz_buffer = np.zeros((height, width, 3), dtype=np.float32)
    xyz_buffer[..., 0] = x_camera
    xyz_buffer[..., 1] = y_camera
    xyz_buffer[..., 2] = z_map

    return np.linalg.norm(xyz_buffer, axis=2)


def z_to_xyz_buffer_batch(y_fov, image_resolution, z_map_batch):
    batch_size = z_map_batch.shape[0]
    height, width = image_resolution

    # Calculate the focal length
    focal_length = (height / 2) / np.tan(y_fov / 2)

    # Create arrays of x and y coordinates
    x_coords = np.arange(width) - (width / 2)
    y_coords = np.arange(height) - (height / 2)

    # Create a grid of x and y coordinates
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Expand the first dimension of x_grid to match the batch size
    x_grid = x_grid[np.newaxis]
    y_grid = y_grid[np.newaxis]

    # Compute the x and y coordinates in the camera coordinate system for each batch
    x_camera_batch = (x_grid * (z_map_batch / focal_length)).astype(np.float32)
    y_camera_batch = (y_grid * (z_map_batch / focal_length)).astype(np.float32)

    # Create the XYZ buffer with batch dimension
    xyz_buffer_batch = np.zeros((batch_size, height, width, 3), dtype=np.float32)
    xyz_buffer_batch[..., 0] = x_camera_batch
    xyz_buffer_batch[..., 1] = y_camera_batch
    xyz_buffer_batch[..., 2] = z_map_batch

    return np.linalg.norm(xyz_buffer_batch, axis=3, keepdims=True)


def create_perspective_camera(yfov, img_size, batch_size=1):
    aspect = img_size[1] / img_size[0]
    camera = pyrender.camera.PerspectiveCamera(yfov=yfov, aspectRatio=aspect, znear=30, zfar=20000.0)
    return camera


def check_winding_order(vertices, normals, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = torch.cross(v1 - v0, v2 - v0)
    dot_products = torch.einsum("ij,ij->i", face_normals, normals[faces[:, 0]])

    incorrect_mask = dot_products < 0
    faces[incorrect_mask] = faces[incorrect_mask].flip(1)

    return faces


def create_objects_combined(geometry, materials, clip_holes=False):
    material_map = {
        "floor": 1,
        "floor_bb": 1,
        "floor_wall": 1,
        "floor_column": 1,
        "ceiling": 2,
        "ceiling_bb": 2,
        "ceiling_wall": 2,
        "ceiling_column": 2,
        "wall": 3,
        "wall_bb": 3,
        "wall_wall": 3,
        "wall_column": 3,
        "door": 4,
        "window": 5,
    }

    merged_verts = []
    merged_faces = []
    merged_normals = []
    merged_colors = []

    num_verts_accumulated = 0  # Track the accumulated number of vertices

    for _, (mesh, material_names) in enumerate(zip(geometry, materials)):
        for verts, faces, normals, material_name in zip(
            mesh.verts_list(), mesh.faces_list(), mesh.verts_normals_list(), material_names
        ):
            verts = verts.to(device).clone()
            faces = faces.to(device).clone()
            normals = normals.to(device).clone()

            colors = torch.full([verts.shape[0], 4], float(material_map[material_name]) / 255.0)
            colors[:, 3] = 1.0

            if not clip_holes or material_name not in ["door", "window"]:
                # Update the face indices by adding the accumulated number of vertices
                faces += num_verts_accumulated

                # Append the current mesh data to the merged arrays
                merged_verts.append(verts)
                merged_faces.append(faces)
                merged_normals.append(normals)
                merged_colors.append(colors)

                # Update the accumulated number of vertices
                num_verts_accumulated += verts.shape[0]

    # Concatenate the merged arrays along the first dimension
    merged_verts = torch.cat(merged_verts, dim=0)
    merged_faces = torch.cat(merged_faces, dim=0)
    merged_normals = torch.cat(merged_normals, dim=0)
    merged_colors = torch.cat(merged_colors, dim=0)

    # merged_faces = check_winding_order(merged_verts, merged_normals, merged_faces)

    primitive = pyrender.Primitive(
        positions=merged_verts.numpy(),
        normals=merged_normals.numpy(),
        color_0=merged_colors.numpy(),
        indices=merged_faces.numpy(),
    )

    return pyrender.Node(mesh=pyrender.Mesh([primitive]))


def render_scene_batched_pyrender(
    geometry, geometry_clipped, poses, materials, materials_clipped, img_size, cam=None, clip_holes=False
):
    global suppress_multisampling
    suppress_multisampling = True

    # define renderer
    renderer = pyrender.OffscreenRenderer(img_size[1], img_size[0])
    renderer._renderer._program_cache = CustomShaderCache()

    render_flags = pyrender.RenderFlags.RGBA
    render_flags |= pyrender.RenderFlags.SKIP_CULL_FACES

    yfov = np.arctan(img_size[0] / img_size[1] / cam[1, 1]) * 2

    if isinstance(geometry_clipped, list):
        scene_node_seg = create_objects_combined(geometry_clipped, materials_clipped, clip_holes=False)
        scene_node_normals = create_objects_combined(geometry, materials, clip_holes=True)
    else:
        raise NotImplementedError("Meshes must be a list")

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0])

    # Prepare all cameras
    cam_nodes = []
    for pose, _ in poses:
        pose = torch.Tensor(pose).to(device)
        cam = torch.Tensor(cam).to(device)

        camera_pose = pose.clone()
        camera_pose[:3, 2] = -1 * camera_pose[:3, 2]  # only rotation is changed
        camera = create_perspective_camera(yfov, img_size)

        scene.add(camera, pose=camera_pose)
        cam_nodes.append(scene.main_camera_node)
        scene._main_camera_node = None

    normals = []
    depths = []
    segs = []

    scene.add_node(scene_node_normals)
    for idx in range(len(poses)):
        # Set camera node
        scene.main_camera_node = cam_nodes[idx]
        # Render image and depth
        normal, depth = renderer.render(scene, flags=render_flags)
        normals.append(normal[:, :, :3])
        depths.append(depth)
    scene.remove_node(scene_node_normals)

    scene.add_node(scene_node_seg)
    for idx in range(len(poses)):
        # Set camera node
        scene.main_camera_node = cam_nodes[idx]
        # Render image and depth
        seg, _ = renderer.render(scene, flags=render_flags)
        segs.append(seg[:, :, 3])
    scene.remove_node(scene_node_seg)

    # Clear the scene
    scene.clear()

    normals = np.stack(normals, axis=0)
    depths = np.stack(depths, axis=0)
    segs = np.stack(segs, axis=0)
    normals = normals / 255 * 2 - 1
    # segs = segs[:, :, :]
    segs[segs > 5] = 0

    depths = z_to_xyz_buffer_batch(yfov, img_size, depths / 1000.0)

    renderer.delete

    suppress_multisampling = False

    return normals.astype(np.float32), depths.astype(np.float32), segs.astype(np.int32)


def render_scene_pyrender(
    geometry,
    geometry_clipped,
    pose,
    materials,
    materials_clipped,
    img_size,
    cam=None,
    clip_holes=False,
    room_idx=None,
):
    global suppress_multisampling
    suppress_multisampling = True
    # define renderer

    renderer = pyrender.OffscreenRenderer(img_size[1], img_size[0])
    renderer._renderer._program_cache = CustomShaderCache()

    render_flags = pyrender.RenderFlags.RGBA
    render_flags |= pyrender.RenderFlags.SKIP_CULL_FACES

    yfov = np.arctan(img_size[0] / img_size[1] / cam[1, 1]) * 2

    if isinstance(geometry_clipped, list):
        if room_idx is None or room_idx > (len(geometry) - 1):
            scene_node_seg = create_objects_combined(geometry_clipped, materials_clipped, clip_holes=False)
            scene_node_normals = create_objects_combined(geometry, materials, clip_holes=True)
        else:
            scene_node_seg = create_objects_combined(
                [geometry_clipped[room_idx]], [materials_clipped[room_idx]], clip_holes=False
            )
            scene_node_normals = create_objects_combined([geometry[room_idx]], [materials[room_idx]], clip_holes=True)
    else:
        raise NotImplementedError("Meshes must be a list")

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0])

    # Prepare camera
    pose = torch.Tensor(pose).to(device)
    cam = torch.Tensor(cam).to(device)
    camera_pose = pose.clone()
    camera_pose[:3, 2] = -1 * camera_pose[:3, 2]  # only rotation is changed
    camera = create_perspective_camera(yfov, img_size)
    scene.add(camera, pose=camera_pose)

    scene.add_node(scene_node_normals)
    # Render image and depth
    normal, depth = renderer.render(scene, flags=render_flags)

    normal = normal[:, :, :3].copy()
    scene.remove_node(scene_node_normals)

    scene.add_node(scene_node_seg)
    # Render image and depth
    seg, _ = renderer.render(scene, flags=render_flags)
    seg = seg[:, :, 3].copy()
    scene.remove_node(scene_node_seg)

    # Clear the scene
    scene.clear()
    renderer.delete()

    # Prepare output
    normal = normal / 255 * 2 - 1
    seg[seg > 5] = 0

    depth = z_to_xyz_buffer(yfov, img_size, depth / 1000.0)

    suppress_multisampling = False

    return normal.astype(np.float32), depth.astype(np.float32), seg.astype(np.int32)
