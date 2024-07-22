import gc
import os

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ..data.load import load_scene_annos
from ..data.transform import denormalize
from ..utils.eval import get_pickle_name
from ..utils.plot_floorplan import crop_non_white, get_floorplan_polygons, plot_floorplan
from ..utils.projection import calculate_bb, project_pano_to_persp_mask_pano, projects_onto_floor_efficient
from ..utils.render import render_multiple_images, render_scene_batched


def rgba_mask(img, alpha=0.4):
    rgba_mask = np.zeros((img.shape[0], img.shape[1], 4))
    rgba_mask[img == 0] = np.array([0, 0, 0, 0])
    rgba_mask[img > 0] = np.array([1.0, 1.0, 1.0, alpha])
    return rgba_mask


def combine_masks(binary_map1, binary_map2):
    intersection = np.logical_and(binary_map1, binary_map2)
    exclusive_to_map1 = np.logical_and(binary_map1, np.logical_not(binary_map2))
    exclusive_to_map2 = np.logical_and(binary_map2, np.logical_not(binary_map1))
    combined_visualization = np.zeros_like(binary_map1, dtype=np.uint8)
    combined_visualization[exclusive_to_map1] = 1
    combined_visualization[exclusive_to_map2] = 2
    combined_visualization[intersection] = 3
    return combined_visualization


def add_bbs_to_ax(bb_gt, bb_est, w, linewidth=1):
    result = []
    rect_args = {"linewidth": linewidth, "facecolor": "none", "antialiased": False}

    def add_bb(bb, edgecolor):
        shifts = [0] + [-w] * (bb[2] > w)
        for shift in shifts:
            rect = patches.Rectangle(
                (bb[0] + shift, bb[1]),
                bb[2] - bb[0],
                bb[3] - bb[1],
                edgecolor=edgecolor,
                **rect_args,
            )
            result.append(rect)

    if bb_gt is not None:
        add_bb(bb_gt, "g")
    if bb_est is not None:
        add_bb(bb_est, "r")

    return result


def plot_with_bbs(ax, image, bb1, bb2, width, title, vmin=0, vmax=1):
    ax.imshow(image, vmin=vmin, vmax=vmax)
    bb_patches = add_bbs_to_ax(bb1, bb2, width)
    for p in bb_patches:
        ax.add_patch(p)
    ax.set_title(title)


def plot_mask_comparison(ax, mask1, mask2, title):
    ax.imshow(combine_masks(mask1, mask2), vmin=0, vmax=3)
    ax.set_title(title)


def plot_img(ax, img, title="", vmin=0, vmax=1, bb1=None, bb2=None, overlay=None, alpha=1.0, linewidth=1):
    ax.imshow(img, vmin=vmin, vmax=vmax)
    if bb1 is not None or bb2 is not None:
        width = img.shape[1]
        bb_patches = add_bbs_to_ax(bb1, bb2, width, linewidth=linewidth)
        for p in bb_patches:
            ax.add_patch(p)
    if overlay is not None:
        ax.imshow(overlay, vmax=2, alpha=alpha)
    if len(title) > 0:
        ax.set_title(title)


def plot_img_and_save(
    img, filename, scene_folder, vmin=0, vmax=1, bb1=None, bb2=None, overlay=None, alpha=1.0, linewidth=1
):
    img = np.array(img)
    w = img.shape[1]
    h = img.shape[0]
    dpi = w / 4
    fig = plt.figure(1, frameon=False)
    fig.set_size_inches(w / dpi, h / dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plot_img(ax, img, vmin=vmin, vmax=vmax, bb1=bb1, bb2=bb2, overlay=overlay, alpha=alpha, linewidth=linewidth)
    fig.savefig(os.path.join(scene_folder, filename), dpi=dpi)
    plt.close("all")
    gc.collect()


def pad_image_to_width(image, target_width, constant_value=255):
    padding_needed = target_width - image.shape[1]
    padding_left = padding_needed // 2
    padding_right = padding_needed - padding_left
    return np.pad(
        image, ((0, 0), (padding_left, padding_right), (0, 0)), mode="constant", constant_values=constant_value
    )


def get_viewport_width(config, scene, pano_pose, persp_pose, cam, persp_depths):
    t_gt = torch.tensor(pano_pose[:3, 3]).unsqueeze(0)
    floor = scene["floor_planes"]
    near_room_indices = projects_onto_floor_efficient(t_gt, floor)
    gt_poses_and_rooms = []
    gt_poses_and_rooms.append((t_gt[0], int(near_room_indices)))
    estimated_pano_normals = render_scene_batched(
        config, scene["geometry"], gt_poses_and_rooms, scene["materials"], mode="depth_normal"
    )
    estimated_depth = torch.tensor(estimated_pano_normals[0, ..., 3])
    bb_gt, viewport_gt = project_pano_to_persp_mask_pano(
        persp_depths[0].squeeze(-1),
        estimated_depth.unsqueeze(0),
        t_gt,
        persp_pose,
        cam,
        scale_translation=False,
    )
    y, x = np.where(viewport_gt[0, 64:65])
    middle_bb = calculate_bb(np.stack([x, y]).transpose(), viewport_gt.shape[2], 1)
    bb_gt[0, 0] = middle_bb[0]
    bb_gt[0, 2] = middle_bb[2]
    return bb_gt


def plot_overview(
    batch, output, config, top_idx=0, top_n=3, refine=True, show_normals=False, save_overview=False, save_all=False
):

    if save_overview or save_all:
        mpl.use("Agg")  # use non interactive backend

    cams = batch["cams"].cpu().numpy()
    # pano_normals = batch["sampled_normals"].cpu()
    pano_semantics = batch["sampled_semantics"].cpu()
    pano_depth = batch["sampled_depth"].cpu()
    pano_normals = batch["sampled_normals"].cpu()
    combined_poses = batch["combined_poses"].cpu()
    scene_id = batch["scene_id"].cpu()

    query_image = batch["image"].cpu()
    gt_poses = batch["gt_poses"].cpu()

    bs = pano_semantics.shape[0]
    bs_img = query_image.shape[0]

    max_score_idx = output["max_score_idx"]
    vp_masks_est = output["vp_masks_est"]
    bbs_est = output["bbs_est"]
    initial_estimates = output["initial_estimates"]
    refined_estimates = output["refined_estimates"]
    query_semantics_decoded = output["query_semantics_decoded"]
    query_layout_decoded = output["query_layout_decoded"]
    xyz_out = output["xyz_out"]
    scene = output["scene"]
    annos = load_scene_annos(config.DATASET.PATH, scene_id)
    no_estimates = initial_estimates.shape[1]

    top_score = max_score_idx[:, top_idx]

    pad_height = int((config.RENDER.IMG_SIZE[0] - config.INPUT.IMG_SIZE[0]) / 2)

    movement_from_reference = torch.linalg.norm(torch.reshape(xyz_out, [bs_img, no_estimates, 3]), dim=-1)

    best_pano_poses = combined_poses[top_score, 0, :3].numpy()  # from batch
    combined_poses = combined_poses.numpy()

    best_sem_panos = pano_semantics[top_score, 0]  # from batch
    best_depth_panos = pano_depth[top_score, 0]  # from batch
    best_norm_panos = (0.5 * (pano_normals[top_score] + 1)).permute([0, 2, 3, 1])

    plot_panos = best_norm_panos if show_normals else best_sem_panos

    vp_masks_est_top = (vp_masks_est[:, top_idx, 0] > 0.5).numpy()  # add to output
    bbs_est_top = bbs_est[:, top_idx].numpy()  # add to output
    initial_poses = initial_estimates[:, top_idx].numpy()  # add to output
    alternative_poses = initial_estimates[:, [i for i in range(top_n) if i != top_idx]].numpy()

    if refine:
        refined_poses = refined_estimates[:, 0].numpy()

    decoded_semantics = torch.argmax(query_semantics_decoded, axis=1)  # add to output
    decoded_normals = (0.5 * (query_layout_decoded + 1)).permute([0, 2, 3, 1])  # add to output

    row_to_add = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).view(1, 1, -1)
    gt_poses_4x4 = torch.cat((gt_poses, row_to_add.expand(bs_img, -1, -1)), dim=1).to(torch.float32).numpy()

    for idx, gt_pose in enumerate(gt_poses_4x4):
        orig_image = denormalize(config, query_image[idx])
        cam = cams[idx]

        # RENDER
        render_poses = [gt_pose, initial_poses[idx]] + ([refined_poses[idx]] if refine else [])
        pose_difference_norm = np.linalg.norm(render_poses[0][:3, 3] - render_poses[-1][:3, 3]) / 10.0
        pose_difference_norm_initial = np.linalg.norm(render_poses[0][:3, 3] - render_poses[1][:3, 3]) / 10.0

        # Check if there was a substantial pose improvement by refinement and only plot those samples
        # if (
        #     movement_from_reference[idx, 0] < 1.0
        #     or pose_difference_norm * 2 > pose_difference_norm_initial
        #     or pose_difference_norm_initial < 50.0
        # ):
        #     continue

        cam[1, 1] = cam[0, 0]
        persp_semantics = render_multiple_images(config, scene, None, render_poses, mode="semantic", cam=cam)
        persp_normals = render_multiple_images(config, scene, None, render_poses, mode="normal", cam=cam)
        persp_normals = [0.5 * (norm + 1) for norm in persp_normals]
        persp_plot = persp_normals if show_normals else persp_semantics
        # Viewport
        persp_depths = render_multiple_images(config, scene, None, render_poses, mode="depth", cam=cam)
        bb2d, vp_masks = zip(
            *(
                project_pano_to_persp_mask_pano(
                    persp_depths[i].squeeze(-1),
                    best_depth_panos[idx].unsqueeze(0),
                    best_pano_poses[idx][np.newaxis, ...],
                    render_poses[i],
                    cam,
                    projection_margin=pad_height,
                )
                for i in range(len(render_poses))
            )
        )

        bb2d = np.concatenate(bb2d, axis=0)
        vp_masks = np.concatenate(vp_masks, axis=0)

        # Plot MINI
        # alpha_mask_vp_est = rgba_mask(vp_masks_est_top[idx], alpha=0.4)
        # fig, axes = plt.subplots(4, 1, figsize=(2, 6))
        # for ax in axes.flatten():
        #     ax.axis("off")

        # plot_img(axes[0], orig_image)
        # plot_img(axes[1], persp_plot[1], vmax=5)
        # plot_img(axes[2], plot_panos[idx], vmax=5, bb2=bbs_est_top[idx], overlay=alpha_mask_vp_est)
        # plot_img(axes[3], plan_image, "")
        # for ax in axes.flatten():
        #     ax.set_aspect("equal")
        # plt.tight_layout()
        # plt.show()

        result_folder = os.path.join(config.OUT_DIR, "results")
        pickle_name = get_pickle_name(config)
        test_folder = os.path.join(result_folder, pickle_name)
        scene_folder = os.path.join(test_folder, f"{int(scene_id):05d}")
        img_name = f"{int(idx):03d}"
        vmax = 1 if show_normals else 5
        print(img_name)

        if save_overview or not save_all:
            comb_masks_1 = combine_masks(vp_masks_est_top[idx], vp_masks[0])
            comb_masks_2 = combine_masks(vp_masks[1], vp_masks[0])
            if refine > 0:
                comb_masks_3 = combine_masks(vp_masks[2], vp_masks[0])

            # Prepare Floorplan
            max_score_img = output["max_score"][:, idx]
            score_normalized = (max_score_img - max_score_img.min()) / (max_score_img.max() - max_score_img.min())

            # plot_poses = [
            #    ((pose[0, :3] * 1000.0, None, max_score_img[pose_idx].item()), "sample_pose")
            #    for pose_idx, pose in enumerate(combined_poses)
            # ]

            floorplan_polygons = get_floorplan_polygons(annos)

            plot_score_map = False  # TODO add parameter
            if plot_score_map:
                bbs_est_img = bbs_est[idx]
                plot_poses_score = []
                bb_gt = get_viewport_width(config, scene, gt_pose, render_poses[0], cam, persp_depths)

                plot_poses_score += [((render_poses[0][:3, 3], bb_gt[0], None), "gt_pose")]

                for pose_idx, pose in enumerate(combined_poses):
                    # bb_sample = bbs_est_img[max_score_idx[idx][pose_idx]]
                    plot_poses_score.append(
                        ((pose[0, :3] * 1000.0, None, score_normalized[pose_idx].item()), "sample_pose_score")
                    )
                plot_poses_score += [((render_poses[1][:3, 3], None, None), "init_pose_ring")]
                plot_poses_score += [
                    ((alternative_poses[idx, i, :3, 3], None, None), "alternative_pose_ring") for i in range(top_n - 1)
                ]

                plan_image_score = plot_floorplan(
                    annos, floorplan_polygons, plot_poses_score, 200, crop_floorplan=True
                )

            plot_poses = []
            for pose_idx, pose in enumerate(combined_poses):
                plot_poses.append(((pose[0, :3] * 1000.0, None, None), "sample_pose"))
            plot_poses += [((combined_poses[top_score[idx].item()][0, :3] * 1000.0, None, None), "selected_reference")]
            plot_poses += [((render_poses[0][:3, 3], None, None), "gt_pose")]
            # plot_poses += [
            #     ((alternative_poses[idx, i, :3, 3], None, None), "alternative_pose") for i in range(top_n - 1)
            # ]
            plot_poses += [((render_poses[1][:3, 3], None, None), "init_pose")]
            plot_poses += [((render_poses[2][:3, 3], None, None), "refined_pose")] if refine else []
            plan_image = plot_floorplan(annos, floorplan_polygons, plot_poses, 200, crop_floorplan=True)
            plan_image_zoom = plot_floorplan(
                annos, floorplan_polygons, plot_poses, 100, crop_floorplan=True, zoom=True
            )

            # Plot
            fig, axes = plt.subplots(4, 4, figsize=(13, 9))
            for ax in axes.flatten():
                ax.axis("off")

            plot_img(axes[0, 0], orig_image, "Original Image")
            plot_img(axes[0, 1], persp_plot[1], "Estimated Pose", vmax=vmax)
            plot_img(axes[0, 3], persp_plot[0], "Ground Truth Pose", vmax=vmax)
            plot_img(axes[1, 0], decoded_semantics[idx], "Decoded Semanctics", vmax=5)
            plot_img(axes[1, 1], decoded_normals[idx], "Decoded Normals")
            plot_img(axes[1, 2], plan_image, "Map")
            plot_img(axes[1, 3], plan_image_zoom, "Map (Zoom)")
            plot_img(axes[2, 0], plot_panos[idx], "Estimated BB", vmax=vmax, bb1=bb2d[0], bb2=bbs_est_top[idx])
            plot_img(axes[2, 1], plot_panos[idx], "Estimated BB (render)", vmax=vmax, bb1=bb2d[0], bb2=bb2d[1])
            plot_img(axes[3, 0], comb_masks_1, "Estimated mask", vmax=3)
            plot_img(axes[3, 1], comb_masks_2, "Estimated mask (render)", vmax=3)

            if refine > 0:
                plot_img(axes[0, 2], persp_plot[2], "Estimated Pose Refine", vmax=vmax)
                plot_img(axes[3, 2], comb_masks_3, "Refined mask (render)", vmax=3)
                plot_img(axes[2, 2], plot_panos[idx], "Refined BB", vmax=vmax, bb1=bb2d[0], bb2=bb2d[2])

            refinement_info = f"Refined distance: {pose_difference_norm:.2f}cm." if refine > 0 else ""
            subtitle_text = (
                f"Image {idx+1}/{bs_img} with {bs} reference points. "
                f"Estimated distance: {pose_difference_norm_initial:.2f}cm. "
                f"{refinement_info} Relative movement to reference: "
                f"{movement_from_reference[idx, 0]:.2f}m"
            )
            plt.suptitle(subtitle_text, fontsize=10, y=0.04)
            if save_overview:
                fig.set_dpi(150)
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                if not os.path.exists(scene_folder):
                    os.makedirs(scene_folder)
                image = Image.fromarray(crop_non_white(img))

                image.save(os.path.join(scene_folder, f"{img_name}.png"))
                plt.close(fig)
            else:
                plt.show()

        if save_all:
            img_f = os.path.join(scene_folder, img_name)
            if not os.path.exists(img_f):
                os.makedirs(img_f)

            save_all_estimates = False  # TODO: add parameter
            if save_all_estimates:
                pano_f = os.path.join(img_f, "all_panos")
                if not os.path.exists(pano_f):
                    os.makedirs(pano_f)
                vp_masks_est_img = (vp_masks_est[idx, :, 0] > 0.5).numpy()
                score_idx_img = max_score_idx[idx]
                bbs_est_img = bbs_est[idx]
                max_score_img = output["max_score"][:, idx]
                for ref_idx in range(bs):
                    pano_idx = score_idx_img[ref_idx].item()

                    # alternative overlay
                    # logits = vp_masks_est[idx, ref_idx, 0].numpy()
                    # prob = 1 / (1 + np.exp(-logits))
                    # overlay = np.dstack((prob, prob, prob, np.clip(prob, 0.0, 0.4)))

                    plot_img_and_save(
                        pano_semantics[pano_idx, 0],
                        f"pano_{ref_idx}_{int(max_score_img[pano_idx]*100.0)}.png",
                        pano_f,
                        vmax=vmax,
                        bb2=bbs_est_img[ref_idx],
                        overlay=rgba_mask(vp_masks_est_img[ref_idx]),
                    )

            bb2d = np.concatenate([bb2d[:, :4], bbs_est_top[idx].reshape(1, -1)], axis=0)
            np.savetxt(os.path.join(img_f, "bb2d.txt"), bb2d, fmt="%.2f", delimiter=", ")

            orig_image.save(os.path.join(img_f, "00_orig.png"))
            # plot_img_and_save(orig_image, "00_orig.png", img_f)
            plot_img_and_save(persp_semantics[0], "01a_gt_pose_s.png", img_f, vmax=vmax)
            plot_img_and_save(persp_semantics[1], "01b_pose_s.png", img_f, vmax=vmax)
            # plot_img_and_save(persp_normals[0], "01d_gt_pose_n.png", img_f)
            # plot_img_and_save(persp_normals[1], "01e_pose_n.png", img_f)
            plot_img_and_save(decoded_semantics[idx], "02a_dec_s.png", img_f, vmax=5)
            Image.fromarray((decoded_normals[idx] * 255).numpy().astype(np.uint8)).save(
                os.path.join(img_f, "02b_dec_n.png")
            )

            # plot_img_and_save(decoded_normals[idx], "02b_dec_n.png", img_f)
            plot_img_and_save(best_sem_panos[idx], "03a_pano_s.png", img_f, vmax=vmax)
            # plot_img_and_save(best_norm_panos[idx], "03b_pano_n.png", img_f)

            Image.fromarray(vp_masks_est_top[idx].astype(np.uint8) * 255).save(os.path.join(img_f, "04a_mask_est.png"))
            Image.fromarray(vp_masks[0].astype(np.uint8) * 255).save(os.path.join(img_f, "04b_mask_gt.png"))
            Image.fromarray(vp_masks[1].astype(np.uint8) * 255).save(os.path.join(img_f, "04c_mask_est_rend.png"))

            plot_img_and_save(plan_image, "05a_map.png", img_f)
            plot_img_and_save(plan_image_zoom, "05b_map_zoom.png", img_f)

            if plot_score_map:
                plot_img_and_save(plan_image_score, "05c_map_score.png", img_f)
            # plot_img_and_save(comb_masks_1, "06a_mask.png", img_f, vmax=3)
            # plot_img_and_save(comb_masks_2, "06b_mask_rend.png", img_f, vmax=3)

            # plot_img_and_save(best_sem_panos[idx], "07a_bb_s.png", img_f, vmax=vmax, bb1=bb2d[0], bb2=bb2d[-1])
            # plot_img_and_save(best_sem_panos[idx], "07b_bb_rend_s.png", img_f, vmax=vmax, bb1=bb2d[0], bb2=bb2d[1])
            # plot_img_and_save(best_norm_panos[idx], "07d_bb_n.png", img_f, bb1=bb2d[0], bb2=bb2d[-1])
            # plot_img_and_save(best_norm_panos[idx], "07e_bb_rend_n.png", img_f, bb1=bb2d[0], bb2=bb2d[1])

            if refine > 0:
                plot_img_and_save(persp_semantics[2], "01c_pose_ref_s.png", img_f, vmax=vmax)
                # plot_img_and_save(persp_normals[2], "01f_pose_ref_n.png", img_f)
                Image.fromarray(vp_masks[2].astype(np.uint8) * 255).save(
                    os.path.join(img_f, "05d_mask_est_rend_ref.png")
                )

            # plot_img_and_save(vp_masks[2], "05d_mask_est_rend_ref.png", img_f)
            # plot_img_and_save(comb_masks_3, "06c_mask_rend_ref.png", img_f, vmax=3)
            # plot_img_and_save(
            #    best_sem_panos[idx], f"07c_bb_rend_ref_s.png", img_f, vmax=vmax, bb1=bb2d[0], bb2=bb2d[2]
            # )
            # plot_img_and_save(best_norm_panos[idx], f"07f_bb_rend_ref_n.png", img_f, bb1=bb2d[0], bb2=bb2d[2])
