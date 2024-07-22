import pickle
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.data_inference import Structured3DPlans_Inference
from ..data.dataset import Structured3DPlans_Perspective
from ..utils.eval import save_eval_poses, scene_evaluation
from ..utils.plot_utils import plot_overview
from ..utils.pose_utils import decode_pose_spvloc
from ..utils.projection import projects_onto_floor_efficient
from ..utils.render import render_scene_batched
from .losses import (
    FocalLoss,
    MultiLossCriterion,
    circular_distance_loss,
    cosine_similarity_loss,
    cosine_similarity_loss_masked,
    cross_entropy_loss_masked,
)
from .modules import (
    BoundingBoxDecoder,
    FeatureCompression,
    ImageModule,
    LayoutDecoder_V2,
    PanoFeatExtract,
    PoseEstimatorMLP,
)


def build_dataloader(config, split, shuffle=True, test=False):

    batch_size = None if test else config.TRAIN.BATCH_SIZE

    num_workers = config.SYSTEM.NUM_WORKERS

    # pin_memory is not needed during the test
    pin_memory = not test

    if test:
        dataset = Structured3DPlans_Inference(config, split, visualise=True)
    else:
        dataset = Structured3DPlans_Perspective(config, split, visualise=True)

    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # worker_init_fn=worker_init_fn,
    )
    return dataloader


class PerspectiveImageFromLayout(pl.LightningModule):
    def __init__(self, config):

        super(PerspectiveImageFromLayout, self).__init__()

        self.save_hyperparameters()

        self.desc_length = config.MODEL.DESC_LENGTH
        self.config = config

        pose_mlp_dim = 640  # TODO: hyperparameter

        self.query_embedder = ImageModule(config)
        self.layout_decoder = LayoutDecoder_V2(config)
        self.panorama_embedder = PanoFeatExtract(config)
        self.bounding_box_decoder = BoundingBoxDecoder(config)
        self.pano_bb_compress = FeatureCompression(512)
        self.pose_estimator_mlp = PoseEstimatorMLP(pose_mlp_dim)

        # Losses
        self.focal_loss = FocalLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if config.TRAIN.PERSP_FROM_PANO_RANGE_OPTIMIZE_EULER:
            self.gnn_loss = MultiLossCriterion(q_loss_fn=circular_distance_loss, sax=0.0, saq=0.0, learn_beta=True)
        else:
            self.gnn_loss = MultiLossCriterion(sax=0.0, saq=0.0, learn_beta=True)

        self.all_estimation_results = []

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.TRAIN.INITIAL_LR
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimiser, self.config.TRAIN.LR_MILESTONES, self.config.TRAIN.LR_GAMMA
        )
        return [optimiser], [scheduler]

    def common_step(self, batch, key="train"):
        reference_histogram = batch["reference_histogram"]

        if self.config.TRAIN.FILTER_PANO_SAMPLES and key == "train":

            # ECCV fix: filter invalid rooms (will only be triggered if _get_room_data is adapted)
            index_mask = (batch["room_idx"] != -1).squeeze()

            valid_filter = reference_histogram[:, 0] < 0.05  # filter invalid
            three_classes_filter = (reference_histogram[:, 1:] > 0.002).sum(
                axis=-1
            ) > 2  # at least three classes should be present
            door_filter = reference_histogram[:, 4] < 0.7  # not more than half of the image should be a door
            window_filer = reference_histogram[:, 5] < 0.7  # not more than half of the image should be a window
            valid_mask = valid_filter * three_classes_filter * door_filter * window_filer

            index_mask = torch.logical_and(valid_mask, index_mask)
            valid_samples = index_mask.sum() / reference_histogram.shape[0]

            # filter data
            if valid_samples > 0:
                filtered_batch = {}
                for filter_key, value in batch.items():
                    if torch.is_tensor(value):
                        filtered_value = value[index_mask]
                        filtered_batch[filter_key] = filtered_value
                batch = filtered_batch
        else:
            valid_samples = torch.tensor(1.0)

        # compute query and reference embeddings
        query_image = batch["image"]
        bounding_mask = batch["bounding_mask"]
        annotations = batch["boundingbox_2d"]
        pano_depth = batch["sampled_depth"].to(self.device)
        pano_normals = batch["sampled_normals"].to(self.device)
        pano_semantics = batch["sampled_semantics"].to(self.device)

        bs = pano_semantics.shape[0]
        samples = pano_semantics.shape[1]

        pano_map = self.prepare_pano_map(pano_normals, pano_semantics, pano_depth)

        query_embed, features = self.query_embedder(query_image)
        features = features.unsqueeze(1).repeat([1, samples, 1, 1, 1])
        pano_map = torch.flatten(pano_map, end_dim=1)
        features = torch.flatten(features, end_dim=1)
        bounding_mask = torch.flatten(bounding_mask, end_dim=1)
        annotations = torch.flatten(annotations, end_dim=1)

        annotations = annotations.unsqueeze(1)

        pano_embed = self.panorama_embedder.forward(pano_map)

        if self.config.MODEL.PREDICT_POSES:
            (
                classifications,
                regression,
                anchors,
                segmentation,
                _,
                xcors,
            ) = self.bounding_box_decoder.forward_best_anchors(pano_embed, features)
            segmentation = torch.squeeze(segmentation, 1)
        else:
            classifications, regression, anchors, segmentation, xcors = self.bounding_box_decoder.forward(
                pano_embed, features
            )
            segmentation = torch.squeeze(segmentation, 1)

        # Calculate classification and regression loss
        classification_loss, regression_loss = self.focal_loss.forward(
            classifications, regression, anchors, annotations
        )

        if self.config.TRAIN.IGNORE_MASK_BCE:
            mask_bce_loss = torch.tensor(0.0)
        else:
            mask_bce_loss = self.bce_loss.forward(segmentation, bounding_mask)

        reference_normals = batch["persp_normal"]
        reference_semantics = batch["persp_semantics"]

        reference_map = torch.cat([reference_normals, reference_semantics], axis=1).unsqueeze(1)

        if self.config.TRAIN.RETRAIN_DECODER:
            # retrain also the layout decoder to better capture the inputs
            layout_ch = 3

            query_decoded = self.layout_decoder(query_embed)
            decoded_layout = query_decoded[:, 0:layout_ch]
            decoded_semantics = query_decoded[:, layout_ch:]

            target_map = F.interpolate(reference_map.squeeze(1).detach().clone(), self.config.MODEL.DECODER_RESOLUTION)

            target_layout = target_map[:, 0:layout_ch]
            target_semantics = target_map[:, layout_ch:].squeeze(1)

            if self.config.MODEL.LEARN_ASPECT:
                visibility_mask = batch["aspect_visibility_mask"]
                loss_decode_layout = cosine_similarity_loss_masked(decoded_layout, target_layout, visibility_mask)
                loss_decode_semantics = cross_entropy_loss_masked(
                    decoded_semantics, target_semantics.long(), visibility_mask
                )
            else:
                loss_decode_layout = cosine_similarity_loss(decoded_layout, target_layout)
                loss_decode_semantics = self.cross_entropy_loss(decoded_semantics, target_semantics.long())
        else:
            loss_decode_semantics = torch.tensor(0.0)
            loss_decode_layout = torch.tensor(0.0)

        if self.config.MODEL.PREDICT_POSES:
            # far samples should not be passed to the pose head
            if self.config.TRAIN.NUM_FAR_SAMPLES > 0:
                xcors = xcors.view(-1, samples, *xcors.shape[-3:])
                samples -= self.config.TRAIN.NUM_FAR_SAMPLES
                xcors = torch.flatten(xcors[:, :samples], end_dim=1)

            combined_poses = batch["combined_poses"]

            pose_features = self.compress_xcors(xcors)  # shape [bs, 640]

            if self.config.MODEL.LEARN_FOV:
                pose_features_normalized = torch.nn.functional.normalize(pose_features, p=2, dim=-1)
                persp_fov = batch["persp_fov"].to(self.device).repeat_interleave(samples).unsqueeze(1).float()
                pose_features = pose_features_normalized * persp_fov

            target_r = combined_poses[:, [-1], 3:]
            target_abs_t = combined_poses[:, [-1], :3] - combined_poses[:, :samples, :3]

            xyz_out, rot_abs_out = self.pose_estimator_mlp(pose_features)
            target_r = target_r.repeat(1, samples, 1)
            rot_abs_out = torch.reshape(rot_abs_out, [bs, samples, 3])
            xyz_out = torch.unsqueeze(xyz_out, 0)

            loss, _, r_loss, t_loss_abs, _ = self.gnn_loss.forward(
                pred_abs_T=xyz_out,
                target_abs_T=target_abs_t,
                pred_R=rot_abs_out,
                targ_R=target_r,
                classification_loss=classification_loss,
                regression_loss=regression_loss,
                mask_bce_losss=mask_bce_loss,
                loss_decode_layout=loss_decode_layout,
                loss_decode_semantics=loss_decode_semantics,
            )

        else:
            loss = self.gnn_loss.forward_no_pose(
                classification_loss, regression_loss, mask_bce_loss, loss_decode_layout, loss_decode_semantics
            )
            t_loss_abs = torch.tensor(0.0)
            r_loss = torch.tensor(0.0)

        stats_to_log = {
            "{}/loss".format(key): loss.item(),
            "{}/classification_loss".format(key): classification_loss.item(),
            "{}/regression_loss".format(key): regression_loss.item(),
            "{}/mask_bce_loss".format(key): mask_bce_loss.item(),
            "{}/loss_decode_semantics".format(key): loss_decode_semantics.item(),
            "{}/loss_decode_layout".format(key): loss_decode_layout.item(),
            "{}/translation_loss_abs".format(key): t_loss_abs.item(),
            "{}/rotation_loss".format(key): r_loss.item(),
            "{}/valid_samples".format(key): valid_samples.item(),
        }

        self.log_dict(stats_to_log, on_step=True, on_epoch=True)

        return loss, stats_to_log

    def training_step(self, batch, batch_idx):
        loss, stats_to_log = self.common_step(batch, "train")
        self.log_command_line(stats_to_log, batch_idx)
        return {"loss": loss, "log": stats_to_log}  # , "log": stats_to_log

    def validation_step(self, batch, batch_idx):
        loss, stats_to_log = self.common_step(batch, "validation")
        self.log_command_line(stats_to_log, batch_idx)
        return {"loss": loss, "log": stats_to_log}

    def test_step(self, batch, batch_idx):
        gt_poses = batch["gt_poses"].cpu()
        scene_id = batch["scene_id"].cpu()
        combined_poses = batch["combined_poses"].cpu()

        refine_iterations = self.config.POSE_REFINE.MAX_ITERS
        export_plots = self.config.TEST.SAVE_PLOT_OUTPUT
        export_all = self.config.TEST.SAVE_PLOT_DETAILS

        prepare_plot = self.config.TEST.PLOT_OUTPUT or export_plots

        eval_gt_map_point = self.config.TEST.EVAL_GT_MAP_POINT
        eval_top3 = self.config.TEST.EVAL_TOP3
        top_n = 3 if self.config.TEST.EVAL_TOP3 else 1

        output, output_plot = self.test_step_full_epoch(
            batch, refine_iterations, top_n=top_n, prepare_plot=prepare_plot
        )

        estiamted_poses = output["initial_estimates"]

        top1, top3, _ = scene_evaluation(gt_poses, estiamted_poses, eval_gt=eval_gt_map_point, eval_top3=eval_top3)

        stats_to_log = {"scene_id": scene_id.item(), "test/top1": top1, "test/top3": top3}

        if prepare_plot:
            merged_output = {**output, **output_plot}
            plot_overview(
                batch,
                merged_output,
                self.config,
                0,
                3,
                refine_iterations > 0,
                save_overview=export_plots,
                save_all=export_all,
            )

        output["combined_poses"] = combined_poses
        output["gt_poses"] = gt_poses
        output["scene_id"] = scene_id

        self.all_estimation_results.append(output)

        self.log_command_line(stats_to_log, batch_idx)
        return {"loss": torch.tensor([0.0]), "log": stats_to_log}

    def train_dataloader(self):
        return build_dataloader(self.config, "train")

    def val_dataloader(self):
        return build_dataloader(self.config, "val", shuffle=False)

    def test_dataloader(self):
        # Convenient to make "test" actually the validation set so you can recheck val acc at any point
        if self.config.TEST.VAL_AS_TEST:
            return build_dataloader(self.config, "val", shuffle=False, test=True)
        return build_dataloader(self.config, "test", shuffle=False, test=True)

    def on_test_epoch_end(self):
        all_gt_poses = torch.concat([i["gt_poses"] for i in self.all_estimation_results])
        all_estimated_poses = torch.concat([i["initial_estimates"] for i in self.all_estimation_results])
        all_refined_poses = torch.concat([i["refined_estimates"] for i in self.all_estimation_results])
        refine_iterations = self.config.POSE_REFINE.MAX_ITERS
        eval_gt_map_point = self.config.TEST.EVAL_GT_MAP_POINT
        eval_top3 = self.config.TEST.EVAL_TOP3

        # save poses
        save_eval_poses(self.config, self.all_estimation_results)

        top1, top3, top_gt = scene_evaluation(
            all_gt_poses, all_estimated_poses, eval_gt=eval_gt_map_point, eval_top3=eval_top3
        )
        top1_2d, top3_2d, _ = scene_evaluation(
            all_gt_poses, all_estimated_poses, False, eval_2d_error=True, eval_top3=eval_top3
        )

        if refine_iterations > 0:
            top1_refined, _, _ = scene_evaluation(all_gt_poses, all_refined_poses, False, eval_top3=eval_top3)
            top1_refined_2d, _, _ = scene_evaluation(
                all_gt_poses, all_refined_poses, False, eval_2d_error=True, eval_top3=eval_top3
            )

        def print_result(data):
            for key, value in data.items():
                if isinstance(value, list):
                    formatted_values = [f"{(val*100.0):.2f}" for val in value]
                    print(f"'{key}': {formatted_values},")
                else:
                    print(f"'{key}': {value:.2f},")

        print("FINAL TOP 1 Result")
        print_result(top1)
        if refine_iterations > 0:
            print("FINAL TOP 1 Result (refined)")
            print_result(top1_refined)

        print("FINAL TOP 1 (2D) Result")
        print_result(top1_2d)

        if refine_iterations > 0:
            print("FINAL TOP 1 (2D) Result (refined)")
            print_result(top1_refined_2d)

        if eval_top3:
            print("FINAL TOP 3 Result")
            print_result(top3)
            print("FINAL TOP 3 (2D) Result")
            print_result(top3_2d)

        if eval_gt_map_point:
            print("FINAL Known Map Point Result")
            print_result(top_gt)

        return None

    def compress_xcors(self, xcors):
        xcors_compressed = self.pano_bb_compress(xcors)
        return xcors_compressed

    def prepare_pano_map(self, pano_normals, pano_semantics, pano_depth, dim=-3):
        encode_depth = self.config.MODEL.PANO_ENCODE_DEPTH
        encode_normals = self.config.MODEL.PANO_ENCODE_NORMALS
        encode_semantics = self.config.MODEL.PANO_ENCODE_SEMANTICS

        normalize_panos = self.config.MODEL.NORMALIZE_PANO_INPUT

        if normalize_panos:
            pano_semantics = pano_semantics / float(self.config.MODEL.DECODER_SEMANTIC_CLASSES - 1)
            if encode_depth:
                pano_depth = pano_depth / self.config.MODEL.PANO_MAX_DEPTH

        if not any([encode_depth, encode_normals, encode_semantics]):
            raise ValueError("At least one encoding flag should be True.")

        # Concatenate along the specified dimension
        # pano_depth * encode_depth,
        concatenated_output = []

        if encode_depth:
            concatenated_output.append(pano_depth)

        if encode_normals:
            concatenated_output.append(pano_normals)

        if encode_semantics:
            concatenated_output.append(pano_semantics)
        # Concatenate along the specified dimension
        return torch.cat(concatenated_output, dim=dim).float()

    def split_chunks(
        self,
        pano_map,
        features_selected,
        max_size,
        return_xcors_only=False,
        return_bbs=False,
    ):
        num_chunks = int(features_selected.shape[0] / max_size + 1.0)

        results_score = []
        results_bbs = []
        results_masks = []

        features_selected_chunks = features_selected.chunk(num_chunks, dim=0)

        pano_encode_time_start = time.time()
        bs_img = int(features_selected.shape[0] / pano_map.shape[0])

        pano_map = torch.cat(
            [self.panorama_embedder.forward(chunk) for chunk in pano_map.chunk(num_chunks, dim=0)], dim=0
        )
        # pano_map = self.panorama_embedder.forward(pano_map)

        # TODO: Idea to save memory, do not flatten this since the memory is not contigous,
        # instead copy the tensor with needed size manually.
        pano_map = pano_map.unsqueeze(1).expand(pano_map.shape[0], bs_img, *pano_map.shape[1:]).flatten(end_dim=1)

        pano_encode_time = time.time() - pano_encode_time_start

        correlation_time_start = time.time()

        xcors = torch.zeros(
            features_selected.shape[0], 512, 8, 18, device=pano_map.device
        )  # TODO: Calculate size, maybe prealloc is not needed
        pano_map_chunks = pano_map.chunk(num_chunks, dim=0)
        idx = 0
        for pano_chunk, features_chunk in zip(pano_map_chunks, features_selected_chunks):
            chunk_size = pano_chunk.shape[0]

            if return_xcors_only:

                xcors[idx : idx + chunk_size] = self.bounding_box_decoder.forward_xcors_only(
                    pano_chunk, features_chunk
                )
            elif return_bbs:
                (
                    max_score,
                    anchors_pred,
                    segmentation,
                    _,
                    xcors[idx : idx + chunk_size],
                ) = self.bounding_box_decoder.forward_regress_boxes(pano_chunk, features_chunk)
                results_score.append(max_score)
                results_bbs.append(anchors_pred)
                results_masks.append(segmentation.cpu())
            else:
                max_score, xcors[idx : idx + chunk_size] = self.bounding_box_decoder.forward_score_and_corr(
                    pano_chunk, features_chunk, test=False
                )
                results_score.append(max_score)

            idx += chunk_size

        results_score = torch.concat(results_score) if len(results_score) > 0 and results_score[0].dim() != 0 else []
        results_masks = torch.concat(results_masks) if len(results_masks) > 0 and results_masks[0].dim() != 0 else []
        results_bbs = torch.concat(results_bbs) if len(results_bbs) > 0 else []

        correlation_time = time.time() - correlation_time_start

        return results_score, xcors, results_bbs, results_masks, (pano_encode_time, correlation_time)

    def test_step_full_epoch(self, batch, refine_iterations=1, top_n=3, top_idx=0, prepare_plot=False, scene=None):
        start_time = time.time()
        pano_normals = batch["sampled_normals"].to(self.device)
        pano_semantics = batch["sampled_semantics"].to(self.device)
        pano_depth = batch["sampled_depth"].to(self.device)
        combined_poses = batch["combined_poses"].to(self.device)
        query_image = batch["image"].to(self.device)
        # sampled_poses = batch["sampled_poses"].to(self.device)
        eval_gt_map_point = self.config.TEST.EVAL_GT_MAP_POINT
        cam_fov = torch.arctan(1.0 / batch["cams"][0, 0, 0]) * 2.0  # assume all cams are the sae
        refine = refine_iterations > 0
        filter_by_score = False

        max_split_size = 100  # 246

        pano_map = self.prepare_pano_map(pano_normals, pano_semantics, pano_depth)

        query_time_start = time.time()
        query_embed, features = self.query_embedder(query_image)
        query_time = time.time() - query_time_start

        if prepare_plot:
            query_decoded = self.layout_decoder(query_embed)  # [bs, 3 + num_classes, h_dec, w_dec]
            query_layout_decoded = query_decoded[:, :3]
            query_semantics_decoded = query_decoded[:, 3:]
            query_layout_decoded = query_layout_decoded / torch.norm(query_layout_decoded, dim=1, keepdim=True)

        bs = pano_semantics.shape[0]

        bs_img = query_image.shape[0]

        if top_n == -1:
            # Estimate a pose for each panorama reference.
            top_n = bs

        features_selected = features.unsqueeze(0).repeat([bs, 1, 1, 1, 1])
        features_selected = torch.flatten(features_selected, end_dim=1)

        max_score, xcors, bbs_est, vp_masks_est, (pano_encode_time, correlation_time) = self.split_chunks(
            pano_map,
            features_selected,
            max_split_size,
            False,
            return_bbs=prepare_plot,
        )

        def extend_tensor(x_in):
            # add last line if bs is smaller than top_n
            while x_in.size(0) < top_n:
                x_in = torch.cat([x_in, x_in[-1].unsqueeze(0)], dim=0)
                # print(f"Too few samples. Less than {top_n} references")
            return x_in

        max_score = max_score.reshape(bs, bs_img)

        max_score_idx = torch.argsort(max_score, dim=0, descending=True)[:top_n]
        max_scores_top = max_score[max_score_idx, torch.arange(bs_img).unsqueeze(0).expand_as(max_score_idx)]

        max_score = extend_tensor(max_score)
        max_score_idx = extend_tensor(max_score_idx)
        max_scores_top = extend_tensor(max_scores_top)

        max_score_idx = torch.transpose(max_score_idx, 0, 1)
        max_scores_top = torch.transpose(max_scores_top, 0, 1)

        if prepare_plot:
            bbs_est = extend_tensor(bbs_est.reshape(bs, bs_img, 4))
            while bbs_est.size(0) < top_n:
                bbs_est = torch.cat([bbs_est, bbs_est[-1].unsqueeze(0)], dim=0)
            bbs_est_top = bbs_est[max_score_idx, torch.arange(bs_img).unsqueeze(1).expand(bs_img, top_n)]
            vp_masks_est = extend_tensor(torch.unflatten(vp_masks_est, 0, (bs, bs_img)))
            vp_masks_est = vp_masks_est[
                max_score_idx.cpu(), torch.arange(bs_img).unsqueeze(1).expand(bs_img, top_n).cpu()
            ]

        if eval_gt_map_point:
            gt_map_points = batch["gt_map_points"].to(self.device)
            max_score_idx = torch.concat([max_score_idx, gt_map_points.unsqueeze(1).to(torch.int64)], axis=-1)
            eval_samples = top_n + 1
        else:
            eval_samples = top_n

        xcors = extend_tensor(torch.unflatten(xcors, 0, (bs, bs_img)))
        xcors = xcors[max_score_idx, torch.arange(bs_img).unsqueeze(1).expand(bs_img, eval_samples)].flatten(end_dim=1)

        pose_time_start = time.time()
        pose_features = self.compress_xcors(xcors)
        combined_poses_local = torch.index_select(combined_poses, 0, max_score_idx.flatten())
        if self.config.MODEL.LEARN_FOV:
            pose_features_normalized = torch.nn.functional.normalize(pose_features, p=2, dim=-1)
            pose_features = pose_features_normalized * cam_fov.to(self.device).float()

        xyz_out, rot_abs_out = self.pose_estimator_mlp(pose_features)
        xyz_abs_t_out = xyz_out + combined_poses_local[:, 0, :3]

        poses = torch.cat([xyz_abs_t_out.squeeze(1), rot_abs_out.squeeze(1)], dim=1)
        decoded_poses = []
        for pose in poses:
            pose = decode_pose_spvloc(pose.squeeze(0)[:3], pose.squeeze(0)[3:])
            decoded_poses.append(pose)

        decoded_poses = torch.stack(decoded_poses)
        decoded_poses = decoded_poses.reshape(bs_img, eval_samples, 4, 4)
        pose_time = time.time() - pose_time_start

        if (prepare_plot or refine) and scene is None:
            encoded_scene = batch["encoded_data"]
            encoding_length = batch["encoding_length"]
            scene = encoded_scene[:encoding_length]
            scene_encoded_np = scene.cpu().numpy()
            scene_encoded_bytes = np.frombuffer(scene_encoded_np, dtype=np.uint8)
            scene = pickle.loads(scene_encoded_bytes)

        if scene is not None:
            floor = scene["floor_planes"]

        initial_estimates = decoded_poses.clone()
        refined_estimates = decoded_poses[:, top_idx : top_idx + 1, :, :]  # copy to

        for refine_it in range(refine_iterations):
            print(f"Refine {refine_it}")
            t_est = refined_estimates[:, 0, :3, 3]
            near_room_indices = projects_onto_floor_efficient(t_est, floor)
            estimated_poses_and_rooms = []
            for idx, near_room_index in enumerate(near_room_indices):
                estimated_position = refined_estimates[idx, 0, :3, 3].cpu().detach().numpy().astype(np.float64)
                estimated_poses_and_rooms.append((estimated_position, int(near_room_index)))

            estimated_pano_normals = render_scene_batched(
                self.config, scene["geometry"], estimated_poses_and_rooms, scene["materials"], mode="depth_normal"
            )
            estimated_normals = torch.tensor(estimated_pano_normals[..., :3]).permute(0, 3, 1, 2)
            estimated_depth = torch.tensor(estimated_pano_normals[..., 3:]).permute(0, 3, 1, 2)

            estimated_semantics = render_scene_batched(
                self.config,
                scene["geometry_clipped"],
                estimated_poses_and_rooms,
                scene["materials_clipped"],
                mode="semantic",
            )

            estimated_semantics = torch.tensor(estimated_semantics).unsqueeze(1)

            pano_map_refine = self.prepare_pano_map(
                estimated_normals, estimated_semantics, estimated_depth, dim=-3
            ).to(self.device)

            max_score_refine, xcors_refine, _, _, _ = self.split_chunks(
                pano_map_refine, features, max_split_size, False
            )

            pose_features_refine = self.compress_xcors(xcors_refine)
            if self.config.MODEL.LEARN_FOV:
                pose_features_normalized = torch.nn.functional.normalize(pose_features_refine, p=2, dim=-1)
                pose_features_refine = pose_features_normalized * cam_fov.to(self.device).float()

            xyz_out_refine, rot_abs_out = self.pose_estimator_mlp(pose_features_refine)

            xyz_abs_t_out_refine = xyz_out_refine.cpu().detach() + (t_est / 1000.0)
            poses = torch.cat([xyz_abs_t_out_refine.squeeze(1), rot_abs_out.squeeze(1).cpu().detach()], dim=1)
            decoded_poses_refine = []
            for pose in poses:
                pose = decode_pose_spvloc(pose.squeeze(0)[:3], pose.squeeze(0)[3:])
                decoded_poses_refine.append(pose)

            decoded_poses_refine = torch.stack(decoded_poses_refine)
            decoded_poses_refine = decoded_poses_refine.reshape(bs_img, 1, 4, 4)  # .repeat(

            t_est_new = decoded_poses_refine[:, 0, :3, 3]
            new_room_indices = projects_onto_floor_efficient(t_est_new, floor)
            index_filter = ((near_room_indices == new_room_indices) & new_room_indices >= 0) & near_room_indices >= 0

            if refine_it == 0:
                best_score = max_score_refine  # take score of first refinement epoch
            else:
                score_filter = (
                    (max_score_refine > best_score).detach().cpu().numpy()
                )  # only update if new score is better than the previous one
                best_score[score_filter] = max_score_refine[score_filter]
            if filter_by_score and refine_it > 0:
                index_filter = index_filter & score_filter
            # refined_estimates[index_filter] = decoded_poses[index_filter, 0:1, :, :]
            refined_estimates[index_filter] = decoded_poses_refine[index_filter]

        scene_id = batch["scene_id"].cpu().numpy() if "scene_id" in batch else ""
        print(
            f"Processing scene {scene_id} with {bs_img} images "
            f"and {bs} references took {time.time() - start_time} seconds."
        )

        if "render_time" in batch:
            timing_dict = {
                "no_references": bs,
                "no_images": bs_img,
                "render": batch["render_time"].cpu().numpy(),
                "sampling": batch["sampling_time"].cpu().numpy(),
                "query": query_time,
                "pano_encode": pano_encode_time,
                "correlation": correlation_time,
                "pose_head": pose_time,
            }
        else:
            timing_dict = {}

        retrieved_positions = torch.unflatten(combined_poses_local, 0, (bs_img, eval_samples))[:, :, 0, :3]

        output = {}
        output["max_score_idx"] = max_score_idx.cpu().detach()
        output["initial_estimates"] = initial_estimates.cpu().detach()
        output["refined_estimates"] = refined_estimates.cpu().detach()
        output["xyz_out"] = xyz_out.cpu().detach()

        # scores
        output["max_scores"] = max_scores_top.cpu().detach()
        output["max_score"] = max_score.cpu().detach()

        output["retrieved_positions"] = retrieved_positions.cpu().detach()
        output["timing"] = timing_dict

        if refine:
            output["xyz_out_refine"] = xyz_out_refine.cpu().detach()
            if isinstance(max_score_refine, list):
                output["max_score_refine"] = torch.tensor([], dtype=torch.float32)
            else:
                output["max_score_refine"] = max_score_refine.cpu().detach()

        output_plot = {}

        if prepare_plot:
            output_plot["bbs_est"] = bbs_est_top.cpu().detach()
            output_plot["bb"] = bbs_est.cpu().detach()
            output_plot["vp_masks_est"] = vp_masks_est.cpu().detach()
            output_plot["query_semantics_decoded"] = query_semantics_decoded.cpu().detach()
            output_plot["query_layout_decoded"] = query_layout_decoded.cpu().detach()
            output_plot["scene"] = scene

        return output, output_plot

    def log_command_line(self, stats_to_log, batch_idx):
        if (batch_idx % self.config.CONSOLE_LOG_INVERVAL) == 0:
            print(stats_to_log)
