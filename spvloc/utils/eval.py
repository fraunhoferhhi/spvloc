import os
import pickle
from datetime import datetime

import torch

from ..utils.pose_utils import get_pose_distances_batch


def index_of_val(arr, value):
    return (arr == value).nonzero()


def scene_evaluation(gt_poses, decoded_poses, eval_gt=False, eval_top3=True, eval_2d_error=False):
    gt_poses = gt_poses.cpu().float().unsqueeze(1)

    err_t, err_r, err_t_2d, err_yaw = get_pose_distances_batch(gt_poses, decoded_poses)
    if eval_2d_error:
        err_t = err_t_2d
        err_r = err_yaw

    err_t_top1 = err_t[:, 0]
    err_r_top1 = err_r[:, 0]

    def eval(err_r, err_t, eval_2d_error):
        suffix = "_2D" if eval_2d_error else ""

        bs_img = len(err_t)
        smaller_10deg = err_r < 10.0
        smaller_30deg = err_r < 30.0
        smaller_5deg = err_r < 5.0

        smaller_100cm = err_t < 1000.0
        smaller_50cm = err_t < 500.0
        smaller_25cm = err_t < 250.0
        smaller_10cm = err_t < 100.0

        def get_recalls(smaller_10cm, smaller_25cm, smaller_50cm, smaller_100cm, angle_threshold=True):
            recall_100cm = (angle_threshold & smaller_100cm).sum() / float(bs_img)
            recall_50cm = (angle_threshold & smaller_50cm).sum() / float(bs_img)
            recall_25cm = (angle_threshold & smaller_25cm).sum() / float(bs_img)
            recall_10cm = (angle_threshold & smaller_10cm).sum() / float(bs_img)
            return [recall_10cm.item(), recall_25cm.item(), recall_50cm.item(), recall_100cm.item()]

        recall_no_angle = get_recalls(smaller_10cm, smaller_25cm, smaller_50cm, smaller_100cm)
        recall_5deg = get_recalls(smaller_10cm, smaller_25cm, smaller_50cm, smaller_100cm, smaller_5deg)
        recall_10deg = get_recalls(smaller_10cm, smaller_25cm, smaller_50cm, smaller_100cm, smaller_10deg)
        recall_30deg = get_recalls(smaller_10cm, smaller_25cm, smaller_50cm, smaller_100cm, smaller_30deg)

        err_t_median = err_t[smaller_100cm].median() / 10.0
        err_r_median = err_r[smaller_100cm].median()

        result = {
            f"err_t_median{suffix}": err_t_median.item(),
            f"err_r_median{suffix}": err_r_median.item(),
            f"recall_dist{suffix}": recall_no_angle,
            f"recall_dist_5deg{suffix}": recall_5deg,
            f"recall_dist_10deg{suffix}": recall_10deg,
            f"recall_dist_30deg{suffix}": recall_30deg,
        }

        return result

    top_gt = {}
    if eval_gt:
        err_t_gt = err_t[:, -1]
        err_r_gt = err_r[:, -1]
        top_gt = eval(err_r_gt, err_t_gt, eval_2d_error)

    top1 = eval(err_r_top1, err_t_top1, eval_2d_error)

    top3 = {}
    if eval_top3:
        err_t_top3, err_t_top3_index = err_t[:, :3].min(dim=1)
        err_r_top3 = err_r[torch.arange(err_r.shape[0]), err_t_top3_index]
        top3 = eval(err_r_top3, err_t_top3, eval_2d_error)

    return top1, top3, top_gt


def get_pickle_name(config, prefix="results"):
    pickle_name = (
        prefix
        + "_ps"
        + str(int(config.TEST.POSE_SAMPLE_STEP))
        + "_ch"
        + str(int(config.TEST.CAMERA_HEIGHT))
        + "_ns"
        + str(int(config.TRAIN.NUM_NEAR_SAMPLES))
        + "_rad"
        + str(int(config.TRAIN.PANO_POSE_OFFSETS[0] * 1000))
        + "_h"
        + str(int(config.TRAIN.PANO_POSE_OFFSETS[2] * 1000))
    )
    if config.TEST.ADVANCED_POSE_SAMPLING:
        pickle_name += "_" + "adv"

    return pickle_name


def save_eval_poses(config, data_to_save):
    # convert to numpy
    for scene in data_to_save:
        for key, value in scene.items():
            if isinstance(value, torch.Tensor):
                scene[key] = value.numpy()
    result_folder = os.path.join(config.OUT_DIR, "results")
    now = datetime.now()
    short_datetime_str = now.strftime("%Y%m%d%H%M")
    pickle_name = get_pickle_name(config)

    test_folder = os.path.join(result_folder, pickle_name)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    with open(os.path.join(test_folder, f"result_{short_datetime_str}.pkl"), "wb") as f:
        pickle.dump(data_to_save, f)
