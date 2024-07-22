import json
import os
import pickle
import sys
import torch

from spvloc.data.data_inference import scene_from_json
from spvloc.model.spvloc import scene_evaluation


def prepare_scene(json_path, scene_id, tmp_path="./temp"):
    f = open(json_path)
    structured3d_json = json.load(f)
    scene, _ = scene_from_json(structured3d_json, name=f"temp_{scene_id}", tmp_path=tmp_path)
    return structured3d_json, scene


def plot_data(results_folder):

    all_folders = sorted(os.listdir(results_folder))

    print_sota = False
    print_ablation = False
    print_summary = True

    # all_folders = ["results_ps1200_ch1400_ns1_rad1400_h300"]
    # all_plot_data = []
    for _, folder in enumerate(all_folders):
        experiment_path = os.path.join(results_folder, folder)
        all_files = os.listdir(experiment_path)
        pickle_files = [file for file in all_files if file.endswith(".pkl")]

        # Load each pickle file and print keys
        for _, pickle_file in enumerate(pickle_files):
            pickle_path = os.path.join(experiment_path, pickle_file)

            with open(pickle_path, "rb") as file:
                data = pickle.load(file)
                # all_scene_ids = np.array([i["scene_id"] for i in data])
                all_gt_poses = torch.concat([torch.tensor(i["gt_poses"]) for i in data])
                # print(all_gt_poses.shape)

                all_estimated_poses = torch.concat([torch.tensor(i["initial_estimates"]) for i in data])
                all_refined_poses = torch.concat([torch.tensor(i["refined_estimates"]) for i in data])

                top1, top3, _ = scene_evaluation(all_gt_poses, all_estimated_poses, False, True)
                top1_2d, top3_2d, _ = scene_evaluation(
                    all_gt_poses, all_estimated_poses, False, True, eval_2d_error=True
                )
                top1_refined, _, _ = scene_evaluation(all_gt_poses, all_refined_poses, False, False)
                # top1_refined_2d, _, _ = scene_evaluation(
                #     all_gt_poses, all_refined_poses, False, False, eval_2d_error=True
                # )

                def print_result(data):
                    for key, value in data.items():
                        if isinstance(value, list):
                            formatted_values = [f"{(val*100.0):.2f}" for val in value]
                            print(f"'{key}': {formatted_values},")
                        else:
                            print(f"'{key}': {value:.2f},")

                print(f"\n======= Experiment: {folder}/{pickle_file}   ========")

                if print_summary:
                    print("FINAL TOP 1 Result")
                    print_result(top1)
                    print("FINAL TOP 1 Result (refined)")
                    print_result(top1_refined)
                    print("FINAL TOP 1 (2D) Result")
                    print_result(top1_2d)
                    print("FINAL TOP 3 Result")
                    print_result(top3)
                    print("FINAL TOP 3 (2D) Result")
                    print_result(top3_2d)

                if print_sota:
                    # SoTA
                    d = []
                    d.append(top1_2d["err_t_median_2D"])
                    d.append(top1_2d["err_r_median_2D"])
                    d.append(top1_2d["recall_dist_2D"][0])
                    d.append(top1_2d["recall_dist_2D"][2])
                    d.append(top1_2d["recall_dist_2D"][3])
                    d.append(top1_2d["recall_dist_30deg_2D"][3])
                    d.append(top3_2d["recall_dist_2D"][3])

                    latex = ""
                    for idx, val in enumerate(d):
                        if idx > 1:
                            val = val * 100.0
                        latex += f"{(val):.2f} & "

                    print(latex)

                # top1 = top1_refined (Ablation refinement)
                if print_ablation:
                    e = []
                    e.append(top1["err_t_median"])
                    e.append(top1["err_r_median"])
                    e.append(top1["recall_dist"][0])
                    e.append(top1["recall_dist"][2])
                    e.append(top1["recall_dist"][3])
                    e.append(top1["recall_dist_30deg"][3])
                    e.append(top3["recall_dist"][3])

                    latex = ""
                    for idx, val in enumerate(e):
                        if idx > 1:
                            val = val * 100.0
                        latex += f"{(val):.2f} & "

                    print(latex)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_data.py results_folder")
        print("Input the path of the results folder.")
        sys.exit(1)
    results_folder = sys.argv[1]
    plot_data(results_folder)
