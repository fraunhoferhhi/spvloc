import json
import os


def expand_dict_with_folders(folder_path_in, input_dict):
    expanded_dict = {}

    for key, x_numbers in input_dict.items():
        expanded_dict[key] = []

        for x_number in x_numbers:
            x_number = x_number.zfill(4)  # Ensure 4-digit format
            y_number = 0  # Start with Y = 0

            while True:
                folder_path = folder_path_in
                folder_name = f"scene_{x_number}{y_number}"
                folder_path = os.path.join(folder_path, folder_name)

                if os.path.exists(folder_path):
                    expanded_dict[key].append(int(x_number + str(y_number)))
                    y_number += 1

                else:
                    break

    return expanded_dict


def scenes_split(split, name="S3D", split_filename=None, dataset_folder=""):
    if not split_filename:
        split_path = "data/datasets/zind/zind_partition.json"
    else:
        split_path = os.path.join(dataset_folder, split_filename)
    if name == "Zillow" and os.path.exists(split_path):
        with open(split_path, "r") as json_file:
            input_split = json.load(json_file)
            if name == "Zillow":
                expanded_dict = expand_dict_with_folders(dataset_folder, input_split)
                return expanded_dict[split]
            else:
                return input_split[split]
    elif name == "S3D":
        splits = {
            "train": (
                list(range(0, 3000)),
                [
                    153,
                    335,
                    683,
                    1151,
                    1192,
                    1753,
                    1852,
                    2205,
                    2209,
                    2223,
                    2339,
                    2357,
                    2401,
                    2956,
                    2309,
                    278,
                    379,
                    1212,
                    1840,
                    1855,
                    2025,
                    2110,
                    2593,
                    3250,
                ],
            ),
            "val": (list(range(3000, 3250)), [2110, 3079, 3086, 3117, 3121, 3239, 3250]),
            "test": (list(range(3250, 3500)), []),
        }
    else:
        print("Unsupported dataset")
        exit()

    ids, to_remove = splits[split]
    ids = [i for i in ids if i not in to_remove]
    return ids
