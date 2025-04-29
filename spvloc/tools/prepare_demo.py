import os
import subprocess
import json
import os
import argparse

from spvloc.tools.download_pretrained_models import download_file

CHECKPOINT_PATH = "ckpt_zind_retrain_demo.ckpt"  # "ckpt_zind_retrain_large_angle"
ZIND_URL = "https://github.com/zillow/zind.git"
ZIND_FOLDER = "sample_tour"

CHECKPOINT_URL = os.path.join("https://cvg.hhi.fraunhofer.de/SPVLoc/", CHECKPOINT_PATH)
DOWNLOAD_PATH = os.path.join("data/pretrained_models", CHECKPOINT_PATH)


def prepare_dataset(clone_dir="external/zind", output_dir="external/zind_sample_tour"):

    if not os.path.exists(clone_dir):
        # Check the Git version
        git_version = subprocess.check_output(["git", "--version"]).decode().strip()
        git_version_number = git_version.split()[2].split(".")[0:3]  # Extract the version number without the suffix
        version_parts = [int(part) for part in git_version_number]

        # Function to check if Git version supports sparse-checkout
        def supports_sparse_checkout(version_parts):
            return version_parts[0] > 2 or (version_parts[0] == 2 and version_parts[1] >= 25)

        # If Git version supports sparse-checkout, use it, otherwise clone the whole repo
        if supports_sparse_checkout(version_parts):
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", ZIND_URL, clone_dir], check=True)
            subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=clone_dir, check=True)
            subprocess.run(["git", "sparse-checkout", "set", ZIND_FOLDER], cwd=clone_dir, check=True)
            subprocess.run(["git", "checkout"], cwd=clone_dir, check=True)
            print(f"Downloaded folder '{ZIND_FOLDER}' from the repository.")
        else:
            subprocess.run(["git", "clone", ZIND_URL, clone_dir], check=True)
            print(f"Cloned the entire repository as sparse-checkout is not supported in this Git version.")

    # Rename folder for compatibility with conversion script
    input_dir = os.path.join(clone_dir, ZIND_FOLDER)
    old_scene_path = os.path.join(input_dir, "000")
    new_scene_path = os.path.join(input_dir, "0000")

    if os.path.exists(old_scene_path):
        os.rename(old_scene_path, new_scene_path)
        print(f"Renamed folder from '000' to '0000'")
    else:
        print(f"Folder '{old_scene_path}' not found, skipping rename.")

    # Run conversion script
    conversion_command = [
        "python",
        "-m",
        "spvloc.tools.zind_to_s3d",
        "-i",
        input_dir,
        "-o",
        output_dir,
        "-m",
        "-idx",
        "0",
    ]

    try:
        subprocess.run(conversion_command, check=True)
        print("Conversion completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running conversion script: {e}")

    # Create dummy partition file
    partition_data = {"test": ["0000"]}
    partition_file_path = os.path.join(output_dir, "zind_partition.json")
    with open(partition_file_path, "w") as f:
        json.dump(partition_data, f, indent=2)

    print(f"Created partition file at: {partition_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretrained models with optional SSL verification.")
    parser.add_argument(
        "--verify", "-v", action="store_true", help="Enable SSL certificate verification (default: disabled)."
    )
    args = parser.parse_args()

    prepare_dataset()
    if not os.path.exists(DOWNLOAD_PATH):
        download_file(CHECKPOINT_URL, DOWNLOAD_PATH, args.verify)
