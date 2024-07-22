import os
import requests
import argparse
import urllib3

urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)

# Define the base URL and file paths
BASE_URL = "https://cvg.hhi.fraunhofer.de/SPVLoc/"
DOWNLOAD_PATH = "data/pretrained_models"

S3D_MODEL_PATH = BASE_URL + "ckpt_s3d.ckpt"
S3D_16_9_MODEL_PATH = BASE_URL + "ckpt_s3d_16_9.ckpt"
ZIND_MODEL_PATH = BASE_URL + "ckpt_zind.ckpt"

S3D_MODEL_PATH_OUT = os.path.join(DOWNLOAD_PATH, "ckpt_s3d.ckpt")
S3D_16_9_MODEL_PATH_OUT = os.path.join(DOWNLOAD_PATH, "ckpt_s3d_16_9.ckpt")
ZIND_MODEL_PATH_OUT = os.path.join(DOWNLOAD_PATH, "ckpt_zind.ckpt")


def download_file(url, output_path, verify):
    print(f"Download: {url}")
    response = requests.get(url, verify=verify)
    if response.status_code == 404:
        print("Warning: The requested resource was not found (404).")
    else:
        with open(output_path, "wb") as f:
            f.write(response.content)


def main(verify_ssl):
    """
    Args:
        verify_ssl (bool): Whether to verify SSL certificates.
    """
    if not os.path.exists(DOWNLOAD_PATH):
        os.makedirs(DOWNLOAD_PATH)

    if not os.path.exists(S3D_MODEL_PATH_OUT):
        download_file(S3D_MODEL_PATH, S3D_MODEL_PATH_OUT, verify_ssl)

    if not os.path.exists(S3D_16_9_MODEL_PATH_OUT):
        download_file(S3D_16_9_MODEL_PATH, S3D_16_9_MODEL_PATH_OUT, verify_ssl)

    if not os.path.exists(ZIND_MODEL_PATH_OUT):
        download_file(ZIND_MODEL_PATH, ZIND_MODEL_PATH_OUT, verify_ssl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretrained models with optional SSL verification.")
    parser.add_argument(
        "--verify", "-v", action="store_true", help="Enable SSL certificate verification (default: disabled)."
    )
    args = parser.parse_args()

    # Run the main function with the SSL verification setting
    main(verify_ssl=args.verify)
