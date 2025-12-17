import json
import os
import requests
from urllib.parse import urlparse
from PIL import Image

# JSON files
train_json_file = 'species196L_train.json'
val_json_file = 'species196L_val.json'

# Output directories
if not os.path.exists('species196'):
    os.makedirs('species196')
train_output_directory = 'species196/train'
val_output_directory = 'species196/val'

# Failed downloads log file
failed_log_file = "failed_download.txt"

# Setup requests session (disable retries for fast fail)
session = requests.Session()

def is_valid_image(path):
    """Check if the downloaded file is a valid image."""
    try:
        with Image.open(path) as img:
            img.verify()  # verify file integrity
        return True
    except Exception:
        return False

def download_and_save_image(url, output_path):
    """Download an image and save it, validating afterwards."""
    try:
        response = session.get(url, timeout=(5, 10), stream=True)
        response.raise_for_status()

        # Ensure the response is an image
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            print(f"Not an image: {url}, Content-Type={content_type}")
            return False

        # Save in chunks
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Validate image integrity
        if not is_valid_image(output_path):
            print(f"Invalid or corrupted image: {url}")
            os.remove(output_path)
            return False

    except Exception as e:
        print(f"Download failed: {url}, Error: {str(e)}")
        return False

    return True

def process_updated_coco_file(input_json_file, output_directory):
    with open(input_json_file, 'r') as f:
        updated_coco_data = json.load(f)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    failed_downloads = []

    for image_info in updated_coco_data['images']:
        if 'category_name' not in image_info:
            continue

        category_name = image_info['category_name']
        origin_url = image_info['origin_url']
        _, file_extension = os.path.splitext(urlparse(origin_url).path)
        file_name = image_info['file_name'] + file_extension

        # Create category folder
        category_directory = os.path.join(output_directory, category_name)
        if not os.path.exists(category_directory):
            os.makedirs(category_directory)

        output_path = os.path.join(category_directory, file_name)

        # Skip if already exists
        if os.path.exists(output_path):
            print(f"Skipping (already exists): {output_path}")
            continue

        print(f"Downloading {origin_url} to {output_path}")
        success = download_and_save_image(origin_url, output_path)
        if not success:
            failed_downloads.append((origin_url, output_path))

    # Log failed downloads
    if failed_downloads:
        with open(failed_log_file, "a") as log_f:
            for url, path in failed_downloads:
                log_f.write(f"URL: {url}, Path: {path}\n")

        print(f"\nLogged {len(failed_downloads)} failed downloads to {failed_log_file}")

if __name__ == "__main__":
    print("Processing train JSON file...")
    process_updated_coco_file(train_json_file, train_output_directory)
    print("\nProcessing validation JSON file...")
    process_updated_coco_file(val_json_file, val_output_directory)
