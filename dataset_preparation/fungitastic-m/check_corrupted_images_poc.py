import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

# Directories to check
dirs = [
        "test/720p",
        "val/720p",    
        "train/720p",
    ]


def check_images_in_dir(directory):
    bad_files = []
    directory = Path(directory)

    for root, _, files in os.walk(directory):
        for fname in tqdm(files, desc=f"Processing {root}", unit="file"):
            fpath = Path(root) / fname
            try:
                # Try to open & fully load the image
                with Image.open(fpath) as img:
                    img.load()
            except (OSError, UnidentifiedImageError) as e:
                bad_files.append(fpath)
                print(f"❌ Corrupted: {fpath} ({e})")
    return bad_files

if __name__ == "__main__":

    all_bad = []
    for d in dirs:
        print(f"\n Checking {d} ...")
        bad = check_images_in_dir(d)
        all_bad.extend(bad)
        print(f"Found {len(bad)} corrupted images in {d}")

    print("\n=== SUMMARY ===")
    if all_bad:
        print(f"Total corrupted images: {len(all_bad)}")
        with open("corrupted_images.txt", "w") as f:
            for bf in all_bad:
                f.write(str(bf) + "\n")
        print("List saved to corrupted_images.txt")
    else:
        print("✅ No corrupted images found!")
