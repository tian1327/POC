"""
Examples:
python pregenerate_reference_images.py \
  --class-json ../data/semi-aves/semi-aves_labels.json \
  --k 8 \
  --dataset semi-aves \
  --seed-file ../data/semi-aves/fewshot16_seed1.txt \
  --image-root ../../../../work/nvme/bfln/dataset_poc/semi-aves \
  --output-dir ../../../../work/nvme/bfln/dataset_poc/semi-aves

python pregenerate_reference_images.py \
  --class-json ../data/fungitastic-m/fungitastic-m_labels.json \
  --k 8 \
  --dataset fungitastic-m \
  --seed-file ../data/fungitastic-m/fewshot16_seed1.txt \
  --image-root ../../../../work/nvme/bfln/dataset_poc/fungitastic-m \
  --output-dir ../../../../work/nvme/bfln/dataset_poc/fungitastic-m

python pregenerate_reference_images.py \
  --class-json ../data/species196_insecta/species196_insecta_labels.json \
  --k 8 \
  --dataset species196_insecta \
  --seed-file ../data/species196_insecta/fewshot16_seed1.txt \
  --image-root ../../../../work/nvme/bfln/dataset_poc/species196_insecta \
  --output-dir ../../../../work/nvme/bfln/dataset_poc/species196_insecta

python pregenerate_reference_images.py \
  --class-json ../data/species196_mollusca/species196_mollusca_labels.json \
  --k 8 \
  --dataset species196_mollusca \
  --seed-file ../data/species196_mollusca/fewshot16_seed1.txt \
  --image-root ../../../../work/nvme/bfln/dataset_poc/species196_mollusca \
  --output-dir ../../../../work/nvme/bfln/dataset_poc/species196_mollusca

python pregenerate_reference_images.py \
  --class-json ../data/species196_weeds/species196_weeds_labels.json \
  --k 8 \
  --dataset species196_weeds \
  --seed-file ../data/species196_weeds/fewshot16_seed1.txt \
  --image-root ../../../../work/nvme/bfln/dataset_poc/species196_weeds \
  --output-dir ../../../../work/nvme/bfln/dataset_poc/species196_weeds
"""

from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from tqdm import tqdm
from typing import Optional


TILE_SIZE = 256

DEFAULT_BG = (0, 0, 0)

# Can customize per dataset 
BG_MAP = {  
    "semi-aves": (0, 0, 0),    
    "fungitastic-m": (0, 0, 0), 
    "species196_insecta": (0, 0, 0),
    "species196_mollusca": (0, 0, 0),
    "species196_weeds": (0, 0, 0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate K-shot reference tiles per class (seed-file only).")
    p.add_argument("--class-json", type=str, required=True,
                   help="Path to labels JSON used to infer number of classes.")
    p.add_argument("--k", type=int, default=16, help="Top-K images per class to tile (default: 16).")
    p.add_argument("--seed-id", type=int, default=1,
                   help="Used only when --seed-file is not set (fewshot{K}_seed{seed_id}.txt).")
    p.add_argument("--seed-file", type=str, default="",
                   help="Optional explicit seed file path. If empty, uses fewshot{K}_seed{seed_id}.txt in CWD.")
    p.add_argument("--dataset", type=str, choices=["fish-vista-m", "fungitastic-m", "semi-aves", "species196_insecta", "species196_mollusca", "species196_weeds"],
                   help="Dataset folder name not path.")
    p.add_argument("--image-root", type=str, default=".",
                   help="Root directory to prepend to relative image paths from the seed file.")
    p.add_argument("--output-dir", type=str, default="",
                   help="Output directory. If empty, uses pregenerated_references_{K}shot/")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle candidates per class before picking (deterministic if --rand-seed set).")
    p.add_argument("--rand-seed", type=int, default=0,
                   help="Random seed for reproducibility when using --shuffle (default: 0 -> no seeding).")
    p.add_argument("--fmt", type=str, default="jpg", choices=["jpg", "png"],
                   help="Output image format (default: jpg).")
    return p.parse_args()


def default_seed_filename(k: int, seed_id: int) -> str:
    return f"fewshot{k}_seed{seed_id}.txt"


def default_output_dir(k: int) -> str:
    return f"pregenerated_references_{k}shot"



def infer_num_classes(labels_json_path: Path) -> int:
    """
    Infer number of classes from a labels JSON whose top-level keys represent classes.
    Handles keys as strings or ints; if keys look like contiguous indices starting at 0,
    returns max_index + 1; otherwise falls back to len(unique keys).
    """
    with labels_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return len(data)

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported labels JSON shape: {type(data)}. Expect dict or list.")

    keys = list(data.keys())
    int_keys = []
    for k in keys:
        try:
            int_keys.append(int(k))
        except Exception:
            int_keys = None
            break

    if int_keys is not None and len(int_keys) > 0:
        mi, ma = min(int_keys), max(int_keys)
        if mi == 0 and len(set(int_keys)) == (ma + 1):
            return ma + 1
        return len(set(int_keys))

    return len(set(keys))


def load_fewshot_map(seed_file: Path) -> Dict[int, List[str]]:
    class_to_paths: Dict[int, List[str]] = {}

    with seed_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if "," in line:
                csv_parts = [t.strip() for t in line.split(",") if t.strip()]
                if len(csv_parts) >= 2:
                    rel_path = csv_parts[0]
                    try:
                        cls_id = int(csv_parts[1])
                        class_to_paths.setdefault(cls_id, []).append(rel_path)
                        continue
                    except ValueError:
                        pass 

            parts = line.replace("\t", " ").split()
            if len(parts) >= 2:
                rel_path = parts[0]
                try:
                    cls_id = int(parts[1])
                    class_to_paths.setdefault(cls_id, []).append(rel_path)
                    continue
                except ValueError:
                    pass



            parts_ws = line.replace("\t", " ").split()
            cls_id = None
            rel_path = ""
            for i in range(len(parts_ws) - 1, -1, -1):
                tok = parts_ws[i].strip().strip(",")
                try:
                    cls_id = int(tok)
                    rel_path = " ".join(parts_ws[:i]).strip().strip(",")
                    break
                except ValueError:
                    continue
            if cls_id is not None and rel_path:
                class_to_paths.setdefault(cls_id, []).append(rel_path)
                continue

            if len(parts_ws) >= 2:
                try:
                    cls_id = int(parts_ws[0].strip().strip(","))
                    rel_path = " ".join(parts_ws[1:]).strip().strip(",")
                    if rel_path:
                        class_to_paths.setdefault(cls_id, []).append(rel_path)
                        continue
                except ValueError:
                    pass

    return class_to_paths


def open_rgb(image_path: Path) -> Image.Image:
    with Image.open(image_path) as im:
        return im.convert("RGB")


def resize_letterbox(im: Image.Image, target: int, DATASET: str) -> Image.Image:
    w, h = im.size
    if w <= 0 or h <= 0:
        return im.resize((target, target))
    scale = min(target / w, target / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    im_resized = im.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (target, target), BG_MAP[DATASET])
    off_x = (target - new_w) // 2
    off_y = (target - new_h) // 2
    canvas.paste(im_resized, (off_x, off_y))
    return canvas


def stitch_grid(images: List[Image.Image], rows: int, cols: int, tile_size: int, DATASET: str) -> Image.Image:
    W = cols * tile_size
    H = rows * tile_size
    out = Image.new("RGB", (W, H), BG_MAP[DATASET])
    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        if r >= rows:
            break
        out.paste(im, (c * tile_size, r * tile_size))
    return out


def compute_grid(k: int) -> Tuple[int, int]:
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k / cols)
    return rows, cols

def pick_k_paths(candidates: List[str], k: int, allow_overlap: bool, shuffle: bool) -> List[str]:
    uniq = list(dict.fromkeys(candidates)) 
    if shuffle:
        random.shuffle(uniq)

    if len(uniq) >= k:
        return uniq[:k]

    if not allow_overlap:
        return uniq

    if not uniq:
        return []

    out = []
    i = 0
    while len(out) < k:
        out.append(uniq[i % len(uniq)])
        i += 1
    return out


def process_class(
    class_id: int,
    rel_paths: List[str],
    image_root: Path,
    k: int,
    rows: int,
    cols: int,
    DATASET:str,
) -> Image.Image | None:
    picked = pick_k_paths(rel_paths, k=k, allow_overlap=True, shuffle=False)
    if not picked:
        return None

    tiles: List[Image.Image] = []
    for rel in picked:
        p = (image_root / rel).resolve()
        try:
            im = open_rgb(p)
            tiles.append(resize_letterbox(im, TILE_SIZE, DATASET))
        except Exception:
            continue

    if not tiles:
        return None

    i = 0
    while len(tiles) < k and len(tiles) > 0:
        tiles.append(tiles[i % len(tiles)])
        i += 1

    tiles = tiles[:k]
    return stitch_grid(tiles, rows, cols, TILE_SIZE, DATASET)


def main():
    args = parse_args()
    DATASET = args.dataset

    if args.shuffle and args.rand_seed:
        random.seed(args.rand_seed)

    labels_json = Path(args.class_json)
    if not labels_json.exists():
        raise FileNotFoundError(f"--class-json not found: {labels_json}")

    num_classes = infer_num_classes(labels_json)

    seed_file = Path(args.seed_file) if args.seed_file else Path(default_seed_filename(args.k, args.seed_id))
    if not seed_file.exists():
        raise FileNotFoundError(
            f"Seed file not found: {seed_file}. "
            f"Pass --seed-file /path/to/file.txt or ensure {default_seed_filename(args.k, args.seed_id)} exists."
        )

    image_root = Path(args.image_root)
    if args.output_dir:
        out_dir = Path(args.output_dir) / default_output_dir(args.k)
    else:
        out_dir = Path(default_output_dir(args.k))
    out_dir.mkdir(parents=True, exist_ok=True)

    class_to_paths = load_fewshot_map(seed_file)

    rows, cols = compute_grid(args.k)
    skipped: List[Tuple[int, str]] = []

    for class_id in tqdm(range(num_classes), desc=f"Generating {args.k}-shot tiles"):
        rels = class_to_paths.get(class_id, [])
        if not rels:
            skipped.append((class_id, "no candidates in seed file"))
            continue

        rels_to_use = rels[:]
        if args.shuffle:
            random.seed(args.rand_seed or 0)
            random.shuffle(rels_to_use)

        grid = process_class(
            class_id=class_id,
            rel_paths=rels_to_use,
            image_root=image_root,
            k=args.k,
            rows=rows,
            cols=cols,
            DATASET=DATASET
        )

        if grid is None:
            skipped.append((class_id, "all candidates failed to open"))
            continue

        ext = ".png" if args.fmt == "png" else ".jpg"
        out_path = out_dir / f"{class_id}{ext}"
        try:
            if args.fmt == "png":
                grid.save(out_path, optimize=True)
            else:
                grid.save(out_path, quality=90, subsampling=0)
        except Exception as e:
            skipped.append((class_id, f"save failed: {e}"))

    if skipped:
        print(f"\nSkipped {len(skipped)} classes:")
        for cid, reason in skipped:
            print(f"  class {cid}: {reason}")
    else:
        print("\nAll classes processed successfully.")


if __name__ == "__main__":
    main()
