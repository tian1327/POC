import argparse
import json, csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import torchvision.transforms as T
import open_clip


# ---------------------------
# Helpers
# ---------------------------
def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def read_listfile(list_path: Path) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with list_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 2:
                raise ValueError(f"List line must have at least 2 tokens: {line}")
            rel = toks[0]
            try:
                y = int(toks[1])
            except Exception:
                raise ValueError(f"Second token must be an int class id: {line}")
            items.append((rel, y))
    if not items:
        raise ValueError(f"No items found in list file: {list_path}")
    return items


# ---------------------------
# Dataset
# ---------------------------
class ListFileImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        list_path: str,
        transform: Optional[T.Compose] = None,
        return_pil: bool = False,
        return_relpath: bool = False,
    ):
        self.root = Path(root)
        self.items = read_listfile(Path(list_path))
        self.transform = transform
        self.return_pil = return_pil
        self.return_relpath = return_relpath
        self.num_classes = 1 + max(y for _, y in self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rel, y = self.items[idx]
        p = self.root / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing image: {p}")
        img = Image.open(p).convert("RGB")

        if self.return_pil:
            return img, y, rel

        if self.transform is None:
            raise ValueError("transform must be provided when return_pil=False")
        x = self.transform(img)
        if self.return_relpath:
            return x, y, rel
        return x, y


# ---------------------------
# Augmentation pipeline 
# ---------------------------
def random_augmentation(n_px: int = 224) -> T.Compose:
    if hasattr(T, "AdjustSharpness"):
        sharpness_tf = T.AdjustSharpness(sharpness_factor=2.0)
    else:
        sharpness_tf = T.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0)

    return T.Compose([
        T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC, antialias=True),

        T.RandomChoice([
            T.CenterCrop(n_px),
            T.RandomCrop(n_px, padding=16, padding_mode="reflect"),
            T.RandomResizedCrop(
                n_px, scale=(0.5, 1.0),
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True
            ),
        ]),

        T.RandomApply([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        ], p=0.5),
        T.RandomApply([sharpness_tf], p=0.5),

        T.RandomApply([
            T.RandomChoice([
                T.RandomHorizontalFlip(p=1.0),
                T.RandomRotation(degrees=30, interpolation=T.InterpolationMode.BILINEAR),
                T.RandomPerspective(distortion_scale=0.3, p=1.0,
                                    interpolation=T.InterpolationMode.BILINEAR),
            ])
        ], p=0.6),

        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])


# ---------------------------
# Class names & prompts
# ---------------------------
DATASET_PROMPT = {
    "_default": "a photo of {}.",
    "semi-aves":   "a photo of a {}, a type of bird.",
    "fungitastic-m":   "a photo of a {}, a type of fungus.",
    "species196_insecta": "a photo of a {}, a type of insect.",
    "species196_weeds":  "a photo of a {}, a type of weed.",
    "species196_mollusca": "a photo of a {}, a type of mollusk.",
}


def build_class_prompt_texts(
    metrics_json: str,
    name_key: str = "most_common_name",
    fallback_keys: Tuple[str, ...] = ("scientific_name", "most_common_name_alt"),
    lowercase: bool = False,
) -> Dict[int, List[str]]:
    with open(metrics_json, "r") as f:
        meta = json.load(f)
    ids = sorted(int(k) for k in meta.keys())
    out: Dict[int, List[str]] = {}
    for cid in ids:
        m = meta[str(cid)]
        name = m.get(name_key, None)
        if not name:
            for fk in fallback_keys:
                if m.get(fk, None):
                    name = m[fk]
                    break
        if not name:
            raise ValueError(f"No name found for class id {cid} in {metrics_json}")
        name = name.strip()
        if lowercase:
            name = name.lower()
        out[cid] = [name]
    return out


def expand_with_templates(cid_to_names: Dict[int, List[str]],
                          template_str: str) -> Dict[int, List[str]]:
    cid_to_texts: Dict[int, List[str]] = {}
    for cid, names in cid_to_names.items():
        name = names[0]
        cid_to_texts[cid] = [template_str.format(name)]
    return cid_to_texts


# ---------------------------
# Text weights (zero-shot)
# ---------------------------
@torch.no_grad()
def compute_text_weights_from_names(
    model,
    tokenizer,
    device: torch.device,
    cid_to_texts: Dict[int, List[str]],
    normalize: bool = True,
) -> torch.Tensor:
    C = 1 + max(cid_to_texts.keys())
    rows: List[torch.Tensor] = []
    for cid in tqdm(range(C), desc="Encoding text prompts"):
        texts = cid_to_texts[cid]
        toks = tokenizer(texts).to(device)
        txt = model.encode_text(toks).float()  
        if normalize:
            txt = l2norm(txt)
        w = txt.mean(dim=0)                   
        if normalize:
            w = l2norm(w)
        rows.append(w)
    W = torch.stack(rows, dim=0)               
    if normalize:
        W = l2norm(W)
    return W


# ---------------------------
# Vision prototypes with K augs
# ---------------------------
@torch.no_grad()
def build_vision_prototypes(
    model,
    loader: DataLoader,
    device: torch.device,
    transform: T.Compose,
    aug_repeats: int = 10,
    encode_chunk_size: int = 512,
) -> torch.Tensor:
    assert transform is not None, "An augmentation transform must be provided"
    C = loader.dataset.num_classes
    buckets: List[List[torch.Tensor]] = [[] for _ in range(C)]

    for batch in tqdm(loader, desc="Building vision prototypes (K augs/img)"):
        images_pil, labels, *_ = batch

        if isinstance(labels, torch.Tensor):
            labels_list = labels.tolist()
        else:
            labels_list = list(labels)

        aug_tensors: List[torch.Tensor] = []
        aug_targets: List[int] = []
        for pil_img, y in zip(images_pil, labels_list):
            for _ in range(aug_repeats):
                x = transform(pil_img)  # (3, H, W)
                aug_tensors.append(x)
                aug_targets.append(int(y))

        if not aug_tensors:
            continue

        # Encode in chunks to avoid OOM
        feats_list: List[torch.Tensor] = []
        big_batch = torch.stack(aug_tensors, dim=0).to(device)  # (B*K, 3, H, W)

        for chunk in torch.split(big_batch, encode_chunk_size, dim=0):
            f = model.encode_image(chunk).float()
            f = l2norm(f)
            feats_list.append(f)
        feats = torch.cat(feats_list, dim=0)  # (B*K, D)

        for f, y in zip(feats, aug_targets):
            buckets[y].append(f)

    rows: List[torch.Tensor] = []
    for cid, feat_list in enumerate(buckets):
        if not feat_list:
            raise RuntimeError(
                f"Empty bucket for class {cid}. "
                "Check your train_list coverage and label remapping."
            )
        m = torch.stack(feat_list, dim=0).mean(dim=0)
        rows.append(l2norm(m))
    return torch.stack(rows, dim=0)  # (C, D)


# ---------------------------
# Fusion & Eval
# ---------------------------
def fuse_text_vision(W_txt: torch.Tensor, W_img: torch.Tensor, alpha: float) -> torch.Tensor:
    W_txt = l2norm(W_txt)
    W_img = l2norm(W_img)
    W = alpha * W_txt + (1.0 - alpha) * W_img
    return l2norm(W)


@torch.no_grad()
def eval_top1(
    model,
    loader: DataLoader,
    W: torch.Tensor,
    device: torch.device,
    logit_scale: float = 1.0,
) -> float:
    WT = W.t().contiguous()  # (D, C)
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        # Support (images, labels) OR (images, labels, rels)
        if len(batch) == 3:
            images, labels, _rels = batch
        else:
            images, labels = batch

        images = images.to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.to(device)

        feats = model.encode_image(images).float()
        feats = l2norm(feats)                   # (B, D)
        logits = feats @ WT                     # (B, C)
        if logit_scale != 1.0:
            logits = logits * logit_scale

        pred = logits.argmax(dim=-1)            # (B,)
        correct += (pred == labels).sum().item()
        total += labels.numel()

    return correct / max(1, total)



# ---------------------------
# Collate (keeps PILs as a list)
# ---------------------------
def collate_pil(batch):
    imgs, ys, rels = zip(*batch)  
    return list(imgs), torch.tensor(ys, dtype=torch.long), list(rels)


# ---------------------------
# Export fused Top-K probabilities
# ---------------------------
@torch.no_grad()
def export_fused_topk_probs(
    model,
    loader: DataLoader,
    W_fused: torch.Tensor,
    device: torch.device,
    topk: int,
    logit_scale: float,
    root_prefix: Path,
):
    WT = W_fused.t().contiguous()  
    rows = []

    for batch in tqdm(loader, desc="Exporting fused Top-K probs", leave=False):
        if len(batch) == 3:
            images, labels, rels = batch
        else:
            images, labels = batch
            rels = [""] * images.size(0)

        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

        images = images.to(device)
        labels = labels.to(device)

        feats = model.encode_image(images).float()
        feats = l2norm(feats)             
        logits = feats @ WT              

        if logit_scale != 1.0:
            logits = logits * logit_scale

        # stable softmax
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(logits, dim=-1)  
        vals, idxs = probs.topk(topk, dim=-1)  
        preds = idxs[:, 0]

        for i in range(images.size(0)):
            rel = rels[i]
            full_path = str((root_prefix / rel).resolve()) if rel else ""
            rows.append({
                "image_path": full_path,
                "pred": int(preds[i].cpu()),
                "label": int(labels[i].cpu()),
                "topk_cls": [int(c) for c in idxs[i].cpu().tolist()],
                "topk_probs": [round(float(p), 3) for p in vals[i].cpu().tolist()],
            })

    return rows


# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser("finer_mmc_aug_export.py — CLIP text/vision/fused heads with fused Top-K export")
    # Model
    p.add_argument("--model_cfg", type=str, default="ViT-B-32")
    p.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    p.add_argument("--device", type=str, default="cuda")
    # Data
    p.add_argument("--dataset_name", type=str, required=True, help="Dataset identifier used to select a single CLIP prompt template.")
    p.add_argument("--dataset_root_train", type=str, required=True)
    p.add_argument("--dataset_root_test", type=str, required=True)
    p.add_argument("--train_list", type=str, required=True)
    p.add_argument("--test_list", type=str, required=True)
    # Names/prompts
    p.add_argument("--metrics_json", type=str, required=True)
    p.add_argument("--name_key", type=str, default="most_common_name")
    p.add_argument("--lowercase_names", action="store_true")
    # Aug/prototypes
    p.add_argument("--use_random_aug", action="store_true",
                   help="Use the custom random augmentation pipeline (recommended for prototypes).")
    p.add_argument("--aug_repeats", type=int, default=10,
                   help="Number of random augmentations per image for prototype building.")
    p.add_argument("--encode_chunk_size", type=int, default=512,
                   help="Chunk size for image encoding during prototype building.")
    # Eval / fusion
    p.add_argument("--alpha", type=float, default=0.7, help="Fusion weight for text vs vision.")
    p.add_argument("--logit_scale_eval", type=float, default=1.0, help="Scale (1/temperature) for accuracy evaluation.")
    p.add_argument("--logit_scale_export", type=float, default=None, help="Scale (1/temperature) for exporting probabilities; falls back to logit_scale_eval if not set.")
    p.add_argument("--skip_acc", action="store_true", help="Skip computing accuracies and summary.")
    # Loader
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    # Export
    p.add_argument("--topk", type=int, default=10, help="Top-K to export from the fused head.")
    p.add_argument("--output_json", type=str, default="", help="If set, write fused Top-K probs to this JSON.")
    p.add_argument("--output_csv", type=str, default="", help="If set, write fused Top-K probs to this CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, _, base_preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_cfg,
        pretrained=args.pretrained,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model_cfg)

    template_str = DATASET_PROMPT.get(args.dataset_name, DATASET_PROMPT["_default"])
    print(f"[INFO] Dataset: {args.dataset_name} | Using template: {template_str}")

    if args.use_random_aug:
        proto_transform = random_augmentation(224)
    else:
        proto_transform = base_preprocess

    train_ds = ListFileImageDataset(
        root=args.dataset_root_train,
        list_path=args.train_list,
        transform=None,          
        return_pil=True          
    )
    test_ds = ListFileImageDataset(
        root=args.dataset_root_test,
        list_path=args.test_list,
        transform=base_preprocess,  
        return_pil=False,
        return_relpath=True         
    )

    cid_to_names = build_class_prompt_texts(
        metrics_json=args.metrics_json,
        name_key=args.name_key,
        lowercase=args.lowercase_names
    )
    cid_to_texts = expand_with_templates(cid_to_names, template_str)

    json_ids = sorted(cid_to_names.keys())
    id_to_row = {cid: i for i, cid in enumerate(json_ids)}

    train_ds.items = [(rel, id_to_row[y]) for (rel, y) in train_ds.items]
    test_ds.items  = [(rel, id_to_row[y]) for (rel, y) in test_ds.items]

    train_ds.num_classes = 1 + max(y for _, y in train_ds.items)
    test_ds.num_classes  = 1 + max(y for _, y in test_ds.items)


    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=collate_pil,   
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )


    C = train_ds.num_classes
    contiguous_cid_to_texts = {i: cid_to_texts[json_ids[i]] for i in range(C)}
    W_txt = compute_text_weights_from_names(
        model, tokenizer, device, contiguous_cid_to_texts, normalize=True
    )
    print(f"[INFO] W_txt: {tuple(W_txt.shape)}")


    W_img = build_vision_prototypes(
        model=model,
        loader=train_loader,
        device=device,
        transform=proto_transform,
        aug_repeats=args.aug_repeats,
        encode_chunk_size=args.encode_chunk_size,
    )
    print(f"[INFO] W_img: {tuple(W_img.shape)}")


    W_fused = fuse_text_vision(W_txt, W_img, alpha=args.alpha)


    if not args.skip_acc:
        acc_txt   = eval_top1(model, test_loader, W_txt,   device, logit_scale=args.logit_scale_eval)
        acc_img   = eval_top1(model, test_loader, W_img,   device, logit_scale=args.logit_scale_eval)
        acc_fused = eval_top1(model, test_loader, W_fused, device, logit_scale=args.logit_scale_eval)

        export_scale = args.logit_scale_export if args.logit_scale_export is not None else args.logit_scale_eval

        summary = {
            "acc_text": acc_txt,
            "acc_vision": acc_img,
            "acc_fused": acc_fused,
            "alpha": args.alpha,
            "aug_repeats": args.aug_repeats,
            "templates_count": 1,
            "dataset_name": args.dataset_name,
            "template_used": template_str,
            "classes_train": train_ds.num_classes,
            "classes_test": test_ds.num_classes,
            "model_cfg": args.model_cfg,
            "pretrained": args.pretrained,
        }
        print(json.dumps(summary, indent=2))

    if args.output_json or args.output_csv:
        recs = export_fused_topk_probs(
            model=model,
            loader=test_loader,
            W_fused=W_fused,
            device=device,
            topk=args.topk,
            logit_scale=export_scale,  
            root_prefix=Path(args.dataset_root_test),
        )

        if args.output_json:
            out_json = Path(args.output_json)
            out_json.parent.mkdir(parents=True, exist_ok=True) 
            payload = {str(i): rec for i, rec in enumerate(recs)}
            with out_json.open("w") as f:
                json.dump(payload, f, indent=2)
            print(f"[INFO] Wrote JSON to {out_json}")

        
        if args.output_csv:
            out_csv = Path(args.output_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)  
            with out_csv.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_path","pred","label","topk_cls","topk_probs"])
                for r in recs:
                    w.writerow([
                        r["image_path"],
                        r["pred"],
                        r["label"],
                        json.dumps(r["topk_cls"]),
                        json.dumps(r["topk_probs"]),
                    ])
            print(f"[INFO] Wrote CSV to {out_csv}")


if __name__ == "__main__":
    main()
