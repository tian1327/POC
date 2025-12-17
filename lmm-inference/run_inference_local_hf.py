from __future__ import annotations
import os, sys, json, argparse, time, csv, re, unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

import pandas as pd
from PIL import Image
import yaml
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ---------------------------
# Files & Mappings
# ---------------------------

TEMPLATE_MAP = {
    # base top-5 (no candidate images)
    "top5-simple": ("top5_base.txt", "top5_text", dict(with_conf=False)),
    "top5-simple-with-confidence": ("top5_base.txt", "top5_text", dict(with_conf=True)),
    "top5-flat": ("top5_base.txt", "top5_flat_taxonomy", dict(with_conf=False)),
    "top5-flat-with-confidence": ("top5_base.txt", "top5_flat_taxonomy", dict(with_conf=True)),
    "top5-sci": ("top5_base.txt", "top5_text", dict(with_conf=False, with_sci=True)),
    "top5-sci-with-confidence": ("top5_base.txt", "top5_text", dict(with_conf=True, with_sci=True)),

    # multimodal top-5 (candidate stitched images)
    "top5-multimodal-16shot": ("top5_multimodal_16shot.txt", "top5_multimodal", dict(with_conf=False, with_sci=True)),
    "top5-multimodal-16shot-with-confidence": ("top5_multimodal_16shot.txt", "top5_multimodal", dict(with_conf=True, with_sci=True)),
    "top5-multimodal-16shot_with_taxonomy-with-confidence": ("top5_multimodal_16shot.txt", "top5_multimodal", dict(with_conf=True, with_sci=True, with_taxonomy=True)),

    # with descriptions (text only)
    "top5_with_descriptions": ("top5_with_descriptions.txt", "top5_desc_text", dict(with_conf=False, with_sci=True)),
    "top5_with_descriptions-with-confidence": ("top5_with_descriptions.txt", "top5_desc_text", dict(with_conf=True, with_sci=True)),

    # multimodal with descriptions
    "top5_multimodal_with_descriptions": ("top5_multimodal_with_descriptions.txt", "top5_desc_multimodal", dict(with_conf=False, with_sci=True)),
    "top5_multimodal_with_descriptions-with-confidence": ("top5_multimodal_with_descriptions.txt", "top5_desc_multimodal", dict(with_conf=True, with_sci=True)),

    # ranking variants
    "top5-sci-with-confidence_ranking": ("top5_base_ranking.txt", "top5_text", dict(with_conf=True, with_sci=True)),
    "top5-multimodal-16shot_ranking": ("top5_multimodal_ranking.txt", "top5_desc_multimodal_rank", dict(with_conf=False, with_sci=True)),
    "top5-multimodal-4shot-with-confidence_ranking": ("top5_multimodal_ranking_4.txt", "top5_desc_multimodal_rank", dict(with_conf=True, with_sci=True)),
    "top5-multimodal-8shot-with-confidence_ranking": ("top5_multimodal_ranking_8.txt", "top5_desc_multimodal_rank", dict(with_conf=True, with_sci=True)),
    "top5-multimodal-16shot-with-confidence_ranking": ("top5_multimodal_ranking.txt", "top5_desc_multimodal_rank", dict(with_conf=True, with_sci=True)),
    "top5-multimodal-16shot_with_taxonomy-with-confidence_ranking": ("top5_multimodal_ranking.txt", "top5_desc_multimodal_rank", dict(with_conf=True, with_sci=True, with_taxonomy=True)),
    
    "top5_multimodal_with_descriptions_ranking": ("top5_multimodal_with_descriptions_ranking.txt", "top5_desc_multimodal_rank", dict(with_conf=False, with_sci=True)),
    "top5_multimodal_with_descriptions_ranking-with-confidence": ("top5_multimodal_with_descriptions_ranking.txt", "top5_desc_multimodal_rank", dict(with_conf=True, with_sci=True)),

    # zeroshot
    "zeroshot": ("zeroshot_identify.txt", "zeroshot_identify", dict()),
    "zeroshot-explanation": ("zeroshot_identify.txt", "zeroshot_identify", dict()),
    "zeroshot-all200": ("zeroshot_all200.txt", "zeroshot_all200", dict()),
    "zeroshot-all200-explanation": ("zeroshot_all200.txt", "zeroshot_all200", dict()),
    "zeroshot-cot":  ("zeroshot_identify_cot.txt", "zeroshot_identify", dict()),
    "zeroshot_cot": ("zeroshot_identify_cot.txt", "zeroshot_identify", dict()),


    # contrastive descriptions
    "top5_with_contrastive_group": ("top5_with_contrastive_group.txt", "top5_contrastive_group_text", dict(with_conf=False, with_sci=True)),
    "top5_with_contrastive_group-with-confidence": ("top5_with_contrastive_group.txt", "top5_contrastive_group_text", dict(with_conf=True, with_sci=True)),
}

CONF_NOTE = (
    "Note on confidence: The confidence shown for candidate 1 (p1) reflects how certain "
    "the underlying model was. Use p1 only as a signal of the model's certainty. "
    "If p1 appears strong and matches the visible evidence, you may lean toward #1. "
    "If p1 appears weak or the image contradicts it, give more weight to visual evidence "
    "and consider other candidates."
)

def _response_format_for(prompt_key: str) -> str:
    WITH_EXPLANATION = {
        "zeroshot-explanation",
        "zeroshot-all200-explanation",
    }
    base = "Most Likely: [Common Name (Scientific name)]"
    return base + ("\nExplanation: [your explanation here]" if prompt_key in WITH_EXPLANATION else "")

# ---------------------------
# IO helpers
# ---------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_test_list(path: Path) -> List[str]:
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            first = ln.split()[0]
            first = first.split(",")[0]
            out.append(first)
    return out

def load_template_text(prompt_dir: Path, filename: str) -> str:
    p = prompt_dir / filename
    return p.read_text(encoding="utf-8")

def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    if not max_side or max_side <= 0:
        return img
    Resampling = getattr(Image, "Resampling", Image)
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_sz = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_sz, Resampling.LANCZOS)


def _first_token(s: str) -> str:
    s = s.strip()
    s = s.split()[0]
    s = s.split(",")[0]
    return s

def _variants_for_path_str(path_str: str, image_dir: Path) -> List[str]:
    t = _first_token(path_str)
    p = Path(t)
    variants = set()
    variants.add(t)
    if not p.is_absolute():
        try:
            variants.add(str((image_dir / p).resolve()))
        except Exception:
            variants.add(str(image_dir / p))
    else:
        try:
            variants.add(str(p.resolve()))
        except Exception:
            variants.add(str(p))
    try:
        abs_p = (image_dir / p) if not p.is_absolute() else p
        abs_p = abs_p.resolve()
        variants.add(str(abs_p.relative_to(image_dir.resolve())))
    except Exception:
        pass
    try:
        parts = list(Path(t).parts)
        imgdir_name = Path(image_dir).name
        if parts and parts[0] == imgdir_name:
            variants.add(str(Path(*parts[1:])))
    except Exception:
        pass
    variants.add(Path(t).name)
    return list(variants)

def build_topk_index(topk_json_path: Path, image_dir: Path) -> Dict[str, Dict[str, Any]]:
    with topk_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    index: Dict[str, Dict[str, Any]] = {}
    if not isinstance(data, dict):
        raise ValueError("topk_json must be a dict")
    looks_like_layout_B = False
    if data and all(k.isdigit() for k in list(data.keys())[:5]):
        for k in list(data.keys())[:5]:
            v = data[k]
            looks_like_layout_B = isinstance(v, dict) and "image_path" in v
            if not looks_like_layout_B:
                break
    if looks_like_layout_B:
        for _, rec in data.items():
            imgp = rec.get("image_path")
            if not imgp:
                continue
            for v in _variants_for_path_str(imgp, image_dir):
                index.setdefault(v, rec)
    else:
        for k, rec in data.items():
            for v in _variants_for_path_str(k, image_dir):
                index.setdefault(v, rec)
    return index

def _load_contrastive_map(path):
    if not path:
        return {}
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            abs_p = rec.get("image_path")
            rel_p = rec.get("rel_path")
            if isinstance(abs_p, str) and abs_p:
                m[abs_p] = rec
            if isinstance(rel_p, str) and rel_p:
                m[rel_p] = rec
    return m

# ---------------------------
# Taxonomy -> names
# ---------------------------

def build_taxonomy_maps(taxonomy_json_path: Path) -> Dict[str, Any]:
    with taxonomy_json_path.open("r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    id2common: Dict[int, str] = {}
    id2sci: Dict[int, str] = {}
    id2genus: Dict[int, str] = {}
    id2family: Dict[int, str] = {}

    def pick(d, keys):
        for k in keys:
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def pick_nested(d, paths):
        for path in paths:
            cur = d
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False; break
            if ok and isinstance(cur, str) and cur.strip():
                return cur.strip()
        return ""

    for cid_raw, rec in taxonomy.items():
        if not isinstance(rec, dict):
            continue
        try:
            cid = int(cid_raw)
        except Exception:
            try:
                cid = int(rec.get("id"))
            except Exception:
                continue

        common = pick(rec, ["most_common_name","common","common_name","en_common_name","en","label","species_common"])
        sci    = pick(rec, ["name","sci","scientific","scientific_name","latin","species","species_scientific"])
        genus  = pick(rec, ["genus","genus_name"]) or pick_nested(rec, [["taxonomy","genus"],["taxonomy","genus","name"]])
        family = pick(rec, ["family","family_name"]) or pick_nested(rec, [["taxonomy","family"],["taxonomy","family","name"]])

        id2common[cid] = common or sci or f"class_{cid}"
        id2sci[cid]    = sci or common or f"class_{cid}"
        id2genus[cid]  = genus or ""
        id2family[cid] = family or ""

    species_items = [(cid, id2common[cid], id2sci[cid]) for cid in sorted(id2common.keys())]
    all_species_list = [f"{cid}. {common} ({sci})" for cid, common, sci in species_items]
    return {
        "id2common": id2common,
        "id2sci": id2sci,
        "id2genus": id2genus,
        "id2family": id2family,
        "all_species_list": all_species_list,
    }

# ---------------------------
# Template rendering -> messages
# ---------------------------

def _inject_text(content: List[Dict[str, Any]], text: str):
    if text:
        content.append({"type": "text", "text": text})

def _confidence_note_text(with_conf: bool) -> str:
    return (CONF_NOTE + "\n") if with_conf else ""

def _load_and_resize(ref_path: Path, resize_side: int, debug: bool = False):
    try:
        with Image.open(ref_path) as im:
            pil = im.convert("RGB")
        orig = pil.size
        pil = resize_keep_aspect(pil, resize_side)
        if debug:
            print(f"[resize] {ref_path.name}: {orig} -> {pil.size} (max_side={resize_side})")
        return pil
    except Exception as ex:
        if debug:
            print(f"[resize] {ref_path.name}: failed to open ({ex}); inserting text note instead")
        return None  # signal failure

def render_top5_text(template_txt: str, with_conf: bool, with_sci: bool,
                     test_img: Path, candidates: List[Tuple[int, Optional[float]]],
                     tax: Dict[str, Any]) -> List[Dict[str, Any]]:
    id2c, id2s = tax["id2common"], tax["id2sci"]
    lines = []
    for i, (cid, p) in enumerate(candidates, 1):
        common = id2c.get(cid, f"class_{cid}")
        sci    = id2s.get(cid, f"class_{cid}") if with_sci else None
        seg = f"{i}. {common}"
        if sci: seg += f" ({sci})"
        if with_conf and (p is not None): seg += f" [p={p}]"
        lines.append(seg)
    species_list = "\n".join(lines)
    tpl = template_txt.replace("{species_list}", species_list)
    tpl = tpl.replace("{confidence_note}", _confidence_note_text(with_conf))

    before, after = tpl, ""
    m = re.search(r"\n\s*\n", tpl)
    if m:
        before = tpl[:m.end()]
        after = tpl[m.end():]

    content: List[Dict[str, Any]] = []
    _inject_text(content, before)
    content.append({"type": "image", "image": str(test_img)})
    _inject_text(content, "\n" + after)
    return [{"role": "user", "content": content}]


def render_top5_flat_taxonomy(template_txt, with_conf, test_img, candidates, tax):
    id2c, id2s = tax["id2common"], tax["id2sci"]
    id2g, id2f = tax.get("id2genus", {}), tax.get("id2family", {})
    lines = []
    for i, (cid, p) in enumerate(candidates, 1):
        sci    = id2s.get(cid, f"class_{cid}")
        common = id2c.get(cid, f"class_{cid}")
        genus  = (id2g.get(cid) or "unknown").strip()
        family = (id2f.get(cid) or "unknown").strip()
        seg = f"{i}. {sci}, also known as {common}, belongs to the genus {genus}, family {family}"
        if with_conf and (p is not None):
            seg += f" [p={p}]"
        lines.append(seg)
    species_list = "\n".join(lines)
    tpl = template_txt.replace("{species_list}", species_list)
    tpl = tpl.replace("{confidence_note}", _confidence_note_text(with_conf))

    m = re.search(r"\n\s*\n", tpl)
    before, after = (tpl[:m.end()], tpl[m.end():]) if m else (tpl, "")
    content = []
    _inject_text(content, before)
    content.append({"type": "image", "image": str(test_img)})
    _inject_text(content, "\n" + after)
    return [{"role": "user", "content": content}]


def render_top5_multimodal(template_txt: str, with_conf: bool, with_sci: bool,
                           test_img: Path, candidates: List[Tuple[int, Optional[float]]],
                           tax: Dict[str, Any], ref_dir: Path, resize_side: int,
                           debug_resize: bool = False, *, with_taxonomy: bool = False,) -> List[Dict[str, Any]]:
    id2c, id2s = tax["id2common"], tax["id2sci"]
    note_txt = _confidence_note_text(with_conf)
    tpl = template_txt.replace("{confidence_note}", "")
    id2g, id2f = tax.get("id2genus", {}), tax.get("id2family", {})

    cand_blocks: List[List[Dict[str, Any]]] = []
    for i, (cid, p) in enumerate(candidates, 1):
        parts: List[Dict[str, Any]] = []
        common = id2c.get(cid, f"class_{cid}")
        sci    = id2s.get(cid, f"class_{cid}") if with_sci else None

        if with_taxonomy:
            genus  = (id2g.get(cid) or "unknown").strip()
            family = (id2f.get(cid) or "unknown").strip()
            line = f"Candidate {i}: {sci}, also known as {common}, belongs to the genus {genus}, family {family}"
            if with_conf and (p is not None):
                line += f" [p={p}]"
        else:
            line = f"Candidate {i}: {common}"
            if sci:
                line += f" ({sci})"
            if with_conf and (p is not None):
                line += f" [p={p}]"

        parts.append({"type": "text", "text": line})

        ref_path = ref_dir / f"{cid}.jpg"
        pil = _load_and_resize(ref_path, resize_side, debug=debug_resize)
        if pil is not None:
            parts.append({"type": "image", "image": pil})
        else:
            parts.append({"type": "text", "text": f"[missing stitched image: {ref_path}]"})
        parts.append({"type": "text", "text": "\n"})
        cand_blocks.append(parts)

    placeholder = "{species_list}"
    if "{species_list_with_descriptions_and_images}" in tpl:
        placeholder = "{species_list_with_descriptions_and_images}"

    if placeholder in tpl:
        before, after = tpl.split(placeholder, 1)
    else:
        before, after = tpl, ""

    b1, b2 = before, ""
    m = re.search(r"\n\s*\n", before)
    if m:
        b1 = before[:m.end()]
        b2 = before[m.end():]

    step3_txt = ""
    m_step3 = re.search(r"(Step 3:.*)", b2, flags=re.S)
    if m_step3:
        pre_step3 = b2[:m_step3.start()].rstrip()
        step3_txt = b2[m_step3.start():].lstrip()
    else:
        pre_step3 = b2

    content: List[Dict[str, Any]] = []
    _inject_text(content, b1)
    content.append({"type": "image", "image": str(test_img)})
    _inject_text(content, "\n" + pre_step3)

    content.append({"type": "text", "text": "\n"})
    for block in cand_blocks:
        content.extend(block)

    if note_txt:
        content.append({"type": "text", "text": "\n" + note_txt})

    if step3_txt:
        _inject_text(content, "\n" + step3_txt)

    _inject_text(content, after)
    return [{"role": "user", "content": content}]

def render_top5_desc_text(template_txt: str, with_conf: bool, with_sci: bool,
                          test_img: Path, candidates: List[Tuple[int, Optional[float]]],
                          tax: Dict[str, Any], descriptions: Optional[Dict[str, str]]) -> List[Dict[str, Any]]:
    id2c, id2s = tax["id2common"], tax["id2sci"]
    blocks = []
    for i, (cid, p) in enumerate(candidates, 1):
        common = id2c.get(cid, f"class_{cid}")
        sci    = id2s.get(cid, f"class_{cid}") if with_sci else None
        line = f"{i}. {common}"
        if sci: line += f" ({sci})"
        if with_conf and (p is not None): line += f" [p={p}]"
        desc = ""
        if descriptions:
            desc = descriptions.get(str(cid)) or descriptions.get(common) or descriptions.get(sci or "")
        if desc:
            line += f"\nDescription: {desc}"
        blocks.append(line)
    species_desc = "\n\n".join(blocks)

    tpl = template_txt.replace("{species_list_with_descriptions}", species_desc)
    tpl = tpl.replace("{confidence_note}", _confidence_note_text(with_conf))

    before, after = tpl, ""
    m = re.search(r"\n\s*\n", tpl)
    if m:
        before = tpl[:m.end()]
        after = tpl[m.end():]

    content: List[Dict[str, Any]] = []
    _inject_text(content, before)
    content.append({"type": "image", "image": str(test_img)})
    _inject_text(content, "\n" + after)
    return [{"role": "user", "content": content}]

def render_top5_desc_multimodal(
    template_txt: str,
    rank_mode: bool,
    with_conf: bool,
    with_sci: bool,
    test_img: Path,
    candidates: List[Tuple[int, Optional[float]]],
    tax: Dict[str, Any],
    descriptions: Optional[Dict[str, str]],
    ref_dir: Path,
    resize_side: int,
    debug_resize: bool = False,
    *,
    with_taxonomy: bool = False,
) -> List[Dict[str, Any]]:
    id2c, id2s = tax["id2common"], tax["id2sci"]
    id2g, id2f = tax.get("id2genus", {}), tax.get("id2family", {})

    def _flat_taxonomy_line(i: int, cid: int, p: Optional[float]) -> str:
        sci    = id2s.get(cid, f"class_{cid}")
        common = id2c.get(cid, f"class_{cid}")
        genus  = (id2g.get(cid) or "unknown").strip()
        family = (id2f.get(cid) or "unknown").strip()
        line = f"{i}. {sci}, also known as {common}, belongs to the genus {genus}, family {family}"
        if with_conf and (p is not None):
            line += f" [p={p}]"
        return line

    wants_desc_placeholder = "{species_list_with_descriptions_and_images}" in template_txt
    wants_list_placeholder = "{species_list}" in template_txt

    # helper: build the header line for one candidate
    def _header_line(i: int, cid: int, p: Optional[float]) -> str:
        common = id2c.get(cid, f"class_{cid}")
        sci    = id2s.get(cid, f"class_{cid}")
        line = f"{i}. {common} ({sci})" if with_sci else f"{i}. {common}"
        if with_conf and (p is not None):
            line += f" [p={p}]"
        return line

    # ============ PATH A: with DESCRIPTIONS ============
    if wants_desc_placeholder:
        cand_blocks: List[List[Dict[str, Any]]] = []
        for i, (cid, p) in enumerate(candidates, 1):
            parts: List[Dict[str, Any]] = []

            # line "Candidate i: Name (Sci) [p=...]"
            common = id2c.get(cid, f"class_{cid}")
            sci    = id2s.get(cid, f"class_{cid}") if with_sci else None
            line = f"Candidate {i}: {common}"
            if sci: line += f" ({sci})"
            if with_conf and (p is not None): line += f" [p={p}]"
            parts.append({"type": "text", "text": line})

            # optional description
            desc = ""
            if descriptions:
                desc = descriptions.get(str(cid)) or descriptions.get(common) or descriptions.get(sci or "")
            if desc:
                parts.append({"type": "text", "text": f"Description: {desc}"})

            if with_taxonomy:
                flat = _flat_taxonomy_line(i, cid, p)
                prefix = f"{i}. "
                if flat.startswith(prefix):
                    flat = flat[len(prefix):]
                flat = f"Candidate {i}: {flat}"
                parts[0] = {"type": "text", "text": flat}

            # stitched reference image
            ref_path = ref_dir / f"{cid}.jpg"
            pil = _load_and_resize(ref_path, resize_side, debug=debug_resize)
            if pil is not None:
                parts.append({"type": "image", "image": pil})
            else:
                parts.append({"type": "text", "text": f"[missing stitched image: {ref_path}]"})
            parts.append({"type": "text", "text": "\n"})
            cand_blocks.append(parts)

        tpl = template_txt.replace("{confidence_note}", _confidence_note_text(with_conf))
        before, after = tpl.split("{species_list_with_descriptions_and_images}", 1)

        # inject: text → query image → (optional mid-text) → blocks → tail
        b1, b2 = before, ""
        m = re.search(r"\n\s*\n", before)
        if m:
            b1 = before[:m.end()]
            b2 = before[m.end():]

        content: List[Dict[str, Any]] = []
        _inject_text(content, b1)
        content.append({"type": "image", "image": str(test_img)})
        _inject_text(content, "\n" + b2)

        content.append({"type": "text", "text": "\n"})
        for block in cand_blocks:
            content.extend(block)
        _inject_text(content, after)
        return [{"role": "user", "content": content}]

    # ============ PATH B: NO DESCRIPTIONS (names-only species list) ============
    if wants_list_placeholder:
        # 1) build the 5-line list ONLY (no per-candidate descriptions)
        lines: List[str] = []
        for i, (cid, p) in enumerate(candidates, 1):
            if with_taxonomy:
                flat = _flat_taxonomy_line(i, cid, p)
                prefix = f"{i}. "
                if flat.startswith(prefix):
                    flat = flat[len(prefix):]
                header = f"Candidate {i}: {flat}"
            else:
                header = _header_line(i, cid, p)
            lines.append(header)
        species_list_text = "\n".join(lines)

        # 2) fill template placeholders
        tpl = (template_txt
               .replace("{species_list}", species_list_text)
               .replace("{confidence_note}", _confidence_note_text(with_conf)))

        # 3) text → query image → candidate stitched refs → tail text
        m = re.search(r"\n\s*\n", tpl)
        before, after = (tpl[:m.end()], tpl[m.end():]) if m else (tpl, "")

        content: List[Dict[str, Any]] = []
        _inject_text(content, before)
        content.append({"type": "image", "image": str(test_img)})

        # attach 5× stitched candidate images (no extra text labels)
        for cid, _ in candidates:
            ref_path = ref_dir / f"{cid}.jpg"
            pil = _load_and_resize(ref_path, resize_side, debug=debug_resize)
            if pil is not None:
                content.append({"type": "image", "image": pil})
            else:
                content.append({"type": "text", "text": f"[missing stitched image: {ref_path}]"})
        _inject_text(content, "\n" + after)
        return [{"role": "user", "content": content}]

    # ============ Guard: neither placeholder present ============
    raise RuntimeError(
        "Ranking template is missing required placeholder. "
        "Expected either '{species_list_with_descriptions_and_images}' or '{species_list}'."
    )


def render_zeroshot_all200(template_txt: str, tax: Dict[str, Any], test_img: Path, response_format: str) -> List[Dict[str, Any]]:
    species_block = "\n".join(tax["all_species_list"])
    tpl = (template_txt
           .replace("{species_block}", species_block)
           .replace("{response_format}", response_format))
    content: List[Dict[str, Any]] = []
    content.append({"type": "image", "image": str(test_img)})
    _inject_text(content, "\n" + tpl)
    return [{"role": "user", "content": content}]

def render_zeroshot_identify(template_txt: str, tax: Dict[str, Any], test_img: Path, response_format: str) -> List[Dict[str, Any]]:
    tpl = template_txt.replace("{response_format}", response_format)
    content: List[Dict[str, Any]] = []
    content.append({"type": "image", "image": str(test_img)})
    _inject_text(content, "\n" + tpl)
    return [{"role": "user", "content": content}]

def render_top5_contrastive_group_text(
    template_txt: str,
    with_conf: bool,
    with_sci: bool,
    test_img: Path,
    candidates: List[Tuple[int, Optional[float]]],
    tax: Dict[str, Any],
    contrastive_map: Dict[str, Any],
) -> List[Dict[str, Any]]:
    id2c, id2s = tax["id2common"], tax["id2sci"]

    lines = []
    for i, (cid, p) in enumerate(candidates, 1):
        common = id2c.get(cid, f"class_{cid}")
        sci = id2s.get(cid, f"class_{cid}") if with_sci else None

        line = f"Candidate {i}: {common}"
        if sci:
            line += f" ({sci})"
        if with_conf and (p is not None):
            line += f" [p={p}]"
        lines.append(line)

    species_block = "\n".join(lines)

    q_abs = str(test_img)
    rel_key = q_abs
    for root in ("/home/ltmask/dataset/semi-aves/", "/mnt/data/semi-aves/", "/home/"):
        if rel_key.startswith(root):
            rel_key = rel_key[len(root):]
            while rel_key.startswith("/"):
                rel_key = rel_key[1:]
            break

    rec = (contrastive_map.get(q_abs) or contrastive_map.get(rel_key) or contrastive_map.get(Path(q_abs).name))
    group_points_text = ""
    if rec:
        if isinstance(rec.get("text"), str) and rec["text"].strip():
            group_points_text = rec["text"].strip()
        else:
            pts = [p for p in (rec.get("points") or []) if isinstance(p, str) and p.strip()]
            if pts:
                group_points_text = "\n".join(f"• {p.strip()}" for p in pts)

    body = template_txt
    body = body.replace("{species_list}", species_block)
    body = body.replace("{contrastive_group_points}", group_points_text)
    body = body.replace("{confidence_note}", _confidence_note_text(with_conf))

    before, after = body, ""
    m = re.search(r"\n\s*\n", body)
    if m:
        before = body[:m.end()]
        after  = body[m.end():]

    content: List[Dict[str, Any]] = []
    _inject_text(content, before)
    content.append({"type": "image", "image": q_abs})
    _inject_text(content, "\n" + after)
    return [{"role": "user", "content": content}]


# ---------------------------
# Prompt printing / CSV-safe
# ---------------------------

def _clean_vision_tokens(s: str) -> str:
    return (s
            .replace("<|vision_start|><|image_pad|><|vision_end|>", "\n[IMAGE]\n")
            .replace("<image>", "\n[IMAGE]\n")
            .replace("<img>", "\n[IMAGE]\n")
            .replace("<|image|>", "\n[IMAGE]\n"))

def _normalize_ws(s: str) -> str:
    s = unicodedata.normalize("NFKC", s.replace("\xa0", " "))
    s = "\n".join(line.rstrip() for line in s.splitlines())
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def print_prompt_for_debug(processor: AutoProcessor, messages: List[Dict[str, Any]]):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text = _strip_chat_wrappers(text)
    cleaned = _normalize_ws(_clean_vision_tokens(text))
    print("----- PROMPT -----")
    print(cleaned)
    print("------------------")

def prompt_text_for_csv(processor: AutoProcessor, messages: List[Dict[str, Any]]) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text = _strip_chat_wrappers(text)
    return _normalize_ws(_clean_vision_tokens(text))

def _strip_chat_wrappers(s: str) -> str:
    s = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>\s*", "", s, flags=re.S)
    s = re.sub(r"\s*<\|im_start\|>user\s*", "", s)
    s = re.sub(r"\s*<\|im_end\|>\s*$", "", s)
    s = re.sub(r"\s*<\|im_start\|>assistant\s*$", "", s)
    return s

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Local HF (template-driven)")

    ap.add_argument("--prompt-template", required=True, choices=TEMPLATE_MAP.keys())
    ap.add_argument("--prompt-dir", required=False, default="./prompt_templates")

    ap.add_argument("--backend", required=True, choices=["nebius", "hyperbolic", "ollama", "huggingface"])
    ap.add_argument("--hf-model-name-or-path", default="Qwen/Qwen2.5-VL-7B-Instruct")

    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--image-paths", required=True)
    ap.add_argument("--taxonomy-json", required=True)
    ap.add_argument("--topk-json")
    ap.add_argument("--ref-image-dir")
    ap.add_argument("--descriptions-json")

    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--error-file", required=True)

    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-first-prompt", action="store_true")
    ap.add_argument("--throttle-sec", type=float, default=0.0)
    ap.add_argument("--config-yaml", default="config.yml")
    ap.add_argument("--max_new_tokens", type=int, default=900)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--no-kv-cache", action="store_true")
    ap.add_argument("--contrastive-map-jsonl", type=str, default=None,
                    help="JSONL built by build_image_contrastive_map.py (per-image group points).")

    ap.add_argument(
        "--resize-candidates", "--resize_candidates", "--candidate-resize", "--candidate_resize",
        dest="resize_candidates", type=int, default=1200,
        help="Max side (px) for stitched candidate images (multimodal templates only)."
    )
    ap.add_argument("--min_pixels", type=int, default=None, help="Qwen VL min pixels (e.g., 256*28*28)")
    ap.add_argument("--max_pixels", type=int, default=None, help="Qwen VL max pixels; set <=0 to disable")

    ap.add_argument("--debug-resize", action="store_true", help="Log original->resized sizes for candidate images")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-resume", action="store_true")

    args = ap.parse_args()
    if args.backend != "huggingface":
        print("NOTE: This runfile uses Hugging Face locally. Other backends are ignored.")

    # Resolve dataset root from config YAML
    cfg = load_yaml(args.config_yaml)
    dataset_root = cfg.get("dataset_path")
    if not dataset_root:
        sys.exit("ERROR: config.yml missing 'dataset_path'")

    def _resolve_under_dataset(p: Optional[str], root: Optional[str]) -> Optional[str]:
        if not p or not root:
            return p
        return p if os.path.isabs(p) else os.path.join(root, p)

    args.image_dir     = _resolve_under_dataset(args.image_dir, dataset_root)
    args.ref_image_dir = _resolve_under_dataset(args.ref_image_dir, dataset_root)

    # Load basics
    image_dir = Path(args.image_dir)
    prompt_dir = Path(args.prompt_dir)
    tax = build_taxonomy_maps(Path(args.taxonomy_json))
    template_file, mode, flags = TEMPLATE_MAP[args.prompt_template]
    print('prompt_templates/' + template_file)
    template_txt = load_template_text(prompt_dir, 'prompt_templates/' + template_file)

    # Data lists
    paths = read_test_list(Path(args.image_paths))
    if args.limit is not None and args.limit > 0:
        paths = paths[:args.limit]
    if not paths:
        sys.exit("No images to process.")
    df = pd.DataFrame({"image_path": paths})

    # Top-k (for top-5 modes)
    needs_top5 = mode.startswith("top5")
    topk_index: Dict[str, Dict[str, Any]] = {}
    if needs_top5:
        if not args.topk_json:
            sys.exit(f"--topk-json is required for '{args.prompt_template}'")
        topk_index = build_topk_index(Path(args.topk_json), image_dir)

    # Optional descriptions
    descriptions = None
    if "desc" in mode and args.descriptions_json:
        try:
            with Path(args.descriptions_json).open("r", encoding="utf-8") as f:
                descriptions = json.load(f)
        except Exception:
            descriptions = None

    contrastive_map = _load_contrastive_map(args.contrastive_map_jsonl) if getattr(args, "contrastive_map_jsonl", None) else {}

    processor_kwargs = {"use_fast": True}
    if args.min_pixels is not None and args.min_pixels > 0:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None and args.max_pixels > 0:
        processor_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.hf_model_name_or_path, **processor_kwargs)

    def build_messages_for(rel_img_path: str) -> List[Dict[str, Any]]:
        q_abs = image_dir / rel_img_path
        if not q_abs.exists():
            raise FileNotFoundError(f"Query image not found: {q_abs}")
        try:
            with Image.open(q_abs) as _im:
                _ = _im.size
        except Exception as ex:
            raise RuntimeError(f"Cannot open query image with PIL: {q_abs} ({ex})")

        if mode == "zeroshot_all200":
            rfmt = _response_format_for(args.prompt_template)
            return render_zeroshot_all200(template_txt, tax, q_abs, rfmt)
        if mode == "zeroshot_identify":
            rfmt = _response_format_for(args.prompt_template)
            return render_zeroshot_identify(template_txt, tax, q_abs, rfmt)

        rec = None
        for v in _variants_for_path_str(rel_img_path, image_dir):
            rec = topk_index.get(v)
            if rec:
                break
        if not rec:
            for v in _variants_for_path_str(str(q_abs), image_dir):
                rec = topk_index.get(v)
                if rec:
                    break
        if not rec:
            raise KeyError(f"No top-k record for {rel_img_path}")

        cls = rec.get("topk_cls") or []
        probs = rec.get("topk_probs") or rec.get("topk_prob") or []
        pairs: List[Tuple[int, Optional[float]]] = []
        for k in range(min(5, len(cls))):
            try:
                cid = int(cls[k])
            except Exception:
                continue
            p = None
            if k < len(probs):
                try:
                    p = float(probs[k])
                except Exception:
                    p = None
            pairs.append((cid, p))

        with_conf = bool(flags.get("with_conf"))
        with_sci  = bool(flags.get("with_sci"))

        if mode == "top5_text":
            return render_top5_text(template_txt, with_conf, with_sci, q_abs, pairs, tax)
        if mode == "top5_flat_taxonomy": 
            return render_top5_flat_taxonomy(template_txt, with_conf, q_abs, pairs, tax)
        if mode == "top5_multimodal":
            if not args.ref_image_dir:
                raise RuntimeError("--ref-image-dir is required for multimodal templates")
            return render_top5_multimodal(template_txt, with_conf, with_sci, q_abs, pairs, tax,
                                          Path(args.ref_image_dir), args.resize_candidates,
                                          debug_resize=args.debug_resize, with_taxonomy=bool(flags.get("with_taxonomy")),)
        if mode == "top5_desc_text":
            return render_top5_desc_text(template_txt, with_conf, with_sci, q_abs, pairs, tax, descriptions)
        if mode == "top5_contrastive_group_text":
            return render_top5_contrastive_group_text(template_txt, with_conf, with_sci, q_abs, pairs, tax, contrastive_map)
        if mode in ("top5_desc_multimodal", "top5_desc_multimodal_rank"):
            if not args.ref_image_dir:
                raise RuntimeError("--ref-image-dir is required for multimodal templates")
            rank_mode = (mode == "top5_desc_multimodal_rank")
            with_taxonomy = bool(flags.get("with_taxonomy"))
            return render_top5_desc_multimodal(template_txt, rank_mode, with_conf, with_sci, q_abs, pairs, tax,
                                               descriptions, Path(args.ref_image_dir), args.resize_candidates,
                                               debug_resize=args.debug_resize, with_taxonomy=with_taxonomy,)

        raise RuntimeError(f"Unknown rendering mode: {mode}")

    msgs0 = build_messages_for(df.iloc[0]["image_path"])
    print_prompt_for_debug(processor, msgs0)

    if args.dry_run:
        errs: List[str] = []
        ok = 0
        for rel in df["image_path"].astype(str).tolist():
            q_abs = image_dir / rel
            if not q_abs.exists():
                errs.append(f"Missing test image: {q_abs}")
                continue
            try:
                with Image.open(q_abs) as _im:
                    _ = _im.size
            except Exception as ex:
                errs.append(f"Cannot open test image with PIL: {q_abs} ({ex})")
                continue

            if needs_top5:
                found = False
                for v in _variants_for_path_str(rel, image_dir):
                    if v in topk_index:
                        found = True; break
                if not found:
                    for v in _variants_for_path_str(str(q_abs), image_dir):
                        if v in topk_index:
                            found = True; break
                if not found:
                    errs.append(f"No topk entry for: {rel}")
                    continue

                if ("multimodal" in mode) and args.ref_image_dir:
                    rec = None
                    for v in _variants_for_path_str(rel, image_dir):
                        rec = topk_index.get(v)
                        if rec: break
                    if not rec:
                        for v in _variants_for_path_str(str(q_abs), image_dir):
                            rec = topk_index.get(v)
                            if rec: break
                    if rec:
                        for k in range(min(5, len(rec.get("topk_cls") or []))):
                            try:
                                cid = int(rec["topk_cls"][k])
                            except Exception:
                                cid = None
                            if cid is None:
                                continue
                            cand_path = Path(args.ref_image_dir) / f"{cid}.jpg"
                            if not cand_path.exists():
                                errs.append(f"Missing ref image: {cand_path} (for {rel})")

            ok += 1

        # Optional: show again on dry-run
        msgs0 = build_messages_for(df.iloc[0]["image_path"])
        print_prompt_for_debug(processor, msgs0)

        print(f"DRY RUN SUMMARY: ok={ok}, errors={len(errs)}")
        if errs:
            print("DRY RUN ERRORS:")
            for e in errs:
                print(" -", e)
        return

    print(f">> Loading Hugging Face model: {args.hf_model_name_or_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.hf_model_name_or_path,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else "auto"),
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    try:
        model.generation_config.use_cache = not args.no_kv_cache
    except Exception:
        pass
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    out_csv = Path(args.output_csv)
    err_file = Path(args.error_file)
    processed: set[Tuple[str, str]] = set()
    if out_csv.exists() and not args.no_resume:
        try:
            with out_csv.open("r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                if {"image_path", "prompt_template"}.issubset(rdr.fieldnames or []):
                    for row in rdr:
                        processed.add((row["image_path"], row["prompt_template"]))
        except Exception:
            pass

    out_cols = ["image_path", "prompt_template", "prompt", "answer"]
    write_header = not (out_csv.exists() and out_csv.stat().st_size > 0)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    err_file.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_csv.open("a", encoding="utf-8", newline="")
    err_f = err_file.open("a", encoding="utf-8", newline="")
    out_writer = csv.writer(out_f); err_writer = csv.writer(err_f, delimiter="\t")

    try:
        if write_header:
            out_writer.writerow(out_cols)

        print(">> Starting generation ...")
        print(f">> Template: {args.prompt_template}  |  File: {template_file}  |  Mode: {mode}")

        images = df["image_path"].astype(str).tolist()
        iterable = tqdm(images, desc="Generating", unit="img")
        for i, rel in enumerate(iterable, 1):
            if not args.no_resume and (rel, args.prompt_template) in processed:
                iterable.set_postfix_str("skipped")
                continue
            try:
                if i <= max(args.warmup, 0):
                    pass

                messages = build_messages_for(rel)

                prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_for_csv = prompt_text_for_csv(processor, messages)

                image_inputs, video_inputs = process_vision_info(messages)
                expects_images = ("<|image_pad|>" in prompt_text)
                if expects_images and (not image_inputs and not video_inputs):
                    raise RuntimeError("Prompt expects images (<|image_pad|>) but no images were loaded for this row.")

                proc_kwargs: Dict[str, Any] = dict(text=[prompt_text], padding=True, return_tensors="pt")
                if image_inputs:
                    proc_kwargs["images"] = image_inputs
                if video_inputs:
                    proc_kwargs["videos"] = video_inputs

                inputs = processor(**proc_kwargs)
                model_device = next(model.parameters()).device
                for k, v in inputs.items():
                    if hasattr(v, "to"):
                        inputs[k] = v.to(model_device)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                with torch.inference_mode():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=(not args.no_kv_cache),
                        do_sample=False,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen)]
                decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                answer = decoded[0] if decoded else ""
                out_writer.writerow([
                    rel, args.prompt_template,
                    prompt_for_csv.replace("\n", "\\n"),
                    answer.replace("\n", "\\n")
                ])
                out_f.flush()

                del inputs, image_inputs, video_inputs, gen, trimmed, decoded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                if args.throttle_sec and args.throttle_sec > 0:
                    time.sleep(args.throttle_sec)

            except Exception as e:
                err_writer.writerow([rel, f"{type(e).__name__}: {e}"]); err_f.flush()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                continue
    finally:
        out_f.close(); err_f.close()

if __name__ == "__main__":
    main()
