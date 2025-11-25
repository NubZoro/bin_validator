#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-24T19:51:05.016Z
"""

import os
os.environ["TRANSFORMERS_NO_TF_IMPORT"] = "1"


import os, glob, json
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch, clip, faiss

from groundingdino.util.inference import Model as GroundingDINOModel

# --- Paths ---
IMG_DIR  = "data/images"
META_DIR = "data/metadata"
OUT_DIR  = "data/outputs"
VIS_DIR  = os.path.join(OUT_DIR, "visuals_dino")
os.makedirs(VIS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(" Device:", device)

def list_imgs(d):
    return sorted([p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith((".jpg", ".png", ".jpeg"))])

def meta_for_image(img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    path = os.path.join(META_DIR, stem + ".json")
    return path if os.path.exists(path) else None

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d, *keys, default=""):
    v = d
    for k in keys:
        if isinstance(v, dict) and k in v:
            v = v[k]
        else:
            return default
    return v

def make_asin_text(rec: dict) -> str:
    name = rec.get("normalizedName") or rec.get("name") or ""
    h = safe_get(rec, "height", "value")
    w = safe_get(rec, "width", "value")
    l = safe_get(rec, "length", "value")
    wt = safe_get(rec, "weight", "value")
    def fmt(x):
        try: return str(round(float(x), 2))
        except: return str(x)
    dims = " x ".join([fmt(l), fmt(w), fmt(h)])
    txt = f"{name}, size {dims} inches, weight {fmt(wt)} pounds"
    return txt.strip()

def build_asin_text_map(meta_path: str) -> Dict[str, str]:
    data = load_json(meta_path)
    asin_text = {}
    for asin, rec in (data.get("BIN_FCSKU_DATA") or {}).items():
        asin_text[asin] = make_asin_text(rec)
    return asin_text


clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def encode_texts_clip(texts: List[str]) -> np.ndarray:
    with torch.no_grad():
        toks = clip.tokenize(texts, truncate=True).to(device)
        e = clip_model.encode_text(toks)
        e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().numpy().astype("float32")

def encode_image_clip(pil_img: Image.Image) -> np.ndarray:
    with torch.no_grad():
        t = clip_preprocess(pil_img).unsqueeze(0).to(device)
        e = clip_model.encode_image(t)
        e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().numpy().astype("float32")

def build_faiss_index(embs: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype("float32"))
    return index


import os
import urllib.request
from groundingdino.util.inference import Model as GroundingDINOModel

# --- 1️⃣ Create folder for model files ---
os.makedirs("grounding_dino_ckpt", exist_ok=True)

# --- 2️⃣ Define file paths ---
CONFIG_PATH = "grounding_dino_ckpt/GroundingDINO_SwinT_OGC.py"
CKPT_PATH   = "grounding_dino_ckpt/groundingdino_swint_ogc.pth"

# --- 3️⃣ Download files if not already present ---
if not os.path.exists(CONFIG_PATH):
    print("⬇️  Downloading Grounding DINO config file...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        CONFIG_PATH
    )

if not os.path.exists(CKPT_PATH):
    print("⬇️  Downloading Grounding DINO checkpoint (~380 MB, may take several minutes)...")
    urllib.request.urlretrieve(
        "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
        CKPT_PATH
    )

print("✅  Files ready at:", os.path.abspath("grounding_dino_ckpt"))

# --- 4️⃣ Load the model ---
dino_model = GroundingDINOModel(
    model_config_path=CONFIG_PATH,
    model_checkpoint_path=CKPT_PATH,
    device=device
)

import numpy as np
import cv2

def detect_regions_dino(img, conf=0.25, max_det=400):
    """
    Compatible with GroundingDINO + Supervision API.
    Returns list of (x1, y1, x2, y2, conf)
    """
    # Convert PIL → OpenCV BGR
    np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Run the model
    result = dino_model.predict_with_caption(
        image=np_img,
        caption="object .",
        box_threshold=conf,
        text_threshold=0.25
    )

    # Handle the two-return format (Detections + phrases)
    detections, phrases = result
    out = []

    if detections is None or len(detections) == 0:
        return out

    w, h = img.size

    # Convert detection boxes to pixel coordinates
    xyxy = detections.xyxy  # tensor or ndarray of shape (N, 4)
    scores = detections.confidence  # detection confidence per box

    for (x1, y1, x2, y2), sc in zip(xyxy, scores):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2, float(sc)))

    return out[:max_det]


def match_regions_to_asins(img: Image.Image, boxes_scored, asin_texts, sim_th=0.20):
    if not asin_texts:
        return []
    asin_list = list(asin_texts.keys())
    text_list = [asin_texts[a] for a in asin_list]
    text_embs = encode_texts_clip(text_list)
    index = build_faiss_index(text_embs)

    dets = []
    for (x1, y1, x2, y2, dino_conf) in boxes_scored:
        crop = img.crop((x1, y1, x2, y2))
        img_emb = encode_image_clip(crop)
        D, I = index.search(img_emb, 1)
        sim, idx = float(D[0][0]), int(I[0][0])
        if sim >= sim_th:
            dets.append({
                "asin": asin_list[idx],
                "score": sim,
                "dino_conf": dino_conf,
                "box": [x1, y1, x2, y2]
            })
    return dets


def draw_boxes(img: Image.Image, dets: List[dict], out_path: str):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        asin, sc = d["asin"], d["score"]
        draw.rectangle([(x1, y1), (x2, y2)], outline="lime", width=2)
        text = f"{asin} ({sc:.2f})"
        draw.text((x1+2, y1+2), text, fill="black", font=font)
    img.save(out_path)


SIM_TH   = 0.20
DINO_CONF= 0.05

imgs = list_imgs(IMG_DIR)
test_img = imgs[31]
meta_path = meta_for_image(test_img)
print(" Image:", test_img)
print(" Metadata:", meta_path)

img = Image.open(test_img).convert("RGB")
asin_texts = build_asin_text_map(meta_path)
required_asins = list(load_json(meta_path)["BIN_FCSKU_DATA"].keys())

# DINO proposals → CLIP matching
boxes_scored = detect_regions_dino(img, conf=DINO_CONF, max_det=400)
print(f"Grounding DINO V2 proposed {len(boxes_scored)} regions")

dets = match_regions_to_asins(img, boxes_scored, asin_texts, sim_th=SIM_TH)

detected_asins = sorted(set([d["asin"] for d in dets]))
print("\nDetected ASINs (existence only):")
for a in detected_asins:
    print(f"   {a}")

missing = [a for a in required_asins if a not in detected_asins]
if missing:
    print("\nPossibly missing / hidden ASINs:")
    for a in missing:
        print(f"   {a}")
else:
    print("\nAll ASINs appear visible in the image!")

# Visualization
out_path = os.path.join(VIS_DIR, os.path.basename(test_img))
draw_boxes(img, dets, out_path)
print("\nAnnotated image saved to:", out_path)


from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# --- Parameters ---
N_IMAGES = 100         # Number of test images
SIM_TH   = 0.20
DINO_CONF= 0.05
MAX_DET  = 400

# --- Storage ---
all_true, all_pred = [], []

imgs = list_imgs(IMG_DIR)[:1000]
print(f"Evaluating on first {len(imgs)} images...\n")

for img_path in tqdm(imgs):
    meta_path = meta_for_image(img_path)
    if meta_path is None:
        continue

    # Load data
    img = Image.open(img_path).convert("RGB")
    asin_texts = build_asin_text_map(meta_path)
    meta_data = load_json(meta_path).get("BIN_FCSKU_DATA", {})
    required_asins = list(meta_data.keys())

    # --- Grounding DINO proposals ---
    boxes_scored = detect_regions_dino(img, conf=DINO_CONF, max_det=MAX_DET)

    # --- CLIP matching ---
    dets = match_regions_to_asins(img, boxes_scored, asin_texts, sim_th=SIM_TH)
    detected_asins = sorted(set([d["asin"] for d in dets]))

    # --- Prepare labels ---
    # 1 = present, 0 = not present
    all_labels = sorted(set(required_asins + detected_asins))
    y_true = [1 if a in required_asins else 0 for a in all_labels]
    y_pred = [1 if a in detected_asins else 0 for a in all_labels]

    all_true.extend(y_true)
    all_pred.extend(y_pred)

# --- Compute metrics ---
precision = precision_score(all_true, all_pred)
recall = recall_score(all_true, all_pred)
f1 = f1_score(all_true, all_pred)

print("\n📊 Evaluation Summary (first 100 images)")
print(f"Precision : {precision:.3f}")
print(f"Recall    : {recall:.3f}")
print(f"F1-score  : {f1:.3f}")


def calculate_iou(boxA, boxB):
    # standard intersection over union
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def is_contained(inner, outer):
    # Checks if 'inner' box is roughly inside 'outer' box
    # Useful when DINO detects the text label separately from the box itself
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    return (ix1 >= ox1) and (iy1 >= oy1) and (ix2 <= ox2) and (iy2 <= oy2)

def smart_count_objects(dets: List[dict], iou_thresh=0.5) -> Tuple[dict, List[dict]]:
    """
    INNOVATIVE METHOD: Semantic-Spatial Deduplication
    1. Groups by ASIN (Semantic).
    2. Sorts by CLIP confidence.
    3. Removes spatial duplicates ONLY within the same ASIN.
    4. Handles 'nesting' (label detected inside object).
    """
    final_dets = []
    counts = {}
    
    # 1. Group by ASIN
    from collections import defaultdict
    grouped = defaultdict(list)
    for d in dets:
        grouped[d['asin']].append(d)
        
    # 2. Process each ASIN group independently
    for asin, items in grouped.items():
        # Sort by score (highest confidence first)
        items.sort(key=lambda x: x['score'], reverse=True)
        
        kept_for_asin = []
        
        while items:
            current = items.pop(0) # Take best remaining
            keep_it = True
            
            # Compare against already kept items of THIS ASIN
            for existing in kept_for_asin:
                iou = calculate_iou(current['box'], existing['box'])
                
                # 3. Spatial Deduplication
                if iou > iou_thresh:
                    keep_it = False
                    break
                
                # 4. Containment Check (if smaller box is inside larger box of same ASIN)
                if is_contained(current['box'], existing['box']) or is_contained(existing['box'], current['box']):
                    keep_it = False
                    break
            
            if keep_it:
                kept_for_asin.append(current)
        
        counts[asin] = len(kept_for_asin)
        final_dets.extend(kept_for_asin)
        
    return counts, final_dets

# --- VISUALIZATION WITH COUNTS ---
def draw_boxes_with_counts(img: Image.Image, dets: List[dict], counts: dict, out_path: str):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        # Try loading a larger font for the summary
        font = ImageFont.truetype("arial.ttf", 16)
        header_font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        
    # Draw Boxes
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        asin, sc = d["asin"], d["score"]
        
        # Color code based on confidence
        color = "lime" if sc > 0.25 else "yellow"
        
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        text = f"{asin}\nConf: {sc:.2f}"
        
        # Draw background for text for readability
        bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(bbox, fill="black")
        draw.text((x1, y1), text, fill="white", font=font)

    # Draw Summary Header (The "Dashboard" view)
    y_offset = 10
    draw.rectangle([(0, 0), (300, len(counts)*30 + 20)], fill=(0,0,0, 180))
    draw.text((10, 5), "INVENTORY COUNT:", fill="white", font=header_font)
    
    for i, (asin, qty) in enumerate(counts.items()):
        line = f"{asin}: {qty}"
        draw.text((10, 30 + i*25), line, fill="cyan", font=font)
        
    

# ============================================================
#   🔥 RUN FULL BENCHMARK ON FIRST 100 IMAGES
# ============================================================

N = 100   # number of images to test
image_list = imgs[:N]

id_tp = id_fp = id_fn = 0

qty_errors = []
qty_matches = 0
qty_detect_recall_hits = 0
total_items_gt = 0
total_items_pred = 0

print(f"\n🔍 Evaluating first {N} images...\n")

for img_path in tqdm(image_list):

    meta_path = meta_for_image(img_path)
    if meta_path is None:
        continue

    # --- Load image and metadata ---
    img = Image.open(img_path).convert("RGB")
    meta_json = load_json(meta_path)

    asin_texts = build_asin_text_map(meta_path)
    gt_asins = list(meta_json["BIN_FCSKU_DATA"].keys())

    # expected quantities per ASIN
    gt_quantities = {
        asin: int(rec.get("quantity", 1))
        for asin, rec in meta_json["BIN_FCSKU_DATA"].items()
    }

    total_items_gt += sum(gt_quantities.values())

    # ===============================
    # 1. DETECT REGIONS
    # ===============================
    boxes_scored = detect_regions_dino(img, conf=DINO_CONF, max_det=400)

    # ===============================
    # 2. CLIP MATCHING
    # ===============================
    raw_dets = match_regions_to_asins(img, boxes_scored, asin_texts, sim_th=SIM_TH)

    # ===============================
    # 3. SMART COUNTING (Your innovation)
    # ===============================
    counts, clean_dets = smart_count_objects(raw_dets, iou_thresh=0.45)

    total_items_pred += sum(counts.values())

    # ===============================
    # IDENTIFICATION METRICS
    # ===============================
    detected_asins = set(counts.keys())
    gt_asins_set = set(gt_asins)

    id_tp += len(detected_asins & gt_asins_set)
    id_fp += len(detected_asins - gt_asins_set)
    id_fn += len(gt_asins_set - detected_asins)

    # ===============================
    # QUANTITY METRICS
    # ===============================
    for asin in gt_asins:
        gt_q = gt_quantities.get(asin, 0)
        pred_q = counts.get(asin, 0)

        if pred_q == gt_q:
            qty_matches += 1

        if pred_q > 0:
            qty_detect_recall_hits += 1

        qty_errors.append(abs(pred_q - gt_q))


# ============================================================
# FINAL METRICS
# ============================================================

ident_prec = id_tp / max(id_tp + id_fp, 1)
ident_rec  = id_tp / max(id_tp + id_fn, 1)
ident_f1   = 2 * ident_prec * ident_rec / max(ident_prec + ident_rec, 1e-6)

qty_mae = np.mean(qty_errors)
qty_acc = qty_matches / (N * 1.0 * np.mean([len(load_json(meta_for_image(i))["BIN_FCSKU_DATA"]) for i in image_list]))
qty_rec = qty_detect_recall_hits / (N * 1.0 * np.mean([len(load_json(meta_for_image(i))["BIN_FCSKU_DATA"]) for i in image_list]))

total_count_mae = abs(total_items_pred - total_items_gt) / N


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n===================== FINAL REPORT =====================\n")

print("🔎 IDENTIFICATION METRICS")
print(f"Precision : {ident_prec:.4f}")
print(f"Recall    : {ident_rec:.4f}")
print(f"F1-score  : {ident_f1:.4f}")

print("\n📦 QUANTITY METRICS")
print(f"Per-ASIN MAE          : {qty_mae:.3f}")
print(f"Per-ASIN Accuracy     : {qty_acc:.3f}")
print(f"Per-ASIN Recall       : {qty_rec:.3f}")
print(f"Total Count MAE/Image : {total_count_mae:.3f}")

print("\n========================================================\n")