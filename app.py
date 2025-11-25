# app.py
# Streamlit app with integrated model2 functionality (Order Validation + Inventory Dashboard)
# Model heavy initialization is lazy and only happens when you enable "Run model verification".

import streamlit as st
import os
import json
from collections import defaultdict
from PIL import Image, ExifTags, ImageDraw, ImageFont
from pathlib import Path
import pandas as pd
import traceback

# ------------------------------
# MUST be the first Streamlit command
# ------------------------------
st.set_page_config(layout="wide", page_title="Bin Order Validator")

# ------------------------------
# Config
# ------------------------------
IMAGES_DIR = "data/images"
METADATA_DIR = "data/metadata"
MODEL_CKPT_DIR = "grounding_dino_ckpt"
ANNOTATED_OUT = "data/outputs/annotated"
os.makedirs(ANNOTATED_OUT, exist_ok=True)
os.makedirs(MODEL_CKPT_DIR, exist_ok=True)

# ------------------------------
# Helpers: read metadata
# ------------------------------
def read_json_sidecar(p: Path):
    sidecar = p.with_suffix(".json")
    if sidecar.exists():
        try:
            return json.load(open(sidecar, "r", encoding="utf-8"))
        except:
            return None
    return None

def read_embedded_exif(p: Path):
    try:
        im = Image.open(p)
        exif = im._getexif()
        if not exif:
            return None
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        for key in ("UserComment", "ImageDescription", "XPComment"):
            val = tags.get(key)
            if not val:
                continue
            if isinstance(val, bytes):
                try:
                    val = val.decode("utf-8", errors="ignore")
                except:
                    val = str(val)
            try:
                return json.loads(val)
            except:
                continue
    except Exception:
        return None
    return None

def build_metadata_index(images_dir=IMAGES_DIR, metadata_dir=METADATA_DIR):
    meta_index = {}
    asin_index = defaultdict(list)
    images_dir = Path(images_dir)
    metadata_dir = Path(metadata_dir)
    if not images_dir.exists():
        return meta_index, asin_index
    for img_path in sorted(images_dir.glob("*.*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = img_path.stem
        sidecar = metadata_dir / f"{image_id}.json"

        data = None
        if sidecar.exists():
            try:
                data = json.load(open(sidecar, "r", encoding="utf-8"))
            except:
                data = None
        if data is None:
            data = read_embedded_exif(img_path)

        if not data:
            meta_index[image_id] = []
            continue

        bin_data = data.get("BIN_FCSKU_DATA", {})
        items = []
        for asin_key, item in bin_data.items():
            qty = item.get("quantity", 0)
            try:
                qty = int(qty)
            except:
                qty = 0
            entry = {
                "asin": asin_key,
                "quantity": qty,
                "name": item.get("name") or item.get("normalizedName") or asin_key,
                "normalizedName": item.get("normalizedName") or item.get("name")
            }
            items.append(entry)
            asin_index[asin_key].append((image_id, qty))
        meta_index[image_id] = items

    # sort images per ASIN by available qty desc
    for asin, lst in asin_index.items():
        asin_index[asin] = sorted(lst, key=lambda x: x[1], reverse=True)
    return meta_index, asin_index

# ------------------------------
# Allocation logic (greedy)
# ------------------------------
def allocate_min_images_for_asin(needed_qty, image_qty_list):
    chosen = []
    remaining = needed_qty
    for img_id, avail in image_qty_list:
        if remaining <= 0:
            break
        use = min(avail, remaining)
        if use > 0:
            chosen.append((img_id, use))
            remaining -= use
    return chosen, remaining

def multi_asin_allocator(order_items, asin_index):
    allocation = defaultdict(lambda: defaultdict(int))
    shortages = {}
    for asin, need in order_items.items():
        imgs = asin_index.get(asin, [])
        chosen, remaining = allocate_min_images_for_asin(need, imgs)
        for img_id, used in chosen:
            allocation[img_id][asin] += used
        if remaining > 0:
            shortages[asin] = remaining
    return allocation, shortages

# ------------------------------
# Build indices from metadata
# ------------------------------
meta_index, asin_index = build_metadata_index(IMAGES_DIR, METADATA_DIR)

# Build product dict (asin -> normalized name)
product_dict = {}
for image_id, items in meta_index.items():
    for it in items:
        asin = it["asin"]
        name = it.get("normalizedName") or it.get("name") or asin
        product_dict[asin] = name

# ------------------------------
# Session state init
# ------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = ""

# Placeholder auth UI (uses auth.py if present)
try:
    from auth import init_user_db, signup_user, login_user
    init_user_db()
    use_auth = True
except Exception:
    use_auth = False

if not st.session_state.logged_in:
    st.title("🔒 Login / Signup")
    if use_auth:
        tab1, tab2 = st.tabs(["Login", "Signup"])
        with tab1:
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                success, msg = login_user(email, password)
                st.info(msg)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.rerun()
        with tab2:
            reg_email = st.text_input("New Email")
            reg_pw = st.text_input("New Password", type="password")
            if st.button("Create Account"):
                success, msg = signup_user(reg_email, reg_pw)
                st.info(msg)
    else:
        st.info("Auth module not found. Running in demo mode.")
        if st.button("Continue as demo user"):
            st.session_state.logged_in = True
            st.session_state.user_email = "demo@example.com"
            st.rerun()
    st.stop()

# ------------------------------
# Lazy Model Integration (functions pulled/adapted from your model2 notebook)
# The heavy work (downloading checkpoint, loading models) happens only when needed.
# ------------------------------
MODEL_AVAILABLE = False
_model_resources = {}

def load_model_resources():
    """
    Lazily load groundingdino + CLIP + FAISS resources.
    Returns dict with keys: device, clip_model, clip_preprocess, dino_model, encode_texts_clip,
    encode_image_clip, detect_regions_dino, match_regions_to_asins, smart_count_objects, draw_boxes_with_counts, build_asin_text_map
    """
    global MODEL_AVAILABLE, _model_resources
    if _model_resources:
        return _model_resources

    try:
        import torch, clip, faiss, urllib.request, glob, cv2, numpy as np
        from groundingdino.util.inference import Model as GroundingDINOModel
        from typing import List, Dict
        from PIL import Image
    except Exception as e:
        st.error(f"Could not import model dependencies: {e}")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # helper functions (copied/adapted)
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
        try:
            data = load_json(meta_path)
        except:
            return {}
        asin_text = {}
        for asin, rec in (data.get("BIN_FCSKU_DATA") or {}).items():
            asin_text[asin] = make_asin_text(rec)
        return asin_text

    # load CLIP
    try:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    except Exception as e:
        st.error(f"Failed to load CLIP model: {e}")
        return {}

    def encode_texts_clip(texts):
        with torch.no_grad():
            toks = clip.tokenize(texts, truncate=True).to(device)
            e = clip_model.encode_text(toks)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy().astype("float32")

    def encode_image_clip(pil_img):
        with torch.no_grad():
            t = clip_preprocess(pil_img).unsqueeze(0).to(device)
            e = clip_model.encode_image(t)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy().astype("float32")

    def build_faiss_index(embs):
        idx = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs.astype("float32"))
        return idx

    # grounding dino config + checkpoint paths
    CONFIG_PATH = os.path.join(MODEL_CKPT_DIR, "GroundingDINO_SwinT_OGC.py")
    CKPT_PATH   = os.path.join(MODEL_CKPT_DIR, "groundingdino_swint_ogc.pth")

    # download if missing (only if internet available). Wrap in try to avoid breaking offline.
    try:
        if not os.path.exists(CONFIG_PATH):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                CONFIG_PATH
            )
        if not os.path.exists(CKPT_PATH):
            urllib.request.urlretrieve(
                "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
                CKPT_PATH
            )
    except Exception:
        # ignore download failures here; attempt to load may still fail
        pass

    # load grounding dino model
    try:
        dino_model = GroundingDINOModel(
            model_config_path=CONFIG_PATH,
            model_checkpoint_path=CKPT_PATH,
            device=device
        )
    except Exception as e:
        st.error(f"Failed to load GroundingDINO model: {e}")
        return {}

    # detection function
    def detect_regions_dino(img, conf=0.25, max_det=400):
        try:
            np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            result = dino_model.predict_with_caption(
                image=np_img,
                caption="object .",
                box_threshold=conf,
                text_threshold=0.25
            )
            detections, phrases = result
            out = []
            if detections is None or len(detections) == 0:
                return out
            xyxy = detections.xyxy
            scores = detections.confidence
            for (x1, y1, x2, y2), sc in zip(xyxy, scores):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if x2 > x1 and y2 > y1:
                    out.append((x1, y1, x2, y2, float(sc)))
            return out[:max_det]
        except Exception:
            return []

    def match_regions_to_asins(img: Image.Image, boxes_scored, asin_texts, sim_th=0.20):
        if not asin_texts:
            return []
        asin_list = list(asin_texts.keys())
        text_list = [asin_texts[a] for a in asin_list]
        text_embs = encode_texts_clip(text_list)
        index = build_faiss_index(text_embs)
        dets = []
        for (x1, y1, x2, y2, dino_conf) in boxes_scored:
            try:
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
            except Exception:
                continue
        return dets

    # smart counting (semantic-spatial deduplication)
    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0
        return iou

    def is_contained(inner, outer):
        ix1, iy1, ix2, iy2 = inner
        ox1, oy1, ox2, oy2 = outer
        return (ix1 >= ox1) and (iy1 >= oy1) and (ix2 <= ox2) and (iy2 <= oy2)

    def smart_count_objects(dets, iou_thresh=0.5):
        from collections import defaultdict
        final_dets = []
        counts = {}
        grouped = defaultdict(list)
        for d in dets:
            grouped[d['asin']].append(d)
        for asin, items in grouped.items():
            items.sort(key=lambda x: x['score'], reverse=True)
            kept_for_asin = []
            while items:
                current = items.pop(0)
                keep_it = True
                for existing in kept_for_asin:
                    iou = calculate_iou(current['box'], existing['box'])
                    if iou > iou_thresh:
                        keep_it = False
                        break
                    if is_contained(current['box'], existing['box']) or is_contained(existing['box'], current['box']):
                        keep_it = False
                        break
                if keep_it:
                    kept_for_asin.append(current)
            counts[asin] = len(kept_for_asin)
            final_dets.extend(kept_for_asin)
        return counts, final_dets

    def draw_boxes_with_counts(img: Image.Image, dets, counts, out_path: str):
        img = img.copy()
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            header_font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            header_font = ImageFont.load_default()
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            asin, sc = d["asin"], d.get("score", 0.0)
            color = "lime" if sc > 0.25 else "yellow"
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
            text = f"{asin}\nConf: {sc:.2f}"
            try:
                bbox = draw.textbbox((x1, y1), text, font=font)
                draw.rectangle(bbox, fill="black")
            except Exception:
                pass
            draw.text((x1, y1), text, fill="white", font=font)
        # Summary
        try:
            draw.rectangle([(0, 0), (300, len(counts)*30 + 20)], fill=(0,0,0,180))
            draw.text((10, 5), "INVENTORY COUNT:", fill="white", font=header_font)
            for i, (asin, qty) in enumerate(counts.items()):
                line = f"{asin}: {qty}"
                draw.text((10, 30 + i*25), line, fill="cyan", font=font)
        except Exception:
            pass
        img.save(out_path)

    # pack and return
    _model_resources = {
        "device": device,
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "dino_model": dino_model,
        "encode_texts_clip": encode_texts_clip,
        "encode_image_clip": encode_image_clip,
        "detect_regions_dino": detect_regions_dino,
        "match_regions_to_asins": match_regions_to_asins,
        "smart_count_objects": smart_count_objects,
        "draw_boxes_with_counts": draw_boxes_with_counts,
        "build_asin_text_map": build_asin_text_map
    }

    MODEL_AVAILABLE = True
    return _model_resources

def run_model_on_image(image_path: str):
    """
    Run the detection+matching+counting pipeline on a single image path.
    Returns a dict: {"counts": {...}, "detections": [...], "annotated": path_or_none}
    """
    try:
        res = load_model_resources()
        if not res:
            return {"error":"model_load_failed"}
        detect = res["detect_regions_dino"]
        match = res["match_regions_to_asins"]
        smart = res["smart_count_objects"]
        draw = res["draw_boxes_with_counts"]
        build_map = res["build_asin_text_map"]

        img_id = Path(image_path).stem
        meta_path = str(Path(METADATA_DIR) / f"{img_id}.json")
        if not Path(meta_path).exists():
            asin_texts = {}
        else:
            asin_texts = build_map(meta_path)

        img = Image.open(image_path).convert("RGB")
        boxes = detect(img, conf=0.05, max_det=400)
        dets = match(img, boxes, asin_texts, sim_th=0.20)
        counts, clean_dets = smart(dets, iou_thresh=0.45)

        annotated_path = os.path.join(ANNOTATED_OUT, f"{img_id}_annotated.jpg")
        try:
            draw(img, clean_dets, counts, annotated_path)
        except Exception:
            annotated_path = None

        return {"counts": counts, "detections": clean_dets, "annotated": annotated_path}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# ------------------------------
# UI + layout
# ------------------------------
st.markdown(
    """
    <style>
      .big-title { font-size:38px; font-weight:800; margin-bottom:6px; color:#ffe8a3; }
      .subtitle { color:#9fb3d4; margin-top:0px; margin-bottom:12px; }
      .nav-card { background:#0b1117; padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,0.02); }
      .fade { animation: fadeIn 0.35s ease-in-out; }
      @keyframes fadeIn { from {opacity:0; transform: translateY(6px);} to {opacity:1; transform: translateY(0);} }
    </style>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([9,1])
with header_left:
    st.markdown(f'<div class="big-title">📦 Bin Order Validator</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Fast order validation — use image metadata and optional AI verification</div>', unsafe_allow_html=True)
with header_right:
    st.markdown(f"<div style='text-align:right; color:#cbd8ea'>Signed in: <strong>{st.session_state.user_email or 'demo'}</strong></div>", unsafe_allow_html=True)

st.write("---")

# navigation
nav_col, main_col = st.columns([1,4])
with nav_col:
    st.markdown('<div class="nav-card">', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Order Validation", "Inventory Dashboard"], index=0 if "page" not in st.session_state else ["Order Validation","Inventory Dashboard"].index(st.session_state.page))
    st.markdown('</div>', unsafe_allow_html=True)
    st.session_state.page = page

# Add little animation wrapper
with main_col:
    st.markdown('<div class="fade">', unsafe_allow_html=True)

    # ------------------------------
    # ORDER VALIDATION PAGE
    # ------------------------------
    if st.session_state.page == "Order Validation":
        st.header("Create Order (Select Product Names)")
        st.caption("Select products and quantities → Validate → optional AI verification")

        # initialize rows if absent
        if "order_rows" not in st.session_state:
            default_name = next(iter(product_dict.values()), "")
            st.session_state.order_rows = [{"name": default_name, "qty": 1}]

        # add row button
        if st.button("➕ Add product row"):
            st.session_state.order_rows.append({"name": next(iter(product_dict.values()), ""), "qty": 1})
            st.rerun()

        # render rows
        product_options = sorted(set(product_dict.values()))
        name_to_asin = {v:k for k,v in product_dict.items()}

        rows = st.session_state.order_rows
        new_rows = []
        for i, r in enumerate(rows):
            c1, c2, c3 = st.columns([4,1,1])
            selected = c1.selectbox(f"Product {i+1}", product_options, index=product_options.index(r["name"]) if r["name"] in product_options else 0, key=f"name_{i}")
            qty = c2.number_input(f"Qty {i+1}", min_value=0, value=int(r.get("qty",1)), key=f"qty_{i}")
            remove = c3.button("Remove", key=f"rm_{i}")
            if not remove:
                new_rows.append({"name": selected, "qty": int(qty)})
        st.session_state.order_rows = new_rows

        # Build order_items
        order_items = { name_to_asin[r["name"]]: r["qty"] for r in st.session_state.order_rows if r["name"] and r["qty"]>0 }

        # Validate button & allocation
        if st.button("✔ Validate Order"):
            allocation, shortages = multi_asin_allocator(order_items, asin_index)
            st.success("Allocation computed")
            st.session_state.last_allocation = allocation
            st.session_state.last_shortages = shortages
            st.rerun()

        # show last results if present
        if "last_allocation" in st.session_state:
            allocation = st.session_state.get("last_allocation", {})
            shortages = st.session_state.get("last_shortages", {})
            st.header("Validation Result")
            if not shortages:
                st.success("✅ Order can be fulfilled (metadata allocation)")
            else:
                st.error("⚠️ Shortages detected")
                for asin, missing in shortages.items():
                    st.write(f"- {product_dict.get(asin, asin)} : missing {missing}")

            # show images used
            if allocation:
                st.subheader("Selected images to fulfill the order (fewest images)")
                chosen_images = sorted(list(allocation.keys()), key=lambda img: sum(allocation[img].values()), reverse=True)
                imgs_per_row = 3
                for idx in range(0, len(chosen_images), imgs_per_row):
                    cols = st.columns(imgs_per_row)
                    for j, img_id in enumerate(chosen_images[idx: idx+imgs_per_row]):
                        col = cols[j]
                        img_path_jpg = Path(IMAGES_DIR) / f"{img_id}.jpg"
                        img_path_png = Path(IMAGES_DIR) / f"{img_id}.png"
                        img_path = img_path_jpg if img_path_jpg.exists() else (img_path_png if img_path_png.exists() else None)
                        col.markdown(f"**Image: {img_id}**")
                        if img_path:
                            col.image(str(img_path), use_column_width=True)
                            col.write("Contributions:")
                            col.write(dict(allocation[img_id]))
                        else:
                            col.write("Image file not found")

            # allocation table
            st.subheader("Allocation table")
            alloc_rows = []
            for img, items in allocation.items():
                row = {"image_id": img}
                for a, q in items.items():
                    row[product_dict.get(a, a)] = q
                alloc_rows.append(row)
            if alloc_rows:
                df_alloc = pd.DataFrame(alloc_rows).fillna(0)
                st.dataframe(df_alloc)
            else:
                st.info("No allocation results to show.")

        # ------------------------------
        # Model Verification Controls
        # ------------------------------
        st.markdown("---")
        st.markdown("### AI Verification (optional)")
        st.markdown("If you enable AI verification the app will run GroundingDINO + CLIP on the images selected by the allocation (or all images if none selected). This is slower and may download model files on first run.")
        enable_model = st.checkbox("Enable AI model verification (GroundingDINO + CLIP)", value=False)

        if enable_model:
            # Try to load model resources (lazy)
            st.info("Loading model resources (may download/initialize heavy files). Be patient on first run.")
            resources = load_model_resources()
            if not resources:
                st.error("Model failed to initialize. Check console/logs for errors and ensure dependencies (groundingdino, clip, faiss) are installed.")
            else:
                st.success(f"Model resources loaded on device: {resources.get('device')}")
                # Button to run verification
                if st.button("🔬 Run AI verification on allocated images"):
                    allocation = st.session_state.get("last_allocation", {})
                    if allocation and len(allocation)>0:
                        images_to_check = sorted(allocation.keys())
                    else:
                        images_to_check = sorted(list(meta_index.keys()))
                    st.info(f"Running verification on {len(images_to_check)} images")
                    progress = st.progress(0)
                    total = len(images_to_check)
                    aggregated_counts = defaultdict(int)
                    per_image = {}
                    for i, img_id in enumerate(images_to_check):
                        img_path_jpg = Path(IMAGES_DIR) / f"{img_id}.jpg"
                        img_path_png = Path(IMAGES_DIR) / f"{img_id}.png"
                        img_path = img_path_jpg if img_path_jpg.exists() else (img_path_png if img_path_png.exists() else None)
                        if not img_path:
                            progress.progress(int(((i+1)/total)*100))
                            continue
                        res = run_model_on_image(str(img_path))
                        if "error" in res:
                            st.warning(f"Model error on {img_id}: {res.get('error')}")
                            per_image[img_id] = res
                        else:
                            per_image[img_id] = res
                            for a, q in res.get("counts", {}).items():
                                aggregated_counts[a] += q
                        progress.progress(int(((i+1)/total)*100))
                    st.session_state.model_aggregated_counts = dict(aggregated_counts)
                    st.session_state.model_per_image = per_image
                    st.success("AI verification finished.")
                    st.rerun()

        # show model results if exist
        if "model_aggregated_counts" in st.session_state:
            st.markdown("### Model verification summary (aggregated)")
            agg = st.session_state.model_aggregated_counts
            if not agg:
                st.warning("Model did not detect items across analyzed images.")
            else:
                rows = []
                for asin, qty in agg.items():
                    rows.append((product_dict.get(asin, asin), asin, qty))
                st.dataframe(pd.DataFrame(rows, columns=["Product Name","ASIN","Detected Qty"]), use_container_width=True)

            if order_items:
                st.markdown("### Compare AI detected counts vs requested order")
                compare_rows = []
                for asin, req in order_items.items():
                    detected = st.session_state.model_aggregated_counts.get(asin, 0)
                    name = product_dict.get(asin, asin)
                    status = "OK" if detected >= req else ("MISSING" if detected==0 else "UNDERSUPPLIED")
                    compare_rows.append((name, asin, req, detected, status))
                cdf = pd.DataFrame(compare_rows, columns=["Product","ASIN","Requested","Detected","Status"])
                st.table(cdf)

            st.markdown("### Per-image AI visualizations")
            for imgid, info in st.session_state.model_per_image.items():
                with st.expander(f"Image {imgid}"):
                    if isinstance(info, dict) and info.get("annotated"):
                        st.image(info["annotated"], caption=f"Annotated {imgid}", use_column_width=True)
                    st.write("Counts:", info.get("counts", {}))
                    if info.get("detections"):
                        st.write("Sample detections:")
                        st.write(info.get("detections")[:10])

    # ------------------------------
    # INVENTORY DASHBOARD PAGE
    # ------------------------------
    elif st.session_state.page == "Inventory Dashboard":
        st.header("📊 Full Inventory Dashboard")
        total_counts = { asin: sum(q for _, q in lst) for asin, lst in asin_index.items() }
        df_inv = pd.DataFrame(
            [(asin, product_dict.get(asin, "Unknown"), total_counts[asin]) for asin in total_counts],
            columns=["ASIN", "Product Name", "TotalAvailable"]
        )
        search = st.text_input("Search by name or ASIN")
        if search:
            df_inv = df_inv[df_inv["Product Name"].str.contains(search, case=False, na=False) | df_inv["ASIN"].str.contains(search, case=False, na=False)]
        sort_option = st.selectbox("Sort by", ["Product Name (A-Z)", "Quantity (High → Low)"])
        if sort_option == "Product Name (A-Z)":
            df_inv = df_inv.sort_values("Product Name", ascending=True)
        else:
            df_inv = df_inv.sort_values("TotalAvailable", ascending=False)
        st.dataframe(df_inv.reset_index(drop=True), use_container_width=True, height=700)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer: quick status
st.write("---")
st.markdown("**Notes:** Model verification is optional. If you enable it the app will attempt to download/initialize GroundingDINO and CLIP on first use. If you don't want downloads, leave AI verification disabled and rely on metadata-only validation.")
