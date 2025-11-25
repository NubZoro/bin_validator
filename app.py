# app.py
import os
import io
import json
import base64
import requests
from pathlib import Path
from collections import defaultdict
from PIL import Image, ExifTags
import streamlit as st
import pandas as pd

# -----------------------------
# Configuration / constants
# -----------------------------
st.set_page_config(layout="wide", page_title="Bin Order Validator (Vision-QA)")

IMAGES_DIR = st.sidebar.text_input("Images dir", value="data/images")
METADATA_DIR = st.sidebar.text_input("Metadata dir", value="data/metadata")

# developer instruction: include uploaded file path from session
UPLOADED_PDF_PATH = "/mnt/data/Assignment #1 [Applied AI for Industry].pdf"

# OpenAI configuration (you must set OPENAI_API_KEY in env or via Streamlit secrets)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
OPENAI_MODEL = "gpt-4o-mini"  # vision-capable variant (adjust if your account uses a different name)

# Timeout / thresholds
QA_SIMPLIFY_MAX_PIXELS = 1600 * 1600  # avoid sending extremely huge bytes (we'll resize images in memory)
VERBOSE = False

# -----------------------------
# Helpers: metadata reading
# -----------------------------
def read_json_sidecar(p: Path):
    sidecar = p.with_suffix(".json")
    if sidecar.exists():
        try:
            return json.load(open(sidecar, "r", encoding="utf-8"))
        except Exception:
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
                    val = val.decode('utf-8', errors='ignore')
                except Exception:
                    val = str(val)
            try:
                return json.loads(val)
            except Exception:
                continue
    except Exception:
        return None
    return None

def build_metadata_index(images_dir=IMAGES_DIR, metadata_dir=METADATA_DIR):
    meta_index = {}
    asin_index = defaultdict(list)
    images_dir_p = Path(images_dir)
    if not images_dir_p.exists():
        return {}, {}
    for img_path in sorted(images_dir_p.glob("*.*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = img_path.stem
        sidecar = Path(metadata_dir) / f"{image_id}.json"

        data = None
        if sidecar.exists():
            try:
                data = json.load(open(sidecar, "r", encoding="utf-8"))
            except Exception:
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
            except Exception:
                qty = 0
            name = item.get("normalizedName") or item.get("name") or asin_key
            entry = {
                "asin": asin_key,
                "quantity": qty,
                "name": name,
                "raw": item
            }
            items.append(entry)
            asin_index[asin_key].append((image_id, qty))
        meta_index[image_id] = items

    for asin, lst in asin_index.items():
        asin_index[asin] = sorted(lst, key=lambda x: x[1], reverse=True)
    return meta_index, asin_index

# -----------------------------
# Allocation logic (greedy)
# -----------------------------
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

# -----------------------------
# Small image utilities
# -----------------------------
def load_image_for_api(path: Path, max_dim=1200):
    # Load, optionally resize to keep payload reasonable
    im = Image.open(path).convert("RGB")
    w, h = im.size
    max_edge = max(w, h)
    if max_edge > max_dim:
        scale = max_dim / max_edge
        new = (int(w * scale), int(h * scale))
        im = im.resize(new, Image.LANCZOS)
    return im

def image_to_base64_bytes(img: Image.Image, fmt="JPEG", quality=85):
    with io.BytesIO() as bio:
        img.save(bio, format=fmt, quality=quality)
        bio.seek(0)
        return bio.read()

# -----------------------------
# OpenAI Vision QA helper (uses simple HTTP to OpenAI / openai-python optional)
# This function attempts to use the OpenAI "responses" or chat endpoint that supports images.
# Exact API call may vary by account; adjust accordingly.
# -----------------------------
def ask_vision_qa_via_api(image_bytes: bytes, questions: list, api_key: str, model=OPENAI_MODEL, timeout=30):
    """
    Sends the image (base64) and a structured system/user prompt that asks per-question responses.
    Returns parsed JSON result or None on failure.
    NOTE: The exact endpoint & payload shape depends on your OpenAI client availability.
    Implementation below uses the "chat completions with images" pattern via the /v1/chat/completions endpoint.
    """
    if api_key is None:
        return None, "No API key configured"

    # prepare image as base64 string
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Build human-friendly prompt:
    question_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    system_prompt = (
        "You are a visual QA assistant. You will be given an image and a list of structured questions. "
        "Answer strictly in JSON with the fields for each question. Do NOT add extra commentary."
    )
    user_prompt = (
        "Image: (base64-encoded JPEG)\n\n"
        f"<<BASE64_IMAGE>>\n\n"
        "Questions:\n" + question_text +
        "\n\nReturn a JSON object with keys 'answers' which is a list of objects: "
        "{'question': <text>, 'present': true/false, 'visible_qty_est': <integer or 0>, 'confidence': <0-1 float>}."
    )

    # Replace token with base64 to keep payload readable
    user_prompt = user_prompt.replace("<<BASE64_IMAGE>>", img_b64)

    # Build request to the Chat Completions endpoint (works if your account supports image in chat)
    # Notes: Many OpenAI accounts support images in chat by passing image data in the message content (base64).
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # attempt safe parse
        # Chat completion content path depends on the model: 'choices'[0]['message']['content']
        content = None
        if "choices" in data and len(data["choices"]) > 0:
            c0 = data["choices"][0]
            content = c0.get("message", {}).get("content") or c0.get("text")
        if not content:
            return None, "No textual content returned by API"
        # try to find the first JSON object in the returned text
        import re
        m = re.search(r"(\{[\s\S]*\})", content)
        if not m:
            # return raw content as fallback
            return {"raw_text": content}, None
        json_text = m.group(1)
        parsed = json.loads(json_text)
        return parsed, None
    except Exception as e:
        return None, str(e)

# -----------------------------
# UI & App logic
# -----------------------------
def main():
    st.sidebar.title("Configuration & Info")
    st.sidebar.markdown("Vision-QA model: **gpt-4o-mini** (set OPENAI_API_KEY in env/secrets).")
    st.sidebar.markdown(f"Uploaded PDF (project doc): `{UPLOADED_PDF_PATH}`")
    if Path(UPLOADED_PDF_PATH).exists():
        st.sidebar.markdown(f"[Open Project PDF]({UPLOADED_PDF_PATH})")

    st.title("📦 Bin Order Validator — Vision-QA Verification")

    # Build metadata index (light and fast)
    meta_index, asin_index = build_metadata_index(IMAGES_DIR, METADATA_DIR)

    # Build product dict (asin -> name)
    product_dict = {}
    for image_id, items in meta_index.items():
        for it in items:
            product_dict[it["asin"]] = it.get("name", it.get("normalizedName", it["asin"]))

    # Authentication (simple demo)
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_email = ""

    if not st.session_state.logged_in:
        st.subheader("🔒 Login (demo)")
        email = st.text_input("Email", value="")
        if st.button("Continue as demo user"):
            st.session_state.logged_in = True
            st.session_state.user_email = email or "demo@example.com"
            st.experimental_rerun()
        st.stop()

    st.markdown(f"Signed in as **{st.session_state.user_email}**")

    # Navigation
    page = st.radio("Navigate", ["Order Validation", "Inventory Dashboard"], index=0)

    # Order Validation page
    if page == "Order Validation":
        st.header("Create Order (Select Product Names)")
        product_options = sorted(set(product_dict.values()))
        name_to_asin = {v: k for k, v in product_dict.items()}

        if "order_rows" not in st.session_state:
            st.session_state.order_rows = [{"name": product_options[0] if product_options else "", "qty": 1}]

        if st.button("➕ Add item"):
            st.session_state.order_rows.append({"name": product_options[0] if product_options else "", "qty": 1})

        new_rows = []
        for i, r in enumerate(st.session_state.order_rows):
            cols = st.columns([4, 1, 1])
            selected_name = cols[0].selectbox(f"Product {i+1}", product_options, index=product_options.index(r["name"]) if r["name"] in product_options else 0, key=f"name_{i}")
            qty = cols[1].number_input(f"Qty {i+1}", min_value=0, value=int(r.get("qty", 1)), key=f"qty_{i}")
            rm = cols[2].button("Remove", key=f"rm_{i}")
            if not rm:
                new_rows.append({"name": selected_name, "qty": int(qty)})
        st.session_state.order_rows = new_rows

        # Build order dict (ASIN -> qty)
        order_items = {}
        for r in st.session_state.order_rows:
            if r["name"] and r["qty"] > 0:
                asin = name_to_asin.get(r["name"])
                if asin:
                    order_items[asin] = order_items.get(asin, 0) + int(r["qty"])

        st.markdown("**Order preview**")
        st.write({product_dict.get(k, k): v for k, v in order_items.items()})

        if st.button("✔️ Validate Order (metadata + vision verification)"):
            if not order_items:
                st.warning("Order is empty.")
            else:
                allocation, shortages = multi_asin_allocator(order_items, asin_index)
                st.success("Allocation computed (metadata layer).")
                st.write("Allocation (metadata):", {img: dict(allocation[img]) for img in allocation})

                # store in session to show
                st.session_state.last_allocation = allocation
                st.session_state.last_shortages = shortages

                # Now run Vision-QA verification for each chosen image
                # Build short per-image list of ASINs to verify (only ones contributing)
                verification_results = {}
                if OPENAI_API_KEY:
                    st.info("Running lightweight Vision-QA verification via OpenAI...")
                    for img_id, contributes in allocation.items():
                        img_path_jpg = Path(IMAGES_DIR) / f"{img_id}.jpg"
                        img_path_png = Path(IMAGES_DIR) / f"{img_id}.png"
                        img_path = img_path_jpg if img_path_jpg.exists() else (img_path_png if img_path_png.exists() else None)
                        if not img_path:
                            verification_results[img_id] = {"error": "image file not found"}
                            continue

                        # Which asins to ask about in this image
                        questions = []
                        for asin, used_qty in contributes.items():
                            product_name = product_dict.get(asin, asin)
                            # structured question: presence + visible quantity
                            questions.append(f"For ASIN {asin} (product name: {product_name}), is this product visibly present in the image? If yes estimate visible quantity (integer).")
                        try:
                            img = load_image_for_api(img_path, max_dim=1200)
                            img_bytes = image_to_base64_bytes(img)
                            parsed, err = ask_vision_qa_via_api(img_bytes, questions, api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
                            if err:
                                verification_results[img_id] = {"error": err}
                            else:
                                verification_results[img_id] = parsed
                        except Exception as e:
                            verification_results[img_id] = {"error": str(e)}

                    st.session_state.vision_verification = verification_results
                else:
                    st.info("No OpenAI API key configured. Skipping Vision verification (metadata-only).")

                st.experimental_rerun()

        # if we have last allocation, show results and verification
        if "last_allocation" in st.session_state:
            allocation = st.session_state.last_allocation
            shortages = st.session_state.last_shortages
            st.header("Validation Result")
            if not shortages:
                st.success("✅ Order can be fulfilled (metadata)")
            else:
                st.error("⚠️ Shortages detected (metadata)")
                for asin, missing in shortages.items():
                    st.write(f"- {product_dict.get(asin, asin)} : missing {missing}")

            # Show images & verification results
            st.subheader("Selected images to fulfill the order")
            chosen_images = sorted(list(allocation.keys()), key=lambda img: sum(allocation[img].values()), reverse=True)
            for img_id in chosen_images:
                cols = st.columns([1, 3])
                with cols[0]:
                    img_path_jpg = Path(IMAGES_DIR) / f"{img_id}.jpg"
                    img_path_png = Path(IMAGES_DIR) / f"{img_id}.png"
                    img_path = img_path_jpg if img_path_jpg.exists() else (img_path_png if img_path_png.exists() else None)
                    if img_path:
                        st.image(str(img_path), width=220, caption=f"Image: {img_id}")
                    else:
                        st.write("Image not found")

                with cols[1]:
                    st.write("Contributions (metadata):")
                    st.write({product_dict.get(k,k): v for k, v in allocation[img_id].items()})
                    vv = st.session_state.get("vision_verification", {}).get(img_id)
                    if vv is None:
                        st.info("No vision verification run (API key missing or not run).")
                    elif isinstance(vv, dict) and vv.get("error"):
                        st.error(f"Vision API error: {vv.get('error')}")
                        if vv.get("raw_text"):
                            st.write(vv.get("raw_text"))
                    else:
                        st.write("Vision-QA result (raw):")
                        st.write(vv)

            # allocation table
            st.subheader("Allocation table (metadata)")
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

    # Inventory Dashboard
    elif page == "Inventory Dashboard":
        st.header("📊 Full Inventory Dashboard")
        total_counts = { asin: sum(q for _, q in lst) for asin, lst in asin_index.items() }
        df_inv = pd.DataFrame([(asin, product_dict.get(asin,"Unknown"), total_counts.get(asin,0)) for asin in total_counts],
                              columns=["ASIN","Product Name","TotalAvailable"])
        search = st.text_input("Search by name or ASIN")
        if search:
            df_inv = df_inv[df_inv["Product Name"].str.contains(search, case=False, na=False) | df_inv["ASIN"].str.contains(search, case=False, na=False)]
        st.dataframe(df_inv.reset_index(drop=True), use_container_width=True, height=600)

if __name__ == "__main__":
    main()
