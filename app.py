# app.py
import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import json
from utils import (
    build_metadata_index, 
    multi_asin_allocator, 
    product_dict_from_index, 
    meta_expected_counts
)
from gradio_client import Client

# -------------------------------
# CONFIG
# -------------------------------
HF_SPACE_URL = "https://nubzoro-bin-validator.hf.space"   # CHANGE THIS

IMAGES_DIR = Path("data/images")
METADATA_DIR = Path("data/metadata")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(layout="wide", page_title="Smart Bin Validator")

# Connect to HF ZeroGPU backend
client = Client(HF_SPACE_URL)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<h1 style='font-size:36px'>📦 Smart Bin Validator</h1>", unsafe_allow_html=True)
st.markdown("HuggingFace ZeroGPU backend for heavy object detection.")

# -------------------------------
# METADATA INDEX
# -------------------------------
with st.spinner("Indexing metadata..."):
    meta_index, asin_index = build_metadata_index(
        images_dir=str(IMAGES_DIR), 
        metadata_dir=str(METADATA_DIR)
    )
    product_dict = product_dict_from_index(meta_index)

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Order Validation", "Inventory Dashboard", "Image Validator"])

product_options = sorted(set(product_dict.values()))
name_to_asin = {v: k for k, v in product_dict.items()}

# ---------------------------------------------------------
# LAYER 2: USE HF SPACE FOR VERIFICATION
# ---------------------------------------------------------
def run_hf_verifier(image_path, asins_to_check):
    """
    Sends image bytes + ASIN list to your GroundingDINO ZeroGPU Space.
    """
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        with st.spinner("Querying HF ZeroGPU model…"):
            result = client.predict(
                img_bytes,                                # file bytes
                json.dumps(asins_to_check),               # JSON list of ASINs
                api_name="/run"                           # must match HF API
            )
        return result

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------
# ORDER VALIDATION PAGE
# ---------------------------------------------------------
if page == "Order Validation":
    st.header("Create Order")

    if "order_rows" not in st.session_state:
        st.session_state.order_rows = [{
            "name": product_options[0] if product_options else "",
            "qty": 1
        }]

    # Add row button
    if st.button("➕ Add item"):
        st.session_state.order_rows.append({
            "name": product_options[0],
            "qty": 1
        })
        st.experimental_rerun()

    # Render rows
    new_rows = []
    for i, row in enumerate(st.session_state.order_rows):
        c1, c2, c3 = st.columns([4,1,1])
        selected = c1.selectbox(
            f"Product {i+1}",
            product_options,
            index=product_options.index(row["name"]),
            key=f"name_{i}"
        )
        qty = c2.number_input(
            f"Qty {i+1}",
            min_value=0,
            value=int(row["qty"]),
            key=f"qty_{i}"
        )
        remove = c3.button("Remove", key=f"rm_{i}")

        if not remove:
            new_rows.append({"name": selected, "qty": int(qty)})

    st.session_state.order_rows = new_rows

    # Convert display names → ASINs
    order_items = {
        name_to_asin[r["name"]]: r["qty"]
        for r in st.session_state.order_rows if r["qty"] > 0
    }

    st.write("Order:", order_items)

    if st.button("✅ Validate Order"):
        allocation, shortages = multi_asin_allocator(order_items, asin_index)
        st.session_state.last_allocation = allocation
        st.session_state.last_shortages = shortages
        st.experimental_rerun()

    # Show allocation
    if "last_allocation" in st.session_state:
        allocation = st.session_state.last_allocation
        shortages = st.session_state.last_shortages

        st.header("Validation Result")
        if not shortages:
            st.success("Order can be fulfilled.")
        else:
            st.error("Shortages detected:")
            st.write(shortages)

        # Show selected images
        st.subheader("Selected Images")
        for img_id in allocation.keys():
            path = IMAGES_DIR / f"{img_id}.jpg"
            if not path.exists():
                path = IMAGES_DIR / f"{img_id}.png"

            st.image(str(path), caption=img_id, use_column_width=True)
            st.write("Products counted:", allocation[img_id])

        # -------------------------
        # RUN HF MODEL
        # -------------------------
        if st.button("🔎 Verify with HF Model"):
            verify_results = {}

            for img_id, parts in allocation.items():
                # load local file path
                f = IMAGES_DIR / f"{img_id}.jpg"
                if not f.exists():
                    f = IMAGES_DIR / f"{img_id}.png"

                asins = list(parts.keys())

                result = run_hf_verifier(str(f), asins)
                verify_results[img_id] = result

            st.subheader("HF Verification Results")
            st.json(verify_results)

# ---------------------------------------------------------
# INVENTORY DASHBOARD
# ---------------------------------------------------------
elif page == "Inventory Dashboard":
    st.header("Inventory Overview")

    total_counts = {
        asin: sum(q for _, q in plist)
        for asin, plist in asin_index.items()
    }
    df = pd.DataFrame([
        (asin, product_dict.get(asin, ""), total_counts[asin])
        for asin in total_counts
    ], columns=["ASIN", "Product Name", "Total Available"])

    search = st.text_input("Search")
    if search:
        df = df[df.apply(lambda r: search.lower() in str(r).lower(), axis=1)]

    st.dataframe(df, use_container_width=True, height=700)

# ---------------------------------------------------------
# IMAGE VALIDATOR (manual)
# ---------------------------------------------------------
else:
    st.header("Manual Image Validator")

    images = sorted([p for p in IMAGES_DIR.glob("*") if p.suffix.lower() in (".jpg",".png")])
    sel = st.selectbox("Pick Image", [""] + [str(p) for p in images])

    if sel:
        path = Path(sel)
        st.image(str(path), use_column_width=True)

        # Metadata expected (from JSON)
        meta_file = METADATA_DIR / f"{path.stem}.json"
        if meta_file.exists():
            meta = json.load(open(meta_file, "r"))
            st.write("Metadata expected:", meta_expected_counts(meta))
        else:
            st.warning("No metadata for this image")

        # Pick ASINs to verify
        selected_names = st.multiselect(
            "Select products to verify",
            [product_dict[a] for a in product_dict.keys()]
        )

        asins_to_check = [
            next(k for k,v in product_dict.items() if v == name)
            for name in selected_names
        ]

        if st.button("Run HF Verification"):
            result = run_hf_verifier(str(path), asins_to_check)
            st.json(result)

# Footer
st.markdown("---")
st.markdown("Powered by HuggingFace ZeroGPU + Streamlit.")
