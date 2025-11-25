# app.py
import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
from utils import build_metadata_index, multi_asin_allocator, product_dict_from_index, meta_expected_counts
from verifier import verify_image_asins
import pandas as pd
import json

# Config
IMAGES_DIR = Path("data/images")
METADATA_DIR = Path("data/metadata")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(layout="wide", page_title="Bin Order Validator (ORB verifier)")

# Header
st.markdown("<h1 style='font-size:36px'>📦 Bin Order Validator</h1>", unsafe_allow_html=True)
st.markdown("Metadata-driven allocation + lightweight ORB visual verification (no PyTorch).")

# Build index on startup (fast)
with st.spinner("Indexing metadata..."):
    meta_index, asin_index = build_metadata_index(images_dir=str(IMAGES_DIR), metadata_dir=str(METADATA_DIR))
    product_dict = product_dict_from_index(meta_index)

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Order Validation", "Inventory Dashboard", "Image Verifier"])

# Shared: product options for dropdowns
product_options = sorted(set(product_dict.values()))
name_to_asin = {v: k for k, v in product_dict.items()}

# --------------------------
# Order Validation Page
# --------------------------
if page == "Order Validation":
    st.header("Create Order (Select product names)")
    # session rows
    if "order_rows" not in st.session_state:
        st.session_state.order_rows = [{"name": product_options[0] if product_options else "", "qty": 1}]

    if st.button("➕ Add item row"):
        st.session_state.order_rows.append({"name": product_options[0] if product_options else "", "qty": 1})
        st.experimental_rerun()

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

    # Prepare order (asin->qty)
    order_items = { name_to_asin[r["name"]]: r["qty"] for r in st.session_state.order_rows if r["name"] and r["qty"]>0 }
    st.write("Order:", {product_dict.get(k,k):v for k,v in order_items.items()})

    if st.button("✅ Validate Order (allocate images)"):
        allocation, shortages = multi_asin_allocator(order_items, asin_index)
        st.session_state.last_allocation = allocation
        st.session_state.last_shortages = shortages
        st.experimental_rerun()

    if "last_allocation" in st.session_state:
        allocation = st.session_state.last_allocation
        shortages = st.session_state.last_shortages or {}
        st.header("Validation Result")
        if not shortages:
            st.success("✅ Order can be fulfilled from inventory metadata.")
        else:
            st.error("⚠️ Shortages detected")
            for asin, miss in shortages.items():
                st.write(f"- {product_dict.get(asin,asin)} missing {miss}")

        # show chosen images grid
        if allocation:
            st.subheader("Images chosen (min set)")
            chosen_images = sorted(list(allocation.keys()), key=lambda img: sum(allocation[img].values()), reverse=True)
            # show 3 per row
            for i in range(0, len(chosen_images), 3):
                cols = st.columns(3)
                for j, img_id in enumerate(chosen_images[i:i+3]):
                    col = cols[j]
                    img_path = IMAGES_DIR / f"{img_id}.jpg"
                    if not img_path.exists():
                        img_path = IMAGES_DIR / f"{img_id}.png"
                    if img_path.exists():
                        col.image(str(img_path), use_column_width=True, caption=f"{img_id}")
                        col.write(dict(allocation[img_id]))
                    else:
                        col.write(f"Image {img_id} not found.")

        # Provide a verification button that runs ORB verifier on the chosen images
        if st.button("🔎 Verify chosen images visually (ORB)"):
            allocation = st.session_state.last_allocation
            verify_results = {}
            with st.spinner("Running ORB verification on chosen images..."):
                for img_id, parts in allocation.items():
                    img_path = IMAGES_DIR / f"{img_id}.jpg"
                    if not img_path.exists():
                        img_path = IMAGES_DIR / f"{img_id}.png"
                    if not img_path.exists():
                        verify_results[img_id] = {"error": "image file not found"}
                        continue
                    # parts: dict asin->qty contributed from metadata
                    asins = list(parts.keys())
                    vr = verify_image_asins(str(img_path), asins, meta_index=meta_index)
                    verify_results[img_id] = vr
            st.subheader("Verification results")
            st.json(verify_results)

# --------------------------
# Inventory Dashboard Page
# --------------------------
elif page == "Inventory Dashboard":
    st.header("Inventory Dashboard")
    total_counts = { asin: sum(q for _, q in lst) for asin, lst in asin_index.items() }
    df_inv = pd.DataFrame([(asin, product_dict.get(asin,"Unknown"), total_counts[asin]) for asin in total_counts],
                           columns=["ASIN","Product Name","TotalAvailable"])
    search = st.text_input("Search by name or ASIN")
    if search:
        df_inv = df_inv[df_inv["Product Name"].str.contains(search, case=False, na=False) | df_inv["ASIN"].str.contains(search, case=False, na=False)]
    st.dataframe(df_inv.sort_values("TotalAvailable", ascending=False).reset_index(drop=True), use_container_width=True, height=700)

# --------------------------
# Image Verifier page (manual)
# --------------------------
elif page == "Image Verifier":
    st.header("Manual ORB Verifier")
    all_images = sorted([p for p in IMAGES_DIR.glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")])
    sel = st.selectbox("Choose image", options=[""] + [str(p) for p in all_images])
    if sel:
        p = Path(sel)
        st.image(str(p), use_column_width=True)
        # list ASINs present in this image from metadata
        meta = None
        sidecar = METADATA_DIR / f"{p.stem}.json"
        if sidecar.exists():
            meta = json.load(open(sidecar,"r",encoding="utf-8"))
            expected = meta_expected_counts(meta)
            st.write("Metadata expected:", expected)
        else:
            st.info("No metadata sidecar for this image.")

        # let user pick ASINs to verify
        known_asins = list(product_dict.keys())
        chosen = st.multiselect("Pick ASINs to verify visually (choose from inventory names)", [product_dict[a] for a in known_asins], default=[])
        asin_lookup = {product_dict[a]:a for a in known_asins}
        asins_to_check = [asin_lookup[name] for name in chosen] if chosen else []
        if st.button("Run ORB verify"):
            if not asins_to_check:
                st.warning("Choose at least one ASIN to verify.")
            else:
                with st.spinner("Running ORB verification..."):
                    vr = verify_image_asins(str(p), asins_to_check, meta_index=meta_index)
                st.write("Verification result (per ASIN):")
                st.json(vr)

# footer
st.markdown("---")
st.markdown("Local ORB verifier is approximate — use manual labeling for the best detection accuracy.")
