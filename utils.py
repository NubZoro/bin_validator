# utils.py
import json
from collections import defaultdict

def read_json(path):
    try:
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def build_metadata_index(images_dir="data/images", metadata_dir="data/metadata"):
    """
    meta_index: image_id -> list of items [{asin, quantity, name, normalizedName}]
    asin_index: asin -> list of (image_id, qty)
    """
    from pathlib import Path
    meta_index = {}
    asin_index = defaultdict(list)
    images_dir = Path(images_dir)
    metadata_dir = Path(metadata_dir)
    for img_p in sorted(images_dir.glob("*.*")):
        if img_p.suffix.lower() not in (".jpg",".jpeg",".png"):
            continue
        image_id = img_p.stem
        sidecar = metadata_dir / f"{image_id}.json"
        data = None
        if sidecar.exists():
            data = read_json(str(sidecar))
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
    # sort asin_index list descending by qty
    for asin, lst in asin_index.items():
        asin_index[asin] = sorted(lst, key=lambda x: x[1], reverse=True)
    return meta_index, asin_index

def product_dict_from_index(meta_index):
    d = {}
    for image_id, items in meta_index.items():
        for it in items:
            d[it["asin"]] = it.get("normalizedName") or it.get("name") or it["asin"]
    return d

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

def meta_expected_counts(sidecar_json):
    if not sidecar_json:
        return {}
    bin_data = sidecar_json.get("BIN_FCSKU_DATA", {})
    out = {}
    for asin, rec in bin_data.items():
        try:
            out[asin] = int(rec.get("quantity", 0))
        except:
            out[asin] = 0
    return out
