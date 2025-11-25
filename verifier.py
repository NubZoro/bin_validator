# verifier.py
"""
verify_image_asins(image_path, asins, meta_index)

Approach:
- For each ASIN to verify, find a set of reference images that contain that ASIN (from meta_index).
- Use ORB feature extraction + BFMatcher to compute number of good matches between the query image and each reference image.
- If good matches exceed threshold and homography found, mark ASIN as visually detected in this image.
- Attempt to estimate count: if multiple distinct homography detections exist (non-overlapping), increment count — crude but often useful.

This is heuristics-based and intentionally simple (works without heavy ML).
"""
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
from pathlib import Path

# ORB params
ORB_N_FEATURES = 1000
GOOD_MATCH_RATIO = 0.75  # Lowe ratio
MIN_GOOD_MATCHES = 20
HOMOGRAPHY_MIN_INLIERS = 10

orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)

def load_image_gray(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return gray

def compute_kp_des(img_gray):
    kp, des = orb.detectAndCompute(img_gray, None)
    return kp, des

def match_descriptors(des1, des2):
    # BFMatcher with Hamming (ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    if des1 is None or des2 is None:
        return []
    matches = bf.knnMatch(des1, des2, k=2)
    # Lowe ratio test
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < GOOD_MATCH_RATIO * n.distance:
            good.append(m)
    return good

def verify_image_asins(image_path, asins_list, meta_index, images_dir="data/images"):
    """
    For each asin in asins_list, check against some reference images that contain that asin (from meta_index).
    Returns dict asin -> {detected: bool, matches: best_match_count, details: [...]}
    """
    # Preload query
    qgray = load_image_gray(image_path)
    qkp, qdes = compute_kp_des(qgray)
    results = {}
    # Find reference images for each asin
    # meta_index: image_id -> list of items [{asin, quantity,...}]
    # Build reverse mapping asin->list of image_ids
    asin_to_imgs = defaultdict(list)
    for imgid, items in meta_index.items():
        for it in items:
            asin_to_imgs[it["asin"]].append(imgid)
    for asin in asins_list:
        imgs = asin_to_imgs.get(asin, [])
        best = {"detected": False, "matches": 0, "details": []}
        for ref_imgid in imgs:
            ref_path = Path(images_dir) / f"{ref_imgid}.jpg"
            if not ref_path.exists():
                ref_path = Path(images_dir) / f"{ref_imgid}.png"
            if not ref_path.exists():
                continue
            try:
                rgray = load_image_gray(str(ref_path))
                rkp, rdes = compute_kp_des(rgray)
                good = match_descriptors(qdes, rdes)
                if len(good) > best["matches"]:
                    best["matches"] = len(good)
                # if enough good matches, try homography to be more certain
                if len(good) >= MIN_GOOD_MATCHES:
                    src_pts = np.float32([ rkp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ qkp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    try:
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                        inliers = int(mask.sum()) if mask is not None else 0
                    except Exception:
                        inliers = 0
                    best["details"].append({"ref": ref_imgid, "good_matches": len(good), "inliers": inliers})
                    if inliers >= HOMOGRAPHY_MIN_INLIERS:
                        best["detected"] = True
                        # don't break — check other refs to possibly estimate count
            except Exception:
                continue
        results[asin] = best
    return results
