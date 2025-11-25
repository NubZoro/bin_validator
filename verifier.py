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
    qgray = load_image_gray(image_path)
    qkp, qdes = compute_kp_des(qgray)
    h, w = qgray.shape[:2]

    # Reverse map ASIN -> reference images
    asin_to_imgs = defaultdict(list)
    for imgid, items in meta_index.items():
        for it in items:
            asin_to_imgs[it["asin"]].append(imgid)

    results = {}

    for asin in asins_list:
        refs = asin_to_imgs.get(asin, [])
        asin_result = {
            "detected": False,
            "best_matches": 0,
            "details": [],
            "quantity_estimate": 0
        }

        orb_boxes = []      # bounding boxes discovered via homography
        template_boxes = [] # boxes from template matching

        for ref_imgid in refs:
            ref_path = Path(images_dir) / f"{ref_imgid}.jpg"
            if not ref_path.exists(): ref_path = Path(images_dir) / f"{ref_imgid}.png"
            if not ref_path.exists(): continue

            # load reference
            rgray = load_image_gray(ref_path)
            rkp, rdes = compute_kp_des(rgray)
            rh, rw = rgray.shape

            # ORB matching
            good = match_descriptors(qdes, rdes)
            asin_result["best_matches"] = max(asin_result["best_matches"], len(good))

            if len(good) >= MIN_GOOD_MATCHES:
                try:
                    src_pts = np.float32([rkp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                    dst_pts = np.float32([qkp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    inliers = int(mask.sum())

                    asin_result["details"].append({
                        "ref": ref_imgid,
                        "good_matches": len(good),
                        "inliers": inliers
                    })

                    if inliers >= HOMOGRAPHY_MIN_INLIERS:
                        asin_result["detected"] = True

                        # project reference corners -> bounding box
                        pts = np.float32([[0,0],[rw,0],[rw,rh],[0,rh]]).reshape(-1,1,2)
                        dst = cv2.perspectiveTransform(pts, M)
                        x1,y1 = dst[:,0,0].min(), dst[:,0,1].min()
                        x2,y2 = dst[:,0,0].max(), dst[:,0,1].max()
                        orb_boxes.append([int(x1),int(y1),int(x2),int(y2)])

                except:
                    pass

            # TEMPLATE MATCHING
            try:
                result = cv2.matchTemplate(qgray, rgray, cv2.TM_CCOEFF_NORMED)
                thresh = 0.5
                loc = np.where(result >= thresh)

                for pt in zip(*loc[::-1]):
                    x1, y1 = pt[0], pt[1]
                    x2, y2 = x1 + rw, y1 + rh
                    template_boxes.append([x1,y1,x2,y2])
            except:
                pass

        # -------------------------
        # QUANTITY ESTIMATION
        # -------------------------

        # 1) ORB homography count
        orb_count = len(orb_boxes)

        # 2) Template match count
        template_count = len(template_boxes)

        # 3) Merge box lists + DBSCAN cluster
        all_boxes = orb_boxes + template_boxes
        if len(all_boxes) > 0:
            centers = np.array([
                [(b[0]+b[2])//2, (b[1]+b[3])//2] for b in all_boxes
            ])
            if len(centers) >= 2:
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=50, min_samples=1).fit(centers)
                cluster_count = len(set(clustering.labels_))
            else:
                cluster_count = 1
        else:
            cluster_count = 0

        # Final quantity score
        final_qty = (
            0.5 * orb_count +
            0.3 * template_count +
            0.2 * cluster_count
        )

        asin_result["quantity_estimate"] = max(1, round(final_qty))

        results[asin] = asin_result

    return results
