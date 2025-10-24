#!/usr/bin/env python3
"""
main.py

SIFT feature detection + matching between two images,
and optional comparison of a query image against every frame of a video.

Outputs:
 - outputs/image_matches.jpg      (drawn keypoint matches between two images)
 - outputs/video_matches.mp4      (video with per-frame match annotations, if --video is used)

Usage examples:
 - Compare two images:
    python main.py --img1 images/query.jpg --img2 images/target.jpg

 - Compare image against video frames:
    python main.py --img1 images/query.jpg --video videos/input.mp4

 - Do both (image match + video processing):
    python main.py --img1 images/query.jpg --img2 images/target.jpg --video videos/input.mp4
"""

import argparse
import os
from typing import Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Utility functions
# ---------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_image_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# ---------------------------
# SIFT matching functions
# ---------------------------

def compute_sift_keypoints_and_descriptors(gray: np.ndarray):
    """Return keypoints, descriptors using SIFT."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_descriptors(desc1, desc2, ratio_test=0.75):
    """
    Use FLANN-based matcher + Lowe's ratio test (typical for SIFT).
    Returns list of good matches (cv2.DMatch objects).
    """
    if desc1 is None or desc2 is None:
        return []

    # FLANN parameters for SIFT (floating point descriptors)
    index_params = dict(algorithm=1, trees=5)  # KDTree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # knnMatch returns list of lists (k=2)
    knn_matches = flann.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m_n in knn_matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio_test * n.distance:
            good_matches.append(m)
    return good_matches

def draw_matches(img1_color, kp1, img2_color, kp2, matches, max_display=50):
    """Return an image visualizing matches (cv2.drawMatches)."""
    # limit number of matches drawn for readability
    matches_to_draw = matches[:max_display]
    out = cv2.drawMatches(
        img1_color, kp1,
        img2_color, kp2,
        matches_to_draw, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return out

# ---------------------------
# Image vs Image routine
# ---------------------------

def process_image_pair(img1_path: str, img2_path: str, out_path: str):
    print(f"[INFO] Loading images: '{img1_path}' and '{img2_path}'")
    img1_color = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2_color = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    if img1_color is None or img2_color is None:
        raise FileNotFoundError("One of the images could not be loaded. Check your paths.")

    gray1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    kp1, desc1 = compute_sift_keypoints_and_descriptors(gray1)
    kp2, desc2 = compute_sift_keypoints_and_descriptors(gray2)
    print(f"[INFO] Detected {len(kp1)} and {len(kp2)} keypoints respectively.")

    good_matches = match_descriptors(desc1, desc2)
    print(f"[INFO] {len(good_matches)} good matches found (after ratio test).")

    matched_img = draw_matches(img1_color, kp1, img2_color, kp2, good_matches, max_display=200)

    ensure_dir(os.path.dirname(out_path) or ".")
    cv2.imwrite(out_path, matched_img)
    print(f"[INFO] Match visualization saved to: {out_path}")

# ---------------------------
# Image vs Video routine
# ---------------------------

def process_video_with_query_image(query_img_path: str, video_path: str, out_video_path: str, display: bool = False):
    """
    For each frame in the input video:
      - compute SIFT keypoints & descriptors
      - match them with the query image descriptors
      - annotate the frame with the number of good matches and draw a small match preview
    Save annotated frames to out_video_path.
    """
    print(f"[INFO] Loading query image: {query_img_path}")
    query_color = cv2.imread(query_img_path, cv2.IMREAD_COLOR)
    if query_color is None:
        raise FileNotFoundError(f"Cannot load query image: {query_img_path}")
    query_gray = cv2.cvtColor(query_color, cv2.COLOR_BGR2GRAY)
    kp_q, desc_q = compute_sift_keypoints_and_descriptors(query_gray)
    print(f"[INFO] Query keypoints: {len(kp_q)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_dir(os.path.dirname(out_video_path) or ".")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_f, desc_f = compute_sift_keypoints_and_descriptors(frame_gray)
        good_matches = match_descriptors(desc_q, desc_f)

        # Draw a small inset with matches between query and current frame (downscale for speed)
        # Prepare images for small match visualization
        thumbnail_query = cv2.resize(query_color, (0,0), fx=0.25, fy=0.25)
        thumbnail_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        # Need to recompute keypoints/descriptors for thumbnails OR we can just show text & count
        # For simplicity, show match count and draw query thumbnail on corner
        overlay = frame.copy()
        # Paste the thumbnail at top-left
        th_h, th_w = thumbnail_query.shape[:2]
        overlay[5:5+th_h, 5:5+th_w] = thumbnail_query

        # Annotate number of matches
        cv2.putText(overlay, f"Good matches: {len(good_matches)}", (10, 10 + th_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        out.write(overlay)

        if display:
            cv2.imshow("Matches (frame)", overlay)
            key = cv2.waitKey(1)
            if key == 27:
                break

        if frame_idx % 50 == 0:
            print(f"[INFO] Processed frame {frame_idx}, good matches: {len(good_matches)}")
        frame_idx += 1

    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()
    print(f"[INFO] Video with match annotations saved to: {out_video_path}")

# ---------------------------
# Argument parsing & main
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SIFT matching: image-image and image-video")
    parser.add_argument("--img1", required=True, help="Path to query image (e.g., images/query.jpg)")
    parser.add_argument("--img2", required=False, help="Path to second image to match (e.g., images/target.jpg)")
    parser.add_argument("--video", required=False, help="Path to video file to compare frames with query image")
    parser.add_argument("--out_image", default="outputs/image_matches.jpg", help="Output path for image match visualization")
    parser.add_argument("--out_video", default="outputs/video_matches.mp4", help="Output path for annotated video")
    parser.add_argument("--display", action="store_true", help="Display frames in a window while processing video (press ESC to stop)")
    return parser.parse_args()

def main():
    args = parse_args()
    ensure_dir("outputs")

    if args.img2:
        process_image_pair(args.img1, args.img2, args.out_image)

    if args.video:
        process_video_with_query_image(args.img1, args.video, args.out_video, display=args.display)

    if not args.img2 and not args.video:
        print("[INFO] No --img2 or --video provided; nothing processed. Provide at least one.")

if __name__ == "__main__":
    main()
