import cv2
import numpy as np

ALPHA = 0.8
MAX_ROTATION = 0.5
MAX_TRANSLATION = 0.05


def stabilize(
    frames,
    ratio_thresh=0.75,
    min_matches=10,
    cancel_axis="x",
    skip_no_matches=False,
    max_rotation_degrees=MAX_ROTATION,
    max_translation=MAX_TRANSLATION,
    alpha=ALPHA,
    ecc=True,
):
    """
    Stabilize the rotation and translation in a video by aligning each frame to the previous frame.

    Args:
        frames (List[np.ndarray]): List of input video frames, each of shape (height, width, channels) or (height, width) for grayscale.
        ratio_thresh (float, optional): Lowe's ratio test threshold. Defaults to 0.75.
        min_matches (int, optional): Minimum number of good matches required. Defaults to 10.
        cancel_axis (str, optional): Axis to cancel stabilization ('x', 'y', or None). Defaults to 'x'.
        skip_no_matches (bool, optional): If True, skip frames with insufficient matches. Defaults to False.
        max_rotation_degrees (float, optional): Maximum allowed rotation in degrees. Defaults to MAX_ROTATION.
        max_translation (float, optional): Maximum allowed translation as fraction of frame size. Defaults to MAX_TRANSLATION.
        alpha (float, optional): Smoothing factor for motion (0 to 1). Defaults to ALPHA.

    Returns:
        tuple: A tuple containing:
            - List[np.ndarray]: Stabilized frames
            - List[np.ndarray]: List of 2x3 affine transformation matrices (H) that transform
              frame i to frame i-1 (forward transformations)

    Raises:
        ValueError: If less than two frames are provided
        ValueError: If insufficient keypoints or matches are found (when skip_no_matches is False)
    """
    if len(frames) < 2:
        raise ValueError("Error: At least two frames are required for stabilization")

    stabilized_frames = [frames[0]]
    transformations = [np.eye(2, 3)]
    previous_deg = 0
    previous_translation = 0
    sift = cv2.SIFT_create()
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-4)
    for i in range(1, len(frames)):
        frame1 = stabilized_frames[-1]
        frame2 = frames[i]

        # Convert images to grayscale
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            gray2 = frame2

        # Detect SIFT features
        kp1, desc1 = sift.detectAndCompute(gray1.astype("uint8"), None)
        kp2, desc2 = sift.detectAndCompute(gray2.astype("uint8"), None)

        # Handle insufficient keypoints
        if len(kp1) < min_matches or len(kp2) < min_matches:
            if skip_no_matches:
                H = np.eye(2, 3)
                transformations.append(H)
                stabilized_frames.append(frame2)
                continue
            else:
                raise ValueError("Insufficient keypoints detected")

        # Match features using FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        # Handle insufficient matches
        if len(good_matches) < min_matches:
            if skip_no_matches:
                H = np.eye(2, 3)
                transformations.append(H)
                stabilized_frames.append(frame2)
                continue
            else:
                raise ValueError(f"Not enough good matches found")

        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        # Compute transformation

        H = cv2.estimateAffinePartial2D(dst_pts, src_pts, cv2.RANSAC)[0][:2]
        if ecc:
            H = cv2.findTransformECC(
                gray1, gray2, H.astype(np.float32), cv2.MOTION_EUCLIDEAN, criteria
            )[1]
        # Apply stabilization constraints
        rotation_rad = np.arctan2(H[1, 0], H[0, 0])
        rotation_deg = np.degrees(rotation_rad)
        rotation_deg = previous_deg * alpha + rotation_deg * (1 - alpha)
        previous_deg = rotation_deg

        if abs(rotation_deg) > max_rotation_degrees:
            clamped_rad = np.radians(
                np.clip(rotation_deg, -max_rotation_degrees, max_rotation_degrees)
            )
            cos_theta = np.cos(clamped_rad)
            sin_theta = np.sin(clamped_rad)
            H[0:2, 0:2] = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        # Apply translation constraints
        if max_translation is not None:
            height, width = frame1.shape[:2]
            if cancel_axis != "x":
                max_pixels_x = width * max_translation
                H[0, 2] = np.clip(H[0, 2], -max_pixels_x, max_pixels_x)
                H[0, 2] = previous_translation * alpha + H[0, 2] * (1 - alpha)
                previous_translation = H[0, 2]
            if cancel_axis != "y":
                max_pixels_y = height * max_translation
                H[1, 2] = np.clip(H[1, 2], -max_pixels_y, max_pixels_y)
                H[1, 2] = previous_translation * alpha + H[1, 2] * (1 - alpha)
                previous_translation = H[1, 2]

        transformations.append(H.copy())
        # Cancel axis if specified
        if cancel_axis == "x":
            H[0, 2] = 0
        if cancel_axis == "y":
            H[1, 2] = 0

        # Store transformation and apply its inverse for stabilization
        # H_inv = cv2.invertAffineTransform(H)[:2]
        aligned_frame = cv2.warpAffine(frame2, H, (frame2.shape[1], frame2.shape[0]))
        stabilized_frames.append(aligned_frame)

    return stabilized_frames, transformations

if __name__ == "__main__":
    from itertools import product
    from tools import (
        video_to_array,
        array_to_video,
    )
    from datetime import datetime

    all_videos = [
        "data\\input\\boat.mp4",
        "data\\input\\Garden.mp4",
        "data\\input\\House.mp4",
        "data\\input\\Lguazu.mp4",
        "data\\input\\Kessaria.mp4",
        "data\\input\\Shinkansen.mp4",
    ]
    real_video_path = "data\\input\\boat.mp4"
    for vido_path in all_videos[:1]:
        real_frames = video_to_array(real_video_path)
        stabilized_real_frames = stabilize(
            real_frames[:150], cancel_axis="x", skip_no_matches=True, ecc=True, alpha=0.8
        )[0]

        # Save real video before and after stabilization

        current_time = datetime.now().strftime("%H%M")
        array_to_video(
            stabilized_real_frames,
            f"data\\my_output\\stabilize_alpha_{0.8}_{current_time}.mp4",
            fps=20,
        )

