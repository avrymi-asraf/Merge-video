import cv2
import numpy as np
from findShift import find_shift


def mean_transformation(M1, M2, alpha):
    """
    Calculates the mean transformation matrix given two transformation matrices
    and a weighting factor alpha.

    Args:
      M1: The first transformation matrix (2x3 or 3x3 NumPy array).
      M2: The second transformation matrix (2x3 or 3x3 NumPy array).
      alpha: The weighting factor for M1 (0 <= alpha <= 1).

    Returns:
      A new transformation matrix representing the weighted average of M1 and M2.
    """

    # 1. Decompose the matrices (assuming affine transformations)
    M1_translation = (M1[0, 2], M1[1, 2])
    M2_translation = (M2[0, 2], M2[1, 2])

    M1_angle = np.arctan2(M1[1, 0], M1[0, 0]) * 180 / np.pi
    M2_angle = np.arctan2(M2[1, 0], M2[0, 0]) * 180 / np.pi

    # 2. Calculate weighted averages
    mean_translation = (
        alpha * M1_translation[0] + (1 - alpha) * M2_translation[0],
        alpha * M1_translation[1] + (1 - alpha) * M2_translation[1],
    )
    mean_angle = alpha * M1_angle + (1 - alpha) * M2_angle

    # 3. Reconstruct the mean transformation matrix
    mean_M = cv2.getRotationMatrix2D((0, 0), mean_angle, 1)  # Rotation only
    mean_M[0, 2] = mean_translation[0]  # Add translation
    mean_M[1, 2] = mean_translation[1]

    return mean_M


def stabilize(
    frames,
    ratio_thresh=0.75,
    min_matches=10,
    cancel_axis="x",
    skip_no_matches=False,
    max_rotation_degrees=30,
    max_translation=None,
    alpha=0.3,
):
    """
    Stabilize the rotation and translation in a video by aligning each frame to the previous frame.

    Args:
        frames (list of np.ndarray): List of frames to be stabilized, each of shape (H, W, C)
        ratio_thresh (float): Ratio threshold for SIFT matching (default: 0.75)
        min_matches (int): Minimum number of good matches required (default: 10)
        cancel_axis (str): Axis to cancel during stabilization ('x', 'y', or None)
        skip_no_matches (bool): Whether to skip frames with insufficient matches
        max_rotation_degrees (float): Maximum allowed rotation in degrees
        max_translation (float, optional): Maximum allowed translation as fraction of image size (0 to 1)

    Returns:
        list of np.ndarray: List of stabilized frames, each of shape (H, W, C)

    Raises:
        ValueError: If less than two frames are provided or if frames have different sizes
    """
    if not frames or len(frames) < 2:
        raise ValueError("Error: At least two frames are required for stabilization")

    stabilized_frames = [frames[0]]
    previous_deg = 0
    preivous_translation = 0

    for i in range(1, len(frames)):
        frame1 = stabilized_frames[-1]
        frame2 = frames[i]

        # Find transformation between frames
        transform = find_shift(
            frame1,
            frame2,
            ratio_thresh,
            min_matches,
            skip_no_matches=skip_no_matches,
        )
        H = transform["homography"]

        # Get inverse transform
        H_inv = cv2.invertAffineTransform(H)[:2]

        # Clamp rotation after creating inverse matrix
        rotation_rad = np.arctan2(H_inv[1, 0], H_inv[0, 0])
        rotation_deg = np.degrees(rotation_rad)
        rotation_deg = previous_deg*alpha + rotation_deg*(1-alpha)
        previous_deg = rotation_deg

        if abs(rotation_deg) > max_rotation_degrees:
            clamped_rad = np.radians(
                np.clip(rotation_deg, -max_rotation_degrees, max_rotation_degrees)
            )
            cos_theta = np.cos(clamped_rad)
            sin_theta = np.sin(clamped_rad)
            H_inv[0:2, 0:2] = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        # Clip translation if max_translation is provided
        if max_translation is not None:
            height, width = frame1.shape[:2]
            if cancel_axis != "x":
                max_pixels_x = width * max_translation
                H_inv[0, 2] = np.clip(H_inv[0, 2], -max_pixels_x, max_pixels_x)
                H_inv[0, 2] = preivous_translation*alpha + H_inv[0, 2]*(1-alpha)
                preivous_translation = H_inv[0, 2]
            if cancel_axis != "y":
                max_pixels_y = height * max_translation
                H_inv[1, 2] = np.clip(H_inv[1, 2], -max_pixels_y, max_pixels_y)
                H_inv[1, 2] = preivous_translation*alpha + H_inv[1, 2]*(1-alpha)
                preivous_translation = H_inv[1, 2]

        # Cancel axis movement if specified
        if cancel_axis == "x":
            H_inv[0, 2] = 0
        if cancel_axis == "y":
            H_inv[1, 2] = 0


        # Apply transformation
        aligned_frame = cv2.warpAffine(
            frame2, H_inv, (frame2.shape[1], frame2.shape[0])
        )
        stabilized_frames.append(aligned_frame)

    return stabilized_frames
