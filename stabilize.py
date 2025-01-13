import cv2
import numpy as np
from findShift import find_shift

def align_frames(frame1, frame2, ratio_thresh=0.75, min_matches=10):
    """
    Align frame2 to frame1 using homography transformation.

    Args:
        frame1 (np.ndarray): Reference frame of shape (H, W, C)
        frame2 (np.ndarray): Frame to be aligned of shape (H, W, C)
        ratio_thresh (float): Ratio threshold for SIFT matching (default: 0.75)
        min_matches (int): Minimum number of good matches required (default: 10)

    Returns:
        tuple: (aligned_frame, H) where:
            - aligned_frame (np.ndarray): Frame2 aligned to frame1
            - H (np.ndarray): 3x3 homography matrix used for alignment

    Raises:
        ValueError: If frames have different sizes or if alignment fails
    """
    if frame1.shape != frame2.shape:
        raise ValueError("Frames must have the same dimensions")

    # Find transformation between frames
    try:
        transform = find_shift(frame1, frame2, ratio_thresh, min_matches)
        H = transform['homography']
    except ValueError as e:
        raise ValueError(f"Frame alignment failed: {str(e)}")

    # Apply homography transformation
    H_inv = cv2.invertAffineTransform(H)[:2]
    aligned_frame = cv2.warpAffine(frame2, H_inv, (frame2.shape[1], frame2.shape[0]))

    return aligned_frame, H_inv
