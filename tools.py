import cv2
import numpy as np


def video_to_array(video_path):
    """
    Convert a video file to an array of frames.

    Args:
        video_path (str): Path to the video file

    Returns:
        list: List of numpy arrays, where each array is a frame
              Each frame has shape (height, width, channels)

    Raises:
        ValueError: If video file cannot be opened or is empty
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video file")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("Error: Video file is empty")

    return frames


def artificial_movement(image, transformation_matrices, clip_frames=False):
    """
    Apply a sequence of transformation matrices to an image and track corner points.

    Args:
        image (numpy.ndarray): Input image of shape (height, width, channels)
        transformation_matrices (list): List of 3x3 transformation matrices
                                     Each matrix represents an affine transformation
        clip_frames (bool): If True, returns cropped frames to remove black regions
                          If False, returns full-size frames with original dimensions

    Returns:
        tuple: (
            list: If clip_frames=True: List of cropped frames, each potentially of different size
                 If clip_frames=False: List of full-size frames (height, width, channels)
            tuple: ((x1, y1), (x2, y2)) coordinates of upper-left and lower-right corners
                  relative to the original frame
        )

    Raises:
        ValueError: If image is empty or transformation matrices are invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Error: Empty input image")

    height, width = image.shape[:2]
    results = [image.copy()]

    # Initialize corner points
    points = np.array(
        [[0, 0, 1], [width - 1, height - 1, 1]]  # Upper left  # Lower right
    )

    # Apply transformations to both images and points
    current_transform = np.eye(3)
    for matrix in transformation_matrices:
        if matrix.shape != (3, 3):
            raise ValueError("Error: Invalid transformation matrix shape")

        # Update cumulative transformation
        current_transform = matrix @ current_transform

        # Transform image
        transformed = cv2.warpAffine(results[-1], matrix[:2, :], (width, height))
        results.append(transformed)

    # Transform points using final cumulative transformation
    final_points = (current_transform @ points.T).T
    final_points = final_points[:, :2] / final_points[:, 2:]

    if clip_frames:
        # Clip coordinates to image bounds
        def clip_coords(point, height, width):
            x = np.clip(point[0], 0, width - 1)
            y = np.clip(point[1], 0, height - 1)
            return (int(x), int(y))

        upper_left = clip_coords(final_points[0], height, width)
        lower_right = clip_coords(final_points[1], height, width)

        # Crop all frames to the valid region
        x1, y1 = upper_left
        x2, y2 = lower_right
        cropped_results = [frame[y1 : y2 + 1, x1 : x2 + 1].copy() for frame in results]

        return cropped_results, (upper_left, lower_right)

    return results, (
        tuple(final_points[0].astype(int)),
        tuple(final_points[1].astype(int)),
    )
