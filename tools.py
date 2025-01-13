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
        frames.append(frame.astype(np.uint8))

    cap.release()

    if not frames:
        raise ValueError("Error: Video file is empty")

    return frames


def create_synthetic_frame(
    size: tuple = (100, 100),
    color: bool = False,
    num_shapes: int = 50,
    shape_size_range: tuple = (2, 8),
) -> np.ndarray:
    """
    Create a synthetic frame with randomly placed shapes, either in random colors or white.

    Args:
        size (tuple): Frame size (height, width)
        color (bool): If True, shapes will be random colors. If False, shapes will be white
        num_shapes (int): Number of shapes to draw
        shape_size_range (tuple): Range of shape sizes (min_size, max_size)

    Returns:
        np.ndarray: Frame with randomly placed shapes
                   Shape is (height, width, 3) for RGB image
    """
    frame = np.zeros((*size, 3), dtype=np.uint8)

    min_size, max_size = shape_size_range
    for _ in range(num_shapes):
        x = np.random.randint(max_size, size[1] - max_size)
        y = np.random.randint(max_size, size[0] - max_size)
        square_size = np.random.randint(min_size, max_size)

        # Generate color for the square
        if color:
            square_color = (
                np.random.randint(0, 256),
                np.random.randint(0, 256),
                np.random.randint(0, 256),
            )
        else:
            square_color = (255, 255, 255)  # White

        frame[y - square_size : y + square_size, x - square_size : x + square_size] = (
            square_color
        )

    return frame.astype(np.uint8)


def artificial_movement(image, transformation_matrices, clip_frames=False):
    """
    Apply a sequence of transformation matrices to an image.

    Args:
        image (numpy.ndarray): Input image of shape (height, width, channels)
        transformation_matrices (list): List of 3x3 transformation matrices
                                     Each matrix represents an affine transformation
        clip_frames (bool): If True, returns cropped frames to remove black regions
                          If False, returns full-size frames with original dimensions

    Returns:
        list: If clip_frames=True: List of cropped frames, each potentially of different size
              If clip_frames=False: List of full-size frames (height, width, channels)
    """
    if image is None or image.size == 0:
        raise ValueError("Error: Empty input image")

    height, width = image.shape[:2]
    results = [image.copy()]

    # Apply transformations to images
    for matrix in transformation_matrices:
        if matrix.shape != (3, 3):
            raise ValueError("Error: Invalid transformation matrix shape")

        # Transform image
        transformed = cv2.warpAffine(
            results[-1], matrix[:2, :].astype(np.float32), (width, height)
        )
        results.append(transformed)

    if clip_frames:
        # Calculate bounds from last frame
        non_black = np.where(results[-1] != 0)
        if len(non_black[0]) > 0:  # If there are non-black pixels
            y1, y2 = non_black[0].min(), non_black[0].max()
            x1, x2 = non_black[1].min(), non_black[1].max()
            # Crop all frames to the valid region
            return [frame[y1 : y2 + 1, x1 : x2 + 1].copy() for frame in results]

    return results
