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


def random_image(
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


def array_to_video(frames, output_path, fps=30, codec='mp4v'):
    """
    Convert an array of frames to a video file.

    Args:
        frames (list or np.ndarray): List of frames, where each frame is a numpy array
                                   Each frame should have shape (height, width, channels)
        output_path (str): Path where the video file will be saved
        fps (int): Frames per second for the output video
        codec (str): Four character code for the video codec (e.g., 'mp4v', 'XVID')

    Raises:
        ValueError: If frames list is empty or frames have inconsistent dimensions
    """
    if not frames or len(frames) == 0:
        raise ValueError("Error: Empty frames list")

    # Get dimensions from first frame
    height, width = frames[0].shape[:2]
    
    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError("Error: Could not create video file")

    try:
        # Write frames to video
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("Error: Inconsistent frame dimensions")
            out.write(frame.astype(np.uint8))
    finally:
        # Release resources
        out.release()




def generate_complex_movement_matrices(num_frames, movement_intensity=1.0, sudden_movement_prob=0.1, noise_level=0.02):
    """
    Generate complex movement transformation matrices with controllable parameters.

    Args:
        num_frames (int): Number of frames to generate matrices for
        movement_intensity (float): Overall scaling factor for movement amplitude (default: 1.0)
        sudden_movement_prob (float): Probability of sudden movements [0-1] (default: 0.1)
        noise_level (float): Standard deviation of continuous noise (default: 0.02)

    Returns:
        list: List of 3x3 transformation matrices (np.ndarray) for each frame
    """
    transformation_matrices = []
    base_frequencies = np.array([0.2, 0.5, 0.8, 1.2, 1.5])
    base_amplitudes = np.array([0.15, 0.08, 0.05, 0.03, 0.02]) * movement_intensity
    phase_shifts = np.random.uniform(0, 2 * np.pi, len(base_frequencies))

    for i in range(num_frames):
        # Create complex rotation with multiple frequencies
        angle = np.sum([
            amp * np.sin(freq * i + phase)
            for amp, freq, phase in zip(base_amplitudes, base_frequencies, phase_shifts)
        ])

        # Add sudden movements occasionally
        if np.random.random() < sudden_movement_prob:
            angle += np.random.normal(0, 0.2 * movement_intensity)
            translation_sudden = np.random.normal(0, 8.0 * movement_intensity, 2)
        else:
            translation_sudden = np.zeros(2)

        # Create complex translations
        translation_x = movement_intensity * np.sum([
            3 * np.sin(0.3 * i + np.pi / 4),
            1 * np.cos(0.5 * i + np.pi / 3),
        ]) + translation_sudden[0]

        translation_y = movement_intensity * np.sum([
            3 * np.cos(0.4 * i + np.pi / 5),
            2 * np.sin(0.6 * i + np.pi / 2),
        ]) + translation_sudden[1]

        # Add continuous random noise
        angle += np.random.normal(0, noise_level)
        translation_x += np.random.normal(0, noise_level * 10)
        translation_y += np.random.normal(0, noise_level * 10)

        transformation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), translation_x],
            [np.sin(angle), np.cos(angle), translation_y],
            [0, 0, 1],
        ])
        transformation_matrices.append(transformation_matrix)

    return transformation_matrices