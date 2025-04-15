import cv2
import numpy as np
import datetime
import os


def video_to_array(video_path):
    """
    Convert a video file to a numpy array of frames.

    Args:
        video_path (str): Path to the video file

    Returns:
        np.ndarray: Array of frames with shape (num_frames, height, width, channels)
                   Each frame is uint8 type with values 0-255

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

    return np.array(frames, dtype=np.uint8)

def array_to_image(image, output_path):
    cv2.imwrite(output_path, image)


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

    height, width = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
            raise ValueError("Error: Could not create video file")

    try:
 
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("Error: Inconsistent frame dimensions")
            out.write(frame.astype(np.uint8))
    finally:
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


def apply_homography(image, homography_matrix, output_size=None):
    """
    Transform an image using a homography matrix.

    Args:
        image (np.ndarray): Input image of shape (height, width, channels)
                           or (height, width) for grayscale images
        homography_matrix (np.ndarray): 3x3 homography transformation matrix
                                      that maps points from the input image to the output image
        output_size (tuple, optional): Size of the output image as (width, height).
                                     If None, the input image size is used.

    Returns:
        np.ndarray: Transformed image with the same number of channels as the input image.
                   If output_size is specified, the returned image will have that size.

    Raises:
        ValueError: If the input image is empty or the homography matrix is invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Error: Empty input image")
    
    if homography_matrix.shape != (3, 3):
        raise ValueError("Error: Homography matrix must be 3x3")
    
    height, width = image.shape[:2]
    
    # Use the input image size if output_size is not specified
    if output_size is None:
        output_size = (width, height)
    
    # Apply the homography transformation
    transformed_image = cv2.warpPerspective(
        image, 
        homography_matrix, 
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return transformed_image


if __name__ == "__main__":
    # Test the apply_homography function
    import matplotlib.pyplot as plt
    
    # Create a test image with a simple pattern
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    test_image[50:250, 50:250] = (255, 255, 255)  # White square
    
    # Add a pattern inside the square to better visualize the transformation
    for i in range(70, 230, 20):
        cv2.line(test_image, (i, 70), (i, 230), (0, 0, 255), 2)  # Vertical red lines
        cv2.line(test_image, (70, i), (230, i), (0, 255, 0), 2)  # Horizontal green lines
    
    # Create a homography matrix for perspective transformation
    # This example creates a perspective effect (like viewing a square from an angle)
    src_points = np.array([[50, 50], [250, 50], [250, 250], [50, 250]], dtype=np.float32)
    dst_points = np.array([[80, 70], [220, 50], [250, 210], [30, 220]], dtype=np.float32)
    homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply the homography
    transformed_image = apply_homography(test_image, homography_matrix)
    
    # Display the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Transformed Image (Homography)")
    plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Test with additional homography examples
    
    # 1. Rotation homography (45 degrees)
    center = (test_image.shape[1] // 2, test_image.shape[0] // 2)
    angle = 45
    scale = 1
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotation_homography = np.eye(3)
    rotation_homography[:2, :] = rotation_matrix
    rotated_image = apply_homography(test_image, rotation_homography)
    
    # 2. Translation homography (move 50px right, 30px down)
    translation_homography = np.eye(3)
    translation_homography[0, 2] = 50  # x translation
    translation_homography[1, 2] = 30  # y translation
    translated_image = apply_homography(test_image, translation_homography)
    
    # Display additional results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Rotated Image (45 degrees)")
    plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Translated Image (50px right, 30px down)")
    plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Homography transformation tests completed successfully!")




def get_timestamp_folder(folder_name):
    """
    Generate a timestamp-based folder name with 6 digits representing day of month, hour, and minutes.

    Args:
        folder_name (str): folder name to add
    
    Returns:
        str: Path to the output folder with timestamp in format 'data/my_output/test_find_shift_DDHHMM'
             where DD is day of month (01-31), HH is hour (00-23), MM is minutes (00-59)
    """
    now = datetime.datetime.now()
    # Day of month (01-31)
    day = now.day
    # Hour (00-23) 
    hour = now.hour
    # Minutes (00-59)
    minute = now.minute
    
    # Format as 6 digits (DDHHMM)
    timestamp = f"{day:02d}{hour:02d}{minute:02d}"
    
    # Create the full path
    output_folder = os.path.join("data", "my_output", f"{folder_name}_{timestamp}")
    
    # Create the directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    return output_folder