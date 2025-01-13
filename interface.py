# %%
from itertools import product
import cv2
import numpy as np
from tools import (
    video_to_array,
    random_image,
    artificial_movement,
    array_to_video,
    generate_complex_movement_matrices,
)
from stabilize import stabilize

# %%
# Test stabilize function with synthetic video
num_frames = 30  # Increased number of frames
synthetic_frames = [
    random_image(size=(200, 200), color=True, num_shapes=30) for _ in range(num_frames)
]
transformation_matrices = []


# Use the function with default parameters
transformation_matrices = generate_complex_movement_matrices(num_frames)

moved_frames = artificial_movement(synthetic_frames[0], transformation_matrices)
array_to_video(moved_frames, "synthetic_moved.mp4")
# %%
stabilized_frames = stabilize(
    moved_frames, cancel_axis="x", min_matches=5, skip_no_matches=True
)

# Save synthetic video before and after stabilization
array_to_video(stabilized_frames, "synthetic_stabilized.mp4")

# %%
# Test stabilize function with real video
real_video_path = "data\\input\\boat.mp4"
real_frames = video_to_array(real_video_path)
for alpha, max_rotation, max_translation in product(
    [0.1, 0.2, 0.4, 0.8], [0.5, 1, 2], [0.05, 0.1, 0.2]
):
    stabilized_real_frames = stabilize(
        real_frames[:100],
        cancel_axis="x",
        min_matches=4,
        skip_no_matches=True,
        max_rotation_degrees=max_rotation,
        max_translation=max_translation,
        alpha=alpha,
    )

    # Save real video before and after stabilization
    # array_to_video(real_frames, 'real_original.mp4')
    array_to_video(
        stabilized_real_frames,
        f"data\\my_output\\alpha_{alpha}_MR_{max_rotation}_MT_{max_translation}.mp4",
    )

# %%
