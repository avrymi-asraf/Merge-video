from itertools import product
from tools import (
    video_to_array,
    array_to_video,
)
from stabilize import stabilize

real_video_path = "data\\input\\boat.mp4"
real_frames = video_to_array(real_video_path)
for alpha, max_rotation, max_translation in product(
    [0.1, 0.2, 0.4, 0.8], [0.5, 1, 2], [0.05, 0.1, 0.2]
):
    stabilized_real_frames = stabilize(
        real_frames[:150],
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
