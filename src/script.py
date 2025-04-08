
from itertools import product
from tools import (
    video_to_array,
    array_to_video,
)
from stabilize import stabilize
import cv2
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
        real_frames[:150],
        cancel_axis="x",
        min_matches=4,
        skip_no_matches=True,
    )[0]

    # Save real video before and after stabilization

    current_time = datetime.now().strftime('%H%M')
    array_to_video(stabilized_real_frames, f'data\\my_output\\stabilize_{current_time}.mp4')
