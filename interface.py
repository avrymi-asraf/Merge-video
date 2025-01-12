#%%
from tools import video_to_array, artificial_movement
import matplotlib.pyplot as plt
import numpy as np
import cv2

#%% Load video
video_arr = video_to_array("data\\input\\boat.mp4")

#%% Create transformation matrices for 5 frames
# Create rotation matrices with increasing angles
angles = np.linspace(0, 15, 5)  # Rotate from 0 to 15 degrees
translations = np.linspace(0, 50, 5)  # Translate from 0 to 50 pixels

transformation_matrices = []
for angle, trans in zip(angles, translations):
    # Create rotation matrix
    angle_rad = np.deg2rad(angle)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotation = np.array([
        [c, -s, trans],
        [s, c, trans],
        [0, 0, 1]
    ])
    transformation_matrices.append(rotation)

#%% Apply transformations
frame = video_arr[200]  # Use frame 200 as example
transformed_frames, corner_points = artificial_movement(frame, transformation_matrices, clip_frames=True)

#%% Display results
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('Artificial Movement Example')

# Plot all frames
for i in range(5):
    axes[i].imshow(transformed_frames[i])
    axes[i].axis('off')
    axes[i].set_title(f'Frame {i}\n{transformed_frames[i].shape[:2]}')

plt.tight_layout()
plt.show()

# %%
