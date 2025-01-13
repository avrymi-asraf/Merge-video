#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from findShift import keypoints, match, compute_homography,find_shift,visualize_matches
from tools import create_synthetic_frame, artificial_movement,video_to_array


#%%
image1, image2 = video_to_array("data\\input\\boat.mp4")[6:8]

# %%
fig,im1,im2 = visualize_matches(image1, image2)
plt.imshow(im1)
# %%
