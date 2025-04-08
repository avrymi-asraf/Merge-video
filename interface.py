# %%
from itertools import product
import cv2
import numpy as np
from src.tools import (
    video_to_array,
    random_image,
    artificial_movement,
    array_to_video,
    generate_complex_movement_matrices,
)
from src.stabilize import stabilize
import matplotlib.pyplot as plt

# %%
real_video_path = "data\\input\\boat.mp4"
real_frames = video_to_array(real_video_path)

# %%


# def stitch_frames(frames, transformations):
#     """
#     Stitches a list of frames horizontally using a list of 2x3 (or 3x3)
#     transformation matrices that encode x-translations (and possibly y-translations).

#     :param frames: List of images (NumPy arrays) of size (H, W, 3).
#     :param transformations: List of transformation matrices, one per frame,
#                             where transformations[i][0,2] is typically the x-displacement of frame i.
#     :return: A stitched image (NumPy array) containing all frames.
#     """

#     # Safety check: each transformation[i] should be a 2x3 or 3x3 matrix
#     # for i in range(len(transformations)):
#     #     assert transformations[i].shape in [(2,3), (3,3)]

#     # 1) Compute each frame's x-offset on the final canvas
#     #    We'll store these in x_offsets so x_offsets[i] is where frame i starts horizontally.
#     x_offsets = [0]

#     for i in range(1, len(frames)):
#         # If your transform encodes frame i's position relative to frame (i-1),
#         # you add that x-displacement to the previous x_offset:
#         x_shift = transformations[i - 1][0, 2]
#         x_offsets.append(x_offsets[-1] + int(round(x_shift)))

#     # 2) Determine final stitched width
#     #    The last frame starts at x_offsets[-1], then extends by the width of the last frame.
#     stitched_width = x_offsets[-1] + frames[-1].shape[1]

#     # 3) Determine the height of the stitched canvas
#     #    For simplicity, we can just take the maximum frame height.
#     frame_height = max(frame.shape[0] for frame in frames)

#     # 4) Create an empty (black) canvas for the stitched result
#     stitched_image = np.zeros((frame_height, stitched_width, 3), dtype=np.uint8)

#     # 5) Copy each frame into the stitched image at its correct offset
#     for i, frame in enumerate(frames):
#         x_start = x_offsets[i]
#         h, w = frame.shape[:2]
#         stitched_image[0:h, x_start : x_start + w] = frame

#     return stitched_image


# # %%
# # Test stabilize function with real video

# stabilized_frames, t = stabilize(
#     real_frames,
#     cancel_axis="x",
#     min_matches=4,
#     skip_no_matches=True,
# )
# colomn_range = (100, 150)
# colomn_frames = [
#     frame[:, colomn_range[0] : colomn_range[1]] for frame in stabilized_frames
# ]
# # %%
# # %%
# out = stitch_frames(colomn_frames, t)
# plt.imshow(out)
# plt.show()
# # %%


# # %%
# # Example usage:
# def align_and_combine(img1, img2, guess=None):
#     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     # Define the motion model
#     # For small shifts/rotations, MOTION_EUCLIDEAN or MOTION_AFFINE often works well
#     warp_mode = cv2.MOTION_EUCLIDEAN

#     # Initialize the warp matrixm
#     if guess is not None:
#         warp_matrix = guess
#     else:
#         warp_matrix = np.eye(2, 3, dtype=np.float32)

#     # Number of iterations and threshold for ECC
#     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-4)

#     # Run ECC to find the warp matrix
#     (cc, warp_matrix) = cv2.findTransformECC(
#         gray1, gray2, warp_matrix, warp_mode, criteria
#     )
#     x_t = int(round(warp_matrix[0, 2]))
#     warp_matrix[0, 2] = 0
#     # Use warpAffine for translation, euclidean, or affine
#     aligned_img2 = cv2.warpAffine(
#         img2,
#         warp_matrix,
#         (img1.shape[1], img1.shape[0]),
#         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
#     )
#     # aligned_img2 = np.array(aligned_img2, dtype=np.uint8)
#     # Simple averaging or max blending to combine the images
#     # Here we do a simple average just as a demo
#     if x_t < 0:
#         aligned_img2, img1 = img1, aligned_img2
#         x_t = -x_t

#     img1 = np.concatenate([img1, aligned_img2[:, -x_t:]], axis=1)
#     aligned_img2 = np.concatenate((img1[:, :x_t], aligned_img2), axis=1)

#     combined = cv2.addWeighted(img1, 0.5, aligned_img2, 0.5, 0)
#     return combined, warp_matrix


# %%

# # %%
# im1 = colomn_frames[0]
# im2 = colomn_frames[1]
# guess = t[1].astype(np.float32)
# out, _ = align_and_combine(im1, im2, guess)
# plt.imshow(out)
# cv2.imwrite("data\\my_output\\stabilize_frame.png", out)


# %%
def concate_images(frames, t, range):
    
    colomn_frames = [frame[:, colomn_range[0] : colomn_range[1]] for frame in stabilized_frames]
    out = frames[0].astype(np.float32)
    for i in range(1, len(frames)):
        new_frame = frames[i]
        gray1 = cv2.cvtColor(out[:, -50:], cv2.COLOR_BGR2GRAY).astype(np.uint8)
        gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        (cc, warp_matrix) = cv2.findTransformECC(
            gray1, gray2, t[i].astype(np.float32), cv2.MOTION_EUCLIDEAN, criteria
        )

        x_t = max(int(warp_matrix[0, 2]), 0)
        warp_matrix[0, 2] = 0

        align_frame = cv2.warpAffine(
            new_frame,
            warp_matrix,
            (new_frame.shape[1], new_frame.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        ).astype(np.float32)
        overlding = align_frame.shape[1] - x_t
        out[:, -overlding:] *= 0.5
        out[:, -overlding:] += align_frame[:, :overlding] * 0.5
        out = np.concatenate((out, align_frame[:, overlding:]), axis=1)
    return out.astype(np.uint8)


# %%


stabilized_frames, t = stabilize(
    real_frames,
    cancel_axis="x",
    skip_no_matches=True,
)
colomn_range = (100, 150)
colomn_frames = [
    frame[:, colomn_range[0] : colomn_range[1]] for frame in stabilized_frames
]

# %%
out = concate_images(colomn_frames[:200], t[:200])
plt.imshow(out)
# %%
