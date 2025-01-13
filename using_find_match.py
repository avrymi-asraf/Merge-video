# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from findShift import keypoints, match, compute_homography
from tools import create_synthetic_frame, artificial_movement,video_to_array

# %%
# Create a synthetic reference image
image = create_synthetic_frame(
    size=(400, 400), color=True, num_shapes=100, shape_size_range=(5, 15)
)
image = video_to_array("data\\input\\boat.mp4")[0]
# Visualize the synthetic image
plt.figure(figsize=(8, 8))
plt.title("Step 1: Generated Synthetic Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

# %%
# Create transformation matrices for artificial movement
translation_matrix = np.array(
    [[1, 0, 50], [0, 1, 30], [0, 0, 1]]  # Move 50 pixels right  # Move 30 pixels down
)

rotation_matrix = cv2.getRotationMatrix2D(
    (200, 200), angle=15, scale=1.0  # Center point  # 15 degrees rotation
)
rotation_matrix = np.vstack([rotation_matrix, [0, 0, 1]])  # Make it 3x3

# Create sequence of transformations
transformations = [translation_matrix, rotation_matrix]

# %%
# Apply artificial movement to create sequence of frames
moved_frames = artificial_movement(image, transformations, clip_frames=False)

# Visualize the transformation sequence
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title("Original Frame")
plt.imshow(cv2.cvtColor(moved_frames[0], cv2.COLOR_BGR2RGB))
plt.subplot(132)
plt.title("After Translation")
plt.imshow(cv2.cvtColor(moved_frames[1], cv2.COLOR_BGR2RGB))
plt.subplot(133)
plt.title("After Rotation")
plt.imshow(cv2.cvtColor(moved_frames[2], cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()

# %%
# Detect and match features between original and transformed
original_frame = moved_frames[0]
transformed_frame = moved_frames[
    -1
]  # Get the last frame (with all transformations applied)
# Detect keypoints
kp1, desc1 = keypoints(original_frame)
kp2, desc2 = keypoints(transformed_frame)
# %%
# Visualize keypoints
img1_keypoints = cv2.drawKeypoints(original_frame, kp1, None, color=(0, 255, 0))
img2_keypoints = cv2.drawKeypoints(transformed_frame, kp2, None, color=(0, 255, 0))

plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.title(f"Original Frame Keypoints ({len(kp1)} points)")
plt.imshow(cv2.cvtColor(img1_keypoints, cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.title(f"Transformed Frame Keypoints ({len(kp2)} points)")
plt.imshow(cv2.cvtColor(img2_keypoints, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()
# %%
# Match keypoints
good_matches = match(desc1, desc2, kp1, kp2)

# Visualize matches
match_img = cv2.drawMatches(
    original_frame,
    kp1,
    transformed_frame,
    kp2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
plt.figure(figsize=(15, 5))
plt.title(f"Step 4: Keypoint Matching ({len(good_matches)} good matches)")
plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
plt.show()

# Compute homography
H, mask = compute_homography(kp1, kp2, good_matches)

# %%
# Visualize the final results
height, width = original_frame.shape[:2]
recovered_image = cv2.warpPerspective(
    transformed_frame, np.linalg.inv(H), (width, height)
)

plt.figure(figsize=(20, 5))
plt.subplot(141)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
plt.subplot(142)
plt.title("Transformed Image")
plt.imshow(cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2RGB))
plt.subplot(143)
plt.title("Recovered Image")
plt.imshow(cv2.cvtColor(recovered_image, cv2.COLOR_BGR2RGB))
plt.subplot(144)
plt.title("Difference (Original - Recovered)")
diff_image = cv2.absdiff(original_frame, recovered_image)
plt.imshow(cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()

# %%
# Print transformation analysis
print("Number of good matches found:", len(good_matches))
print("\nOriginal transformation matrices:")
print("Translation:")
print(translation_matrix)
print("\nRotation:")
print(rotation_matrix)
print("\nRecovered transformation matrix (homography):")
print(H)

# Visualize transformation error
fig = plt.figure(figsize=(8, 4))
plt.title("Transformation Error Analysis")
combined = np.concatenate([original_frame, recovered_image], axis=1)
plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
plt.axvline(x=width, color="r", linestyle="--")
plt.text(width / 2, -10, "Original", ha="center")
plt.text(width * 1.5, -10, "Recovered", ha="center")
plt.show()

# %%
