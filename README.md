# Exercise 4: Stereo Mosaicing

## Goal

Implement the "Stereo Mosaicing" algorithm to create a video of n different panoramas from an input video scanning a scene horizontally. [cite: 57]

### Steps
* Compute rigid transformations between consecutive frames. 
* Stabilize rotations and y translations.
* Align all frames to the same coordinate system using motion composition.
* Create a mosaic by pasting strips of the correct width and location.
* Set a convergence point for the panoramas.
* Record and process your own videos to test the algorithm. 

## Files and Functions

### tools.py
*   [X] `video_to_array`: Convert a video into an array of frames.
*   [X] `artificial_movement`: Apply artificial movement to a frame or an array of frames according to a given set of transformation matrices.
*   [X] `create_synthetic_frame`: Create a synthetic frame with a given size, color, number of shapes, and shape size range.
*   [X] `array_to_video` convert np.array of frames to vidoe.

### findShift.py
*   [X] `keypoints`: Detect keypoints in an image using sift algorithm.
*   [X] `match`: matche for the points using RANSAC algorithem.
*   [X] `compute_homography`: Compute the homography matrix between two images.
*   [X] `find_shift`: Find the shift between two images.
*   [X] `visualize_matches`: visualize matches between two images.

### stabilize.py
*  [ ] `align_frames`: given two frames find the homography matrix between them and align the second frame to the.
*  [ ] `stabilize`: Stabilize the rotation and y translation in video by align the y translation and rotation for every frame to the frevius frame.