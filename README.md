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
*  [X] `video_to_array`: Convert a video into an array of frames.
*  [X] `artificial_movement`: Apply artificial movement to a frame or an array of frames according to a given set of transformation matrices.
*  [X] `random_image`: Create a synthetic frame with a given size, color, number of shapes, and shape size range.
*  [X] `array_to_video` convert np.array of frames to vidoe.

### findShift.py
*  [X] `calc_translations`: Given array of frames, find the rigid transformation between them using SIFT feature matching, return list of transformations between consecutive frames, and the list of cumulative transformations between the first frame and each of the other frames.
*  [ ] `create_canvas`: Create a canvas to hold the mosaic of the video, by calculating the size of the mosaic based on the matrix multiplication of the transformation matrices that result from the `findShift_SIFT` function.
*  [X] `visualize_matches`: Visualize matches between two images using either SIFT or Lucas-Kanade.

