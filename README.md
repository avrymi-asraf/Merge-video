# Merge-video

## Poereps of the project
Align consecutive frames 
a. Lucas Kanade or Point Correspondences, you choose 
b. Create a transformation matrix for every two consecutive frames 
c. Debug: Use synthetic videos: no motion, simple translation, rotation...    

Stabilize Rotations & Y translations 
a. Debug: Use synthetic videos: no motion, simple translation, rotation... 
b. Create - warp frames to get a stable video (only horizontal motion)    

Use motion composition to align all frames to same coordinates.
a. Compute canvas size from motion matrices
b. Create - paste aligned frames into canvas on top of each other

Create mosaic by pasting strips using correct width & location
a. Start with synthetic videos at constant translation & no rotation...
b. Back Warping from canvas to frame...

Set convergence point (With no setting - this is infinity)
a. Depth point that does not move between mosaics



## filse
### Tools.py
contains the tools used in the project
- [ ] functoin extract video to array of numpy arrays  `video_to_array`
- [ ] functoin thet revice image and make move according the set of tranformatoin matrixs `artificial_movementmes`


### Lucas_Kanade.py
contains the implementation of the Lucas Kanade algorithm
 - [ ] function `lucas_kanade` that takes two frames and returns the transformation matrix

