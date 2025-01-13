import cv2
import numpy as np
import matplotlib.pyplot as plt

def keypoints(image):
    """
    Detect keypoints in an image using SIFT algorithm.

    Args:
        image (np.ndarray): Input image array of shape (H, W) or (H, W, C).

    Returns:
        tuple: (keypoints, descriptors) where:
            - keypoints (list): List of cv2.KeyPoint objects
            - descriptors (np.ndarray): Array of shape (N, 128) containing SIFT descriptors

    Raises:
        ValueError: If the input image is None or empty
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")

    # Convert image to grayscale if it's in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray.astype('uint8'), None)
    return keypoints, descriptors

def match(desc1, desc2, kp1, kp2, ratio_thresh=0.75):
    """
    Match keypoints between two images using FLANN matcher and RANSAC.

    Args:
        desc1 (np.ndarray): Descriptors from first image, shape (N, 128)
        desc2 (np.ndarray): Descriptors from second image, shape (M, 128)
        kp1 (list): Keypoints from first image
        kp2 (list): Keypoints from second image
        ratio_thresh (float): Ratio test threshold for Lowe's ratio test

    Returns:
        list: List of good matches that passed the ratio test and RANSAC

    Raises:
        ValueError: If descriptors or keypoints are None or empty
    """
    if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
        raise ValueError("Invalid input: descriptors or keypoints are empty")

    # Initialize FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches using knnMatch
    matches = flann.knnMatch(desc1, desc2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches

def compute_homography(kp1, kp2, matches):
    """
    Compute the homography matrix between two images using matched keypoints.

    Args:
        kp1 (list): Keypoints from first image
        kp2 (list): Keypoints from second image
        matches (list): List of good matches between the keypoints

    Returns:
        np.ndarray: 3x3 homography matrix that transforms points from image1 to image2
        np.ndarray: Mask indicating which matches were considered inliers

    Raises:
        ValueError: If there are not enough matches to compute homography
    """
    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography (minimum 4 required)")

    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography matrix using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H, mask

def find_shift(img1, img2, ratio_thresh=0.75, ransac_thresh=5.0, min_matches=10):
    """
    Find the complete transformation (translation and rotation) between two images.

    Args:
        img1 (np.ndarray): First image array of shape (H, W) or (H, W, C)
        img2 (np.ndarray): Second image array of shape (H, W) or (H, W, C)
        ratio_thresh (float): Ratio test threshold for Lowe's ratio test (default: 0.75)
        ransac_thresh (float): Maximum allowed reprojection error in RANSAC (default: 5.0)
        min_matches (int): Minimum number of good matches required (default: 10)

    Returns:
        dict: Transformation parameters containing:
            - 'translation' (tuple): (dx, dy) shift in pixels
            - 'rotation' (float): rotation angle in degrees
            - 'scale' (float): scale factor
            - 'confidence' (float): transformation confidence score (0-1)
            - 'homography' (np.ndarray): 3x3 homography matrix
            - 'inliers_mask' (np.ndarray): Boolean mask of inlier matches
            - 'num_matches' (int): Number of good matches found

    Raises:
        ValueError: If images are None/empty or if insufficient matches are found
    """
    if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
        raise ValueError("Input images are empty or None")

    # Detect keypoints and compute descriptors
    kp1, desc1 = keypoints(img1)
    kp2, desc2 = keypoints(img2)

    # Match keypoints with custom ratio threshold
    good_matches = match(desc1, desc2, kp1, kp2, ratio_thresh=ratio_thresh)

    if len(good_matches) < min_matches:
        raise ValueError(f"Not enough good matches found (minimum {min_matches} required, got {len(good_matches)})")

    # Compute homography with custom RANSAC threshold
    H, mask = compute_homography(kp1, kp2, good_matches)

    # Decompose homography matrix into rotation, translation and scale
    # Get image center for rotation reference
    h, w = img1.shape[:2]
    center = (w / 2, h / 2)
    
    # Decompose homography
    _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, np.array([[0,0,1]]))
    
    # Select the most probable solution (usually the first one for simple transformations)
    R = Rs[0]
    T = Ts[0]
    
    # Extract rotation angle (in degrees)
    rotation_angle = np.degrees(np.arctan2(R[1,0], R[0,0]))
    
    # Extract scale (average of x and y scaling)
    scale = np.sqrt((H[0,0]**2 + H[1,0]**2 + H[0,1]**2 + H[1,1]**2) / 2)
    
    # Extract translation
    dx = H[0,2]
    dy = H[1,2]
    
    # Calculate confidence score based on inlier ratio
    num_inliers = np.sum(mask)
    confidence = num_inliers / len(good_matches) if len(good_matches) > 0 else 0

    return {
        'translation': (dx, dy),
        'rotation': rotation_angle,
        'scale': scale,
        'confidence': confidence,
        'homography': H,
        'inliers_mask': mask,
        'num_matches': len(good_matches)
    }

def visualize_matches(img1, img2, title="Image Matching Visualization"):
    """
    Visualize the matching process between two images and show the transformed result.

    Args:
        img1 (np.ndarray): First image array of shape (H, W) or (H, W, C)
        img2 (np.ndarray): Second image array of shape (H, W) or (H, W, C)
        title (str): Title for the visualization plot

    Returns:
        tuple: (fig, matches_img, warped_img) where:
            - fig: matplotlib figure object
            - matches_img (np.ndarray): Image showing the matches
            - warped_img (np.ndarray): Image showing img1 transformed to align with img2
    """
    if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
        raise ValueError("Input images are empty or None")

    # Detect keypoints and compute descriptors
    kp1, desc1 = keypoints(img1)
    kp2, desc2 = keypoints(img2)

    # Match keypoints
    good_matches = match(desc1, desc2, kp1, kp2)

    # Compute homography
    H, mask = compute_homography(kp1, kp2, good_matches)
    print("\nHomography Matrix:")
    print(H)

    # Create match visualization
    matches_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Warp image1 to align with image2
    h, w = img2.shape[:2]
    warped_img = cv2.warpPerspective(img1, H, (w, h))

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 5))
    plt.subplot(141)
    plt.title(f"Image 1 Keypoints ({len(kp1)})")
    plt.imshow(cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0)))
    plt.axis('off')

    plt.subplot(142)
    plt.title(f"Image 2 Keypoints ({len(kp2)})")
    plt.imshow(cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0)))
    plt.axis('off')

    plt.subplot(143)
    plt.title(f"Matches ({len(good_matches)})")
    plt.imshow(matches_img)
    plt.axis('off')

    plt.subplot(144)
    plt.title("Transformed Image 1")
    plt.imshow(warped_img)
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    return fig, matches_img, warped_img
