import cv2
import os
from tools import array_to_video, array_to_image, video_to_array
from tqdm import tqdm
import numpy as np


def find_good_matches(descriptors_im1, descriptors_im2, k=2, threshold=0.75):
    """
    Match feature descriptors between two images using Lowe's ratio test.

    Args:
        descriptors_im1 (np.ndarray): Descriptors from the first image, shape (n, 128) for SIFT.
        descriptors_im2 (np.ndarray): Descriptors from the second image, shape (m, 128) for SIFT.
        k (int): Number of nearest neighbors to find for each descriptor. Defaults to 2.
        threshold (float): Threshold for Lowe's ratio test (0.0-1.0). Defaults to 0.75.

    Returns:
        list: A list of cv2.DMatch objects representing good matches that passed the ratio test.
    """

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches_2nn = bf.knnMatch(descriptors_im1, descriptors_im2, k=k)

    return [
        best_match
        for best_match, second_best_match in matches_2nn
        if best_match.distance < threshold * second_best_match.distance
    ]


def ransac(src_points, dest_point, matches, top_of_inliners=0.5):
    """
    Apply RANSAC algorithm to filter out outlier matches between two images.

    Args:
        match_point_last (np.ndarray): Points from the previous frame, shape (n, 1, 2) where n is the number of matches.
        match_point_curr (np.ndarray): Points from the current frame, shape (n, 1, 2) where n is the number of matches.
        matches (list): List of cv2.DMatch objects representing potential matches between frames.

    Returns:
        list: Filtered list of cv2.DMatch objects containing only inlier matches.
    """
    H, inliers_mask = cv2.estimateAffinePartial2D(
        src_points, dest_point, method=cv2.RANSAC
    )
    inlier_matches = [match for i, match in enumerate(matches) if inliers_mask[i] == 1]
    inlier_matches.sort(key=lambda x: x.distance)
    inlier_matches = inlier_matches[: int(len(inlier_matches) * top_of_inliners)]

    return inlier_matches


def compute_transformation_matrix(
    keypoints_last, keypoints_current, matches, top_k=200
):
    """
    Calculate transformation matrix between two frames based on matched keypoints.

    Args:
        keypoints_last (list): Keypoints from the previous frame, list of cv2.KeyPoint objects.
        keypoints_current (list): Keypoints from the current frame, list of cv2.KeyPoint objects.
        matches (list): List of cv2.DMatch objects representing inlier matches between frames.
        top_k (int): Number of top matches to consider for transformation calculation. Defaults to 200.

    Returns:
        np.ndarray: A 3x3 transformation matrix representing translation between frames.
    """

    source_points = np.float32(
        [keypoints_last[m.queryIdx].pt for m in matches]
    ).reshape(-1, 2)
    destination_points = np.float32(
        [keypoints_current[m.trainIdx].pt for m in matches]
    ).reshape(-1, 2)
    diff = source_points - destination_points
    dx, dy = np.median(diff[:, 0]), np.median(diff[:, 1])
    matrix = np.eye(3)
    matrix[0, 2] = dx
    matrix[1, 2] = dy
    return matrix


def calc_all_translations(frames):
    """
    Calculate translation matrices between consecutive frames in a video sequence.

    Args:
        frames (list): List of video frames as numpy arrays, shape (height, width, 3).

    Returns:
        tuple: A tuple containing:
            - cum_translations (list): List of cumulative 3x3 transformation matrices.
            - translations (list): List of frame-to-frame 3x3 transformation matrices.
    """
    translations = np.empty((len(frames) - 1, 3, 3), dtype=np.float32)
    cumulative_translations = [np.eye(3)]
    sift = cv2.SIFT_create()
    for i in range(len(frames) - 1):
        last_frame = frames[i]
        current_frame = frames[i + 1]
        src_points, descriptors_last = sift.detectAndCompute(last_frame, None)
        dest_points, descriptors_current = sift.detectAndCompute(current_frame, None)
        matches = find_good_matches(descriptors_last, descriptors_current)
        match_src_points = np.array(
            [src_points[m.queryIdx].pt for m in matches], dtype=np.float32
        ).reshape(-1, 1, 2)
        match_dest_points = np.array(
            [dest_points[m.trainIdx].pt for m in matches], dtype=np.float32
        ).reshape(-1, 1, 2)

        god_matches = ransac(match_src_points, match_dest_points, matches)
        translations[i] = compute_transformation_matrix(
            src_points,
            dest_points,
            god_matches,
        )
        cumulative_translations.append(cumulative_translations[-1] @ translations[i])
    cumulative_translations = cumulative_translations[1:]
    return cumulative_translations, translations


def apply_backward_warp(frame, forward_transform_matrix, height, width):
    """
    Apply backward warping to transform a frame into the panoramic canvas space.

    Args:
        target_canvas (np.ndarray): The target canvas image, shape (canvas_height, canvas_width, 3).
        frame (np.ndarray): Source frame to warp, shape (frame_height, frame_width, 3).
        transform_matrix (np.ndarray): 3x3 transformation matrix mapping from canvas to frame.

    Returns:
        np.ndarray: Warped frame in the canvas space, shape (canvas_height, canvas_width, 3).
    """
    backward_matrix = np.linalg.inv(forward_transform_matrix)
    return cv2.warpPerspective(
        frame,
        backward_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )


def cat_canvaces_for_dynamic(canvaces, frame_width):
    """
    Create dynamic panorama by strategically trimming each canvas.

    Args:
        canvaces (list): List of panorama canvases as numpy arrays, shape (canvas_height, canvas_width, 3).
        frame_width (int): Width of the original video frames.

    Returns:
        list: List of trimmed panorama canvases for dynamic panorama visualization.
    """
    panorama_width = canvaces[0].shape[1]
    num_canvaces = len(canvaces)
    out = []
    for canvas_index in range(num_canvaces):
        canvas_left_offset = canvas_index * (frame_width // num_canvaces)
        canvas_right_offset = (num_canvaces - canvas_index) * (
            frame_width // num_canvaces
        )
        cropped_canvas = canvaces[canvas_index][
            :, canvas_left_offset : panorama_width - canvas_right_offset
        ]
        out.append(cropped_canvas)
    return out


def cat_canvaces_for_viewpoint(canvaces, frame_width):
    """
    Create viewpoint panorama by trimming each canvas uniformly.

    Args:
        canvaces (list): List of panorama canvases as numpy arrays, shape (canvas_height, canvas_width, 3).
        frame_width (int): Width of the original video frames.

    Returns:
        list: List of trimmed panorama canvases for viewpoint visualization.
    """
    panorama_width = canvaces[0].shape[1]
    return [
        canvas[:, frame_width : panorama_width - frame_width] for canvas in canvaces
    ]


def create_canvaces(frames, num_of_canvaces, move_between=20):
    """
    Create panoramic canvases from a sequence of video frames.

    Args:
        frames (list): List of video frames as numpy arrays, shape (height, width, 3).
        num_of_canvaces (int): Number of panoramic canvases to create.
        move_between (int): Number of pixels to move between canvases. Defaults to 20.

    Returns:
        list: List of panorama canvases as numpy arrays, shape (canvas_height, canvas_width, 3).
    """
    cumulative_translations, translations = calc_all_translations(frames)
    frame_height, frame_width = frames[0].shape[:2]

    total_dx = translations[:, 0, 2].sum()
    max_dy = np.ceil(max(mat[1, 2] for mat in cumulative_translations))
    min_dy = np.floor(min(mat[1, 2] for mat in cumulative_translations))

    canvas_width = int(np.ceil(total_dx + frame_width))
    canvas_height = int(np.ceil(frame_height + max_dy - min_dy))

    canvaces = np.empty(
        (num_of_canvaces, canvas_height, canvas_width, 3), dtype=np.uint8
    )
    for canva_index in range(1, num_of_canvaces):

        current_position_x = 0
        strip_start_in_frame = (
            canva_index * (frame_width // num_of_canvaces) + move_between
        )
        for frame, frame_translation_matrix, cum_translation in zip(
            frames[:-2], translations, cumulative_translations
        ):
            dx = round(frame_translation_matrix[0, 2])
            canvas_start_pos = current_position_x + strip_start_in_frame
            strip_end_in_canvas = current_position_x + strip_start_in_frame + dx
            canvas_with_frame = apply_backward_warp(
                frame, cum_translation, canvas_height, canvas_width
            )
            canvaces[canva_index][:, canvas_start_pos:strip_end_in_canvas] = (
                canvas_with_frame[:, canvas_start_pos:strip_end_in_canvas]
            )
            current_position_x += dx

    return canvaces


if __name__ == "__main__":
    # remove the padding in the iguazu video
    # path = "data/input/iguazu.mp4"
    # frames = video_to_array(path)
    # out = [frame[:, 170 : frame.shape[1] - 170] for frame in frames]
    # array_to_video(out, "data/input/iguazu_clean.mp4", fps=24)

    # load the big canvaces without pastprossing
    # path = "data/input/iguazu.mp4"
    # frames = video_to_array(path)
    # canvaces = create_canvaces(frames, 24)
    # array_to_video(canvaces, "data/input/iguazu_big_canvaces.mp4", fps=5)

    ## dynamic panorama
    path = "data/input/iguazu_clean.mp4"
    name_file = os.path.basename(path).split(".")[0]
    output_path = "data/my_output/" + name_file
    frames = video_to_array(path)
    canvaces = create_canvaces(frames, 30)
    panoramas = cat_canvaces_for_dynamic(canvaces, frames[0].shape[1])

    array_to_video(panoramas, output_path + "_dynamic.mp4", fps=12)
    array_to_image(panoramas[0], output_path + "_first.png")
    array_to_image(panoramas[len(panoramas) // 2], output_path + "_middle.png")
    array_to_image(panoramas[-1], output_path + "_last.png")

    # viewpoint panorama
    path = "data/input/boat.mp4"
    name_file = os.path.basename(path).split(".")[0]
    output_path = "data/my_output/" + name_file
    frames = video_to_array(path)
    canvaces = create_canvaces(frames, 24)
    panoramas = cat_canvaces_for_viewpoint(canvaces, frames[0].shape[1])
    array_to_video(panoramas, output_path + "_viewpoint.mp4", fps=8)
    array_to_image(panoramas[0], output_path + "_viewpoint_first.png")
    array_to_image(
        panoramas[len(panoramas) // 2], output_path + "_viewpoint_middle.png"
    )
    array_to_image(panoramas[-1], output_path + "_viewpoint_last.png")
