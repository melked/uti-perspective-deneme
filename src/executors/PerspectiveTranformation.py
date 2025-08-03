import cv2
import numpy as np
import math
from itertools import combinations


# --- Helper Functions ---

def order_points(pts):
    """
    Order the 4 points of a quadrilateral in the order: top-left, top-right,
    bottom-right, bottom-left.
    """
    # Convert points to a numpy array
    pts = np.array(pts)
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point has the smallest sum, and the bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    # The top-right point has the smallest difference, while the bottom-left has the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect

def four_point_transform(image, pts):
    """
    Apply a four point perspective transform to an image given 4 source points.
    """
    # Obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image, which will be the maximum distance between
    # the top-right and top-left x-coordinates or the bottom-right and bottom-left x-coordinates
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image, which will be the maximum distance between
    # the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Now that we have the dimensions of the new image, construct the set of destination
    # points to obtain a "birds eye view" (i.e., top-down perspective) of the image,
    # specifying points in the top-left, top-right, bottom-right, and bottom-left order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the warped image
    return warped, rect


def preprocess_image_clahe(img, clipLimit=3.0, tileGridSize=(8,8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement."""
    # Convert to LAB color space as L channel is luminance
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the original A and B channels
    limg = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final_img

def preprocess_image_grayscale(img):
    """Convert image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def preprocess_image_adaptive_threshold(img_gray, block_size=11, c_value=2):
    """Apply adaptive thresholding to a grayscale image."""
    # Block size must be odd and greater than 1
    if block_size % 2 == 0 or block_size <= 1:
        block_size = 11 # Default if invalid

    # Apply adaptive Gaussian thresholding
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value)
    return thresh

def preprocess_image_canny_edges(img_gray, threshold_min=30, threshold_max=140):
    """Apply Canny edge detection to a grayscale image."""
    return cv2.Canny(img_gray, threshold_min, threshold_max)

def find_lines_hough(edges, rho=1, theta=np.pi/180, threshold=250):
    """Find lines in an edge image using Standard Hough Transform."""
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    if lines is not None:
        # Reshape to a list of lines (rho, theta)
        return lines[:, 0, :]
    return np.array([])

def find_lines_probabilistic_hough(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10):
    """Find line segments in an edge image using Probabilistic Hough Transform."""
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is not None:
        # Reshape to a list of line segments (x1, y1, x2, y2)
        return lines.reshape(-1, 4)
    return np.array([])

def get_intersections_from_lines(lines):
    """Calculate intersection points from a list of lines (rho, theta)."""
    intersections = []
    if len(lines) < 2:
        return np.array([], dtype=np.float32)

    for i, j in combinations(range(len(lines)), 2):
        rho1, theta1 = lines[i]
        rho2, theta2 = lines[j]

        # Convert polar coordinates to line equations (Ax + By = C)
        # A = cos(theta), B = sin(theta), C = rho
        A1, B1, C1 = np.cos(theta1), np.sin(theta1), rho1
        A2, B2, C2 = np.cos(theta2), np.sin(theta2), rho2

        # Calculate determinant
        denom = A1 * B2 - A2 * B1

        # If determinant is close to zero, lines are parallel
        if abs(denom) < 1e-6:
            continue

        # Calculate intersection point (x, y)
        px = (B2 * C1 - B1 * C2) / denom
        py = (A1 * C2 - A2 * C1) / denom

        # Add intersection point
        intersections.append([px, py])

    return np.array(intersections, dtype=np.float32)


def get_intersections_from_segments(segments, img_shape):
    """Calculate intersection points from a list of line segments (x1, y1, x2, y2)."""
    # This is a simplified approach. For accurate intersections of segments,
    # you'd need to check if the intersection point lies within both segments.
    # For finding document corners, extending lines to intersections is often more effective.
    # This function can be used for completeness but might not be the primary method for corners.

    # Convert segments to lines (for get_intersections_from_lines)
    lines_polar = []
    for x1, y1, x2, y2 in segments:
        # Calculate rho and theta from two points
        # Vector from (x1,y1) to (x2,y2) is (dx, dy) = (x2-x1, y2-y1)
        # Normal vector is (-dy, dx) or (dy, -dx)
        dx, dy = x2 - x1, y2 - y1
        # The line equation is dy*(x - x1) - dx*(y - y1) = 0
        # Which is dy*x - dx*y + (dx*y1 - dy*x1) = 0
        # In polar coordinates: x*cos(theta) + y*sin(theta) = rho
        # We can relate the two forms.
        # dy = cos(theta), -dx = sin(theta) -> theta = atan2(-dx, dy)
        # rho = dy*x1 - dx*y1  /  sqrt(dy^2 + (-dx)^2)
        # This can be complex. A simpler approach is to convert the segment endpoints
        # back to polar form if the segment is sufficiently long, or find intersections
        # by extending the line defined by the segment.

        # A common way is to extend the segments to the image boundaries and find intersections.
        # Or use the get_intersections function which takes cartesian points.

        # Let's use get_intersections (which takes cartesian line points) by extending segments.
        # This assumes get_intersections works with arbitrary line points, not just those from HoughLines.
        # We need to convert segments to lines that span the image.

        # Simplified approach: Use get_intersections with the segments treated as lines.
        # This is NOT mathematically precise as it doesn't extend lines to find far-off intersections.
        # A better approach is to use the polar lines method or extend segments properly.

        # Re-implementing extension logic more carefully for get_intersections
        height, width = img_shape[:2]
        # Calculate line equation (ax + by + c = 0)
        # (y2 - y1)x + (x1 - x2)y + (x2y1 - x1y2) = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2

        # Find points where the line intersects image boundaries
        boundary_points = []
        # Top edge (y=0): ax + c = 0 => x = -c/a (if a != 0)
        if a != 0:
            x_at_y0 = -c / a
            if 0 <= x_at_y0 <= width:
                boundary_points.append((int(x_at_y0), 0))
        # Bottom edge (y=height): ax + b*height + c = 0 => x = (-b*height - c)/a (if a != 0)
        if a != 0:
            x_at_yh = (-b * height - c) / a
            if 0 <= x_at_yh <= width:
                boundary_points.append((int(x_at_yh), height))
        # Left edge (x=0): by + c = 0 => y = -c/b (if b != 0)
        if b != 0:
            y_at_x0 = -c / b
            if 0 <= y_at_x0 <= height:
                boundary_points.append((0, int(y_at_x0)))
        # Right edge (x=width): b*width + by + c = 0 => y = (-a*width - c)/b (if b != 0)
        if b != 0:
            y_at_xw = (-a * width - c) / b
            if 0 <= y_at_xw <= height:
                boundary_points.append((width, int(y_at_xw)))

        # Remove duplicate points (can happen at corners)
        unique_boundary_points = list(set(boundary_points))

        # If we have at least two distinct boundary points, define the extended line
        if len(unique_boundary_points) >= 2:
            p1, p2 = unique_boundary_points[0], unique_boundary_points[1]
            extended_lines.append((p1[0], p1[1], p2[0], p2[1]))
        elif len(unique_boundary_points) == 1:
             # If only one boundary point, extend from the closest segment endpoint to it
             bp = unique_boundary_points[0]
             dist1 = np.linalg.norm(np.array([x1, y1]) - np.array(bp))
             dist2 = np.linalg.norm(np.array([x2, y2]) - np.array(bp))
             if dist1 < dist2:
                 extended_lines.append((x1, y1, bp[0], bp[1]))
             else:
                 extended_lines.append((x2, y2, bp[0], bp[1]))
        elif len(unique_boundary_points) < 2 and len(segments) > 1:
             # If less than 2 boundary points (e.g., segment is entirely inside or parallel to boundary)
             # Simple extension based on segment direction
             dx = x2 - x1
             dy = y2 - y1
             # Extend from endpoints far beyond image boundaries
             length = max(width, height) * 2 # Extend sufficiently
             p1_ext = (int(x1 - dx * length), int(y1 - dy * length))
             p2_ext = (int(x2 + dx * length), int(y2 + dy * length))
             extended_lines.append((p1_ext[0], p1_ext[1], p2_ext[0], p2_ext[1]))


    if extended_lines:
         # Use the general get_intersections function with the extended line points
         intersections = get_intersections_from_lines_cartesian(img_shape, extended_lines)
         return intersections

    return np.array([], dtype=np.float32) # Return empty if no extended lines

def get_intersections_from_lines_cartesian(img_shape, lines_cartesian):
    """Calculate intersection points from a list of lines in Cartesian form (x1, y1, x2, y2)."""
    height, width = img_shape[:2]
    intersections = []
    if len(lines_cartesian) < 2:
        return np.array([], dtype=np.float32)

    for i, line1 in enumerate(lines_cartesian):
        x1, y1, x2, y2 = line1
        for j, line2 in enumerate(lines_cartesian):
            if j <= i:
                continue
            x3, y3, x4, y4 = line2

            # Calculate determinant
            denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)

            # If determinant is close to zero, lines are parallel
            if abs(denom) < 1e-6:
                continue

            # Calculate intersection point (px, py)
            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

            # Optional: Filter intersections outside image boundaries (or within a margin)
            # This helps in focusing on intersections relevant to the document.
            # Using a margin to include points slightly outside
            margin_x = width * 0.1
            margin_y = height * 0.1
            if -margin_x <= px <= width + margin_x and -margin_y <= py <= height + margin_y:
                 intersections.append([px, py])

    return np.array(intersections, dtype=np.float32)


def select_outermost_corners(points, k=4):
    """
    Selects the outermost k (default 4) corners from a set of points.
    Simple method based on distance from centroid.
    """
    if points is None or len(points) < k:
        return None

    # Find the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate distance from centroid to each point
    distances = np.linalg.norm(points - centroid, axis=1)

    # Select the indices of the k points with the largest distances
    outermost_indices = np.argsort(distances)[-k:]

    return points[outermost_indices].astype(np.float32)

def select_best_corners(points, img_shape):
    """
    Selects the best 4 corners from a set of points, prioritizing points near image corners
    and forming a reasonable quadrilateral (convex, sufficient area, reasonable aspect ratio).
    """
    if points is None or len(points) < 4:
        return None

    h, w = img_shape[:2]
    image_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Strategy 1: Find 4 points closest to the image corners
    closest_to_image_corners = []
    # Ensure we don't select the same point multiple times
    selected_indices = set()
    for img_corner in image_corners:
        distances = np.linalg.norm(points - img_corner, axis=1)
        # Find the closest point that hasn't been selected yet
        sorted_indices = np.argsort(distances)
        for idx in sorted_indices:
            if idx not in selected_indices:
                closest_to_image_corners.append(points[idx])
                selected_indices.add(idx)
                break # Move to the next image corner

    potential_corners = np.array(closest_to_image_corners, dtype=np.float32)

    # Check if these 4 points form a reasonable quadrilateral
    if len(potential_corners) == 4:
        reordered_potential = order_points(potential_corners)
        # Check convexity, area, and aspect ratio
        try:
            # Convexity check (cross product of adjacent edge vectors should have the same sign)
            v1 = reordered_potential[1] - reordered_potential[0]
            v2 = reordered_potential[2] - reordered_potential[1]
            v3 = reordered_potential[3] - reordered_potential[2]
            v4 = reordered_potential[0] - reordered_potential[3]

            cross_products = [
                np.cross(v1, v2),
                np.cross(v2, v3),
                np.cross(v3, v4),
                np.cross(v4, v1)
            ]

            signs = np.sign(cross_products)
            # Check if all signs are non-negative or all are non-positive (allowing for collinear points where cross product is 0)
            is_convex = (np.all(signs >= 0) or np.all(signs <= 0))

            # Area check
            area = cv2.contourArea(reordered_potential)
            img_area = h * w
            area_ratio = area / img_area if img_area > 0 else 0

            # Aspect ratio check
            side1_len = np.linalg.norm(reordered_potential[0] - reordered_potential[1])
            side2_len = np.linalg.norm(reordered_potential[1] - reordered_potential[2])
            # Avoid division by zero
            aspect_ratio = max(side1_len, side2_len) / (min(side1_len, side2_len) + 1e-6) if min(side1_len, side2_len) > 0 else float('inf')

            # Define reasonable thresholds (can be adjusted based on typical document shapes)
            min_area_ratio = 0.05  # Quadrilateral should cover at least 5% of image area
            max_aspect_ratio = 5.0 # Aspect ratio should not be extremely skewed

            if is_convex and area_ratio > min_area_ratio and aspect_ratio < max_aspect_ratio:
                # print("Selected corners are convex, have sufficient area, and reasonable aspect ratio.")
                return reordered_potential # Found good corners

        except Exception as e:
            # print(f"Error during convexity/area/aspect ratio check for image-corner-based selection: {e}")
            pass # Continue to fallback if check fails


    # Strategy 2: Fallback to selecting the outermost 4 points
    # print("Image-corner-based selection failed or produced poor results. Falling back to outermost points.")
    fallback_corners = select_outermost_corners(points, k=4)

    if fallback_corners is not None and len(fallback_corners) == 4:
        reordered_fallback = order_points(fallback_corners)
        # Re-check convexity, area, and aspect ratio for fallback corners
        try:
            v1 = reordered_fallback[1] - reordered_fallback[0]
            v2 = reordered_fallback[2] - reordered_fallback[1]
            v3 = reordered_fallback[3] - reordered_fallback[2]
            v4 = reordered_fallback[0] - reordered_fallback[3]

            cross_products = [
                np.cross(v1, v2),
                np.cross(v2, v3),
                np.cross(v3, v4),
                np.cross(v4, v1)
            ]

            signs = np.sign(cross_products)
            is_convex = (np.all(signs >= 0) or np.all(signs <= 0))

            area = cv2.contourArea(reordered_fallback)
            img_area = h * w
            area_ratio = area / img_area if img_area > 0 else 0

            side1_len = np.linalg.norm(reordered_fallback[0] - reordered_fallback[1])
            side2_len = np.linalg.norm(reordered_fallback[1] - reordered_fallback[2])
            aspect_ratio = max(side1_len, side2_len) / (min(side1_len, side2_len) + 1e-6) if min(side1_len, side2_len) > 0 else float('inf')

            min_area_ratio = 0.05
            max_aspect_ratio = 5.0

            if is_convex and area_ratio > min_area_ratio and aspect_ratio < max_aspect_ratio:
                # print("Fallback outermost corners are convex, have sufficient area, and reasonable aspect ratio.")
                return reordered_fallback # Use fallback if it looks reasonable
            else:
                 # print("Fallback outermost corners do not meet criteria.")
                 return None # Fallback also failed

        except Exception as e:
             # print(f"Error during fallback convexity/area/aspect ratio check: {e}")
             return None # Fallback check failed


    # If both methods fail, return None
    # print("Both initial and fallback corner selection methods failed.")
    return None


# --- Main Perspective Correction Logic ---

def correct_perspective(img_bgr,
                        preprocess_method='clahe', # 'clahe', 'none'
                        thresholding_method='adaptive', # 'adaptive', 'otsu', 'none'
                        edge_detection_method='canny', # 'canny', 'sobel', 'none'
                        line_detection_method='hough_probabilistic', # 'hough_probabilistic', 'hough_standard', 'none'
                        corner_detection_method='hough_intersections', # 'hough_intersections', 'contour', 'shi_tomasi', 'harris', 'none'
                        adaptive_block_size=11,
                        adaptive_c_value=2,
                        canny_threshold_min=30,
                        canny_threshold_max=140,
                        sobel_threshold_min=50,
                        sobel_threshold_max=200,
                        hough_rho=1,
                        hough_theta=np.pi/180,
                        hough_threshold_standard=150,
                        hough_threshold_probabilistic=50,
                        hough_minLineLength=50,
                        hough_maxLineGap=10,
                        contour_area_threshold_ratio=0.05,
                        shi_tomasi_maxCorners=100,
                        shi_tomasi_qualityLevel=0.01,
                        shi_tomasi_minDistance=10,
                        harris_blockSize=2,
                        harris_ksize=3,
                        harris_k=0.04,
                        harris_threshold=0.01):
    """
    Apply perspective correction to an image using a flexible combination of
    preprocessing and detection methods.

    Args:
        img_bgr (np.ndarray): The input image in BGR format.
        preprocess_method (str): Method for initial image enhancement ('clahe', 'none').
        thresholding_method (str): Method for thresholding ('adaptive', 'otsu', 'none').
        edge_detection_method (str): Method for edge detection ('canny', 'sobel', 'none').
        line_detection_method (str): Method for line detection ('hough_probabilistic', 'hough_standard', 'none').
        corner_detection_method (str): Method for corner detection ('hough_intersections', 'contour', 'shi_tomasi', 'harris', 'none').
        adaptive_block_size (int): Block size for adaptive thresholding.
        adaptive_c_value (int): C value for adaptive thresholding.
        canny_threshold_min (int): Minimum threshold for Canny.
        canny_threshold_max (int): Maximum threshold for Canny.
        sobel_threshold_min (int): Minimum threshold for Sobel.
        sobel_threshold_max (int): Maximum threshold for Sobel.
        hough_rho (float): Distance resolution for Hough transform.
        hough_theta (float): Angle resolution for Hough transform.
        hough_threshold_standard (int): Accumulator threshold for Standard Hough.
        hough_threshold_probabilistic (int): Accumulator threshold for Probabilistic Hough.
        hough_minLineLength (int): Minimum line length for Probabilistic Hough.
        hough_maxLineGap (int): Maximum gap between segments to link for Probabilistic Hough.
        contour_area_threshold_ratio (float): Minimum area ratio for contour detection.
        shi_tomasi_maxCorners (int): Maximum number of corners for Shi-Tomasi.
        shi_tomasi_qualityLevel (float): Minimum accepted quality of image corners for Shi-Tomasi.
        shi_tomasi_minDistance (int): Minimum possible Euclidean distance between corners for Shi-Tomasi.
        harris_blockSize (int): Neighborhood size for Harris detector.
        harris_ksize (int): Aperture parameter for Sobel operator for Harris.
        harris_k (float): Harris detector free parameter in the equation.
        harris_threshold (float): Threshold for Harris corner response.

    Returns:
        tuple: A tuple containing:
            - warped_img (np.ndarray): The perspective corrected image.
            - src_corners (np.ndarray): The source corners used for transformation.
            - intermediate_images (dict): Dictionary of intermediate processing steps.
        Returns (None, None, None) if corners cannot be detected.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input image is empty or None.")

    # Ensure image is in BGR format (convert grayscale or RGBA if needed)
    if img_bgr.ndim == 2:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[-1] == 4:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    else:
        img = img_bgr.copy() # Work on a copy

    intermediate_images = {}

    # 1. Preprocessing
    preprocessed = img.copy()
    if preprocess_method == 'clahe':
        preprocessed = preprocess_image_clahe(preprocessed)
    # Add other preprocessing methods here if needed (e.g., blurring, sharpening)
    intermediate_images['preprocessed'] = preprocessed.copy()

    # Convert to grayscale for detection methods
    gray = preprocess_image_grayscale(preprocessed)
    intermediate_images['gray'] = gray.copy()

    # 2. Thresholding (if selected)
    thresh = None
    if thresholding_method == 'adaptive':
        thresh = preprocess_image_adaptive_threshold(gray, adaptive_block_size, adaptive_c_value)
        intermediate_images['thresholded'] = thresh.copy()
    elif thresholding_method == 'otsu':
         _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
         intermediate_images['thresholded'] = thresh.copy()
    # If thresholding_method is 'none', use the grayscale image for edge detection/corner detection directly

    # Determine which image to use for edge/corner detection based on thresholding
    img_for_detection = thresh if thresh is not None else gray

    # 3. Edge Detection (if selected and needed for line detection)
    edges = None
    if edge_detection_method == 'canny':
        edges = preprocess_image_canny_edges(img_for_detection, canny_threshold_min, canny_threshold_max)
        intermediate_images['edges'] = edges.copy()
    elif edge_detection_method == 'sobel':
         # Apply Sobel in X and Y direction
         sobelx = cv2.Sobel(img_for_detection, cv2.CV_64F, 1, 0, ksize=5)
         sobely = cv2.Sobel(img_for_detection, cv2.CV_64F, 0, 1, ksize=5)
         # Compute the gradient magnitude
         edges = cv2.magnitude(sobelx, sobely)
         # Normalize to 8-bit
         max_edge_val = edges.max()
         if max_edge_val > 0:
            edges = np.uint8(edges * 255 / max_edge_val)
         else:
            edges = np.zeros_like(edges, dtype=np.uint8)
         # Apply threshold to get binary edges
         _, edges = cv2.threshold(edges, sobel_threshold_min, sobel_threshold_max, cv2.THRESH_BINARY)
         intermediate_images['edges'] = edges.copy()
    # If edge_detection_method is 'none', proceed to corner detection directly

    # 4. Corner Detection
    corners = None
    detection_output_img = img.copy() # Initialize image for visualizing detection results

    if corner_detection_method == 'hough_intersections':
        if edges is not None and line_detection_method != 'none':
            lines = None
            if line_detection_method == 'hough_standard':
                 lines = find_lines_hough(edges, hough_rho, hough_theta, hough_threshold_standard)
                 # Visualize standard Hough lines
                 if lines is not None and len(lines) > 0:
                      # Convert polar lines to cartesian points for drawing
                      h, w = img.shape[:2]
                      for rho, theta in lines:
                          a = np.cos(theta)
                          b = np.sin(theta)
                          x0 = a * rho
                          y0 = b * rho
                          x1 = int(x0 + 1000 * (-b))
                          y1 = int(y0 + 1000 * (a))
                          x2 = int(x0 - 1000 * (-b))
                          y2 = int(y0 - 1000 * (a))
                          cv2.line(detection_output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            elif line_detection_method == 'hough_probabilistic':
                 lines = find_lines_probabilistic_hough(edges, hough_rho, hough_theta, hough_threshold_probabilistic, hough_minLineLength, hough_maxLineGap)
                 # Visualize probabilistic Hough lines
                 if lines is not None and len(lines) > 0:
                      for x1, y1, x2, y2 in lines:
                          cv2.line(detection_output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)


            if lines is not None and len(lines) > 0:
                 # Get intersections from detected lines/segments
                 if line_detection_method == 'hough_standard':
                     intersections = get_intersections_from_lines(lines)
                 elif line_detection_method == 'hough_probabilistic':
                      # We need to convert segments to extended lines for intersections
                      # Re-using the get_intersections_from_segments logic which extends lines
                      intersections = get_intersections_from_segments(lines, img.shape)
                 else:
                      intersections = np.array([], dtype=np.float32) # Should not happen if line_detection_method is 'none'

                 if intersections is not None and len(intersections) >= 4:
                     corners = select_best_corners(intersections, img.shape)
                     # Visualize detected corner points
                     if corners is not None:
                         for corner in corners:
                              x, y = corner.ravel()
                              cv2.circle(detection_output_img, (int(x), int(y)), 5, (255, 0, 0), -1)


    elif corner_detection_method == 'contour':
        # Contour detection works best on binary images (thresholded)
        if img_for_detection is not None:
            document_contours = find_document_contours(img_for_detection, area_threshold_ratio=contour_area_threshold_ratio)
            if document_contours:
                # Assuming the largest contour is the document
                potential_corners = get_corners_from_contours([document_contours[0]])
                if potential_corners is not None and len(potential_corners) >= 4:
                    corners = select_best_corners(potential_corners, img.shape)
                    # Visualize the detected contour and corners
                    if document_contours:
                         cv2.drawContours(detection_output_img, [document_contours[0]], -1, (0, 255, 0), 3)
                    if corners is not None:
                         for corner in corners:
                              x, y = corner.ravel()
                              cv2.circle(detection_output_img, (int(x), int(y)), 5, (255, 0, 0), -1)


    elif corner_detection_method == 'shi_tomasi':
        # Shi-Tomasi works directly on grayscale or preprocessed images
        corners = detect_corners_shi_tomasi(
            img_for_detection,
            maxCorners=shi_tomasi_maxCorners,
            qualityLevel=shi_tomasi_qualityLevel,
            minDistance=shi_tomasi_minDistance
        )
        if corners is not None and len(corners) >= 4:
             corners = select_best_corners(corners, img.shape)
             # Visualize detected corner points
             if corners is not None:
                 for corner in corners:
                     x, y = corner.ravel()
                     cv2.circle(detection_output_img, (int(x), int(y)), 5, (255, 0, 0), -1)


    elif corner_detection_method == 'harris':
         # Harris works directly on grayscale or preprocessed images
         corners = detect_corners_harris(
             img_for_detection,
             blockSize=harris_blockSize,
             ksize=harris_ksize,
             k=harris_k,
             threshold=harris_threshold
         )
         if corners is not None and len(corners) >= 4:
              corners = select_best_corners(corners, img.shape)
              # Visualize detected corner points
              if corners is not None:
                 for corner in corners:
                     x, y = corner.ravel()
                     cv2.circle(detection_output_img, (int(x), int(y)), 5, (255, 0, 0), -1)


    intermediate_images['detection_output'] = detection_output_img

    # 5. Apply Perspective Transform
    warped_img = None
    src_corners = None
    if corners is not None and len(corners) == 4:
        src_corners = order_points(corners) # Ensure consistent order
        warped_img, _ = four_point_transform(img, src_corners)
        intermediate_images['warped'] = warped_img.copy()
    else:
        # If 4 corners are not found, return None or the original image
        # raise ValueError("Could not detect 4 corners for perspective transformation.")
        print("Warning: Could not detect 4 distinct corners. Perspective transformation skipped.")
        return None, None, intermediate_images # Return None for warped image and corners

    return warped_img, src_corners, intermediate_images


def correct_perspective_auto(img_bgr):
    """
    Attempt perspective correction using a series of predefined parameter combinations.
    Tries combinations on both the original and negative images.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input image is empty or None.")

    # Ensure image is in BGR format
    if img_bgr.ndim == 2:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[-1] == 4:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    else:
        img = img_bgr.copy() # Work on a copy


    # Define a list of parameter dictionaries to try
    # These are example combinations, can be expanded or refined
    parameter_tries = [
        # Basic Hough + Canny
        {"preprocess_method": 'clahe', "thresholding_method": 'adaptive', "edge_detection_method": 'canny', "line_detection_method": 'hough_probabilistic', "corner_detection_method": 'hough_intersections'},
        {"preprocess_method": 'clahe', "thresholding_method": 'otsu', "edge_detection_method": 'canny', "line_detection_method": 'hough_probabilistic', "corner_detection_method": 'hough_intersections'},
        {"preprocess_method": 'none', "thresholding_method": 'adaptive', "edge_detection_method": 'canny', "line_detection_method": 'hough_probabilistic', "corner_detection_method": 'hough_intersections'},

        # Contour based
        {"preprocess_method": 'clahe', "thresholding_method": 'adaptive', "edge_detection_method": 'none', "line_detection_method": 'none', "corner_detection_method": 'contour'},
        {"preprocess_method": 'clahe', "thresholding_method": 'otsu', "edge_detection_method": 'none', "line_detection_method": 'none', "corner_detection_method": 'contour'},
        {"preprocess_method": 'none', "thresholding_method": 'adaptive', "edge_detection_method": 'none', "line_detection_method": 'none', "corner_detection_method": 'contour'},

        # Shi-Tomasi / Harris
        {"preprocess_method": 'clahe', "thresholding_method": 'none', "edge_detection_method": 'none', "line_detection_method": 'none', "corner_detection_method": 'shi_tomasi'},
        {"preprocess_method": 'none', "thresholding_method": 'none', "edge_detection_method": 'none', "line_detection_method": 'none', "corner_detection_method": 'shi_tomasi'},
         {"preprocess_method": 'clahe', "thresholding_method": 'none', "edge_detection_method": 'none', "line_detection_method": 'none', "corner_detection_method": 'harris'},
        {"preprocess_method": 'none', "thresholding_method": 'none', "edge_detection_method": 'none', "line_detection_method": 'none', "corner_detection_method': 'harris'},

        # Add more specific parameter variations if needed
        # Example: Different adaptive thresholding block sizes
        {"preprocess_method": 'clahe', "thresholding_method": 'adaptive', "adaptive_block_size": 21, "edge_detection_method": 'canny', "line_detection_method": 'hough_probabilistic', "corner_detection_method": 'hough_intersections'},

    ]

    print(f"Attempting perspective correction with {len(parameter_tries) * 2} parameter combinations (original + negative image)...")

    # Try on original image
    print("Trying on original image...")
    for i, params in enumerate(parameter_tries):
        try:
            print(f"  Try {i+1}/{len(parameter_tries)} (Original): {params}")
            # Pass all defined parameters, default values will be used if not in params dict
            warped_img, src_corners, intermediate_images = correct_perspective(img, **params)
            if warped_img is not None:
                print("  ✅ Success!")
                return warped_img, src_corners, intermediate_images # Return successful result
        except Exception as e:
            print(f"  ⛔ Try {i+1}/{len(parameter_tries)} (Original) failed: {e}")
            continue # Try next combination

    # If original image failed, try on negative image
    print("\nTrying on negative image...")
    img_neg = cv2.bitwise_not(img)
    for i, params in enumerate(parameter_tries):
        try:
            print(f"  Try {i+1}/{len(parameter_tries)} (Negative): {params}")
            warped_img, src_corners, intermediate_images = correct_perspective(img_neg, **params)
            if warped_img is not None:
                print("  ✅ Success!")
                # Return successful result (Note: warped_img is from the negative, you might want to re-warp the original with these corners)
                # For simplicity here, we return the warped negative image. A more robust approach
                # would be to re-calculate the transform matrix from original image corners if found
                # on negative and apply to original.
                 print("  Applying successful parameters to original image...")
                 # Re-run correct_perspective with original image and the successful parameters
                 warped_img_orig, src_corners_orig, intermediate_images_orig = correct_perspective(img, **params)
                 if warped_img_orig is not None:
                      return warped_img_orig, src_corners_orig, intermediate_images_orig
                 else:
                      # If re-applying to original fails (unlikely if corners were good), return the warped negative
                      print("  Re-applying to original image failed, returning warped negative.")
                      return warped_img, src_corners, intermediate_images # Return the warped negative image
        except Exception as e:
            print(f"  ⛔ Try {i+1}/{len(parameter_tries)} (Negative) failed: {e}")
            continue # Try next combination

    # If all attempts fail
    print("\n❌ All perspective correction attempts failed.")
    return None, None, {} # Return None if all attempts fail


# --- Example Usage (outside any class) ---

if __name__ == '__main__':
    # Example of how to use the functions
    # Load an example image (replace with your image path)
    # Make sure you have an image file accessible, e.g., in your Colab environment.
    # If using Google Drive, you'll need to mount it first.
    # from google.colab import drive
    # drive.mount('/content/drive')
    # image_path = '/content/drive/MyDrive/path/to/your/image.jpg' # Example Google Drive path
    # Or upload an image directly to Colab session storage
    image_path = 'test_image.jpg' # Example: Assuming test_image.jpg is uploaded to Colab


    # Create a dummy image for demonstration if a file is not found
    if not os.path.exists(image_path):
        print(f"Image file not found at {image_path}. Creating a dummy image for demonstration.")
        # Create a simple dummy image with a rectangle
        dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)
        # Draw a white rectangle that is slightly skewed
        pts = np.array([[50, 80], [550, 50], [580, 320], [80, 350]], dtype=np.int32)
        cv2.fillPoly(dummy_img, [pts], (255, 255, 255))
        img = dummy_img
    else:
        img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from {image_path}")
    else:
        print("Image loaded successfully.")
        # Attempt automatic perspective correction
        warped_img, src_corners, intermediate_images = correct_perspective_auto(img)

        if warped_img is not None:
            print("Perspective correction successful!")
            # You can display or save the warped image and intermediate steps
            # from google.colab.patches import cv2_imshow
            # cv2_imshow(warped_img)
            # cv2.imwrite('corrected_image.jpg', warped_img)

            # Optional: Display intermediate steps
            # print("\nIntermediate Images:")
            # for name, intermediate_img in intermediate_images.items():
            #      print(f"- {name}")
                 # cv2_imshow(intermediate_img)

        else:
            print("Perspective correction failed for all attempts.")