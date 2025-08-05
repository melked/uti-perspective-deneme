import os
import sys
import cv2
import numpy as np
from typing import Optional

# Adjusting sys.path without using __file__
# Assuming the script is run from a specific directory or that the SDKs path is known relative to the current working directory
# If the structure is fixed, a relative path from the current working directory could work.
# For Colab, you might need to mount Google Drive or adjust based on where the SDKs are placed.
# Replacing with a placeholder that might work in some Colab setups if the structure is relative to the notebook's location:
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../../")))

# Alternative if the path is fixed relative to Colab's default content directory:
# sys.path.append('/content/sdks/novavision/src') # Example, adjust based on actual path

# Safest approach might be to use an absolute path if known or prompt the user for the path to the SDKs.
# For now, commenting out the problematic line and assuming necessary modules are in the Python path or can be imported directly
# or that the user will handle the sys.path setup separately.
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../")) # Problematic line


from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel

# Placeholder imports - Replace with actual imports if sys.path issue is resolved
class Image:
    @staticmethod
    def get_frame(img, redis_db):
        # Placeholder: In a real scenario, this would load the image
        # For testing, we might simulate loading a dummy image or expect img to be the actual image data
        if img is None:
             return None # Or raise an error
        # Assuming img is already a numpy array for simplicity in this example
        # In a real use case, you'd retrieve it from Redis using img identifier
        class MockImageObj:
            def __init__(self, value):
                self.value = value
        return MockImageObj(img)

    @staticmethod
    def set_frame(img, package_uID, redis_db):
         # Placeholder: In a real scenario, this would save the image to Redis
         # For testing, we might just return the img object or a representation
         return img # Or return an identifier/key


class Component:
    def __init__(self, request, bootstrap):
        self.request = request
        self.bootstrap_config = bootstrap
        self.redis_db = None # Placeholder
        self.uID = "mock_uid" # Placeholder

class Executor:
    def __init__(self, request_data):
        self.request_data = request_data # Assuming request_data is a dict/json string
        # In a real scenario, you'd parse request_data and initialize Component
        # For this example, we'll create a mock request object
        class MockRequest:
            def __init__(self, data):
                self.data = data
                self.model = None # Will be set by component
            def get_param(self, key):
                return self.data.get(key)

        self.request = MockRequest(eval(request_data)) # Using eval for simplicity, use json.loads in real app
        self.bootstrap = {} # Mock bootstrap config
        self.component = PerspectiveTransformation(self.request, self.bootstrap)

    def run(self):
        return self.component.run()


def build_response(context):
    # Placeholder for response building
    return {
        "status": "success",
        "context": context.context,
        "output_image": context.image # In a real app, this might be a key/identifier
    }

# Placeholder for PackageModel - Define if needed based on request.data structure
class PackageModel:
     def __init__(self, inputImage=None, **kwargs):
         self.inputImage = inputImage
         # Handle other potential parameters


# ========== Yardımcı Fonksiyonlar ==========

def _is_valid_quad(pts: np.ndarray, image_shape: tuple) -> bool:
    """Dörtgenin geçerli olup olmadığını kontrol eder."""
    if pts is None or len(pts) != 4:
        return False
    h, w = image_shape[:2]
    for x, y in pts:
        if x < 0 or y < 0 or x > w or y > h:
            return False
    area = cv2.contourArea(pts.astype(np.float32))
    # Adjusted minimum area check - can be tuned
    if area < (w * h * 0.01):  # reduced minimum area to 1%
        return False
    return True


def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(round(max(widthA, widthB)))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(round(max(heightA, heightB)))

    # Destination points for the warped image
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Apply the perspective transformation
    # Using INTER_LANCZOS4 for better quality potentially
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped


def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _find_quad_from_contours(binary_img: np.ndarray, ref_image: np.ndarray, min_area_ratio=0.05) -> np.ndarray:
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _full_image_quad(ref_image)

    # Sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = ref_image.shape[0] * ref_image.shape[1]
    min_area = img_area * min_area_ratio

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # Check if the approximation has 4 points and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    # If no suitable contour found, return the full image corners
    return _full_image_quad(ref_image)


def _unsharp_mask(image, ksize=(5, 5), strength=1.5):
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)


def _gamma_correction(image: np.ndarray, gamma=1.0) -> np.ndarray:
    # Ensure image is float for calculation, then convert back to uint8
    # Also handle color channels if necessary
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    invGamma = 1.0 / gamma
    corrected_image = np.power(image, invGamma)

    # Convert back to uint8 (0-255)
    corrected_image = np.clip(corrected_image * 255.0, 0, 255).astype(np.uint8)
    return corrected_image


def _auto_gamma_correction(image: np.ndarray) -> np.ndarray:
    # Apply gamma correction based on average pixel intensity
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    # Adjust gamma based on brightness
    if mean < 80:
        gamma = 1.8 # Darker images need more correction
    elif mean > 180:
        gamma = 0.6 # Brighter images need less correction
    else:
        gamma = 1.0 # Normal brightness
    return _gamma_correction(image, gamma)


def _mask_background_lab_range(image: np.ndarray) -> np.ndarray:
    # Attempt to create a mask based on LAB color space ranges often associated with backgrounds
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Define ranges for typical background colors (adjust as needed)
    # These ranges are examples and might need tuning based on specific images
    mask_a = cv2.inRange(A, 110, 140) # Example range for A channel
    mask_b = cv2.inRange(B, 110, 140) # Example range for B channel

    # Combine masks
    color_mask = cv2.bitwise_or(mask_a, mask_b)

    # Further refine using adaptive thresholding on L channel
    L_blur = cv2.GaussianBlur(L, (5, 5), 0)
    light_mask = cv2.adaptiveThreshold(
        L_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 5
    )

    # Combine light mask with color mask (inverting color_mask to find non-background)
    combined = cv2.bitwise_and(light_mask, cv2.bitwise_not(color_mask))


    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    return combined

def _auto_detect_document_corners_foreground_priority(image: np.ndarray) -> np.ndarray:
    # Use GrabCut-based masking to prioritize foreground objects
    mask = _detect_foreground_object_mask(image)
    return _find_quad_from_contours(mask, image, min_area_ratio=0.02) # Reduced min area for foreground


def _detect_foreground_object_mask(image: np.ndarray) -> np.ndarray:
    """Creates a mask to separate foreground object from complex background using GrabCut."""
    # Convert to HSV and use saturation channel
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    # Threshold on saturation to find potentially colored foreground
    _, sat_mask = cv2.threshold(s_channel, 40, 255, cv2.THRESH_BINARY)

    # Use Canny edge detection
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    # Combine saturation and edge masks
    combined_mask = cv2.bitwise_or(sat_mask, edges)

    # Apply morphological closing to connect nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Prepare mask for GrabCut
    # Initialize with probable foreground (based on combined mask) and probable background (elsewhere)
    mask_gc = np.zeros(image.shape[:2], np.uint8)
    mask_gc[combined_mask > 0] = cv2.GC_PR_FGD
    # Use GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # Rect is None as we are using an initial mask
    cv2.grabCut(image, mask_gc, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # Create final mask based on GrabCut results (sure foreground and probable foreground)
    final_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    return final_mask


def _auto_detect_document_corners_sharpen_adaptive(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply bilateral filter for noise reduction while preserving edges
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # Apply unsharp mask for sharpening
    sharpened = _unsharp_mask(blur)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    # Morphological operations to clean up the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return _find_quad_from_contours(morph, image)


def _auto_detect_document_corners_clahe_canny(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    # Apply Canny edge detection
    edges = cv2.Canny(clahe_img, 50, 150)
    # Apply morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return _find_quad_from_contours(morph, image)


def _auto_detect_document_corners_bright_blur(image: np.ndarray) -> np.ndarray:
    # Apply gamma correction suitable for bright images
    gamma_corrected = _gamma_correction(image, gamma=0.6) # Using 0.6 for brighter images as per original logic

    gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Apply unsharp mask
    sharp = _unsharp_mask(clahe_img, ksize=(5, 5), strength=1.5)

    # Apply Canny edge detection
    edges = cv2.Canny(sharp, 30, 120)

    # Apply morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    pts = _find_quad_from_contours(closed, image)
    return pts


def _mask_background_complex(image: np.ndarray) -> np.ndarray:
    # More complex background masking using LAB and thresholding
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Thresholding on A and B channels to isolate certain color ranges
    _, mask_a = cv2.threshold(A, 135, 255, cv2.THRESH_BINARY_INV)
    _, mask_b = cv2.threshold(B, 135, 255, cv2.THRESH_BINARY)

    # Combine color masks
    color_mask = cv2.bitwise_and(mask_a, mask_b)

    # Adaptive thresholding on L channel for lightness variation
    L_blur = cv2.GaussianBlur(L, (5, 5), 0)
    light_mask = cv2.adaptiveThreshold(
        L_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 5
    )

    # Edge detection on L channel
    edge_mask = cv2.Canny(L_blur, 40, 120)

    # Combine light, edge, and color masks
    combined = cv2.bitwise_or(light_mask, edge_mask)
    combined = cv2.bitwise_and(combined, color_mask) # Use color_mask directly

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    return combined


def _filter_lines_by_angle(lines, angle_tol=10):
    if lines is None:
        return None
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate angle in degrees, ensuring it's within 0-180 range
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle = angle % 180
        if angle > 90:
            angle = 180 - angle # Ensure angle is between 0 and 90

        # Check if angle is close to 0 or 90 degrees (horizontal or vertical)
        if (abs(angle - 0) < angle_tol) or (abs(angle - 90) < angle_tol):
            filtered.append(line)
    return np.array(filtered) if filtered else None


def _auto_detect_document_corners_hough_improved(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    # Filter lines to keep mostly horizontal and vertical ones
    lines = _filter_lines_by_angle(lines, angle_tol=15)
    if lines is None or len(lines) < 4:
        # If not enough lines, return full image quad
        return _full_image_quad(image)

    # Extract endpoints of filtered lines
    all_points = np.vstack([lines[:, 0, :2], lines[:, 0, 2:]])

    # Find the bounding box of these points
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)

    # Return the corners of the bounding box as the document quad
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)


def _texture_mask_gabor(image: np.ndarray, ksize=31, sigma=4.0, theta=np.pi/4, lambd=10.0, gamma=0.5) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gabor filter to highlight textures
    g_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
    # Threshold the filtered image
    _, mask = cv2.threshold(filtered, 50, 255, cv2.THRESH_BINARY)
    return mask

def _auto_detect_document_corners_lab_adaptive(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Enhance contrast in L channel using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_L = clahe.apply(L)

    # Adaptive thresholding on enhanced L channel
    thresh_L = cv2.adaptiveThreshold(
        clahe_L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 2
    )

    # Combine with edge detection on the original image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    combined = cv2.bitwise_or(thresh_L, edges)

    # Morphological operations to clean up the combined mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    return _find_quad_from_contours(morph, image)


def _auto_detect_document_corners_color_segmentation(image: np.ndarray) -> np.ndarray:
    # Simple color segmentation (example: looking for a dominant color range)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Example: white/light colors - adjust range based on expected document color
    # These ranges are for isolating light areas which are often documents
    lower_light = np.array([0, 0, 180])
    upper_light = np.array([180, 30, 255])
    mask_light = cv2.inRange(hsv, lower_light, upper_light)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morph = cv2.morphologyEx(mask_light, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    return _find_quad_from_contours(morph, image, min_area_ratio=0.02) # Reduced min area


def auto_detect_document_corners_dynamic(image: np.ndarray) -> np.ndarray:
    # Apply auto gamma correction first
    corrected = _auto_gamma_correction(image)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    contrast = gray.max() - gray.min()
    brightness = np.mean(gray)

    pts = None
    full_quad = _full_image_quad(corrected)

    # 1) Try Foreground Priority (GrabCut based) - good for complex backgrounds
    pts = _auto_detect_document_corners_foreground_priority(corrected)
    if not np.allclose(pts, full_quad, atol=1):
        return pts

    # 2) Try LAB space masking - good for separating certain color ranges
    mask = _mask_background_lab_range(corrected)
    pts = _find_quad_from_contours(mask, corrected)
    if not np.allclose(pts, full_quad, atol=1):
        return pts

    # 3) Try Color Segmentation - simple but effective for distinct document colors
    pts = _auto_detect_document_corners_color_segmentation(corrected)
    if not np.allclose(pts, full_quad, atol=1):
         return pts

    # 4) Try LAB adaptive approach - combines LAB and adaptive thresholding
    pts = _auto_detect_document_corners_lab_adaptive(corrected)
    if not np.allclose(pts, full_quad, atol=1):
        return pts


    # 5) Fallback methods based on image characteristics (contrast/brightness)

    # If contrast is low, try sharpening and adaptive thresholding
    if contrast < 40:
        pts = _auto_detect_document_corners_sharpen_adaptive(corrected)
        if not np.allclose(pts, full_quad, atol=1):
            return pts

    # If image is bright, try methods optimized for bright images
    if brightness > 200:
        pts = _auto_detect_document_corners_bright_blur(corrected)
        if not np.allclose(pts, full_quad, atol=1):
            return pts
        pts = _auto_detect_document_corners_clahe_canny(corrected)
        if not np.allclose(pts, full_quad, atol=1):
            return pts
        mask = _mask_background_complex(corrected)
        pts = _find_quad_from_contours(mask, corrected)
        if not np.allclose(pts, full_quad, atol=1):
            return pts
        # If still no good quad, try Hough line detection for straight edges
        pts = _auto_detect_document_corners_hough_improved(corrected)
        if not np.allclose(pts, full_quad, atol=1):
            return pts


    # General fallback sequence if previous methods fail
    pts = _auto_detect_document_corners_clahe_canny(corrected)
    if np.allclose(pts, full_quad, atol=1):
        mask = _mask_background_complex(corrected)
        pts = _find_quad_from_contours(mask, corrected)
        if np.allclose(pts, full_quad, atol=1):
             pts = _auto_detect_document_corners_hough_improved(corrected)
             if np.allclose(pts, full_quad, atol=1):
                 # Final fallback, consider increasing min_area_ratio for noisy images
                 pts = _find_quad_from_contours(_mask_background_lab_range(corrected), corrected, min_area_ratio=0.1)

    return pts


class PerspectiveTransformation(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        # Ensure request.data is a dictionary
        if isinstance(self.request.data, str):
             try:
                 self.request.data = json.loads(self.request.data) # Use json.loads
             except json.JSONDecodeError:
                 raise ValueError("Invalid JSON data in request.")

        # Ensure PackageModel exists and is initialized correctly
        try:
            self.request.model = PackageModel(**(self.request.data))
        except TypeError as e:
             raise TypeError(f"Failed to initialize PackageModel. Check data structure: {e}")

        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        # Bootstrap logic if needed
        return {}

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None.")
        # Convert image data type if necessary and normalize
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Convert grayscale to BGR if needed
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Convert BGRA to BGR if needed
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        # Retrieve image from Redis or other source
        # Using placeholder Image class
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db) # Assuming self.image is an identifier or the image itself for the placeholder
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")

        src_img = self._prepare_image(img_obj.value)

        # Auto-detect document corners
        pts = auto_detect_document_corners_dynamic(src_img)

        # Perform perspective transformation
        warped = _four_point_transform(src_img, pts)

        # Update the image object with the warped image
        img_obj.value = warped
        # Save the transformed image (placeholder)
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        # Update context with results
        self.context["src_quad"] = pts.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        # Build and return the response
        return build_response(context=self)

# Example of how the Executor might be used (needs actual request data)
# Executor('{"inputImage": "image_identifier_or_data"}').run()

# The Executor call at the end of the cell depends on how the component system works.
# If it's meant to be triggered with actual request data via sys.argv[1],
# ensure sys.argv[1] contains a valid JSON string representing the request.
# If running manually for testing, you'd need to provide a mock request.

# For this notebook environment, the Executor line will likely fail without a proper sys.argv[1].
# Commenting it out for now to avoid errors if running the cell directly without providing arguments.
Executor(sys.argv[1]).run()