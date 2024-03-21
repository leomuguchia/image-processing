import cv2
import numpy as np

def estimate_depth(image):
    # Compute depth map (distance from camera)
    y, x = np.indices(image[:, :, 0].shape)
    depth_map = np.sqrt((x - image.shape[1] / 2) ** 2 + (y - image.shape[0] / 2) ** 2)

    # Normalize depth map to [0, 255] range for visualization
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    return depth_map.astype(np.float32)

def analyze_texture(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define parameters for Gabor filters
    ksize = 15  # Kernel size
    theta = np.pi / 4  # Orientation
    sigma = 3  # Standard deviation
    lambd = 8  # Wavelength
    gamma = 0.5  # Aspect ratio

    # Create Gabor filter bank
    gabor_kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        gabor_kernels.append(cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F))

    # Apply Gabor filters to the image
    texture_features = [cv2.filter2D(gray_image, cv2.CV_8UC3, kernel) for kernel in gabor_kernels]

    return texture_features

def detect_edges(image, low_threshold=30, high_threshold=150, blur_kernel_size=(5, 5)):
    # Preprocess image with Gaussian blur
    blurred = cv2.GaussianBlur(image, blur_kernel_size, 0)
    # Convert image to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def analyze_color(image):
    # Convert the image to the LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Compute mean and standard deviation of each channel
    mean_lab = cv2.mean(lab_image)
    std_lab = cv2.meanStdDev(lab_image)

    # Compute dominant color using K-means clustering
    pixels = lab_image.reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color_lab = np.uint8(centers[0])

    # Convert dominant color from LAB to BGR for display
    dominant_color_bgr = cv2.cvtColor(np.array([[dominant_color_lab]], dtype=np.uint8), cv2.COLOR_LAB2BGR)[0][0]

    # Compute color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist /= hist.sum()

    return mean_lab, std_lab, dominant_color_bgr, hist



def overlay_images(background, foreground, color_info):
    try:
        # Unpack color information
        _, _, dominant_color_bgr, _ = color_info

        # Resize background to match foreground dimensions
        background_resized = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))

        # Extract alpha channel from foreground and normalize it
        alpha_channel = foreground[:, :, 3] / 255.0

        # Remove alpha channel from foreground
        foreground_rgb = foreground[:, :, :3]

        # Adjust color of the foreground based on background color
        foreground_adjusted = np.zeros_like(foreground_rgb, dtype=np.uint8)
        for c in range(3):
            foreground_adjusted[:,:,c] = np.clip(foreground_rgb[:,:,c] + (dominant_color_bgr[c] - np.mean(foreground_rgb[:,:,c])) * 0.5, 0, 255)

        # Perform alpha blending manually
        blended = np.zeros_like(background_resized, dtype=np.float32)
        for c in range(3):
            blended[:,:,c] = alpha_channel * foreground_adjusted[:,:,c] + (1 - alpha_channel) * background_resized[:,:,c]

        blended_image = blended.astype(np.uint8)

        return blended_image

    except Exception as e:
        print(f"Error in overlaying images: {str(e)}")


           
def main():
    # Load the original image
    original_path = "sample_images/originalPerson.jpg"
    original_image = cv2.imread(original_path)

    # Load the foreground image
    foreground_path = "sample_images/NBGperson.png"
    foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)

    # Load the new background image
    background_path = "sample_images/background.jpeg"
    background = cv2.imread(background_path)

    if original_image is None or foreground is None or background is None:
        print("Error: Failed to load images.")
        return

    # Estimate the depth map
    depth_map = estimate_depth(original_image)

    # Analyze texture
    texture_features = analyze_texture(background)

    # Detect edges
    edge_map = detect_edges(background)

    # Analyze color
    color_info = analyze_color(background)

    # Overlay images
    result_image = overlay_images(background, foreground, color_info)

    # Save the result to a file
    cv2.imwrite("result.jpg", result_image)

    # Provide feedback to the user
    print("Result image saved as 'result.jpg'")



if __name__ == "__main__":
    main()
