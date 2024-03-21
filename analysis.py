import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_texture(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define parameters for Gabor filters
    ksize = 31  # Kernel size
    theta = np.pi / 4  # Orientation
    sigma = 3  # Standard deviation
    lambd = 8  # Wavelength
    gamma = 0.5  # Aspect ratio

    # Create Gabor filter bank
    gabor_kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        gabor_kernels.append(cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F))

    # Apply Gabor filters to the image
    texture_features = []
    for kernel in gabor_kernels:
        filtered_image = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)
        texture_features.append(filtered_image)

    return texture_features

def detect_edges(image, low_threshold=30, high_threshold=150, blur_kernel_size=(5, 5)):
    # Preprocess image with Gaussian blur
    blurred = cv2.GaussianBlur(image, blur_kernel_size, 0)
    # Convert image to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def estimate_depth(image):
     # Compute depth map (distance from camera)
    depth_map = np.zeros_like(image[:, :, 0], dtype=np.float32)
    for y in range(depth_map.shape[0]):
        for x in range(depth_map.shape[1]):
            # Compute depth based on pixel position (you may need to adjust these calculations)
            depth_map[y, x] = np.sqrt((x - image.shape[1] / 2)**2 + (y - image.shape[0] / 2)**2)

    # Normalize depth map to [0, 255] range for visualization
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    return depth_map

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

def overlay_images(background, foreground, texture_features, edge_map, depth_map, color_info, output_path='result.jpg'):
    try:
        # Extract relevant information from color analysis
        mean_lab, std_lab, dominant_color_bgr, hist = color_info

        # Normalize depth map to [0, 1] range
        normalized_depth = depth_map / 255.0

        # Ensure texture features have the same shape as the background
        texture_features_resized = []
        for feature in texture_features:
            resized_feature = cv2.resize(feature, (background.shape[1], background.shape[0]))
            texture_features_resized.append(resized_feature)
            print("Resized Texture Feature Shape:", resized_feature.shape)

       # Blend texture features based on depth map
        blended_texture = np.zeros_like(background, dtype=np.float32)
        for c in range(3):
            # Expand dimensions of normalized_depth to match the shape of texture_features_resized[c]
            normalized_depth_expanded = normalized_depth[:, :, np.newaxis]  # Shape: (295, 236, 1)
            # Expand dimensions of texture_features_resized to match the shape of normalized_depth
            texture_feature_expanded = texture_features_resized[c][:, :, np.newaxis]  # Shape: (295, 236, 1)
            # Blend texture feature and background based on depth map
            blended_texture[:, :, c] = (1 - normalized_depth_expanded) * background[:, :, c] + normalized_depth_expanded * texture_feature_expanded

        print("Blended Texture Shape:", blended_texture.shape)


        # Apply edge map to the blended texture
        blended_edges = np.where(edge_map[:, :, None] > 0, blended_texture, background)

        # Convert blended_edges to LAB color space for color adjustment
        blended_color = cv2.cvtColor(blended_edges, cv2.COLOR_BGR2LAB)

        # Adjust LAB channels based on color analysis
        for i in range(3):
            blended_color[:, :, i] = (blended_color[:, :, i] - np.mean(blended_color[:, :, i])) * (std_lab[i] / std_lab[i]) + mean_lab[i]

        # Convert blended_color back to BGR color space
        blended_color = cv2.cvtColor(blended_color, cv2.COLOR_LAB2BGR)

        # Apply dominant color adjustment
        mean_dominant_color = np.mean(dominant_color_bgr)
        blended_color += (dominant_color_bgr - mean_dominant_color)

        # Overlay foreground on top of the blended image
        if foreground.shape[2] == 4:  # Check if foreground has an alpha channel
            alpha = foreground[:, :, 3] / 255.0
            blended_final = np.uint8(alpha[:, :, None] * foreground[:, :, :3] + (1 - alpha[:, :, None]) * blended_color)
        else:
            # If foreground doesn't have an alpha channel, simply overlay it on top of the blended image
            blended_final = blended_color.copy()
            foreground_resized = cv2.resize(foreground, (background.shape[1], background.shape[0]))
            blended_final = cv2.addWeighted(blended_final, 1, foreground_resized, 1, 0)

        # Save output image
        cv2.imwrite(output_path, blended_final)

        print("Overlaying images completed. Result saved as 'result.jpg'.")
    except Exception as e:
        print("An error occurred:", e)
  
        
def main():
    # Load original image
    original_path = "sample_images/originalPerson.jpg"
    original_image = cv2.imread(original_path)
    
    # Load the background image
    background_path = "sample_images/personNEWBG.jpg"
    background = cv2.imread(background_path)

    # Load the foreground image
    foreground_path = "sample_images/NBGperson.png"  
    foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)

    if background is None:
        print("Error: Failed to load background image.")
        return

    if foreground is None:
        print("Error: Failed to load foreground image.")
        return
    
    # Estimate the depth map
    depth_map = estimate_depth(original_image)

    # Analyze texture
    texture_features = analyze_texture(background)

    # Detect edges
    edge_map = detect_edges(background)

    # Estimate depth
    depth_map = estimate_depth(background)

    # Analyze color
    color_info = analyze_color(background)

    # Overlay images
    overlay_images(background, foreground, texture_features, edge_map, depth_map, color_info)

if __name__ == "__main__":
    main()
