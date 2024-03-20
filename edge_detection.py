import cv2

def detect_edges(image, low_threshold=30, high_threshold=150, blur_kernel_size=(5, 5)):
    # Preprocess image with Gaussian blur
    blurred = cv2.GaussianBlur(image, blur_kernel_size, 0)
    # Convert image to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def main():
    # Load foreground image
    foreground_path = 'sample_images/input.jpg'
    foreground = cv2.imread(foreground_path)

    # Detect edges of the foreground object
    foreground_edges = detect_edges(foreground)

    # Display the edge-detected image
    cv2.imshow('Foreground Edges', foreground_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
