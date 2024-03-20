import cv2
from PIL import Image

def extract_foreground(image_path):
    print("Image Path:", image_path)  # Print the image path for debugging purposes

    # Attempt to open the image file using PIL
    try:
        with Image.open(image_path) as img:
            print("Image opened successfully using PIL.")
    except Exception as e:
        print("Error opening the image using PIL:", e)
        return None

    # Read the input image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Failed to read the image file.")
        return None

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment the foreground
    _, binary_mask = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert the binary mask
    inverted_mask = cv2.bitwise_not(binary_mask)

    # Apply the mask to the original image to extract the foreground
    foreground = cv2.bitwise_and(image, image, mask=inverted_mask)

    return foreground

# Path to the input image
image_path = 'sample_images/original.jpg'

# Extract the foreground from the input image
foreground_image = extract_foreground(image_path)

if foreground_image is not None:
    # Display the extracted foreground
    cv2.imshow('Foreground Image', foreground_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Foreground extraction failed.")
