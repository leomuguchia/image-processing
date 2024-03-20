import cv2

def denoise_image(image):
    # Split the image into color channels
    b, g, r = cv2.split(image)
    
    # Apply Non-Local Means Denoising to each color channel
    b_denoised = cv2.fastNlMeansDenoising(b, None, h=10, templateWindowSize=7, searchWindowSize=21)
    g_denoised = cv2.fastNlMeansDenoising(g, None, h=10, templateWindowSize=7, searchWindowSize=21)
    r_denoised = cv2.fastNlMeansDenoising(r, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Merge the denoised color channels
    denoised = cv2.merge((b_denoised, g_denoised, r_denoised))
    
    return denoised

def multi_pass_resize(image, scale_factor, passes=2):
    resized_image = image.copy()
    for _ in range(passes):
        resized_image = cv2.resize(resized_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    return resized_image

def apply_advanced_interpolation(image):
    # Apply advanced interpolation (e.g., Lanczos)
    resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    return resized_image

def preserve_edges(image):
    # Apply edge preservation
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

def resize_image(image_path):
    try:
        # Load the input image
        original_image = cv2.imread(image_path)
        
        # Apply denoising
        denoised_image = denoise_image(original_image)

        # Multi-pass resizing
        resized_image = multi_pass_resize(denoised_image, 2)

        # Apply advanced interpolation
        # resized_image = apply_advanced_interpolation(resized_image)

        # Apply edge preservation
        resized_image = preserve_edges(resized_image)

        # Display the original and resized images
        cv2.imshow('Original Image', original_image)
        cv2.imshow('Resized Image', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("An error occurred:", e)

# Path to the input image
image_path = 'sample_images/goku.jpg'  # Path to the provided image 'person.png'

# Resize the image using advanced techniques
resize_image(image_path)
