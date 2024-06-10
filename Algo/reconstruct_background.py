import cv2
import numpy as np

def overlay_images():
    foreground_path = input("Enter the path for the foreground image: ").strip()
    background_path = input("Enter the path for the background image: ").strip()
    output_path = 'result.jpg' 
    
    try:
        foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED) 
        background = cv2.imread(background_path)

        if foreground is None:
            print("Error: Failed to load foreground image from the specified path.")
            return
        if background is None:
            print("Error: Failed to load background image from the specified path.")
            return

        background_resized = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
        alpha_channel = foreground[:,:,3] / 255.0
        foreground_rgb = foreground[:,:,:3]
        blended = np.zeros_like(background_resized, dtype=np.float32)
        for c in range(3):
            blended[:,:,c] = alpha_channel * foreground_rgb[:,:,c] + (1 - alpha_channel) * background_resized[:,:,c]

        blended = np.clip(blended, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, blended)

        print("Overlaying images completed. Result saved as 'result.jpg'.")
    except Exception as e:
        print("An error occurred:", e)

def main():
    overlay_images()

if __name__ == "__main__":
    main()
