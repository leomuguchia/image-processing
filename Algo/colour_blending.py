import cv2
import numpy as np

def analyze_background(background):
    lab_bg = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)

    l_channel_bg = lab_bg[:,:,0]
    brightness_bg = np.mean(l_channel_bg)
    contrast_bg = np.std(l_channel_bg)

    r_mean = np.mean(background[:,:,2])
    g_mean = np.mean(background[:,:,1])
    b_mean = np.mean(background[:,:,0])
    color_temperature_bg = (-0.2661239 * r_mean - 0.6780686 * g_mean + 0.9471741 * b_mean) * 1000

    return brightness_bg, contrast_bg, color_temperature_bg

def adjust_foreground(foreground, brightness_bg, contrast_bg, color_temperature_bg):
    try:
        foreground_rgb = foreground[:, :, :3]
        alpha_channel = foreground[:, :, 3]
        lab_fg = cv2.cvtColor(foreground_rgb, cv2.COLOR_BGR2LAB)
        mask = alpha_channel > 0
        lab_fg_adjusted = np.copy(lab_fg)
        lab_fg_adjusted[mask, 0] = np.clip((lab_fg[mask, 0] - np.mean(lab_fg[mask, 0])) * (contrast_bg / np.std(lab_fg[mask, 0])) + brightness_bg, 0, 255)

        foreground_adjusted_rgb = cv2.cvtColor(lab_fg_adjusted, cv2.COLOR_LAB2BGR)
        foreground_adjusted = np.dstack((foreground_adjusted_rgb, alpha_channel))

        return foreground_adjusted

    except Exception as e:
        print(f"Error in adjusting foreground: {str(e)}")
        return None


def overlay_images(background, foreground_adjusted):
    try:
        background_resized = cv2.resize(background, (foreground_adjusted.shape[1], foreground_adjusted.shape[0]))
        alpha_channel = foreground_adjusted[:, :, 3] / 255.0
        foreground_rgb = foreground_adjusted[:, :, :3]
        
        blended = np.zeros_like(background_resized, dtype=np.float32)
        for c in range(3):
            blended[:,:,c] = alpha_channel * foreground_rgb[:,:,c] + (1 - alpha_channel) * background_resized[:,:,c]

        blended_image = blended.astype(np.uint8)

        return blended_image

    except Exception as e:
        print(f"Error in overlaying images: {str(e)}")

def main():
    # foreground should have no bg
    # yet to implement bg removal so use remove.bg online
    # background is a background image
    foreground_path = "sample_images/woman.png"
    background_path = "sample_images/clearbg.jpeg"
    foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
    background = cv2.imread(background_path)

    if foreground is None or background is None:
        print("Error: Failed to load images.")
        return

    brightness_bg, contrast_bg, color_temperature_bg = analyze_background(background)
    foreground_adjusted = adjust_foreground(foreground, brightness_bg, contrast_bg, color_temperature_bg)
    result_image = overlay_images(background, foreground_adjusted)
    cv2.imwrite("result.jpg", result_image)

    print("Result image saved as 'result.jpg'")

if __name__ == "__main__":
    main()