import numpy as np


def crop_image(image, crop_coordinates):
    try:
        # Extract the coordinates of the crop
        x, y, w, h = crop_coordinates

        # Crop the region of interest from the original image
        cropped_region = image[y:y + h, x:x + w]

        # Keep the remaining image
        remaining_image = image[0:0 + y, x:x + w]

        return cropped_region, remaining_image

    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None


def merge_images(image1, image2):
    try:
        # Resize images to ensure they have the same width
        width = max(image1.shape[1], image2.shape[1])

        # Merge images row-wise using np.vstack
        merged_image = np.vstack((image1, image2))

        return merged_image

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def windowing(image, window_center, window_width):
    try:
        # Calculate the window min and max values
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2

        # Clip pixel values to be within the specified window
        windowed_image = np.clip(image, window_min, window_max)

        # Normalize pixel values to the range [0, 255] for display
        windowed_image = ((windowed_image - window_min) / (window_max - window_min) * 255).astype(np.uint8)

        return windowed_image

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point2) - np.array(point1)) ** 2))
