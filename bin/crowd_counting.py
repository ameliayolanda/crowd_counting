# Universitat de les Illes Balears
# Intelligent Systems
# 11761 - Images and Video Analysis
# Project 1

# Yolanda, Amelia
# Alsatouf, Abdulrahman

# IMPORTS
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utilities as utl


# FUNCTIONALITY
def filter_white_regions(binary_image, min_region_area, max_region_area, min_aspect_ratio, max_aspect_ratio):
    try:
        # Apply morphological opening
        kernel = np.ones((5, 5), np.uint8)
        opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        # Apply morphological closing to connect small white regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

        # Find contours in the closed image
        contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask for the regions to keep
        keep_mask = np.zeros_like(closed_image)

        # Iterate through the contours and keep only the regions within the specified area range
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_region_area < area < max_region_area \
                    and min_aspect_ratio < (
                    cv2.boundingRect(contour)[2] / cv2.boundingRect(contour)[3]) < max_aspect_ratio:
                cv2.drawContours(keep_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Apply the keep mask to the binary image
        result_image = cv2.bitwise_and(binary_image, binary_image, mask=keep_mask)

        return result_image

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def count_persons(image, processed_image, remaining_image, labels, tol):
    try:
        # Find contours in the binary image
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Store the estimated number of persons
        num_estimated_persons = len(contours)

        # Define a tolerance value for bounding boxes
        tol = tol
        region_of_interest = image.copy()
        bboxes = np.zeros([len(contours), 4])
        # Draw bounding boxes
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            # Bounding boxes
            cv2.rectangle(region_of_interest, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Compute coordinates for bounding boxes with tolerance
            tolx, toly = int(tol * w), int(tol * h)
            # Bounding boxes with tolerance
            cv2.rectangle(region_of_interest, (x - tolx, y - toly), (x + w + tolx, y + h + toly), (255, 0, 0), 2)
            # Store coordinates of each bounding box
            bboxes[i, :] = [x - tolx, y - toly + 410, w + tolx, h + toly]

        num_persons = 0
        for bbox in bboxes:
            x, y, w, h = bbox
            for j in range(len(labels)):
                if x <= labels[j, 0] <= x + w and y <= labels[j, 1] <= y + h:
                    num_persons = num_persons + 1

        # Merge the region of interest with the remaining image
        output_image = utl.merge_images(remaining_image, region_of_interest)

        return num_estimated_persons, num_persons, output_image

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


# CROWD COUNTING ALGORITHM
def crowd_counting(image_path, images, labels, region_of_interest, tol, plotting=False):
    try:
        estimated_persons = []
        detected_persons = []

        for image_name in images:
            # Read the image
            image = cv2.imread(image_path + image_name)

            # Extract the coordinates of each annotated person
            annotated_persons = labels[labels['image'] == image_name][['x', 'y']].values

            # Convert BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Crop the image to avoid capturing background noise
            cropped_image, remaining_image = utl.crop_image(image_rgb, region_of_interest)

            # Convert the bgr image to grayscale
            grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            # Apply windowing
            windowed_image = utl.windowing(grayscale_image, 120, 109)

            # Apply Otsu's thresholding
            _, binary_image = cv2.threshold(windowed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply negative transformation
            negative_image = 255 - binary_image

            # Filter connected white regions
            filtered_image = filter_white_regions(negative_image,
                                                  min_region_area=100, max_region_area=10000,
                                                  min_aspect_ratio=0.3, max_aspect_ratio=5)

            # Count the number of persons on the image
            count_estimated, count_detected, output_image = count_persons(cropped_image,
                                                                          filtered_image,
                                                                          remaining_image,
                                                                          annotated_persons,
                                                                          tol)

            # Store the number of estimated and detected persons
            estimated_persons.append(count_estimated)
            detected_persons.append(count_detected)

            if plotting:
                # Plot the resulting image
                plt.imshow(output_image)
                # Extract coordinates of annotated persons for plotting
                x, y = zip(*annotated_persons)
                # Plot points on the image
                plt.scatter(x, y, color='yellow', marker='o', s=2)
                # Turn off axis labels
                plt.axis('off')
                # Display the image
                plt.show()

        return np.array(estimated_persons), np.array(detected_persons)

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


# EXAMPLE USAGE
image_path = '../data/'
images = np.array(['1660284000.jpg', '1660287600.jpg', '1660291200.jpg', '1660309200.jpg', '1660302000.jpg',
                   '1660294800.jpg', '1660298400.jpg', '1660320000.jpg', '1660316400.jpg', '1660305600.jpg'])
labels = pd.read_csv(filepath_or_buffer=image_path + 'labels.csv', delimiter=',')
annotated_persons = np.genfromtxt(fname=image_path + 'labels_aggregated.csv', delimiter=',')
annotated_persons = annotated_persons[1:11, 1][::-1]
region_of_interest = [0, 1080 - 670, 1920, 670]
tol = 0.15  # Tolerance rate for expanding bounding boxes
estimated_persons, detected_persons = crowd_counting(image_path=image_path,
                                                     images=images,
                                                     labels=labels,
                                                     region_of_interest=region_of_interest,
                                                     tol=tol,
                                                     plotting=True)


# PERFORMANCE EVALUATION
print(f'Number of annotated persons per image: {annotated_persons}')
print(f'Estimated number of persons per image (predicted positive): {estimated_persons}')
print(f'Mean Squared Error: {(1 / annotated_persons.shape[0]) * ((sum(annotated_persons - estimated_persons)) ** 2)}')
print('')
print(f'Detected number of persons per image (true positive): {detected_persons}')
print(f'Average accuracy per image: {round(np.mean(detected_persons / annotated_persons) * 100, 2)}%')
print('')
print(f'Correlation Coefficient (annotated vs. detected persons): '
      f'{round(np.corrcoef(annotated_persons, detected_persons)[0, 1], 4)}')
