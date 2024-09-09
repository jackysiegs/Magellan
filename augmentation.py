import cv2
import numpy as np
import os
import random
import pandas as pd

# Create or open the augmented global metadata file
augmented_metadata_file = "data/images/augmented_global_metadata.csv"

# Augmentation Functions
def random_rotation(image, angle_range=(-10, 10)):
    angle = random.uniform(angle_range[0], angle_range[1])
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (w, h))

def random_flip(image):
    return cv2.flip(image, random.choice([-1, 0, 1]))  # Random flip

def random_brightness(image, brightness_range=(0.7, 1.3)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(brightness_range[0], brightness_range[1]), 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_crop(image, crop_size=(640, 640)):
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    start_x = random.randint(0, w - crop_w)
    start_y = random.randint(0, h - crop_h)
    return image[start_y:start_y + crop_h, start_x:start_x + crop_w]

# Function to save augmented images and update augmented global metadata
def augment_images(city_metadata_file, city_folder):
    # Read the city-specific metadata.csv
    metadata = pd.read_csv(city_metadata_file, header=None, names=['ID', 'Latitude', 'Longitude', 'State', 'City', 'File Path'])
    
    augmented_metadata = []
    for idx, row in metadata.iterrows():
        img_path = row['File Path'].strip()  # Strip any extra whitespace/newline characters
        img_id = row['ID']
        state = row['State']
        city = row['City']
        lat = row['Latitude']
        lon = row['Longitude']

        # Debug: Print the image path being processed
        print(f"Processing file: {img_path}")

        # Load the original image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue

        # Apply augmentations
        augmented_img = random_rotation(img)
        augmented_img = random_flip(augmented_img)
        augmented_img = random_brightness(augmented_img)
        augmented_img = random_crop(augmented_img)

        # Save the augmented image
        new_image_name = f"augmented_street_view_{img_id}.jpg"
        new_image_path = os.path.join(city_folder, new_image_name)
        print(f"Saving augmented image to: {new_image_path}")
        cv2.imwrite(new_image_path, augmented_img)

        # Append to augmented metadata
        augmented_metadata.append([f"{img_id}_aug", lat, lon, state, city, new_image_path])

    # Convert augmented metadata to DataFrame
    augmented_metadata_df = pd.DataFrame(augmented_metadata, columns=['ID', 'Latitude', 'Longitude', 'State', 'City', 'File Path'])

    # Append augmented metadata to the new augmented metadata file
    augmented_metadata_df.to_csv(augmented_metadata_file, mode='a', header=False, index=False)
    print("Augmented images and metadata added to augmented_global_metadata.csv")

# Traverse through the Data/Images folder and process each city's metadata.csv
base_folder = "data/images"  # Path to the root directory containing state and city folders

for state_folder in os.listdir(base_folder):
    state_folder_path = os.path.join(base_folder, state_folder)
    if os.path.isdir(state_folder_path):
        for city_folder in os.listdir(state_folder_path):
            city_folder_path = os.path.join(state_folder_path, city_folder)
            if os.path.isdir(city_folder_path):
                # Check for the existence of metadata.csv in the city folder
                city_metadata_file = os.path.join(city_folder_path, 'metadata.csv')
                if os.path.exists(city_metadata_file):
                    print(f"Processing city: {city_folder}")
                    augment_images(city_metadata_file, city_folder_path)
                else:
                    print(f"No metadata found for {city_folder}")
