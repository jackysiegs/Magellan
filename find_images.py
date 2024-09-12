import requests
from tqdm import tqdm
import os
import io 
from random import uniform
from PIL import Image
import csv
from csv import writer

# Configuration
output_folder = 'data/images/'  # Output folder for the images
global_metadata_path = 'data/images/global_metadata.csv'
image_count = 5 # Amount of images to pull
api_key = ''  # Your Google Street View API Key, replace YOUR_API_KEY_HERE with your actual key
url = 'https://maps.googleapis.com/maps/api/streetview'
geocode_url = 'https://maps.googleapis.com/maps/api/geocode/json'


def generate_random_coords():
    # Approximate bounds around Dallas, Texas
    lat_north = 33.00
    lat_south = 32.62
    lon_east = -96.46
    lon_west = -96.999

    return (uniform(lat_south, lat_north), uniform(lon_west, lon_east))

def check_image_validity(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')  # Convert to RGB to analyze colors
    pixels = list(image.getdata())
    grey_threshold = 10 # Define how close R, G, and B values must be to be considered grey
    grey_pixels = sum(abs(r - g) <= grey_threshold and abs(g - b) <= grey_threshold and abs(r - b) <= grey_threshold for r, g, b in pixels)
    # Return False if >50% of pixels are grey, indicating the image is invalid/not desired
    return grey_pixels / len(pixels) < 0.7

def get_next_image_number():
    """Get the next image number from the global metadata file."""
    if not os.path.exists(global_metadata_path):
        return 0  # Start from 0 if the global metadata file doesn't exist
    with open(global_metadata_path, mode='r', newline='') as file:
        last_line = None
        for last_line in csv.reader(file): pass
        if last_line and last_line[0].isdigit():
            return int(last_line[0]) + 1
    return 0

def reverse_geocode(coords):
    params = {'latlng': f'{coords[0]},{coords[1]}', 'key': api_key}
    response = requests.get(geocode_url, params=params).json()
    if response['status'] == 'OK':
        address_components = response['results'][0]['address_components']
        city = state = None
        for component in address_components:
            if 'locality' in component['types']:
                city = component['long_name']
            if 'administrative_area_level_1' in component['types']:
                state = component['short_name']
        return state, city
    else:
        return None, None, None, None

def append_to_csv(path, row):
    with open(path, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(row)

def main():
    os.makedirs(output_folder, exist_ok=True)
    next_image_number = get_next_image_number()

    if not os.path.exists(global_metadata_path):
        append_to_csv(global_metadata_path, ['Image Number', 'Latitude', 'Longitude', 'State', 'City', 'Image Path'])
    
    coord_output_file = open(os.path.join(output_folder, 'picture_coords.csv'), 'w', newline='')
    csv_writer = writer(coord_output_file)
    
    coords_list = []
    
    for _ in tqdm(range(image_count)):
        coords = generate_random_coords()
        state, city = reverse_geocode(coords)
        if state and city:  # Ensuring valid geocode response
            # Define folder path based on state and city
            folder_path = os.path.join(output_folder, state, city)
            os.makedirs(folder_path, exist_ok=True)
            
            # Define image and CSV file paths
            image_number = _
            image_path = os.path.join(folder_path, f'street_view_{next_image_number + _}.jpg')
            csv_path = os.path.join(folder_path, 'metadata.csv')

            if not os.path.exists(csv_path):
                append_to_csv(csv_path, ['Image Number', 'Latitude', 'Longitude', 'State', 'City', 'Image Path'])
            
            # Fetch and save the street view image
            params = {
                'key': api_key,
                'size': '640x640',
                'location': f'{coords[0]},{coords[1]}',
                'heading': str(uniform(0, 360)),
                'pitch': '20',
                'fov': '90'
            }
            response = requests.get(url, params=params)
            if response.ok and check_image_validity(response.content):
                with open(image_path, "wb") as file:
                    file.write(response.content)
                
                # Append metadata to CSV
                metadata_row = [next_image_number + _, coords[0], coords[1], state, city, image_path]
                append_to_csv(csv_path, metadata_row)
                append_to_csv(global_metadata_path, metadata_row)

if __name__ == '__main__':
    main()
