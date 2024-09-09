import csv
import matplotlib
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_coordinates(lat, lon):
    # US latitude and longitude range
    lat_min, lat_max = 24.396308, 49.384358  # Latitude range for US
    lon_min, lon_max = -125.0, -66.93457     # Longitude range for US
    normalized_lat = 2 * (lat - lat_min) / (lat_max - lat_min) - 1
    normalized_lon = 2 * (lon - lon_min) / (lon_max - lon_min) - 1
    return normalized_lat, normalized_lon

def denormalize_coordinates(norm_lat, norm_lon):
    lat_min, lat_max = 24.396308, 49.384358  # Latitude range for US
    lon_min, lon_max = -125.0, -66.93457     # Longitude range for US
    lat = (norm_lat + 1) * (lat_max - lat_min) / 2 + lat_min
    lon = (norm_lon + 1) * (lon_max - lon_min) / 2 + lon_min
    return lat, lon

class GeoGuessCityDataset(Dataset):
    def __init__(self, base_folder, transform=None):
        self.base_folder = base_folder
        self.transform = transform
        self.metadata = self.load_all_metadata()

    def load_all_metadata(self):
        all_metadata = []
        for state_folder in os.listdir(self.base_folder):
            state_folder_path = os.path.join(self.base_folder, state_folder)
            if os.path.isdir(state_folder_path):
                for city_folder in os.listdir(state_folder_path):
                    city_folder_path = os.path.join(state_folder_path, city_folder)
                    if os.path.isdir(city_folder_path):
                        city_metadata_file = os.path.join(city_folder_path, 'metadata.csv')
                        if os.path.exists(city_metadata_file):
                            city_metadata = pd.read_csv(city_metadata_file, header=None, names=['ID', 'Latitude', 'Longitude', 'State', 'City', 'File Path'])
                            all_metadata.append(city_metadata)
        return pd.concat(all_metadata, ignore_index=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.iloc[idx]['File Path']
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found, skipping: {img_path}")
            return None

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        latitude = float(self.metadata.iloc[idx]['Latitude'])
        longitude = float(self.metadata.iloc[idx]['Longitude'])
        normalized_lat, normalized_lon = normalize_coordinates(latitude, longitude)
        coordinates = torch.tensor([normalized_lat, normalized_lon], dtype=torch.float32)

        region = assign_region(self.metadata.iloc[idx]['State'])
        if region is None:
            print(f"Warning: Unknown state found, skipping: {self.metadata.iloc[idx]['State']}")
            return None

        return {'image': image, 'region': torch.tensor(region, dtype=torch.long), 'labels': coordinates}

class GeoGuessAugmentedDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.metadata = pd.read_csv(csv_file, header=None, names=['ID', 'Latitude', 'Longitude', 'State', 'City', 'File Path'])
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.iloc[idx]['File Path']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        latitude = self.metadata.iloc[idx]['Latitude']
        longitude = self.metadata.iloc[idx]['Longitude']
        normalized_lat, normalized_lon = normalize_coordinates(latitude, longitude)
        coordinates = torch.tensor([normalized_lat, normalized_lon], dtype=torch.float32)

        region = assign_region(self.metadata.iloc[idx]['State'])
        if region is None:
            print(f"Warning: Unknown state found, skipping: {self.metadata.iloc[idx]['State']}")
            return None

        return {'image': image, 'region': torch.tensor(region, dtype=torch.long), 'labels': coordinates}

region_map = {
    'West': ['WA', 'OR', 'CA', 'NV', 'ID', 'MT', 'WY', 'UT', 'CO', 'AK', 'HI'],
    'Midwest': ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH'],
    'South': ['TX', 'OK', 'AR', 'LA', 'MS', 'AL', 'GA', 'FL', 'TN', 'KY', 'WV', 'VA', 'SC', 'NC'],
    'Northeast': ['PA', 'NJ', 'NY', 'CT', 'RI', 'MA', 'VT', 'NH', 'ME', 'DE', 'MD', 'DC'],
    'Southwest': ['AZ', 'NM', 'NV', 'UT'],
    'Northwest': ['WA', 'OR', 'ID', 'MT', 'WY']
}
region_labels = {'West': 0, 'Midwest': 1, 'South': 2, 'Northeast': 3, 'Southwest': 4, 'Northwest': 5}

region_bounds = {
    'West': {'lat_min': 32.0, 'lat_max': 49.0, 'lon_min': -125.0, 'lon_max': -114.0},
    'Midwest': {'lat_min': 36.5, 'lat_max': 49.0, 'lon_min': -105.0, 'lon_max': -80.0},
    'South': {'lat_min': 24.5, 'lat_max': 36.5, 'lon_min': -106.5, 'lon_max': -75.0},
    'Northeast': {'lat_min': 38.0, 'lat_max': 47.5, 'lon_min': -80.0, 'lon_max': -66.0},
    'Southwest': {'lat_min': 31.0, 'lat_max': 37.0, 'lon_min': -114.0, 'lon_max': -103.0},
    'Northwest': {'lat_min': 41.0, 'lat_max': 49.0, 'lon_min': -125.0, 'lon_max': -110.0}
}

def assign_region(state):
    for region, states in region_map.items():
        if state in states:
            return region_labels[region]
    return None

def constrain_coordinates_to_region(region, lat, lon):
    bounds = region_bounds[region]
    lat_min, lat_max = bounds['lat_min'], bounds['lat_max']
    lon_min, lon_max = bounds['lon_min'], bounds['lon_max']

    lat = torch.clamp(lat, lat_min, lat_max)
    lon = torch.clamp(lon, lon_min, lon_max)

    return lat, lon

class GeoGuessNet(nn.Module):
    def __init__(self, num_regions):
        super(GeoGuessNet, self).__init__()
        
        self.shared_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.shared_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.shared_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.region_fc1 = nn.Linear(128 * 80 * 80, 512)
        self.region_fc2 = nn.Linear(512, 128)
        self.region_out = nn.Linear(128, num_regions)
        
        self.coord_fc1 = nn.Linear(128 * 80 * 80, 512)
        self.coord_fc2 = nn.Linear(512, 128)
        self.coord_out = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.shared_conv1(x)))
        x = self.pool(F.relu(self.shared_conv2(x)))
        x = self.pool(F.relu(self.shared_conv3(x)))
        
        x_flat = x.view(-1, 128 * 80 * 80)
        
        region_x = F.relu(self.region_fc1(x_flat))
        region_x = F.relu(self.region_fc2(region_x))
        region_out = self.region_out(region_x)
        
        coord_x = F.relu(self.coord_fc1(x_flat))
        coord_x = F.relu(self.coord_fc2(coord_x))
        coord_out = self.coord_out(coord_x)
        
        return region_out, coord_out

def custom_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    images = torch.stack([b['image'] for b in batch])
    region_labels = torch.tensor([b['region'] for b in batch], dtype=torch.long)
    coord_labels = torch.stack([b['labels'] for b in batch])
    return {'images': images, 'region_labels': region_labels, 'coord_labels': coord_labels}

def train_model(model, train_loader, criterion_region, criterion_coords, optimizer, num_epochs):
    # Open the CSV file once
    with open('epoch_batch_loss_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Batch', 'Regular_Loss', 'Coordinate_Loss', 'Total_Loss', 'GPU_Memory_Allocated', 'Images Found', 'Images Skipped', 'Skipped Files'])  # Add a column for skipped files

        total_images_found = 0
        total_images_skipped = 0
        all_skipped_files = []  # This will accumulate all skipped files across batches

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            valid_batches = 0
            epoch_images_found = 0
            epoch_images_skipped = 0

            for batch_idx, batch in enumerate(train_loader):
                if batch is None:
                    epoch_images_skipped += train_loader.batch_size  # Track skipped images
                    continue  # Skip empty batches
                
                skipped_files = []  # List to track skipped image paths in the current batch
                valid_images = []
                valid_region_labels = []
                valid_coord_labels = []

                # Process each item in the batch
                for i, item in enumerate(batch['images']):
                    if batch['images'][i] is not None:
                        valid_images.append(batch['images'][i])
                        valid_region_labels.append(batch['region_labels'][i])
                        valid_coord_labels.append(batch['coord_labels'][i])
                    else:
                        skipped_files.append(batch['file_paths'][i])  # Track the file path of the skipped image

                # Add skipped files to the global list
                all_skipped_files.extend(skipped_files)

                if len(valid_images) == 0:
                    # Skip this batch entirely if no valid images
                    epoch_images_skipped += len(batch['images'])
                    continue

                valid_batches += 1
                images = torch.stack(valid_images).to(device)
                region_labels = torch.stack(valid_region_labels).to(device)
                coord_labels = torch.stack(valid_coord_labels).to(device)

                # Track number of images processed in this batch
                epoch_images_found += len(images)
                epoch_images_skipped += len(skipped_files)

                # Forward pass
                region_preds, coord_preds = model(images)

                # Get predicted regions
                predicted_regions = torch.argmax(region_preds, dim=1).cpu().numpy()

                # Constrain predicted coordinates based on predicted region
                constrained_coords = []
                for i, predicted_region in enumerate(predicted_regions):
                    region_name = list(region_map.keys())[predicted_region]
                    norm_lat, norm_lon = coord_preds[i]
                    lat, lon = denormalize_coordinates(norm_lat, norm_lon)
                    constrained_lat, constrained_lon = constrain_coordinates_to_region(region_name, lat, lon)
                    constrained_coords.append([constrained_lat, constrained_lon])

                constrained_coords = torch.tensor(constrained_coords, dtype=torch.float32).to(device)

                # Calculate losses
                loss_region = criterion_region(region_preds, region_labels)
                loss_coords = criterion_coords(constrained_coords, coord_labels)

                # Combine losses
                total_loss = loss_region + loss_coords

                # Backward pass and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()

                # GPU memory allocation check
                gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # Convert to GB

                # Write batch-level losses, GPU memory usage, and total skipped files to CSV
                writer.writerow([epoch + 1, batch_idx + 1, loss_region.item(), loss_coords.item(), total_loss.item(), gpu_memory_allocated, len(images), len(all_skipped_files), ",".join(all_skipped_files)])

                # Print loss for each batch with GPU memory usage
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {total_loss.item()}, GPU Memory: {gpu_memory_allocated:.2f} GB, Skipped Files: {len(skipped_files)}")

            total_images_found += epoch_images_found
            total_images_skipped += epoch_images_skipped

            # Calculate average loss for the epoch
            epoch_loss = running_loss / valid_batches
            print(f"Epoch [{epoch+1}/{num_epochs}] - Found: {epoch_images_found}, Skipped: {epoch_images_skipped}, Avg Loss: {epoch_loss:.4f}")

        # Final summary of all images found and skipped
        print(f"Training complete. Total images found: {total_images_found}, Total images skipped: {total_images_skipped}")

def train_on_regular_images(model, base_folder, num_epochs):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    geo_dataset = GeoGuessCityDataset(base_folder=base_folder, transform=transform)
    train_loader = DataLoader(geo_dataset, batch_size=50, shuffle=True, num_workers=4, collate_fn=custom_collate)

    criterion_region = nn.CrossEntropyLoss()
    criterion_coords = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    regular_losses = train_model(model, train_loader, criterion_region, criterion_coords, optimizer, num_epochs)

    print(f"Training on regular images complete. Moving to augmented data training.")
    
    return regular_losses

def train_on_augmented_images(model, augmented_metadata_file, num_epochs):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    augmented_dataset = GeoGuessAugmentedDataset(csv_file=augmented_metadata_file, transform=transform)
    train_loader = DataLoader(augmented_dataset, batch_size=50, shuffle=True, num_workers=4, collate_fn=custom_collate)

    criterion_region = nn.CrossEntropyLoss()
    criterion_coords = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    augmented_losses = train_model(model, train_loader, criterion_region, criterion_coords, optimizer, num_epochs)

    print(f"Training on augmented images complete.")
    
    return augmented_losses

if __name__ == '__main__':
    model = GeoGuessNet(num_regions=6)
    model.to(device)
    base_folder = 'data/images'
    augmented_metadata_file = 'data/images/augmented_global_metadata.csv'

    print("Training on regular images...")
    regular_losses = train_on_regular_images(model, base_folder, num_epochs=20)

    print("Training on augmented images...")
    augmented_losses = train_on_augmented_images(model, augmented_metadata_file, num_epochs=20)

    torch.save(model.state_dict(), 'geoguess_model.pth')
    print("Model saved successfully after both training phases!")
