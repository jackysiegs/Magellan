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

# Dataset class for city-specific metadata (regular images)
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
        
        # Check if the file exists, and skip if it doesn't
        if not os.path.exists(img_path):
            print(f"Warning: Image not found, skipping: {img_path}")
            return None  # This will skip the image

        # Load the image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Ensure latitude and longitude are floats
        latitude = float(self.metadata.iloc[idx]['Latitude'])
        longitude = float(self.metadata.iloc[idx]['Longitude'])
        coordinates = torch.tensor([latitude, longitude], dtype=torch.float32)

        # Get the region (from the State)
        region = assign_region(self.metadata.iloc[idx]['State'])
        if region is None:  # Skip if the region is unknown
            print(f"Warning: Unknown state found, skipping: {self.metadata.iloc[idx]['State']}")
            return None

        return {'image': image, 'region': torch.tensor(region, dtype=torch.long), 'labels': coordinates}

# Dataset class for augmented images
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

        # Labels
        latitude = self.metadata.iloc[idx]['Latitude']
        longitude = self.metadata.iloc[idx]['Longitude']
        coordinates = torch.tensor([latitude, longitude], dtype=torch.float32)

        # Get the region (from the State)
        region = assign_region(self.metadata.iloc[idx]['State'])
        if region is None:  # Skip if the region is unknown
            print(f"Warning: Unknown state found, skipping: {self.metadata.iloc[idx]['State']}")
            return None

        return {'image': image, 'region': torch.tensor(region, dtype=torch.long), 'labels': coordinates}


# Function to assign regions based on the state
region_map = {
    'West': ['WA', 'OR', 'CA', 'NV', 'ID', 'MT', 'WY', 'UT', 'CO', 'AK', 'HI'],
    'Midwest': ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH'],
    'South': ['TX', 'OK', 'AR', 'LA', 'MS', 'AL', 'GA', 'FL', 'TN', 'KY', 'WV', 'VA', 'SC', 'NC'],
    'Northeast': ['PA', 'NJ', 'NY', 'CT', 'RI', 'MA', 'VT', 'NH', 'ME', 'DE', 'MD', 'DC'],
    'Southwest': ['AZ', 'NM', 'NV', 'UT'],
    'Northwest': ['WA', 'OR', 'ID', 'MT', 'WY']
}
region_labels = {'West': 0, 'Midwest': 1, 'South': 2, 'Northeast': 3, 'Southwest': 4, 'Northwest': 5}

def assign_region(state):
    for region, states in region_map.items():
        if state in states:
            return region_labels[region]
    return None  # Return None instead of -1 for unknown states


# Model definition for region classification and coordinate prediction
class GeoGuessNet(nn.Module):
    def __init__(self, num_regions):
        super(GeoGuessNet, self).__init__()
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers for region classification
        self.fc1 = nn.Linear(128 * 80 * 80, 512)
        self.fc2 = nn.Linear(512, 128)
        self.region_out = nn.Linear(128, num_regions)  # Output for regions
        
        # Fully connected layers for coordinate regression
        self.coord_fc1 = nn.Linear(128 * 80 * 80, 512)
        self.coord_fc2 = nn.Linear(512, 128)
        self.coord_out = nn.Linear(128, 2)  # Output latitude and longitude

    def forward(self, x):
        # Shared convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x_flat = x.view(-1, 128 * 80 * 80)
        
        # Region classification branch
        region_x = F.relu(self.fc1(x_flat))
        region_x = F.relu(self.fc2(region_x))
        region_out = self.region_out(region_x)
        
        # Coordinate regression branch
        coord_x = F.relu(self.coord_fc1(x_flat))
        coord_x = F.relu(self.coord_fc2(coord_x))
        coord_out = self.coord_out(coord_x)
        
        return region_out, coord_out

# Training function
def custom_collate(batch):
    # Filter out None values caused by missing images or unknown regions
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    images = torch.stack([b['image'] for b in batch])
    region_labels = torch.tensor([b['region'] for b in batch], dtype=torch.long)
    coord_labels = torch.stack([b['labels'] for b in batch])
    return {'images': images, 'region_labels': region_labels, 'coord_labels': coord_labels}

# Updated train_model function with the CSV file open/close logic outside the loop
def train_model(model, train_loader, criterion_region, criterion_coords, optimizer, num_epochs):
    # Open the CSV file once
    with open('epoch_batch_loss_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Batch', 'Regular_Loss', 'Coordinate_Loss', 'Total_Loss', 'GPU_Memory_Allocated'])  # Add headers

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            valid_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                if batch is None:
                    continue  # Skip empty batches
                valid_batches += 1

                images = batch['images'].to(device)
                region_labels = batch['region_labels'].to(device)
                coord_labels = batch['coord_labels'].to(device)

                # Forward pass
                region_preds, coord_preds = model(images)

                # Calculate losses
                loss_region = criterion_region(region_preds, region_labels)
                loss_coords = criterion_coords(coord_preds, coord_labels)

                # Combine losses
                total_loss = loss_region + loss_coords

                # Backward pass and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()

                # GPU memory allocation check
                gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # Convert to GB

                # Write batch-level losses and GPU memory usage to CSV
                writer.writerow([epoch + 1, batch_idx + 1, loss_region.item(), loss_coords.item(), total_loss.item(), gpu_memory_allocated])

                # Print loss for each batch with GPU memory usage
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {total_loss.item()}, GPU Memory: {gpu_memory_allocated:.2f} GB")

            # Calculate average loss for the epoch
            epoch_loss = running_loss / valid_batches
            print(f"Epoch [{epoch+1}/{num_epochs}] finished with Average Loss: {epoch_loss}")


        # CSV file will close automatically when exiting the "with" block


# Updated train_on_regular_images to return losses
def train_on_regular_images(model, base_folder, num_epochs):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    geo_dataset = GeoGuessCityDataset(base_folder=base_folder, transform=transform)
    train_loader = DataLoader(geo_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=custom_collate)

    # Loss functions and optimizer
    criterion_region = nn.CrossEntropyLoss()
    criterion_coords = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Return the loss from the training process
    return train_model(model, train_loader, criterion_region, criterion_coords, optimizer, num_epochs)


# Train on augmented images (Phase 2)
def train_on_augmented_images(model, augmented_metadata_file, num_epochs):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    augmented_dataset = GeoGuessAugmentedDataset(csv_file=augmented_metadata_file, transform=transform)
    train_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=custom_collate)

    # Loss functions and optimizer
    criterion_region = nn.CrossEntropyLoss()
    criterion_coords = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    train_model(model, train_loader, criterion_region, criterion_coords, optimizer, num_epochs)

# Main training function
if __name__ == '__main__':
    model = GeoGuessNet(num_regions=6)
    model.to(device)
    base_folder = 'data/images'  # Path to regular images folder
    augmented_metadata_file = 'data/images/augmented_global_metadata.csv'  # Path to augmented images metadata

    # # Train on regular images (Phase 1)
    print("Training on regular images...")
    regular_losses = train_on_regular_images(model, base_folder, num_epochs=20)

    print("Training on augmented images...")
    augmented_losses = train_on_augmented_images(model, augmented_metadata_file, num_epochs=20)

    # Save the model after both phases
    torch.save(model.state_dict(), 'geoguess_model.pth')
    print("Model saved succesfully!")
