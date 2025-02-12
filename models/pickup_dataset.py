import csv
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms
from torchvision.models import resnet34
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Normalize latitude and longitude to [0, 1]
        latitude_normalized = (latitude + 90) / 180  # Latitude from [-90, 90] to [0, 1]
        longitude_normalized = (longitude + 180) / 360  # Longitude from [-180, 180] to [0, 1]

        coordinates = torch.tensor([latitude_normalized, longitude_normalized], dtype=torch.float32)

        # Get the region (from the State)
        region = assign_region(self.metadata.iloc[idx]['State'])
        if region is None:  # Skip if the region is unknown
            print(f"Warning: Unknown state found, skipping: {self.metadata.iloc[idx]['State']}")
            return None

        return {'image': image, 'region': torch.tensor(region, dtype=torch.long), 'labels': coordinates}

# GeoGuessNet model architecture remains the same
class GeoGuessNet(nn.Module):
    def __init__(self, num_regions):
        super(GeoGuessNet, self).__init__()

        # Use a pre-trained ResNet34 backbone
        self.resnet = resnet34(pretrained=True)

        # Remove the final fully connected layer of ResNet34 and keep the feature extractor
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # Adjust for the 512 * 20 * 20 feature map from ResNet34 with 640x640 inputs
        self.fc_region = nn.Linear(512 * 20 * 20, 128)
        self.bn_region = nn.BatchNorm1d(128)  # Apply batch normalization
        self.region_out = nn.Linear(128, num_regions)

        # Coordinate regression that takes region as an additional input
        self.coord_fc1 = nn.Linear(512 * 20 * 20 + num_regions, 128)
        self.bn_coord = nn.BatchNorm1d(128)
        self.coord_fc2 = nn.Linear(128, 64)
        self.coord_out = nn.Linear(64, 2)  # Output normalized latitude and longitude

    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        region_x = F.relu(self.bn_region(self.fc_region(features)))
        region_out = self.region_out(region_x)
        region_probs = F.softmax(region_out, dim=1)
        coord_input = torch.cat([features, region_probs], dim=1)
        coord_x = F.relu(self.bn_coord(self.coord_fc1(coord_input)))
        coord_x = F.relu(self.coord_fc2(coord_x))
        coord_out = self.coord_out(coord_x)
        return region_out, coord_out

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

# Loading the model from the saved state
def load_model(model_path, num_regions):
    model = GeoGuessNet(num_regions=num_regions)

    # Load the saved model weights
    model.load_state_dict(torch.load(model_path))

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model

def unfreeze_resnet_layers(model):
    for param in model.resnet.parameters():
        param.requires_grad = True

def custom_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    images = torch.stack([b['image'] for b in batch])
    region_labels = torch.tensor([b['region'] for b in batch], dtype=torch.long)
    coord_labels = torch.stack([b['labels'] for b in batch])
    return {'images': images, 'region_labels': region_labels, 'coord_labels': coord_labels}

# Function to resume training on augmented images (Phase 3)
def resume_training_on_augmented_images(model, augmented_metadata_file, criterion_region, criterion_coords, optimizer, num_epochs, csv_file_name, starting_epoch=1):
    unfreeze_resnet_layers(model)  # Unfreeze the ResNet layers for fine-tuning on augmented images

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    augmented_dataset = GeoGuessAugmentedDataset(csv_file=augmented_metadata_file, transform=transform)
    train_loader = DataLoader(augmented_dataset, batch_size=64, shuffle=True, num_workers=4, collate_fn=custom_collate)

    # Resume training from the saved model state
    with open(csv_file_name, 'a', newline='') as f:
        writer = csv.writer(f)

        best_loss = float('inf')
        patience = 5
        epochs_without_improvement = 0
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        for epoch in range(starting_epoch, num_epochs + 1):
            model.train()
            running_loss = 0.0
            valid_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                if batch is None:
                    continue
                valid_batches += 1

                images = batch['images'].to(device)
                region_labels = batch['region_labels'].to(device)
                coord_labels = batch['coord_labels'].to(device)

                region_preds, coord_preds = model(images)
                loss_region = criterion_region(region_preds, region_labels)
                loss_coords = criterion_coords(coord_preds, coord_labels)

                total_loss = 0.5 * loss_region + 1.0 * loss_coords

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += total_loss.item()

                gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
                writer.writerow([epoch, batch_idx + 1, loss_region.item(), loss_coords.item(), total_loss.item(), gpu_memory_allocated])

                print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {total_loss.item()}, GPU Memory: {gpu_memory_allocated:.2f} GB")

            epoch_loss = running_loss / valid_batches
            print(f"Epoch [{epoch}/{num_epochs}] finished with Average Loss: {epoch_loss}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), 'best_model.pth')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            scheduler.step()

# Main function to resume training on augmented images
if __name__ == '__main__':
    # Define parameters
    num_regions = 6  # Number of regions in your model
    model_path = 'best_model.pth'  # Path to the saved model
    augmented_metadata_file = 'data/images/augmented_global_metadata.csv'  # Path to augmented metadata
    csv_file_name = 'aug_epoch_batch_loss_resume.csv'
    num_epochs = 20  # Number of epochs to train augmented images

    # Load the model
    model = load_model(model_path, num_regions)

    # Define loss functions and optimizer
    criterion_region = nn.CrossEntropyLoss()
    criterion_coords = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Resume training from the first epoch of training on augmented images
    resume_training_on_augmented_images(model, augmented_metadata_file, criterion_region, criterion_coords, optimizer, num_epochs, csv_file_name, starting_epoch=1)
