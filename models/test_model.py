import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the region map and labels
region_map = {
    'West': ['WA', 'OR', 'CA', 'NV', 'ID', 'MT', 'WY', 'UT', 'CO', 'AK', 'HI'],
    'Midwest': ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH'],
    'South': ['TX', 'OK', 'AR', 'LA', 'MS', 'AL', 'GA', 'FL', 'TN', 'KY', 'WV', 'VA', 'SC', 'NC'],
    'Northeast': ['PA', 'NJ', 'NY', 'CT', 'RI', 'MA', 'VT', 'NH', 'ME', 'DE', 'MD', 'DC'],
    'Southwest': ['AZ', 'NM', 'NV', 'UT'],
    'Northwest': ['WA', 'OR', 'ID', 'MT', 'WY']
}
region_labels = {'West': 0, 'Midwest': 1, 'South': 2, 'Northeast': 3, 'Southwest': 4, 'Northwest': 5}

# Function to assign regions based on state
def assign_region(state):
    for region, states in region_map.items():
        if state in states:
            return region_labels[region]
    return None  # Return None if state not found in the map

# Define the model class
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
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x_flat = torch.flatten(x, 1)  # Dynamic flattening

        # Region classification branch
        region_x = nn.functional.relu(self.fc1(x_flat))
        region_x = nn.functional.relu(self.fc2(region_x))
        region_out = self.region_out(region_x)
        
        # Coordinate regression branch
        coord_x = nn.functional.relu(self.coord_fc1(x_flat))
        coord_x = nn.functional.relu(self.coord_fc2(coord_x))
        coord_out = self.coord_out(coord_x)
        
        return region_out, coord_out

# Load the trained model
def load_model():
    model = GeoGuessNet(num_regions=6)  # Adjust num_regions as needed
    model.load_state_dict(torch.load('geoguess_model.pth'))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess and predict for a single image
def predict_location(image_path, model):
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to GPU/CPU

    # Run the image through the model
    with torch.no_grad():
        region_pred, coord_pred = model(image)
    
    # Convert region prediction to human-readable format
    region_idx = torch.argmax(region_pred, dim=1).item()
    for region_name, region_label in region_labels.items():
        if region_label == region_idx:
            predicted_region = region_name
            break
    
    # Get the predicted coordinates (latitude and longitude)
    predicted_coords = coord_pred.squeeze().cpu().numpy()
    
    return predicted_region, predicted_coords

# Main function to run prediction
def main():
    # Path to the test image
    image_path = "data/images/CT/Hartford/street_view_106727.jpg"  # Replace with your image path
    
    # Load the trained model
    model = load_model()
    
    # Make predictions for the image
    predicted_region, predicted_coords = predict_location(image_path, model)
    
    if predicted_region and predicted_coords is not None:
        print(f"Predicted Region: {predicted_region}")
        print(f"Predicted Coordinates: Latitude {predicted_coords[0]}, Longitude {predicted_coords[1]}")

if __name__ == '__main__':
    main()
