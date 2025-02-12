import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet34

# GeoGuessNet model definition
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
        self.coord_out = nn.Linear(64, 2)  # Output latitude and longitude

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

# Helper function to denormalize predicted coordinates
def denormalize_coordinates(coord_tensor):
    latitude = coord_tensor[0] * 180 - 90  # Convert from [0, 1] back to [-90, 90]
    longitude = coord_tensor[1] * 360 - 180  # Convert from [0, 1] back to [-180, 180]
    return latitude, longitude

# Load the trained model
def load_trained_model(model_path, num_regions):
    model = GeoGuessNet(num_regions=num_regions)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()  # Set to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

# Preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to get the model's prediction for region and coordinates
def predict_region_and_coordinates(model, image_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():  # Disable gradient calculation for inference
        region_preds, coord_preds = model(image_tensor)
        
        # Get predicted region
        region_probs = F.softmax(region_preds, dim=1)
        predicted_region = torch.argmax(region_probs, dim=1).item()
        
        # Denormalize predicted coordinates
        predicted_coords = coord_preds.squeeze(0).cpu()
        latitude, longitude = denormalize_coordinates(predicted_coords)
        
    return predicted_region, latitude, longitude

# Example usage
if __name__ == "__main__":
    # Path to your trained model and test image
    model_path = 'best_model.pth'
    test_image_path = 'data/images/CA/Mount Shasta/street_view_106822.jpg'
    # Load the model
    model = load_trained_model(model_path, num_regions=6)  # Assuming 6 regions
    
    # Preprocess the test image
    image_tensor = preprocess_image(test_image_path)
    
    # Get predictions
    predicted_region, predicted_latitude, predicted_longitude = predict_region_and_coordinates(model, image_tensor)
    
    # Output the results
    print(f"Predicted Region: {predicted_region}")
    print(f"Predicted Coordinates: Latitude = {predicted_latitude}, Longitude = {predicted_longitude}")
