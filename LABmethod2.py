#For creating L and ab Channels where we'll store the images in this manner
def create_training_data(image_paths):
    L_channels = []
    ab_channels = []
    
#addding the new L and ab images after every image and gets converted to lists
    for image_path in image_paths:
 L_channels, ab_channels = extract_lab_channels(image_path)
        L_channels.append(L_channel)
        ab_channels.append(ab_channel)
    
    # Convert lists to numpy arrays
    L_channels = np.array(L_channels)
    ab_channels = np.array(ab_channels)
    
    return L_channels, ab_channels




##For creating the Model

import torch
import torch.nn as nn
import torch.optim as optim


class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.conv12(x)
        return x


# Initialize the model
model = ColorizationNet()

#For Training the Model
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Convert data to PyTorch tensors
L_train, ab_train = torch.from_numpy(L_channels).unsqueeze(1).float(), torch.from_numpy(ab_channels).float()


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(L_train)
    
    # Compute loss
    loss = criterion(outputs, ab_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

