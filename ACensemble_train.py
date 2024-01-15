import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
import kerastuner as kt
from sklearn.model_selection import train_test_split

# Set the dataset directory
dataset_dir = 'dataset/'

# Function to load training data
def load_trainingdata():
    TOA_xtrain_path = os.path.join(dataset_dir, 'TOA_xtrain.npy')
    angles_xtrain_path = os.path.join(dataset_dir, 'angles_xtrain.npy')
    AOT_xtrain_path = os.path.join(dataset_dir, 'AOT_xtrain.npy')
    y_trainiCOR_path = os.path.join(dataset_dir, 'y_trainiCOR.npy')

    TOA_xtrain = np.load(TOA_xtrain_path)
    angles_xtrain = np.load(angles_xtrain_path)
    AOT_xtrain = np.load(AOT_xtrain_path)
    ytrain_iCOR = np.load(y_trainiCOR_path)

    # Similarly, load validation data

    return TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR

# Function to load testing data
def load_testingdata():
    TOA_xtesting_path = os.path.join(dataset_dir, 'TOA_xtesting.npy')
    angles_xtesting_path = os.path.join(dataset_dir, 'angles_xtesting.npy')
    AOT_xtesting_path = os.path.join(dataset_dir, 'AOT_xtesting.npy')

    TOA_xtesting = np.load(TOA_xtesting_path)
    angles_xtesting = np.load(angles_xtesting_path)
    AOT_xtesting = np.load(AOT_xtesting_path)

    # Similarly, load other testing data

    return TOA_xtesting, angles_xtesting, AOT_xtesting

# Load training data
TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR = load_trainingdata()

# Load testing data
TOA_xtesting, angles_xtesting, AOT_xtesting = load_testingdata()

# Convert NumPy arrays to PyTorch tensors
TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR, TOA_xvali, angles_xvali, AOT_xvali, y_vali_iCOR = map(torch.tensor, 
                                                                                                    [TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR, TOA_xvali, angles_xvali, AOT_xvali, y_vali_iCOR])

# Combine inputs and targets into a PyTorch dataset
train_dataset = TensorDataset(TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR)
val_dataset = TensorDataset(TOA_xvali, angles_xvali, AOT_xvali, y_vali_iCOR)

# DataLoader for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Define the model architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv2 = nn.Conv3d(16, 16, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3 = nn.Conv3d(16, 16, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(288, 324)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(324, 400)  # You can replace these sizes with the best hyperparameters
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(400, 300)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(300, 200)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(200, 100)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(100, 5)

    def forward(self, TOA_input, angles_input, AOT_input):
        x = F.relu(self.conv1(TOA_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.reshape(x, (x.size(0), -1))
        x = torch.cat([x, angles_input, AOT_input], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)
        x = F.sigmoid(self.fc6(x))
        return x

# Instantiate the model
model = MyModel()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adjust the learning rate

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    for TOA_batch, angles_batch, AOT_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(TOA_batch, angles_batch, AOT_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for TOA_batch, angles_batch, AOT_batch, y_batch in val_loader:
            val_outputs = model(TOA_batch, angles_batch, AOT_batch)
            val_loss = criterion(val_outputs, y_batch)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'my_model.pth')

# Now you can use the trained model for predictions